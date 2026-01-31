//! Streaming Inference HTTP Server
//!
//! Provides HTTP endpoints for streaming LLM inference with:
//! - Server-Sent Events (SSE) for unidirectional streaming
//! - WebSocket support for bidirectional communication
//! - Bearer token authentication
//! - OpenAI-compatible API endpoints
//! - Custom ABI endpoints
//! - Admin endpoints for model hot-reload
//!
//! Endpoints:
//! - POST /v1/chat/completions - OpenAI-compatible chat completions
//! - POST /api/stream - Custom ABI streaming endpoint
//! - GET  /api/stream/ws - WebSocket upgrade for streaming
//! - GET  /health - Health check endpoint
//! - GET  /metrics - Metrics snapshot for dashboards
//! - POST /admin/reload - Hot-reload model without server restart

const std = @import("std");
const mod = @import("mod.zig");
const sse = @import("sse.zig");
const websocket = @import("websocket.zig");
const backends = @import("backends/mod.zig");
const formats = @import("formats/mod.zig");
const shared_utils = @import("../../shared/utils.zig");
const observability = @import("../../observability/mod.zig");
const net_utils = shared_utils.net;
const json_utils = shared_utils.json;
const recovery = @import("recovery.zig");
const RecoveryConfig = recovery.RecoveryConfig;
const RecoveryEvent = recovery.RecoveryEvent;

pub const StreamingServerError = std.mem.Allocator.Error || error{
    InvalidAddress,
    InvalidRequest,
    Unauthorized,
    BackendError,
    StreamError,
    WebSocketError,
    RequestTooLarge,
    UnsupportedBackend,
    ModelReloadFailed,
    ModelReloadTimeout,
    CircuitBreakerOpen,
};

const max_body_bytes = 1024 * 1024; // 1MB max request body
const heartbeat_interval_ms: u64 = 15000; // 15 second heartbeats
const admin_reload_drain_timeout_ns: u64 = 30_000_000_000; // 30 second drain timeout
const admin_reload_poll_interval_ns: u64 = 100_000_000; // 100ms poll interval

/// Server configuration
pub const ServerConfig = struct {
    /// Listen address (e.g., "127.0.0.1:8080")
    address: []const u8 = "127.0.0.1:8080",
    /// Bearer token for authentication (null = no auth required)
    auth_token: ?[]const u8 = null,
    /// Allow health endpoint without auth
    allow_health_without_auth: bool = true,
    /// Default backend for inference
    default_backend: backends.BackendType = .local,
    /// Heartbeat interval in milliseconds (0 = disabled)
    heartbeat_interval_ms: u64 = heartbeat_interval_ms,
    /// Maximum concurrent streams
    max_concurrent_streams: u32 = 100,
    /// Enable OpenAI-compatible endpoints
    enable_openai_compat: bool = true,
    /// Enable WebSocket support
    enable_websocket: bool = true,
    /// Path to default local model (optional, for local backend)
    default_model_path: ?[]const u8 = null,
    /// Pre-load model on server start (reduces first-request latency)
    preload_model: bool = false,
    /// Enable error recovery (circuit breakers, retry, session caching)
    enable_recovery: bool = true,
    /// Recovery configuration (only used if enable_recovery is true)
    recovery_config: RecoveryConfig = .{},
};

/// Streaming inference server
pub const StreamingServer = struct {
    allocator: std.mem.Allocator,
    config: ServerConfig,
    backend_router: backends.BackendRouter,
    active_streams: std.atomic.Value(u32),
    start_time_ms: i64,

    const Self = @This();

    /// Initialize the streaming server
    ///
    /// If `config.default_model_path` is set and `config.preload_model` is true,
    /// the model will be loaded during initialization. Otherwise, the model
    /// will be loaded lazily on the first request.
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !Self {
        // Initialize backend router with optional recovery support
        const recovery_cfg: ?RecoveryConfig = if (config.enable_recovery)
            config.recovery_config
        else
            null;
        var backend_router = try backends.BackendRouter.initWithRecovery(allocator, recovery_cfg);
        errdefer backend_router.deinit();

        // Configure local backend with model path if provided
        if (config.default_model_path) |model_path| {
            const local_backend = try backend_router.getBackend(.local);

            // Store path for lazy loading or preload immediately
            if (config.preload_model) {
                try local_backend.impl.local.loadModel(model_path);
            } else {
                // Store path for lazy loading on first request
                if (local_backend.impl.local.model_path) |old_path| {
                    allocator.free(old_path);
                }
                local_backend.impl.local.model_path = try allocator.dupe(u8, model_path);
            }
        }

        return .{
            .allocator = allocator,
            .config = config,
            .backend_router = backend_router,
            .active_streams = std.atomic.Value(u32).init(0),
            .start_time_ms = shared_utils.unixMs(),
        };
    }

    /// Deinitialize
    pub fn deinit(self: *Self) void {
        self.backend_router.deinit();
        self.* = undefined;
    }

    /// Start the server (blocking)
    pub fn serve(self: *Self) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        const listen_addr = try self.resolveAddress(io, self.config.address);
        var server = try listen_addr.listen(io, .{ .reuse_address = true });
        defer server.deinit(io);

        std.debug.print("Streaming inference server listening on {s}\n", .{self.config.address});
        std.debug.print("  - SSE endpoint: POST /api/stream\n", .{});
        if (self.config.enable_openai_compat) {
            std.debug.print("  - OpenAI endpoint: POST /v1/chat/completions\n", .{});
        }
        if (self.config.enable_websocket) {
            std.debug.print("  - WebSocket endpoint: GET /api/stream/ws\n", .{});
        }

        while (true) {
            var stream = server.accept(io) catch |err| {
                std.debug.print("Streaming server accept error: {t}\n", .{err});
                continue;
            };
            defer stream.close(io);

            self.handleConnection(io, stream) catch |err| {
                std.debug.print("Streaming server connection error: {t}\n", .{err});
            };
        }
    }

    /// Handle a single connection
    fn handleConnection(self: *Self, io: std.Io, stream: std.Io.net.Stream) !void {
        var send_buffer: [8192]u8 = undefined;
        var recv_buffer: [8192]u8 = undefined;
        var connection_reader = stream.reader(io, &recv_buffer);
        var connection_writer = stream.writer(io, &send_buffer);
        var http_server: std.http.Server = .init(
            &connection_reader.interface,
            &connection_writer.interface,
        );

        // Create connection context for streaming
        var conn_ctx = ConnectionContext{
            .io = io,
            .stream = stream,
            .send_buffer = &send_buffer,
        };

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };

            self.dispatchRequest(&request, &conn_ctx) catch |err| {
                std.debug.print("Streaming request error: {t}\n", .{err});
                const error_body = if (err == StreamingServerError.Unauthorized)
                    "{\"error\":{\"message\":\"unauthorized\",\"type\":\"authentication_error\"}}"
                else
                    "{\"error\":{\"message\":\"internal server error\",\"type\":\"server_error\"}}";
                const status: std.http.Status = if (err == StreamingServerError.Unauthorized)
                    .unauthorized
                else
                    .internal_server_error;
                self.respondJson(&request, error_body, status) catch {};
                return;
            };
        }
    }

    /// Connection context for streaming responses
    const ConnectionContext = struct {
        io: std.Io,
        stream: std.Io.net.Stream,
        send_buffer: *[8192]u8,

        /// Write raw bytes to connection using the writer interface
        pub fn write(self: *ConnectionContext, data: []const u8) !void {
            // Get writer and write all bytes (handles partial writes)
            var writer = self.stream.writer(self.io, self.send_buffer);
            try writer.interface.writeAll(data);
        }

        /// Flush is a no-op for stream writers (writeAll ensures delivery)
        pub fn flush(_: *ConnectionContext) !void {
            // writeAll ensures all bytes are written, no explicit flush needed
        }

        /// Write SSE headers to start streaming response
        pub fn writeSseHeaders(self: *ConnectionContext) !void {
            const headers =
                "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/event-stream\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: keep-alive\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "\r\n";
            try self.write(headers);
        }
    };

    /// Send a recovery event as an SSE message.
    fn sendRecoveryEvent(self: *Self, conn_ctx: *ConnectionContext, event: RecoveryEvent) void {
        const json = event.toJson(self.allocator) catch return;
        defer self.allocator.free(json);

        // Format as SSE event
        conn_ctx.write("event: recovery\ndata: ") catch return;
        conn_ctx.write(json) catch return;
        conn_ctx.write("\n\n") catch return;
        conn_ctx.flush() catch return;
    }

    /// Check circuit breaker and find available backend.
    fn getAvailableBackend(self: *Self, preferred: backends.BackendType) !*backends.Backend {
        // Check circuit breaker if recovery is enabled
        if (!self.backend_router.isBackendAvailable(preferred)) {
            // Try to find a fallback backend
            if (self.backend_router.findAvailableBackend(preferred)) |fallback| {
                return self.backend_router.getBackend(fallback);
            }
            return error.CircuitBreakerOpen;
        }
        return self.backend_router.getBackend(preferred);
    }

    fn getMetrics(self: *Self) ?*mod.StreamingMetrics {
        const recovery_mgr = self.backend_router.getRecovery() orelse return null;
        return recovery_mgr.getMetrics();
    }

    /// Dispatch request to appropriate handler
    fn dispatchRequest(
        self: *Self,
        request: *std.http.Server.Request,
        conn_ctx: *ConnectionContext,
    ) !void {
        const target = request.head.target;
        const path = splitTarget(target).path;

        // Health endpoint
        if (std.mem.eql(u8, path, "/health")) {
            if (!self.config.allow_health_without_auth) {
                try self.validateAuth(request);
            }
            return self.respondJson(request, "{\"status\":\"ok\"}", .ok);
        }

        // Metrics endpoint
        if (std.mem.eql(u8, path, "/metrics")) {
            if (!self.config.allow_health_without_auth) {
                try self.validateAuth(request);
            }
            return self.handleMetrics(request);
        }

        // All other endpoints require auth
        try self.validateAuth(request);

        // OpenAI-compatible endpoint
        if (self.config.enable_openai_compat and std.mem.eql(u8, path, "/v1/chat/completions")) {
            return self.handleOpenAIChatCompletions(request, conn_ctx);
        }

        // Custom ABI streaming endpoint
        if (std.mem.eql(u8, path, "/api/stream")) {
            return self.handleStreamRequest(request, conn_ctx);
        }

        // WebSocket upgrade
        if (self.config.enable_websocket and std.mem.eql(u8, path, "/api/stream/ws")) {
            return self.handleWebSocketUpgrade(request, conn_ctx);
        }

        // Models list (OpenAI-compatible)
        if (self.config.enable_openai_compat and std.mem.eql(u8, path, "/v1/models")) {
            return self.handleModelsList(request);
        }

        // Admin model reload endpoint (no auth required)
        if (std.mem.eql(u8, path, "/admin/reload")) {
            return self.handleAdminReload(request);
        }

        return self.respondJson(
            request,
            "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}",
            .not_found,
        );
    }

    /// Handle OpenAI-compatible chat completions
    fn handleOpenAIChatCompletions(self: *Self, request: *std.http.Server.Request, conn_ctx: *ConnectionContext) !void {
        if (request.head.method != .POST) {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}",
                .method_not_allowed,
            );
        }

        // Parse request body
        const body = try self.readRequestBody(request);
        defer self.allocator.free(body);

        // Parse OpenAI request format
        const chat_request = try formats.openai.parseRequest(self.allocator, body);
        defer chat_request.deinit(self.allocator);

        // Check if streaming is requested
        if (chat_request.stream) {
            return self.streamOpenAIResponse(conn_ctx, chat_request);
        } else {
            return self.nonStreamingOpenAIResponse(request, chat_request);
        }
    }

    /// Stream OpenAI-format response with true SSE streaming
    ///
    /// Implements Server-Sent Events (SSE) streaming for real-time token delivery.
    /// Writes HTTP headers directly to the connection, then streams each token
    /// as a separate SSE event in OpenAI-compatible format.
    fn streamOpenAIResponse(
        self: *Self,
        conn_ctx: *ConnectionContext,
        chat_request: formats.openai.ChatCompletionRequest,
    ) !void {
        // Check concurrent stream limit
        const current = self.active_streams.fetchAdd(1, .seq_cst);
        if (current >= self.config.max_concurrent_streams) {
            _ = self.active_streams.fetchSub(1, .seq_cst);
            // For rate limit errors, we can't use respondJson as we don't have request
            // Write error response directly
            const error_response =
                "HTTP/1.1 429 Too Many Requests\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: 71\r\n" ++
                "\r\n" ++
                "{\"error\":{\"message\":\"too many concurrent streams\",\"type\":\"rate_limit_error\"}}";
            try conn_ctx.write(error_response);
            try conn_ctx.flush();
            return;
        }
        defer _ = self.active_streams.fetchSub(1, .seq_cst);

        // Get backend (with circuit breaker check if recovery enabled)
        const backend = self.getAvailableBackend(self.config.default_backend) catch |err| {
            if (err == error.CircuitBreakerOpen) {
                const error_response =
                    "HTTP/1.1 503 Service Unavailable\r\n" ++
                    "Content-Type: application/json\r\n" ++
                    "Retry-After: 30\r\n" ++
                    "Content-Length: 78\r\n" ++
                    "\r\n" ++
                    "{\"error\":{\"message\":\"backend temporarily unavailable\",\"type\":\"service_unavailable_error\"}}";
                try conn_ctx.write(error_response);
                try conn_ctx.flush();
                return;
            }
            return err;
        };
        const backend_type = backend.backend_type;
        const metrics = self.getMetrics();
        const prompt = try chat_request.formatPrompt(self.allocator);
        defer self.allocator.free(prompt);

        // Write SSE headers to start streaming
        try conn_ctx.writeSseHeaders();

        // Stream tokens as SSE events
        if (metrics) |m| {
            m.recordStreamStart(backend_type);
        }
        var stream_iter = backend.streamTokens(prompt, chat_request.toGenerationConfig()) catch |err| {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
            // Send error event to client
            try conn_ctx.write("event: error\ndata: {\"type\":\"stream_error\",\"message\":\"failed to start stream\"}\n\n");
            try conn_ctx.flush();
            return err;
        };
        defer stream_iter.deinit();

        var stream_ok = false;
        errdefer if (!stream_ok) {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
        };

        // Initialize heartbeat timer
        var heartbeat_timer = std.time.Timer.start() catch null;
        const heartbeat_interval_ns: u64 = self.config.heartbeat_interval_ms * 1_000_000;

        const stream_start_ms = shared_utils.unixMs();
        var last_token_ms = stream_start_ms;
        var token_index: u32 = 0;
        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
            // Check if heartbeat is needed (before sending token)
            if (self.config.heartbeat_interval_ms > 0) {
                if (heartbeat_timer) |*timer| {
                    if (timer.read() >= heartbeat_interval_ns) {
                        // Send SSE comment as heartbeat (keeps connection alive)
                        try conn_ctx.write(": heartbeat\n\n");
                        try conn_ctx.flush();
                        timer.reset();
                    }
                }
            }

            if (metrics) |m| {
                const now_ms = shared_utils.unixMs();
                const latency_ms: u64 = if (now_ms >= last_token_ms)
                    @intCast(now_ms - last_token_ms)
                else
                    0;
                m.recordTokenLatency(backend_type, latency_ms);
                last_token_ms = now_ms;
            }

            // Format as OpenAI streaming chunk (SSE data payload)
            const chunk_json = try formats.openai.formatStreamChunk(
                self.allocator,
                token.text,
                chat_request.model,
                token_index,
                token.is_end,
            );
            defer self.allocator.free(chunk_json);

            // Write SSE event directly (simpler format for OpenAI compat)
            // Format: "data: {json}\n\n"
            try conn_ctx.write("data: ");
            try conn_ctx.write(chunk_json);
            try conn_ctx.write("\n\n");
            try conn_ctx.flush();

            token_index += 1;
            if (token.is_end) break;
        }

        // Send final [DONE] message per OpenAI spec
        try conn_ctx.write("data: [DONE]\n\n");
        try conn_ctx.flush();

        // Record successful stream completion
        stream_ok = true;
        if (metrics) |m| {
            const end_ms = shared_utils.unixMs();
            const duration_ms: u64 = if (end_ms >= stream_start_ms)
                @intCast(end_ms - stream_start_ms)
            else
                0;
            m.recordStreamComplete(backend_type, duration_ms);
        }
        self.backend_router.recordSuccess(backend_type);
    }

    /// Non-streaming OpenAI response
    fn nonStreamingOpenAIResponse(
        self: *Self,
        request: *std.http.Server.Request,
        chat_request: formats.openai.ChatCompletionRequest,
    ) !void {
        const backend = try self.backend_router.getBackend(self.config.default_backend);
        const prompt = try chat_request.formatPrompt(self.allocator);
        defer self.allocator.free(prompt);

        // Generate complete response
        const response_text = try backend.generate(prompt, chat_request.toGenerationConfig());
        defer self.allocator.free(response_text);

        // Format as OpenAI response
        const response_body = try formats.openai.formatResponse(
            self.allocator,
            response_text,
            chat_request.model,
        );
        defer self.allocator.free(response_body);

        return self.respondJson(request, response_body, .ok);
    }

    /// Handle custom ABI streaming request with true SSE streaming
    ///
    /// Implements Server-Sent Events (SSE) streaming for the custom ABI endpoint.
    /// Each token is delivered as a separate SSE event in real-time.
    fn handleStreamRequest(self: *Self, request: *std.http.Server.Request, conn_ctx: *ConnectionContext) !void {
        if (request.head.method != .POST) {
            const error_response =
                "HTTP/1.1 405 Method Not Allowed\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: 29\r\n" ++
                "\r\n" ++
                "{\"error\":\"method not allowed\"}";
            try conn_ctx.write(error_response);
            try conn_ctx.flush();
            return;
        }

        const body = try self.readRequestBody(request);
        defer self.allocator.free(body);

        // Parse ABI request format
        const stream_request = try parseAbiStreamRequest(self.allocator, body);
        defer stream_request.deinit(self.allocator);

        // Check concurrent stream limit
        const current = self.active_streams.fetchAdd(1, .seq_cst);
        if (current >= self.config.max_concurrent_streams) {
            _ = self.active_streams.fetchSub(1, .seq_cst);
            const error_response =
                "HTTP/1.1 429 Too Many Requests\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: 35\r\n" ++
                "\r\n" ++
                "{\"error\":\"too many concurrent streams\"}";
            try conn_ctx.write(error_response);
            try conn_ctx.flush();
            return;
        }
        defer _ = self.active_streams.fetchSub(1, .seq_cst);

        // Write SSE headers to start streaming
        try conn_ctx.writeSseHeaders();

        // Initialize SSE encoder
        var sse_encoder = sse.SseEncoder.init(self.allocator, .{
            .include_id = true,
            .event_prefix = "abi.",
        });
        defer sse_encoder.deinit();

        // Get backend and stream tokens
        const backend_type = stream_request.backend orelse self.config.default_backend;
        const metrics = self.getMetrics();
        const backend = try self.backend_router.getBackend(backend_type);

        if (metrics) |m| {
            m.recordStreamStart(backend_type);
        }

        var stream_iter = backend.streamTokens(stream_request.prompt, stream_request.config) catch |err| {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
            return err;
        };
        defer stream_iter.deinit();

        var stream_ok = false;
        errdefer if (!stream_ok) {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
        };

        // Initialize heartbeat timer
        var heartbeat_timer = std.time.Timer.start() catch null;
        const heartbeat_interval_ns: u64 = self.config.heartbeat_interval_ms * 1_000_000;

        const stream_start_ms = shared_utils.unixMs();
        var last_token_ms = stream_start_ms;

        // Send start event
        const start_event = mod.StreamEvent.startEvent();
        const start_sse = try sse_encoder.encode(start_event);
        defer self.allocator.free(start_sse);
        try conn_ctx.write(start_sse);
        try conn_ctx.flush();

        // Stream tokens as SSE events
        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
            // Check if heartbeat is needed (before sending token)
            if (self.config.heartbeat_interval_ms > 0) {
                if (heartbeat_timer) |*timer| {
                    if (timer.read() >= heartbeat_interval_ns) {
                        // Send SSE heartbeat comment (keeps connection alive)
                        const heartbeat_data = try sse_encoder.encodeHeartbeat();
                        defer self.allocator.free(heartbeat_data);
                        try conn_ctx.write(heartbeat_data);
                        try conn_ctx.flush();
                        timer.reset();
                    }
                }
            }

            if (metrics) |m| {
                const now_ms = shared_utils.unixMs();
                const latency_ms: u64 = if (now_ms >= last_token_ms)
                    @intCast(now_ms - last_token_ms)
                else
                    0;
                m.recordTokenLatency(backend_type, latency_ms);
                last_token_ms = now_ms;
            }

            // Create token event with text
            const token_event = mod.StreamEvent{
                .event_type = .token,
                .token = .{
                    .id = @intCast(token.id orelse 0),
                    .text = token.text,
                    .is_end = token.is_end,
                },
            };

            // Encode and write SSE event
            const sse_data = try sse_encoder.encode(token_event);
            defer self.allocator.free(sse_data);

            try conn_ctx.write(sse_data);
            try conn_ctx.flush();

            if (token.is_end) break;
        }

        // Send end event
        const end_event = mod.StreamEvent.endEvent();
        const end_sse = try sse_encoder.encode(end_event);
        defer self.allocator.free(end_sse);
        try conn_ctx.write(end_sse);
        try conn_ctx.flush();

        stream_ok = true;
        if (metrics) |m| {
            const end_ms = shared_utils.unixMs();
            const duration_ms: u64 = if (end_ms >= stream_start_ms)
                @intCast(end_ms - stream_start_ms)
            else
                0;
            m.recordStreamComplete(backend_type, duration_ms);
        }
        self.backend_router.recordSuccess(backend_type);
    }

    /// Handle WebSocket upgrade
    fn handleWebSocketUpgrade(
        self: *Self,
        request: *std.http.Server.Request,
        conn_ctx: *ConnectionContext,
    ) !void {
        // Check for WebSocket upgrade headers
        const upgrade_header = findHeaderInBuffer(request.head_buffer, "upgrade");
        if (upgrade_header == null or !std.ascii.eqlIgnoreCase(upgrade_header.?, "websocket")) {
            return self.respondJson(
                request,
                "{\"error\":\"expected websocket upgrade\"}",
                .bad_request,
            );
        }

        // WebSocket handshake
        const ws_key = findHeaderInBuffer(request.head_buffer, "sec-websocket-key") orelse {
            return self.respondJson(
                request,
                "{\"error\":\"missing sec-websocket-key\"}",
                .bad_request,
            );
        };

        // Compute accept key
        const accept_key = try websocket.computeAcceptKey(self.allocator, ws_key);
        defer self.allocator.free(accept_key);

        // Send upgrade response
        const upgrade_headers = [_]std.http.Header{
            .{ .name = "upgrade", .value = "websocket" },
            .{ .name = "connection", .value = "upgrade" },
            .{ .name = "sec-websocket-accept", .value = accept_key },
        };

        try request.respond("", .{
            .status = .switching_protocols,
            .extra_headers = &upgrade_headers,
        });

        // Handle WebSocket connection using the existing connection context
        try self.handleWebSocketConnection(conn_ctx, self.allocator);
    }

    /// Handle WebSocket connection after upgrade
    ///
    /// Implements bidirectional streaming over WebSocket:
    /// - Client sends: `{"prompt":"...", "backend":"local", "max_tokens":100}`
    /// - Server streams: `{"type":"start|token|end|error", "data":"..."}`
    /// - Client can cancel: `{"type":"cancel"}`
    fn handleWebSocketConnection(
        self: *Self,
        conn_ctx: *ConnectionContext,
        allocator: std.mem.Allocator,
    ) !void {
        var ws_handler = try websocket.WebSocketHandler.init(allocator, .{});
        defer ws_handler.deinit();
        ws_handler.state = .open;

        // Track cancellation state across messages
        var cancel_requested = std.atomic.Value(bool).init(false);

        // Frame read buffer
        var frame_buffer: [65536]u8 = undefined;

        while (ws_handler.state == .open) {
            // Read data from connection
            var reader = conn_ctx.stream.reader(conn_ctx.io, conn_ctx.send_buffer);
            const bytes_read = reader.interface.readSliceShort(&frame_buffer) catch |err| {
                // Connection closed or error
                if (err == error.EndOfStream) break;
                if (self.config.auth_token != null) {
                    // Silently close on read errors for authenticated connections
                    break;
                }
                continue;
            };

            if (bytes_read == 0) break;

            // Parse WebSocket frame
            const parse_result = ws_handler.parseFrame(frame_buffer[0..bytes_read]) catch |err| {
                // Send protocol error and close
                const close_frame = try ws_handler.sendClose(.protocol_error, "invalid frame");
                defer allocator.free(close_frame);
                try conn_ctx.write(close_frame);
                std.debug.print("WebSocket parse error: {t}\n", .{err});
                break;
            };
            defer allocator.free(parse_result.frame.payload);

            const frame = parse_result.frame;

            switch (frame.opcode) {
                .text => {
                    // Handle JSON message
                    try self.handleWebSocketMessage(
                        conn_ctx,
                        &ws_handler,
                        allocator,
                        frame.payload,
                        &cancel_requested,
                    );
                },
                .binary => {
                    // Binary not supported for this API
                    const error_msg = try websocket.createStreamingMessage(allocator, "error", "binary messages not supported");
                    defer allocator.free(error_msg);
                    const error_frame = try ws_handler.sendText(error_msg);
                    defer allocator.free(error_frame);
                    try conn_ctx.write(error_frame);
                },
                .close => {
                    // Echo close frame and exit
                    const close_frame = try ws_handler.sendClose(.normal, "");
                    defer allocator.free(close_frame);
                    try conn_ctx.write(close_frame);
                    ws_handler.state = .closed;
                    break;
                },
                .ping => {
                    // Respond with pong
                    const pong_frame = try ws_handler.encodeFrame(.pong, frame.payload, true);
                    defer allocator.free(pong_frame);
                    try conn_ctx.write(pong_frame);
                },
                .pong => {
                    // Heartbeat response - ignore
                },
                else => {
                    // Unknown opcode
                },
            }
        }
    }

    /// Handle a WebSocket JSON message
    fn handleWebSocketMessage(
        self: *Self,
        conn_ctx: *ConnectionContext,
        ws_handler: *websocket.WebSocketHandler,
        allocator: std.mem.Allocator,
        payload: []const u8,
        cancel_requested: *std.atomic.Value(bool),
    ) !void {
        // Check for cancel message
        if (extractJsonString(payload, "type")) |msg_type| {
            if (std.mem.eql(u8, msg_type, "cancel")) {
                cancel_requested.store(true, .seq_cst);
                const cancel_msg = try websocket.createStreamingMessage(allocator, "cancelled", "");
                defer allocator.free(cancel_msg);
                const cancel_frame = try ws_handler.sendText(cancel_msg);
                defer allocator.free(cancel_frame);
                try conn_ctx.write(cancel_frame);
                return;
            }
        }

        // Parse as stream request
        const stream_request = parseAbiStreamRequest(allocator, payload) catch {
            const error_msg = try websocket.createStreamingMessage(allocator, "error", "invalid request format");
            defer allocator.free(error_msg);
            const error_frame = try ws_handler.sendText(error_msg);
            defer allocator.free(error_frame);
            try conn_ctx.write(error_frame);
            return;
        };
        defer stream_request.deinit(allocator);

        // Reset cancel flag for new request
        cancel_requested.store(false, .seq_cst);

        // Check concurrent stream limit
        const current = self.active_streams.fetchAdd(1, .seq_cst);
        if (current >= self.config.max_concurrent_streams) {
            _ = self.active_streams.fetchSub(1, .seq_cst);
            const error_msg = try websocket.createStreamingMessage(allocator, "error", "too many concurrent streams");
            defer allocator.free(error_msg);
            const error_frame = try ws_handler.sendText(error_msg);
            defer allocator.free(error_frame);
            try conn_ctx.write(error_frame);
            return;
        }
        defer _ = self.active_streams.fetchSub(1, .seq_cst);

        // Get backend and start streaming
        const backend_type = stream_request.backend orelse self.config.default_backend;
        const metrics = self.getMetrics();
        const backend = self.backend_router.getBackend(backend_type) catch {
            const error_msg = try websocket.createStreamingMessage(allocator, "error", "backend unavailable");
            defer allocator.free(error_msg);
            const error_frame = try ws_handler.sendText(error_msg);
            defer allocator.free(error_frame);
            try conn_ctx.write(error_frame);
            return;
        };

        if (metrics) |m| {
            m.recordStreamStart(backend_type);
        }

        var stream_iter = backend.streamTokens(stream_request.prompt, stream_request.config) catch {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
            const error_msg = try websocket.createStreamingMessage(allocator, "error", "failed to start stream");
            defer allocator.free(error_msg);
            const error_frame = try ws_handler.sendText(error_msg);
            defer allocator.free(error_frame);
            try conn_ctx.write(error_frame);
            return;
        };
        defer stream_iter.deinit();

        var stream_ok = false;
        errdefer if (!stream_ok) {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
        };

        const stream_start_ms = shared_utils.unixMs();
        var last_token_ms = stream_start_ms;

        // Send start event
        const start_msg = try websocket.createStreamingMessage(allocator, "start", "");
        defer allocator.free(start_msg);
        const start_frame = try ws_handler.sendText(start_msg);
        defer allocator.free(start_frame);
        try conn_ctx.write(start_frame);

        // Stream tokens
        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
            // Check for cancellation
            if (cancel_requested.load(.seq_cst)) {
                break;
            }

            if (metrics) |m| {
                const now_ms = shared_utils.unixMs();
                const latency_ms: u64 = if (now_ms >= last_token_ms)
                    @intCast(now_ms - last_token_ms)
                else
                    0;
                m.recordTokenLatency(backend_type, latency_ms);
                last_token_ms = now_ms;
            }

            // Send token event
            const token_msg = try websocket.createStreamingMessage(allocator, "token", token.text);
            defer allocator.free(token_msg);
            const token_frame = try ws_handler.sendText(token_msg);
            defer allocator.free(token_frame);
            try conn_ctx.write(token_frame);

            if (token.is_end) break;
        }

        // Send end event (unless cancelled)
        if (!cancel_requested.load(.seq_cst)) {
            const end_msg = try websocket.createStreamingMessage(allocator, "end", "");
            defer allocator.free(end_msg);
            const end_frame = try ws_handler.sendText(end_msg);
            defer allocator.free(end_frame);
            try conn_ctx.write(end_frame);

            if (metrics) |m| {
                const end_ms = shared_utils.unixMs();
                const duration_ms: u64 = if (end_ms >= stream_start_ms)
                    @intCast(end_ms - stream_start_ms)
                else
                    0;
                m.recordStreamComplete(backend_type, duration_ms);
            }
            self.backend_router.recordSuccess(backend_type);
            stream_ok = true;
        } else {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
            stream_ok = true;
        }
    }

    /// Handle metrics endpoint
    fn handleMetrics(self: *Self, request: *std.http.Server.Request) !void {
        const metrics = self.getMetrics() orelse {
            return self.respondJson(request, "{\"status\":\"disabled\"}", .service_unavailable);
        };

        const snap = metrics.snapshot();

        var active_streams: u32 = 0;
        for (snap.backend_active_streams) |count| {
            if (count > 0) {
                active_streams += @intCast(count);
            }
        }

        var primary_idx: usize = 0;
        var max_requests: u64 = 0;
        for (snap.backend_requests, 0..) |count, i| {
            if (count > max_requests) {
                max_requests = count;
                primary_idx = i;
            }
        }

        const ttft_p50 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.50);
        const ttft_p95 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.95);
        const ttft_p99 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.99);

        const now_ms = shared_utils.unixMs();
        const uptime_ms: u64 = if (now_ms >= self.start_time_ms)
            @intCast(now_ms - self.start_time_ms)
        else
            0;

        const throughput_tps: f64 = if (uptime_ms > 0)
            @as(f64, @floatFromInt(snap.total_tokens)) / (@as(f64, @floatFromInt(uptime_ms)) / 1000.0)
        else
            0.0;

        var json = std.ArrayList(u8).empty;
        defer json.deinit(self.allocator);

        try json.print(
            self.allocator,
            "{{\"status\":\"ok\",\"uptime_ms\":{d},\"active_streams\":{d},\"max_streams\":{d}," ++
                "\"queue_depth\":{d},\"total_tokens\":{d},\"total_requests\":{d},\"total_errors\":{d}," ++
                "\"ttft_ms_p50\":{d},\"ttft_ms_p95\":{d},\"ttft_ms_p99\":{d},\"throughput_tps\":{d:.2}}}",
            .{
                uptime_ms,
                active_streams,
                self.config.max_concurrent_streams,
                0,
                snap.total_tokens,
                snap.total_streams,
                snap.total_errors,
                ttft_p50,
                ttft_p95,
                ttft_p99,
                throughput_tps,
            },
        );

        return self.respondJson(request, json.items, .ok);
    }

    fn histogramPercentile(hist: *const observability.Histogram, p: f64) u64 {
        if (hist.bounds.len == 0) return 0;

        var total: u64 = 0;
        for (hist.buckets) |count| total += count;
        if (total == 0) return 0;

        const target = @as(u64, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(total)) * p)));
        var cumulative: u64 = 0;
        for (hist.buckets, 0..) |count, i| {
            cumulative += count;
            if (cumulative >= target) {
                if (i < hist.bounds.len) return hist.bounds[i];
                return hist.bounds[hist.bounds.len - 1];
            }
        }

        return hist.bounds[hist.bounds.len - 1];
    }

    /// Handle models list (OpenAI-compatible)
    fn handleModelsList(self: *Self, request: *std.http.Server.Request) !void {
        const models_json = try self.backend_router.listModelsJson(self.allocator);
        defer self.allocator.free(models_json);
        return self.respondJson(request, models_json, .ok);
    }

    /// Handle admin model hot-reload
    ///
    /// Waits for active streams to drain (up to 30s timeout), then reloads
    /// the model from the specified path. If the new model fails to load,
    /// the server continues without a model (no rollback).
    ///
    /// Request: POST /admin/reload {"model_path": "/path/to/model.gguf"}
    /// Response: {"status": "ok", "message": "model reloaded successfully"}
    fn handleAdminReload(self: *Self, request: *std.http.Server.Request) !void {
        // Only accept POST method
        if (request.head.method != .POST) {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}",
                .method_not_allowed,
            );
        }

        // Parse request body
        const body = try self.readRequestBody(request);
        defer self.allocator.free(body);

        // Extract model_path from JSON
        const model_path = extractJsonString(body, "model_path") orelse {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"missing model_path field\",\"type\":\"invalid_request_error\"}}",
                .bad_request,
            );
        };

        // Wait for active streams to drain
        var timer = std.time.Timer.start() catch {
            // Timer failed - proceed with reload anyway
            return self.performModelReload(request, model_path);
        };

        while (self.active_streams.load(.seq_cst) > 0) {
            if (timer.read() >= admin_reload_drain_timeout_ns) {
                return self.respondJson(
                    request,
                    "{\"error\":{\"message\":\"timeout waiting for active streams to drain\",\"type\":\"timeout_error\"}}",
                    .request_timeout,
                );
            }
            // Brief delay to avoid pure busy-waiting (Zig 0.16 compatible)
            const poll_timer = std.time.Timer.start() catch continue;
            var pt = poll_timer;
            while (pt.read() < admin_reload_poll_interval_ns) {
                std.atomic.spinLoopHint();
            }
        }

        return self.performModelReload(request, model_path);
    }

    /// Perform the actual model reload on the local backend
    fn performModelReload(self: *Self, request: *std.http.Server.Request, model_path: []const u8) !void {
        // Get local backend from router
        const backend = self.backend_router.getBackend(.local) catch {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"local backend unavailable\",\"type\":\"backend_error\"}}",
                .internal_server_error,
            );
        };

        // Attempt to load new model
        // Note: LocalBackend.loadModel() -> Engine.loadModelImpl() already unloads old model
        backend.impl.local.loadModel(model_path) catch {
            // Per requirements: leave server without model on failure (no rollback)
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"model reload failed\",\"type\":\"model_error\"}}",
                .internal_server_error,
            );
        };

        // Success
        return self.respondJson(
            request,
            "{\"status\":\"ok\",\"message\":\"model reloaded successfully\"}",
            .ok,
        );
    }

    /// Validate bearer token authentication
    fn validateAuth(self: *Self, request: *std.http.Server.Request) StreamingServerError!void {
        const expected_token = self.config.auth_token orelse return;

        const auth_header = findHeaderInBuffer(request.head_buffer, "authorization") orelse {
            return StreamingServerError.Unauthorized;
        };

        const bearer_prefix = "Bearer ";
        if (!std.mem.startsWith(u8, auth_header, bearer_prefix)) {
            return StreamingServerError.Unauthorized;
        }

        const provided_token = auth_header[bearer_prefix.len..];
        if (provided_token.len != expected_token.len or
            !timingSafeEqual(provided_token, expected_token))
        {
            return StreamingServerError.Unauthorized;
        }
    }

    /// Read request body
    fn readRequestBody(self: *Self, request: *std.http.Server.Request) ![]u8 {
        var buffer: [4096]u8 = undefined;
        const reader = request.readerExpectContinue(&buffer) catch
            return StreamingServerError.InvalidRequest;
        return self.readAll(reader, max_body_bytes);
    }

    fn readAll(self: *Self, reader: *std.Io.Reader, limit: usize) ![]u8 {
        var list = std.ArrayListUnmanaged(u8).empty;
        errdefer list.deinit(self.allocator);

        var chunk: [4096]u8 = undefined;
        while (true) {
            const n = reader.readSliceShort(chunk[0..]) catch
                return StreamingServerError.InvalidRequest;
            if (n == 0) break;
            if (list.items.len + n > limit) return StreamingServerError.RequestTooLarge;
            try list.appendSlice(self.allocator, chunk[0..n]);
            if (n < chunk.len) break;
        }
        return list.toOwnedSlice(self.allocator);
    }

    /// Resolve address string to IP address
    fn resolveAddress(self: *Self, io: std.Io, address: []const u8) !std.Io.net.IpAddress {
        var host_port = net_utils.parseHostPort(self.allocator, address) catch
            return StreamingServerError.InvalidAddress;
        defer host_port.deinit(self.allocator);
        return std.Io.net.IpAddress.resolve(io, host_port.host, host_port.port) catch
            return StreamingServerError.InvalidAddress;
    }

    /// Send JSON response
    fn respondJson(
        self: *Self,
        request: *std.http.Server.Request,
        body: []const u8,
        status: std.http.Status,
    ) !void {
        _ = self;
        const headers = [_]std.http.Header{
            .{ .name = "content-type", .value = "application/json" },
        };
        try request.respond(body, .{
            .status = status,
            .extra_headers = &headers,
        });
    }
};

/// ABI streaming request format
pub const AbiStreamRequest = struct {
    prompt: []const u8,
    backend: ?backends.BackendType,
    config: backends.GenerationConfig,
    stream_id: ?[]const u8,

    pub fn deinit(self: *const AbiStreamRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt);
        if (self.stream_id) |id| allocator.free(id);
    }
};

/// Parse ABI stream request from JSON
fn parseAbiStreamRequest(allocator: std.mem.Allocator, body: []const u8) !AbiStreamRequest {
    // Simple JSON parsing - in production would use a proper JSON parser
    const prompt = extractJsonString(body, "prompt") orelse return StreamingServerError.InvalidRequest;
    const prompt_copy = try allocator.dupe(u8, prompt);
    errdefer allocator.free(prompt_copy);

    const backend_str = extractJsonString(body, "backend");
    const backend: ?backends.BackendType = if (backend_str) |b|
        backends.BackendType.fromString(b)
    else
        null;

    const max_tokens = extractJsonInt(body, "max_tokens") orelse 1024;
    const temperature = extractJsonFloat(body, "temperature") orelse 0.7;

    const stream_id = if (extractJsonString(body, "stream_id")) |id|
        try allocator.dupe(u8, id)
    else
        null;

    return .{
        .prompt = prompt_copy,
        .backend = backend,
        .config = .{
            .max_tokens = @intCast(max_tokens),
            .temperature = @floatCast(temperature),
        },
        .stream_id = stream_id,
    };
}

// Helper functions

fn splitTarget(target: []const u8) struct { path: []const u8, query: []const u8 } {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| {
        return .{ .path = target[0..idx], .query = target[idx + 1 ..] };
    }
    return .{ .path = target, .query = "" };
}

fn findHeaderInBuffer(buffer: []const u8, header_name: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, buffer, "\r\n");
    _ = it.next(); // Skip request line

    while (it.next()) |line| {
        if (line.len == 0) break;
        const colon_idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon_idx];

        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            const value_start = colon_idx + 1;
            if (value_start >= line.len) return "";
            return std.mem.trim(u8, line[value_start..], " \t");
        }
    }
    return null;
}

fn timingSafeEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

fn extractJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    const value_start = key_pos + search_key.len;

    // Skip whitespace
    var pos = value_start;
    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1; // Skip opening quote

    const str_start = pos;
    while (pos < json.len and json[pos] != '"') : (pos += 1) {
        if (json[pos] == '\\' and pos + 1 < json.len) pos += 1; // Skip escaped char
    }

    return json[str_start..pos];
}

fn extractJsonInt(json: []const u8, key: []const u8) ?i64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and (json[pos] >= '0' and json[pos] <= '9')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseInt(i64, json[num_start..pos], 10) catch null;
}

fn extractJsonFloat(json: []const u8, key: []const u8) ?f64 {
    // Build search key without allocation using a fixed buffer
    var key_buf: [256]u8 = undefined;
    const search_key = std.fmt.bufPrint(&key_buf, "\"{s}\":", .{key}) catch return null;

    const key_pos = std.mem.indexOf(u8, json, search_key) orelse return null;
    var pos = key_pos + search_key.len;

    while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t')) : (pos += 1) {}

    const num_start = pos;
    while (pos < json.len and ((json[pos] >= '0' and json[pos] <= '9') or json[pos] == '.')) : (pos += 1) {}

    if (pos == num_start) return null;
    return std.fmt.parseFloat(f64, json[num_start..pos]) catch null;
}

// Tests
test "streaming server config defaults" {
    const config = ServerConfig{};
    try std.testing.expectEqualStrings("127.0.0.1:8080", config.address);
    try std.testing.expect(config.auth_token == null);
    try std.testing.expect(config.enable_openai_compat);
    try std.testing.expect(config.enable_websocket);
}

test "heartbeat configuration" {
    // Default heartbeat interval is 15 seconds
    const default_config = ServerConfig{};
    try std.testing.expectEqual(@as(u64, 15000), default_config.heartbeat_interval_ms);

    // Custom heartbeat interval
    const custom_config = ServerConfig{ .heartbeat_interval_ms = 5000 };
    try std.testing.expectEqual(@as(u64, 5000), custom_config.heartbeat_interval_ms);

    // Heartbeat can be disabled by setting to 0
    const disabled_config = ServerConfig{ .heartbeat_interval_ms = 0 };
    try std.testing.expectEqual(@as(u64, 0), disabled_config.heartbeat_interval_ms);
}

test "heartbeat interval conversion to nanoseconds" {
    // Verify the nanosecond conversion used in streaming loops
    const interval_ms: u64 = 15000;
    const interval_ns: u64 = interval_ms * 1_000_000;
    try std.testing.expectEqual(@as(u64, 15_000_000_000), interval_ns);
}

test "split target" {
    const parts = splitTarget("/api/stream?model=gpt");
    try std.testing.expectEqualStrings("/api/stream", parts.path);
    try std.testing.expectEqualStrings("model=gpt", parts.query);
}

test "extract json string" {
    const json = "{\"prompt\":\"hello world\",\"model\":\"gpt-4\"}";
    const prompt = extractJsonString(json, "prompt");
    try std.testing.expect(prompt != null);
    try std.testing.expectEqualStrings("hello world", prompt.?);
}

test "extract json int" {
    const json = "{\"max_tokens\":1024,\"other\":\"value\"}";
    const max_tokens = extractJsonInt(json, "max_tokens");
    try std.testing.expect(max_tokens != null);
    try std.testing.expectEqual(@as(i64, 1024), max_tokens.?);
}

test "websocket message parsing - cancel message" {
    // Test that cancel messages are correctly identified
    const cancel_json = "{\"type\":\"cancel\"}";
    const msg_type = extractJsonString(cancel_json, "type");
    try std.testing.expect(msg_type != null);
    try std.testing.expectEqualStrings("cancel", msg_type.?);
}

test "websocket message parsing - stream request" {
    const allocator = std.testing.allocator;

    const request_json = "{\"prompt\":\"Hello world\",\"backend\":\"local\",\"max_tokens\":100}";
    const request = try parseAbiStreamRequest(allocator, request_json);
    defer request.deinit(allocator);

    try std.testing.expectEqualStrings("Hello world", request.prompt);
    try std.testing.expectEqual(backends.BackendType.local, request.backend.?);
    try std.testing.expectEqual(@as(u32, 100), request.config.max_tokens);
}

test "websocket handler initialization" {
    const allocator = std.testing.allocator;

    var handler = try websocket.WebSocketHandler.init(allocator, .{});
    defer handler.deinit();

    try std.testing.expectEqual(websocket.ConnectionState.connecting, handler.state);
}

test "websocket frame encoding for streaming" {
    const allocator = std.testing.allocator;

    var handler = try websocket.WebSocketHandler.init(allocator, .{});
    defer handler.deinit();

    // Encode a token message
    const msg = try websocket.createStreamingMessage(allocator, "token", "hello");
    defer allocator.free(msg);

    const frame = try handler.sendText(msg);
    defer allocator.free(frame);

    // Verify frame structure: FIN + text opcode = 0x81
    try std.testing.expectEqual(@as(u8, 0x81), frame[0]);
    // Payload length should be > 0
    try std.testing.expect(frame[1] > 0);
}

test "admin reload request parsing" {
    // Test JSON parsing for reload endpoint
    const json = "{\"model_path\":\"/path/to/model.gguf\"}";
    const model_path = extractJsonString(json, "model_path");
    try std.testing.expect(model_path != null);
    try std.testing.expectEqualStrings("/path/to/model.gguf", model_path.?);
}

test "admin reload missing model_path" {
    // Test that missing model_path is detected
    const json = "{\"other_field\":\"value\"}";
    const model_path = extractJsonString(json, "model_path");
    try std.testing.expect(model_path == null);
}

test "admin reload error types" {
    // Test that new error types are available
    const err1: StreamingServerError = StreamingServerError.ModelReloadFailed;
    const err2: StreamingServerError = StreamingServerError.ModelReloadTimeout;
    try std.testing.expect(err1 != err2);
}
