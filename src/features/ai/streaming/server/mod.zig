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
const time = @import("../../../../foundation/mod.zig").time;
const mod = @import("../mod.zig");
const sse = @import("../sse.zig");
const websocket = @import("../websocket.zig");
const backends = @import("../backends/mod.zig");
const formats = @import("../formats/mod.zig");
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const build_options = @import("build_options");
const observability = if (build_options.feat_profiling) @import("../../../observability/mod.zig") else @import("../../../observability/stub.zig");
const net_utils = shared_utils.net;
const json_utils = shared_utils.json;
const recovery = @import("../recovery.zig");
const RecoveryConfig = recovery.RecoveryConfig;
const RecoveryEvent = recovery.RecoveryEvent;
const request_types = @import("../request_types.zig");

// Import sub-modules
const helpers = @import("helpers.zig");
const handlers = @import("handlers.zig");

// Re-export sub-module types
pub const ConnectionContext = handlers.ConnectionContext;
pub const splitTarget = helpers.splitTarget;
pub const findHeaderInBuffer = helpers.findHeaderInBuffer;
pub const timingSafeEqual = helpers.timingSafeEqual;
pub const histogramPercentile = handlers.histogramPercentile;

pub const AbiStreamRequest = request_types.AbiStreamRequest;
const parseAbiStreamRequest = request_types.parseAbiStreamRequest;
const extractJsonString = request_types.extractJsonString;

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
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !Self {
        const recovery_cfg: ?RecoveryConfig = if (config.enable_recovery)
            config.recovery_config
        else
            null;
        var backend_router = try backends.BackendRouter.initWithRecovery(allocator, recovery_cfg);
        errdefer backend_router.deinit();

        if (config.default_model_path) |model_path| {
            const local_backend = try backend_router.getBackend(.local);

            if (config.preload_model) {
                try local_backend.impl.local.loadModel(model_path);
            } else {
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

        std.log.info("Streaming inference server listening on {s}", .{self.config.address});
        std.log.info("  SSE endpoint: POST /api/stream", .{});
        if (self.config.enable_openai_compat) {
            std.log.info("  OpenAI endpoint: POST /v1/chat/completions", .{});
        }
        if (self.config.enable_websocket) {
            std.log.info("  WebSocket endpoint: GET /api/stream/ws", .{});
        }

        while (true) {
            var stream = server.accept(io) catch |err| {
                std.log.err("Streaming server accept error: {t}", .{err});
                continue;
            };
            defer stream.close(io);

            self.handleConnection(io, stream) catch |err| {
                std.log.err("Streaming server connection error: {t}", .{err});
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
                std.log.err("Streaming request error: {t}", .{err});
                const error_body = if (err == StreamingServerError.Unauthorized)
                    "{\"error\":{\"message\":\"unauthorized\",\"type\":\"authentication_error\"}}"
                else
                    "{\"error\":{\"message\":\"internal server error\",\"type\":\"server_error\"}}";
                const status: std.http.Status = if (err == StreamingServerError.Unauthorized)
                    .unauthorized
                else
                    .internal_server_error;
                self.respondJson(&request, error_body, status) catch |resp_err| {
                    std.log.err("Streaming server: failed to send error response: {t}", .{resp_err});
                };
                return;
            };
        }
    }

    /// Send a recovery event as an SSE message.
    fn sendRecoveryEvent(self: *Self, conn_ctx: *ConnectionContext, event: RecoveryEvent) void {
        const json = event.toJson(self.allocator) catch return;
        defer self.allocator.free(json);

        conn_ctx.write("event: recovery\ndata: ") catch return;
        conn_ctx.write(json) catch return;
        conn_ctx.write("\n\n") catch return;
        conn_ctx.flush() catch return;
    }

    /// Check circuit breaker and find available backend.
    fn getAvailableBackend(self: *Self, preferred: backends.BackendType) !*backends.Backend {
        if (!self.backend_router.isBackendAvailable(preferred)) {
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

        const body = try self.readRequestBody(request);
        defer self.allocator.free(body);

        const chat_request = try formats.openai.parseRequest(self.allocator, body);
        defer chat_request.deinit(self.allocator);

        if (chat_request.stream) {
            return self.streamOpenAIResponse(conn_ctx, chat_request);
        } else {
            return self.nonStreamingOpenAIResponse(request, chat_request);
        }
    }

    /// Stream OpenAI-format response with true SSE streaming
    fn streamOpenAIResponse(
        self: *Self,
        conn_ctx: *ConnectionContext,
        chat_request: formats.openai.ChatCompletionRequest,
    ) !void {
        const current = self.active_streams.fetchAdd(1, .seq_cst);
        if (current >= self.config.max_concurrent_streams) {
            _ = self.active_streams.fetchSub(1, .seq_cst);
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

        try conn_ctx.writeSseHeaders();

        if (metrics) |m| {
            m.recordStreamStart(backend_type);
        }
        var stream_iter = backend.streamTokens(prompt, chat_request.toGenerationConfig()) catch |err| {
            if (metrics) |m| {
                m.recordStreamFailure(backend_type);
            }
            self.backend_router.recordFailure(backend_type);
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

        var heartbeat_timer = time.Timer.start() catch null;
        const hb_interval_ns: u64 = self.config.heartbeat_interval_ms * 1_000_000;

        const stream_start_ms = shared_utils.unixMs();
        var last_token_ms = stream_start_ms;
        var token_index: u32 = 0;
        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
            if (self.config.heartbeat_interval_ms > 0) {
                if (heartbeat_timer) |*timer| {
                    if (timer.read() >= hb_interval_ns) {
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

            const chunk_json = try formats.openai.formatStreamChunk(
                self.allocator,
                token.text,
                chat_request.model,
                token_index,
                token.is_end,
            );
            defer self.allocator.free(chunk_json);

            try conn_ctx.write("data: ");
            try conn_ctx.write(chunk_json);
            try conn_ctx.write("\n\n");
            try conn_ctx.flush();

            token_index += 1;
            if (token.is_end) break;
        }

        try conn_ctx.write("data: [DONE]\n\n");
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

    /// Non-streaming OpenAI response
    fn nonStreamingOpenAIResponse(
        self: *Self,
        request: *std.http.Server.Request,
        chat_request: formats.openai.ChatCompletionRequest,
    ) !void {
        const backend = try self.backend_router.getBackend(self.config.default_backend);
        const prompt = try chat_request.formatPrompt(self.allocator);
        defer self.allocator.free(prompt);

        const response_text = try backend.generate(prompt, chat_request.toGenerationConfig());
        defer self.allocator.free(response_text);

        const response_body = try formats.openai.formatResponse(
            self.allocator,
            response_text,
            chat_request.model,
        );
        defer self.allocator.free(response_body);

        return self.respondJson(request, response_body, .ok);
    }

    /// Handle custom ABI streaming request with true SSE streaming
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

        const stream_request = try parseAbiStreamRequest(self.allocator, body);
        defer stream_request.deinit(self.allocator);

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

        try conn_ctx.writeSseHeaders();

        var sse_encoder = sse.SseEncoder.init(self.allocator, .{
            .include_id = true,
            .event_prefix = "abi.",
        });
        defer sse_encoder.deinit();

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

        var heartbeat_timer = time.Timer.start() catch null;
        const hb_interval_ns: u64 = self.config.heartbeat_interval_ms * 1_000_000;

        const stream_start_ms = shared_utils.unixMs();
        var last_token_ms = stream_start_ms;

        const start_event = mod.StreamEvent.startEvent();
        const start_sse = try sse_encoder.encode(start_event);
        defer self.allocator.free(start_sse);
        try conn_ctx.write(start_sse);
        try conn_ctx.flush();

        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
            if (self.config.heartbeat_interval_ms > 0) {
                if (heartbeat_timer) |*timer| {
                    if (timer.read() >= hb_interval_ns) {
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

            const token_event = mod.StreamEvent{
                .event_type = .token,
                .token = .{
                    .id = @intCast(token.id orelse 0),
                    .text = token.text,
                    .is_end = token.is_end,
                },
            };

            const sse_data = try sse_encoder.encode(token_event);
            defer self.allocator.free(sse_data);

            try conn_ctx.write(sse_data);
            try conn_ctx.flush();

            if (token.is_end) break;
        }

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
        const upgrade_header = findHeaderInBuffer(request.head_buffer, "upgrade");
        if (upgrade_header == null or !std.ascii.eqlIgnoreCase(upgrade_header.?, "websocket")) {
            return self.respondJson(
                request,
                "{\"error\":\"expected websocket upgrade\"}",
                .bad_request,
            );
        }

        const ws_key = findHeaderInBuffer(request.head_buffer, "sec-websocket-key") orelse {
            return self.respondJson(
                request,
                "{\"error\":\"missing sec-websocket-key\"}",
                .bad_request,
            );
        };

        const accept_key = try websocket.computeAcceptKey(self.allocator, ws_key);
        defer self.allocator.free(accept_key);

        const upgrade_headers = [_]std.http.Header{
            .{ .name = "upgrade", .value = "websocket" },
            .{ .name = "connection", .value = "upgrade" },
            .{ .name = "sec-websocket-accept", .value = accept_key },
        };

        try request.respond("", .{
            .status = .switching_protocols,
            .extra_headers = &upgrade_headers,
        });

        try self.handleWebSocketConnection(conn_ctx, self.allocator);
    }

    /// Handle WebSocket connection after upgrade
    fn handleWebSocketConnection(
        self: *Self,
        conn_ctx: *ConnectionContext,
        allocator: std.mem.Allocator,
    ) !void {
        var ws_handler = try websocket.WebSocketHandler.init(allocator, .{});
        defer ws_handler.deinit();
        ws_handler.state = .open;

        var cancel_requested = std.atomic.Value(bool).init(false);

        var frame_buffer: [65536]u8 = undefined;

        while (ws_handler.state == .open) {
            var reader = conn_ctx.stream.reader(conn_ctx.io, conn_ctx.send_buffer);
            const bytes_read = reader.interface.readSliceShort(&frame_buffer) catch |err| {
                if (err == error.EndOfStream) break;
                if (self.config.auth_token != null) {
                    break;
                }
                continue;
            };

            if (bytes_read == 0) break;

            const parse_result = ws_handler.parseFrame(frame_buffer[0..bytes_read]) catch |err| {
                const close_frame = try ws_handler.sendClose(.protocol_error, "invalid frame");
                defer allocator.free(close_frame);
                try conn_ctx.write(close_frame);
                std.log.err("WebSocket parse error: {t}", .{err});
                break;
            };
            defer allocator.free(parse_result.frame.payload);

            const frame = parse_result.frame;

            switch (frame.opcode) {
                .text => {
                    try self.handleWebSocketMessage(
                        conn_ctx,
                        &ws_handler,
                        allocator,
                        frame.payload,
                        &cancel_requested,
                    );
                },
                .binary => {
                    const error_msg = try websocket.createStreamingMessage(allocator, "error", "binary messages not supported");
                    defer allocator.free(error_msg);
                    const error_frame = try ws_handler.sendText(error_msg);
                    defer allocator.free(error_frame);
                    try conn_ctx.write(error_frame);
                },
                .close => {
                    const close_frame = try ws_handler.sendClose(.normal, "");
                    defer allocator.free(close_frame);
                    try conn_ctx.write(close_frame);
                    ws_handler.state = .closed;
                    break;
                },
                .ping => {
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

        const stream_request = parseAbiStreamRequest(allocator, payload) catch {
            const error_msg = try websocket.createStreamingMessage(allocator, "error", "invalid request format");
            defer allocator.free(error_msg);
            const error_frame = try ws_handler.sendText(error_msg);
            defer allocator.free(error_frame);
            try conn_ctx.write(error_frame);
            return;
        };
        defer stream_request.deinit(allocator);

        cancel_requested.store(false, .seq_cst);

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

        const start_msg = try websocket.createStreamingMessage(allocator, "start", "");
        defer allocator.free(start_msg);
        const start_frame = try ws_handler.sendText(start_msg);
        defer allocator.free(start_frame);
        try conn_ctx.write(start_frame);

        while (true) {
            const maybe_token = stream_iter.next() catch |err| return err;
            if (maybe_token == null) break;
            const token = maybe_token.?;
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

            const token_msg = try websocket.createStreamingMessage(allocator, "token", token.text);
            defer allocator.free(token_msg);
            const token_frame = try ws_handler.sendText(token_msg);
            defer allocator.free(token_frame);
            try conn_ctx.write(token_frame);

            if (token.is_end) break;
        }

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

        var active_streams_count: u32 = 0;
        for (snap.backend_active_streams) |count| {
            if (count > 0) {
                active_streams_count += @intCast(count);
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

        var json = std.ArrayListUnmanaged(u8).empty;
        defer json.deinit(self.allocator);

        try json.print(
            self.allocator,
            "{{\"status\":\"ok\",\"uptime_ms\":{d},\"active_streams\":{d},\"max_streams\":{d}," ++
                "\"queue_depth\":{d},\"total_tokens\":{d},\"total_requests\":{d},\"total_errors\":{d}," ++
                "\"ttft_ms_p50\":{d},\"ttft_ms_p95\":{d},\"ttft_ms_p99\":{d},\"throughput_tps\":{d:.2}}}",
            .{
                uptime_ms,
                active_streams_count,
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

    /// Handle models list (OpenAI-compatible)
    fn handleModelsList(self: *Self, request: *std.http.Server.Request) !void {
        const models_json = try self.backend_router.listModelsJson(self.allocator);
        defer self.allocator.free(models_json);
        return self.respondJson(request, models_json, .ok);
    }

    /// Handle admin model hot-reload
    fn handleAdminReload(self: *Self, request: *std.http.Server.Request) !void {
        if (request.head.method != .POST) {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}",
                .method_not_allowed,
            );
        }

        const body = try self.readRequestBody(request);
        defer self.allocator.free(body);

        const model_path = extractJsonString(body, "model_path") orelse {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"missing model_path field\",\"type\":\"invalid_request_error\"}}",
                .bad_request,
            );
        };

        var timer = time.Timer.start() catch {
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
            const poll_timer = time.Timer.start() catch continue;
            var pt = poll_timer;
            while (pt.read() < admin_reload_poll_interval_ns) {
                std.atomic.spinLoopHint();
            }
        }

        return self.performModelReload(request, model_path);
    }

    /// Perform the actual model reload on the local backend
    fn performModelReload(self: *Self, request: *std.http.Server.Request, model_path: []const u8) !void {
        const backend = self.backend_router.getBackend(.local) catch {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"local backend unavailable\",\"type\":\"backend_error\"}}",
                .internal_server_error,
            );
        };

        backend.impl.local.loadModel(model_path) catch {
            return self.respondJson(
                request,
                "{\"error\":{\"message\":\"model reload failed\",\"type\":\"model_error\"}}",
                .internal_server_error,
            );
        };

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
        const response_headers = [_]std.http.Header{
            .{ .name = "content-type", .value = "application/json" },
        };
        try request.respond(body, .{
            .status = status,
            .extra_headers = &response_headers,
        });
    }
};

test {
    _ = @import("../request_types.zig");
    _ = @import("../server_test.zig");
    _ = @import("helpers.zig");
    _ = @import("handlers.zig");
}

test {
    std.testing.refAllDecls(@This());
}
