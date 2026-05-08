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
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const backends = @import("../backends/mod.zig");
const mod = @import("../mod.zig");
const recovery = @import("../recovery.zig");
const RecoveryConfig = recovery.RecoveryConfig;
const RecoveryEvent = recovery.RecoveryEvent;

// Sub-modules
const config_mod = @import("config.zig");
const routing_mod = @import("routing.zig");
const handlers_mod = @import("handlers.zig");
const helpers_mod = @import("helpers.zig");
const openai_mod = @import("openai.zig");
const websocket_handler_mod = @import("websocket_handler.zig");
const admin_mod = @import("admin.zig");

// Re-export sub-module types
pub const ServerConfig = config_mod.ServerConfig;
pub const StreamingServerError = config_mod.StreamingServerError;

pub const ConnectionContext = handlers_mod.ConnectionContext;
pub const histogramPercentile = handlers_mod.histogramPercentile;

pub const splitTarget = helpers_mod.splitTarget;
pub const findHeaderInBuffer = helpers_mod.findHeaderInBuffer;
pub const timingSafeEqual = helpers_mod.timingSafeEqual;

pub const AbiStreamRequest = @import("../request_types.zig").AbiStreamRequest;

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

        const listen_addr = try routing_mod.resolveAddress(self.allocator, io, self.config.address);
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

            routing_mod.dispatchRequest(self, &request, &conn_ctx) catch |err| {
                std.log.err("Streaming request error: {t}", .{err});
                const error_body = if (err == StreamingServerError.Unauthorized)
                    "{\"error\":{\"message\":\"unauthorized\",\"type\":\"authentication_error\"}}"
                else
                    "{\"error\":{\"message\":\"internal server error\",\"type\":\"server_error\"}}";
                const status: std.http.Status = if (err == StreamingServerError.Unauthorized)
                    .unauthorized
                else
                    .internal_server_error;
                routing_mod.respondJson(&request, error_body, status) catch |resp_err| {
                    std.log.err("Streaming server: failed to send error response: {t}", .{resp_err});
                };
                return;
            };
        }
    }

    /// Send a recovery event as an SSE message.
    pub fn sendRecoveryEvent(self: *Self, conn_ctx: *ConnectionContext, event: RecoveryEvent) void {
        const json = event.toJson(self.allocator) catch return;
        defer self.allocator.free(json);

        conn_ctx.write("event: recovery\ndata: ") catch return;
        conn_ctx.write(json) catch return;
        conn_ctx.write("\n\n") catch return;
        conn_ctx.flush() catch return;
    }

    /// Check circuit breaker and find available backend.
    pub fn getAvailableBackend(self: *Self, preferred: backends.BackendType) !*backends.Backend {
        if (!self.backend_router.isBackendAvailable(preferred)) {
            if (self.backend_router.findAvailableBackend(preferred)) |fallback| {
                return self.backend_router.getBackend(fallback);
            }
            return error.CircuitBreakerOpen;
        }
        return self.backend_router.getBackend(preferred);
    }

    /// Get metrics from recovery manager if available.
    pub fn getMetrics(self: *Self) ?*mod.StreamingMetrics {
        const recovery_mgr = self.backend_router.getRecovery() orelse return null;
        return recovery_mgr.getMetrics();
    }
};

test {
    _ = @import("../request_types.zig");
    _ = @import("../server_test.zig");
    _ = helpers_mod;
    _ = handlers_mod;
    _ = config_mod;
    _ = routing_mod;
    _ = openai_mod;
    _ = websocket_handler_mod;
    _ = admin_mod;
}

test {
    std.testing.refAllDecls(@This());
}
