//! WebSocket Endpoint Handlers
//!
//! Handles WebSocket upgrade, connection lifecycle, and message processing
//! for bidirectional streaming.

const std = @import("std");
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const websocket = @import("../websocket.zig");
const backends = @import("../backends/mod.zig");
const request_types = @import("../request_types.zig");
const handlers = @import("handlers.zig");
const helpers = @import("helpers.zig");
const routing = @import("routing.zig");

const ConnectionContext = handlers.ConnectionContext;
const findHeaderInBuffer = helpers.findHeaderInBuffer;
const extractJsonString = request_types.extractJsonString;
const parseAbiStreamRequest = request_types.parseAbiStreamRequest;

/// Handle WebSocket upgrade request.
pub fn handleWebSocketUpgrade(
    server: anytype,
    request: *std.http.Server.Request,
    conn_ctx: *ConnectionContext,
) !void {
    const upgrade_header = findHeaderInBuffer(request.head_buffer, "upgrade");
    if (upgrade_header == null or !std.ascii.eqlIgnoreCase(upgrade_header.?, "websocket")) {
        return routing.respondJson(
            request,
            "{\"error\":\"expected websocket upgrade\"}",
            .bad_request,
        );
    }

    const ws_key = findHeaderInBuffer(request.head_buffer, "sec-websocket-key") orelse {
        return routing.respondJson(
            request,
            "{\"error\":\"missing sec-websocket-key\"}",
            .bad_request,
        );
    };

    const accept_key = try websocket.computeAcceptKey(server.allocator, ws_key);
    defer server.allocator.free(accept_key);

    const upgrade_headers = [_]std.http.Header{
        .{ .name = "upgrade", .value = "websocket" },
        .{ .name = "connection", .value = "upgrade" },
        .{ .name = "sec-websocket-accept", .value = accept_key },
    };

    try request.respond("", .{
        .status = .switching_protocols,
        .extra_headers = &upgrade_headers,
    });

    try handleWebSocketConnection(server, conn_ctx, server.allocator);
}

/// Handle WebSocket connection after upgrade.
fn handleWebSocketConnection(
    server: anytype,
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
            if (server.config.auth_token != null) {
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
                try handleWebSocketMessage(
                    server,
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

/// Handle a WebSocket JSON message.
fn handleWebSocketMessage(
    server: anytype,
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

    const current = server.active_streams.fetchAdd(1, .seq_cst);
    if (current >= server.config.max_concurrent_streams) {
        _ = server.active_streams.fetchSub(1, .seq_cst);
        const error_msg = try websocket.createStreamingMessage(allocator, "error", "too many concurrent streams");
        defer allocator.free(error_msg);
        const error_frame = try ws_handler.sendText(error_msg);
        defer allocator.free(error_frame);
        try conn_ctx.write(error_frame);
        return;
    }
    defer _ = server.active_streams.fetchSub(1, .seq_cst);

    const backend_type = stream_request.backend orelse server.config.default_backend;
    const metrics = server.getMetrics();
    const backend = server.backend_router.getBackend(backend_type) catch {
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
        server.backend_router.recordFailure(backend_type);
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
        server.backend_router.recordFailure(backend_type);
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
        server.backend_router.recordSuccess(backend_type);
        stream_ok = true;
    } else {
        if (metrics) |m| {
            m.recordStreamFailure(backend_type);
        }
        server.backend_router.recordFailure(backend_type);
        stream_ok = true;
    }
}

test {
    std.testing.refAllDecls(@This());
}
