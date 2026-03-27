//! Request Routing and HTTP Utilities
//!
//! Request dispatch, authentication validation, JSON response helpers,
//! request body reading, and address resolution.

const std = @import("std");
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const net_utils = shared_utils.net;
const backends = @import("../backends/mod.zig");
const helpers = @import("helpers.zig");
const config_mod = @import("config.zig");
const openai = @import("openai.zig");
const websocket_handler = @import("websocket_handler.zig");
const admin = @import("admin.zig");

const ServerConfig = config_mod.ServerConfig;
const StreamingServerError = config_mod.StreamingServerError;
const ConnectionContext = @import("handlers.zig").ConnectionContext;
const findHeaderInBuffer = helpers.findHeaderInBuffer;
const splitTarget = helpers.splitTarget;
const timingSafeEqual = helpers.timingSafeEqual;

/// Dispatch request to appropriate handler.
/// Called by StreamingServer.handleConnection for each HTTP request.
pub fn dispatchRequest(
    server: anytype,
    request: *std.http.Server.Request,
    conn_ctx: *ConnectionContext,
) !void {
    const target = request.head.target;
    const path = splitTarget(target).path;

    // Health endpoint
    if (std.mem.eql(u8, path, "/health")) {
        if (!server.config.allow_health_without_auth) {
            try validateAuth(server, request);
        }
        return respondJson(request, "{\"status\":\"ok\"}", .ok);
    }

    // Metrics endpoint
    if (std.mem.eql(u8, path, "/metrics")) {
        if (!server.config.allow_health_without_auth) {
            try validateAuth(server, request);
        }
        return admin.handleMetrics(server, request);
    }

    // All other endpoints require auth
    try validateAuth(server, request);

    // OpenAI-compatible endpoint
    if (server.config.enable_openai_compat and std.mem.eql(u8, path, "/v1/chat/completions")) {
        return openai.handleOpenAIChatCompletions(server, request, conn_ctx);
    }

    // Custom ABI streaming endpoint
    if (std.mem.eql(u8, path, "/api/stream")) {
        return openai.handleStreamRequest(server, request, conn_ctx);
    }

    // WebSocket upgrade
    if (server.config.enable_websocket and std.mem.eql(u8, path, "/api/stream/ws")) {
        return websocket_handler.handleWebSocketUpgrade(server, request, conn_ctx);
    }

    // Models list (OpenAI-compatible)
    if (server.config.enable_openai_compat and std.mem.eql(u8, path, "/v1/models")) {
        return admin.handleModelsList(server, request);
    }

    // Admin model reload endpoint
    if (std.mem.eql(u8, path, "/admin/reload")) {
        return admin.handleAdminReload(server, request);
    }

    return respondJson(
        request,
        "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}",
        .not_found,
    );
}

/// Validate bearer token authentication.
pub fn validateAuth(server: anytype, request: *std.http.Server.Request) StreamingServerError!void {
    const expected_token = server.config.auth_token orelse return;

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

/// Send JSON response.
pub fn respondJson(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
) !void {
    const response_headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "application/json" },
    };
    try request.respond(body, .{
        .status = status,
        .extra_headers = &response_headers,
    });
}

/// Read request body up to MAX_BODY_BYTES.
pub fn readRequestBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ![]u8 {
    var buffer: [4096]u8 = undefined;
    const reader = request.readerExpectContinue(&buffer) catch
        return StreamingServerError.InvalidRequest;
    return readAll(allocator, reader, config_mod.MAX_BODY_BYTES);
}

/// Read all bytes from a reader up to a limit.
pub fn readAll(allocator: std.mem.Allocator, reader: *std.Io.Reader, limit: usize) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch
            return StreamingServerError.InvalidRequest;
        if (n == 0) break;
        if (list.items.len + n > limit) return StreamingServerError.RequestTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
        if (n < chunk.len) break;
    }
    return list.toOwnedSlice(allocator);
}

/// Resolve address string to IP address.
pub fn resolveAddress(allocator: std.mem.Allocator, io: std.Io, address: []const u8) !std.Io.net.IpAddress {
    var host_port = net_utils.parseHostPort(allocator, address) catch
        return StreamingServerError.InvalidAddress;
    defer host_port.deinit(allocator);
    return std.Io.net.IpAddress.resolve(io, host_port.host, host_port.port) catch
        return StreamingServerError.InvalidAddress;
}

test {
    std.testing.refAllDecls(@This());
}
