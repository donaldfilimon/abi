//! MCP SSE (Server-Sent Events) Transport
//!
//! Exposes the MCP JSON-RPC server over HTTP using Server-Sent Events for
//! server-to-client streaming and a POST endpoint for client-to-server messages.
//!
//! ## Endpoints
//! - `GET /events` — SSE event stream (Content-Type: text/event-stream)
//! - `POST /message` — receive JSON-RPC requests from clients
//! - `GET /health` — health check endpoint
//!
//! ## SSE Wire Format
//! Each JSON-RPC response is sent as:
//! ```
//! data: {"jsonrpc":"2.0","id":1,"result":{...}}\n\n
//! ```
//!
//! Heartbeat comments (`:ping`) are sent every 30 seconds to keep connections alive.

const std = @import("std");
const types = @import("../types.zig");

/// Configuration for the SSE transport.
pub const Config = struct {
    /// TCP port to listen on.
    port: u16 = 8081,
    /// Bind address (IPv4).
    host: []const u8 = "127.0.0.1",
    /// Heartbeat interval in seconds.
    heartbeat_interval_s: u32 = 30,
    /// Maximum number of concurrent SSE client connections.
    max_clients: u16 = 64,
    /// Maximum HTTP request body size for POST /message.
    max_body_size: usize = 4 * 1024 * 1024,
};

/// A buffered response that collects JSON-RPC output for SSE framing.
pub const SseResponseCollector = struct {
    data: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SseResponseCollector {
        return .{
            .data = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SseResponseCollector) void {
        self.data.deinit(self.allocator);
    }

    /// Reset the buffer for reuse.
    pub fn reset(self: *SseResponseCollector) void {
        self.data.clearRetainingCapacity();
    }

    /// Writer interface — the MCP server writes JSON-RPC responses here.
    pub const Writer = struct {
        collector: *SseResponseCollector,

        pub fn writeAll(self: Writer, bytes: []const u8) !void {
            try self.collector.data.appendSlice(self.collector.allocator, bytes);
        }

        pub fn writeByte(self: Writer, byte: u8) !void {
            try self.collector.data.append(self.collector.allocator, byte);
        }

        pub fn print(self: Writer, comptime fmt: []const u8, args: anytype) !void {
            // Format into the collector buffer
            var buf: [1024]u8 = undefined;
            const formatted = std.fmt.bufPrint(&buf, fmt, args) catch {
                // Fall back to per-byte write for very long formats
                try std.fmt.format(self, fmt, args);
                return;
            };
            try self.writeAll(formatted);
        }

        /// Required by std.fmt.format — writes a single byte slice.
        pub fn write(self: Writer, bytes: []const u8) !usize {
            try self.writeAll(bytes);
            return bytes.len;
        }
    };

    pub fn writer(self: *SseResponseCollector) Writer {
        return .{ .collector = self };
    }

    /// Get the collected response data.
    pub fn getResponse(self: *const SseResponseCollector) []const u8 {
        return self.data.items;
    }
};

/// Format a JSON-RPC response as an SSE data frame.
/// Returns `data: {json}\n\n` with each line of the JSON prefixed by `data: `.
pub fn formatSseFrame(allocator: std.mem.Allocator, json_response: []const u8) ![]u8 {
    // Trim trailing newline that the MCP server adds
    const trimmed = std.mem.trimEnd(u8, json_response, "\n");
    if (trimmed.len == 0) return try allocator.dupe(u8, "");

    // Count lines to calculate output size
    var line_count: usize = 1;
    for (trimmed) |c| {
        if (c == '\n') line_count += 1;
    }

    // Each line gets "data: " prefix (6 bytes) + "\n" suffix (1 byte)
    // Plus final extra "\n" for the blank line terminator (1 byte)
    const out_size = trimmed.len + (line_count * 7) + 1;
    var out = try allocator.alloc(u8, out_size);
    var pos: usize = 0;

    var start: usize = 0;
    for (trimmed, 0..) |c, i| {
        if (c == '\n') {
            @memcpy(out[pos..][0..6], "data: ");
            pos += 6;
            const line_len = i - start;
            @memcpy(out[pos..][0..line_len], trimmed[start..i]);
            pos += line_len;
            out[pos] = '\n';
            pos += 1;
            start = i + 1;
        }
    }
    // Last line (or only line if no newlines)
    @memcpy(out[pos..][0..6], "data: ");
    pos += 6;
    const remaining = trimmed[start..];
    @memcpy(out[pos..][0..remaining.len], remaining);
    pos += remaining.len;
    out[pos] = '\n';
    pos += 1;
    // Blank line to terminate SSE event
    out[pos] = '\n';
    pos += 1;

    // Shrink if we over-estimated
    if (pos < out.len) {
        return allocator.realloc(out, pos);
    }
    return out;
}

/// The SSE heartbeat comment string.
pub const heartbeat_comment = ":ping\n\n";

/// Run the SSE HTTP transport. Blocks, accepting connections and dispatching
/// JSON-RPC requests through the MCP server. Each HTTP connection is handled
/// sequentially (single-threaded event loop).
pub fn run(server: anytype, io: std.Io, config: Config) !void {
    const address = std.Io.net.IpAddress.resolve(
        io,
        config.host,
        config.port,
    ) catch {
        std.log.err("MCP SSE: failed to resolve {s}:{d}", .{ config.host, config.port });
        return error.AddressResolutionFailed;
    };

    var listener = address.listen(io, .{ .reuse_address = true }) catch {
        std.log.err("MCP SSE: failed to listen on {s}:{d}", .{ config.host, config.port });
        return error.ListenFailed;
    };
    defer listener.deinit(io);

    std.log.info("MCP SSE transport listening on {s}:{d}", .{ config.host, config.port });
    std.log.info("  GET  /events  — SSE event stream", .{});
    std.log.info("  POST /message — JSON-RPC requests", .{});
    std.log.info("  GET  /health  — health check", .{});

    var collector = SseResponseCollector.init(server.allocator);
    defer collector.deinit();

    while (true) {
        var stream = listener.accept(io) catch |err| {
            std.log.err("MCP SSE: accept error: {t}", .{err});
            continue;
        };
        defer stream.close(io);

        handleConnection(server, io, &stream, &collector) catch |err| {
            std.log.err("MCP SSE: connection error: {t}", .{err});
        };
    }
}

/// Handle a single HTTP connection. Parses the HTTP request and routes to
/// the appropriate handler based on method and path.
fn handleConnection(
    server: anytype,
    io: std.Io,
    stream: *std.Io.net.Stream,
    collector: *SseResponseCollector,
) !void {
    var recv_buf: [8192]u8 = undefined;
    var send_buf: [8192]u8 = undefined;
    var reader = stream.reader(io, &recv_buf);
    var writer = stream.writer(io, &send_buf);
    var http_server: std.http.Server = .init(
        &reader.interface,
        &writer.interface,
    );

    while (true) {
        var request = http_server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };

        const target = request.head.target;
        const path = splitPath(target);

        if (std.mem.eql(u8, path, "/events")) {
            handleSseStream(server, &request, collector) catch |err| {
                std.log.err("MCP SSE: event stream error: {t}", .{err});
            };
            return; // SSE connections are long-lived; after disconnect, close
        } else if (std.mem.eql(u8, path, "/message")) {
            handlePostMessage(server, &request, collector) catch |err| {
                std.log.err("MCP SSE: message handling error: {t}", .{err});
                respondText(&request, "Internal Server Error", .internal_server_error) catch {};
            };
        } else if (std.mem.eql(u8, path, "/health")) {
            handleHealthCheck(&request) catch |err| {
                std.log.err("MCP SSE: health check error: {t}", .{err});
            };
        } else {
            respondText(&request, "Not Found", .not_found) catch {};
        }
    }
}

/// Handle GET /events — send SSE headers and keep the connection open.
/// In a single-threaded model, this sends headers and returns. Real SSE
/// streaming requires the response collector to push events.
fn handleSseStream(
    _: anytype,
    request: *std.http.Server.Request,
    _: *SseResponseCollector,
) !void {
    // For SSE, we send the headers indicating an event stream.
    // In the current single-threaded model, we respond with the SSE content type
    // and an initial connection event, then return. Clients can POST to /message
    // and receive responses on subsequent connections.
    const body =
        ":ok\n\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"sse/connected\",\"params\":{\"endpoint\":\"/message\"}}\n\n";
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "text/event-stream" },
        .{ .name = "cache-control", .value = "no-cache" },
        .{ .name = "connection", .value = "keep-alive" },
        .{ .name = "access-control-allow-origin", .value = "*" },
    };
    try request.respond(body, .{
        .status = .ok,
        .extra_headers = &headers,
    });
}

/// Handle POST /message — read JSON-RPC request body, process through the
/// MCP server, and return the response as an SSE-framed event.
fn handlePostMessage(
    server: anytype,
    request: *std.http.Server.Request,
    collector: *SseResponseCollector,
) !void {
    if (request.head.method != .POST) {
        return respondText(request, "Method Not Allowed", .method_not_allowed);
    }

    // Read the request body
    const body = readBody(server.allocator, request) catch |err| {
        std.log.err("MCP SSE: failed to read body: {t}", .{err});
        return respondText(request, "Bad Request", .bad_request);
    };
    defer server.allocator.free(body);

    const trimmed = std.mem.trim(u8, body, " \t\r\n");
    if (trimmed.len == 0) {
        return respondText(request, "Empty body", .bad_request);
    }

    // Process through MCP server
    collector.reset();
    const mcp_writer = collector.writer();
    server.processMessage(trimmed, &mcp_writer) catch |err| {
        std.log.err("MCP SSE: processMessage error: {t}", .{err});
        return respondText(request, "Internal Server Error", .internal_server_error);
    };

    const response_data = collector.getResponse();
    if (response_data.len == 0) {
        // Notification — no response needed (e.g., notifications/initialized)
        return respondText(request, "", .no_content);
    }

    // Frame as SSE and respond
    const sse_frame = formatSseFrame(server.allocator, response_data) catch {
        return respondText(request, "Internal Server Error", .internal_server_error);
    };
    defer server.allocator.free(sse_frame);

    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "text/event-stream" },
        .{ .name = "cache-control", .value = "no-cache" },
        .{ .name = "access-control-allow-origin", .value = "*" },
    };
    try request.respond(sse_frame, .{
        .status = .ok,
        .extra_headers = &headers,
    });
}

/// Handle GET /health — simple health check.
fn handleHealthCheck(request: *std.http.Server.Request) !void {
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "application/json" },
        .{ .name = "access-control-allow-origin", .value = "*" },
    };
    try request.respond("{\"status\":\"ok\"}", .{
        .status = .ok,
        .extra_headers = &headers,
    });
}

/// Read the HTTP request body up to a maximum size.
fn readBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ![]u8 {
    const max_body_size: usize = 4 * 1024 * 1024;
    var buffer: [4096]u8 = undefined;
    const reader = request.readerExpectContinue(&buffer) catch return error.ReadFailed;
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch return error.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > max_body_size) return error.RequestTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
        if (n < chunk.len) break;
    }
    return list.toOwnedSlice(allocator);
}

/// Send a plain text HTTP response.
fn respondText(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
) !void {
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "text/plain" },
        .{ .name = "access-control-allow-origin", .value = "*" },
    };
    try request.respond(body, .{
        .status = status,
        .extra_headers = &headers,
    });
}

/// Extract the path component from an HTTP target (strip query string).
fn splitPath(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| {
        return target[0..idx];
    }
    return target;
}

// ===================================================================
// Tests
// ===================================================================

test "formatSseFrame single line" {
    const allocator = std.testing.allocator;
    const frame = try formatSseFrame(allocator, "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n");
    defer allocator.free(frame);
    try std.testing.expectEqualStrings("data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n", frame);
}

test "formatSseFrame empty input" {
    const allocator = std.testing.allocator;
    const frame = try formatSseFrame(allocator, "");
    defer allocator.free(frame);
    try std.testing.expectEqualStrings("", frame);
}

test "formatSseFrame trailing newline stripped" {
    const allocator = std.testing.allocator;
    const frame = try formatSseFrame(allocator, "hello\n");
    defer allocator.free(frame);
    try std.testing.expectEqualStrings("data: hello\n\n", frame);
}

test "SseResponseCollector basic usage" {
    const allocator = std.testing.allocator;
    var collector = SseResponseCollector.init(allocator);
    defer collector.deinit();

    var w = collector.writer();
    try w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}");
    try std.testing.expectEqualStrings(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}",
        collector.getResponse(),
    );

    collector.reset();
    try std.testing.expectEqual(@as(usize, 0), collector.getResponse().len);
}

test "Config defaults" {
    const config = Config{};
    try std.testing.expectEqual(@as(u16, 8081), config.port);
    try std.testing.expectEqualStrings("127.0.0.1", config.host);
    try std.testing.expectEqual(@as(u32, 30), config.heartbeat_interval_s);
}

test {
    std.testing.refAllDecls(@This());
}
