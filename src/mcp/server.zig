const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const json_helpers = @import("json_helpers.zig");
const state = @import("state.zig");
const rpc = @import("rpc.zig");
const shutdown = @import("shutdown.zig");

const DEFAULT_HTTP_PORT: u16 = 8080;
pub const HTTP_PORT_ENV = "ABI_MCP_HTTP_PORT";
pub const HTTP_TOKEN_ENV = "ABI_MCP_HTTP_TOKEN";

// Resolved HTTP listen port. The MCP executable's `main` reads the
// `ABI_MCP_HTTP_PORT` value from the captured process environment (which lives
// in the `abi` module, reachable from `main` but not from this transport file
// under the repo import rules) and pushes it here via `setHttpPort`. Defaults
// to `DEFAULT_HTTP_PORT` when unset/invalid.
var configured_http_port: u16 = DEFAULT_HTTP_PORT;
var configured_http_token: ?[]const u8 = null;

const JsonRpcRequest = protocol.JsonRpcRequest;
const McpMethod = protocol.McpMethod;
const validateRequest = protocol.validateRequest;
const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const valueToJson = json_helpers.valueToJson;
const appendJsonString = json_helpers.appendJsonString;

// --- Shutdown Coordination ---

pub const requestShutdown = shutdown.request;
pub const isShutdownRequested = shutdown.isRequested;
pub const installSignalHandlers = shutdown.installSignalHandlers;
pub const processJsonRpc = rpc.processJsonRpc;

// --- Stdio Transport ---

pub fn runStdioLoop(allocator: std.mem.Allocator, io: std.Io) !void {
    var read_buf: [4096]u8 = undefined;
    var line_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer line_buf.deinit(allocator);

    // Portable stdin read via the std `Io` File API (works on POSIX + Windows);
    // replaces the POSIX-only `std.posix.read(STDIN_FILENO, ...)`.
    const stdin = std.Io.File.stdin();
    while (!isShutdownRequested()) {
        var bufs = [_][]u8{&read_buf};
        const n = stdin.readStreaming(io, &bufs) catch break;
        if (n == 0) break;

        for (read_buf[0..n]) |byte| {
            if (byte == '\n') {
                const line = std.mem.trimEnd(u8, line_buf.items, "\r");
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                processRequest(arena.allocator(), io, line) catch |err| {
                    std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
                };
                line_buf.clearRetainingCapacity();
                continue;
            }

            if (line_buf.items.len >= MAX_REQUEST_SIZE) {
                std.log.warn("dropping overlong MCP request line", .{});
                writeError(io, null, -32700, "Parse error");
                line_buf.clearRetainingCapacity();
                continue;
            }

            try line_buf.append(allocator, byte);
        }
    }

    if (line_buf.items.len > 0) {
        const line = std.mem.trimEnd(u8, line_buf.items, "\r");
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        processRequest(arena.allocator(), io, line) catch |err| {
            std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
        };
    }
}

fn processRequest(allocator: std.mem.Allocator, io: std.Io, line: []const u8) !void {
    if (line.len == 0) return;

    validateRequest(line) catch |err| {
        const code: i32 = switch (err) {
            error.RequestTooLarge => -32700,
            error.InvalidJsonFormat => -32700,
            else => -32700,
        };
        writeError(io, null, code, "Parse error");
        return err;
    };

    const request = std.json.parseFromSlice(JsonRpcRequest, allocator, line, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        writeError(io, null, -32700, "Parse error");
        return err;
    };
    defer request.deinit();

    if (!std.mem.eql(u8, request.value.jsonrpc, "2.0")) {
        writeError(io, request.value.id, -32600, "Invalid Request");
        return;
    }

    const method = McpMethod.fromString(request.value.method);
    const result_json = switch (method) {
        .initialize => handlers.handleInitializeJson(allocator, request.value.params) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/list" => handlers.handleToolsListJson(allocator) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/call" => handlers.handleToolsCallJson(allocator, request.value.params) catch |err| {
            writeError(io, request.value.id, -32603, handlers.errorMessage(err));
            return err;
        },
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => blk: {
            // Only signal shutdown; `main` tears down the shared scheduler/store
            // after it joins the HTTP thread, so teardown never races a peer
            // transport's in-flight tool call (see src/mcp/main.zig).
            requestShutdown();
            break :blk try allocator.dupe(u8, "null");
        },
        .@"resources/list" => try allocator.dupe(u8, "{\"resources\":[]}"),
        .@"prompts/list" => try allocator.dupe(u8, "{\"prompts\":[]}"),
        .unknown => {
            writeError(io, request.value.id, -32601, "Method not found");
            return;
        },
    };
    defer allocator.free(result_json);

    writeResult(allocator, io, request.value.id, result_json);
}

fn buildError(gpa: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), id: ?std.json.Value, code: i32, message: []const u8) !void {
    try buf.appendSlice(gpa, "{\"jsonrpc\":\"2.0\"");
    try appendId(gpa, buf, id);
    try buf.appendSlice(gpa, ",\"error\":{\"code\":");
    try buf.print(gpa, "{d}", .{code});
    try buf.appendSlice(gpa, ",\"message\":");
    try appendJsonString(buf, gpa, message);
    try buf.appendSlice(gpa, "}}\n");
}

fn writeError(io: std.Io, id: ?std.json.Value, code: i32, message: []const u8) void {
    const gpa = std.heap.page_allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(gpa);

    // Don't drop the error frame silently: log like the write path below so an
    // OOM while serializing leaves a trace instead of a vanished response.
    buildError(gpa, &buf, id, code, message) catch |err| {
        std.log.warn("failed to build MCP error response: {s}", .{@errorName(err)});
        return;
    };

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP error response: {s}", .{@errorName(err)});
    };
}

fn buildResult(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), id: ?std.json.Value, result_json: []const u8) !void {
    try buf.appendSlice(allocator, "{\"jsonrpc\":\"2.0\"");
    try appendId(allocator, buf, id);
    try buf.appendSlice(allocator, ",\"result\":");
    try buf.appendSlice(allocator, result_json);
    try buf.appendSlice(allocator, "}\n");
}

fn writeResult(allocator: std.mem.Allocator, io: std.Io, id: ?std.json.Value, result_json: []const u8) void {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);

    buildResult(allocator, &buf, id, result_json) catch |err| {
        std.log.warn("failed to build MCP response: {s}", .{@errorName(err)});
        return;
    };

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP response: {s}", .{@errorName(err)});
    };
}

fn appendId(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), id: ?std.json.Value) !void {
    if (id) |value| {
        const id_json = try valueToJson(value, allocator);
        defer allocator.free(id_json);
        try out.appendSlice(allocator, ",\"id\":");
        try out.appendSlice(allocator, id_json);
    } else {
        try out.appendSlice(allocator, ",\"id\":null");
    }
}

fn writeStdoutAll(io: std.Io, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll(bytes);
    try stdout.flush();
}

// --- HTTP/SSE Transport ---

pub fn runHttpServer(allocator: std.mem.Allocator, io: std.Io) void {
    const net = std.Io.net;
    const port = configuredHttpPort();
    const address = net.IpAddress.parseIp4("127.0.0.1", port) catch {
        std.log.warn("failed to parse HTTP listen address", .{});
        return;
    };
    var tcp_server = address.listen(io, .{
        .reuse_address = true,
    }) catch |err| {
        std.log.warn("failed to bind HTTP server on port {d}: {s}; set {s} to choose another loopback port", .{ port, @errorName(err), HTTP_PORT_ENV });
        return;
    };
    defer tcp_server.deinit(io);

    std.log.info("MCP HTTP/SSE server listening on http://127.0.0.1:{d}, auth={s}", .{ port, if (configuredHttpToken() == null) "off" else "bearer" });

    while (!isShutdownRequested()) {
        const conn = tcp_server.accept(io) catch |err| {
            if (isShutdownRequested()) break;
            std.log.warn("HTTP accept error: {s}", .{@errorName(err)});
            continue;
        };
        handleHttpConnectionWithAuth(allocator, io, conn, configuredHttpToken()) catch |err| {
            std.log.warn("HTTP connection error: {s}", .{@errorName(err)});
        };
    }
}

pub fn wakeHttpServer(io: std.Io) void {
    const net = std.Io.net;
    const address = net.IpAddress.parseIp4("127.0.0.1", configuredHttpPort()) catch return;
    const stream = address.connect(io, .{ .mode = .stream }) catch return;
    defer stream.close(io);
}

/// Set the HTTP listen port from a raw env value (typically
/// `ABI_MCP_HTTP_PORT`). Empty/invalid/out-of-range/zero falls back to the
/// default. Called once at MCP startup before the server thread spawns.
pub fn setHttpPort(raw: ?[]const u8) void {
    configured_http_port = if (raw) |r| (parseHttpPort(r) orelse DEFAULT_HTTP_PORT) else DEFAULT_HTTP_PORT;
}

pub fn configuredHttpPort() u16 {
    return configured_http_port;
}

/// Set optional bearer-token auth for HTTP/SSE from a raw env value (typically
/// `ABI_MCP_HTTP_TOKEN`). Empty/whitespace disables HTTP auth; stdio is never
/// affected.
pub fn setHttpToken(raw: ?[]const u8) void {
    configured_http_token = if (raw) |r| parseHttpToken(r) else null;
}

pub fn configuredHttpToken() ?[]const u8 {
    return configured_http_token;
}

fn parseHttpPort(raw: []const u8) ?u16 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    const port = std.fmt.parseInt(u16, trimmed, 10) catch return null;
    if (port == 0) return null;
    return port;
}

fn parseHttpToken(raw: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    return trimmed;
}

/// Outcome of reading a full HTTP request off the connection.
const HttpReadResult = union(enum) {
    /// Bytes received, framed by `\r\n\r\n` (and Content-Length when present).
    request: []const u8,
    /// The peer closed before any bytes arrived.
    empty,
    /// The request exceeded MAX_REQUEST_SIZE before completing.
    too_large,
};

/// Read a complete HTTP request into `buf`, reassembling across TCP segments.
///
/// The request can span multiple reads: clients routinely flush headers before
/// the body, so a single `conn.read` may return only part of the request. This
/// first accumulates until the `\r\n\r\n` header terminator, parses any
/// `Content-Length`, then continues reading until the declared body has arrived.
///
/// Termination is bounded three ways so the loop can never hang or grow without
/// limit: `buf` is a fixed MAX_REQUEST_SIZE-sized array (a full buffer yields
/// `.too_large`), a 0-length read (EOF) stops the loop, and the body target is
/// capped at the buffer length.
fn readHttpRequest(io: std.Io, conn: std.Io.net.Stream, buf: []u8) HttpReadResult {
    var total: usize = 0;
    var header_end: ?usize = null; // index just past "\r\n\r\n"
    var want_total: ?usize = null; // header_end + Content-Length, when known

    while (true) {
        if (header_end == null) {
            if (std.mem.indexOf(u8, buf[0..total], "\r\n\r\n")) |idx| {
                const end = idx + 4;
                header_end = end;
                const declared = parseContentLength(buf[0..end]) orelse 0;
                // Cap the body target at the buffer; an over-cap declaration is
                // caught as `.too_large` once the buffer fills.
                const target = end + declared;
                want_total = if (target > buf.len) buf.len + 1 else target;
            }
        }

        if (want_total) |want| {
            if (total >= want) break;
        }

        if (total >= buf.len) {
            // Buffer full before the request completed: either the headers alone
            // overflow, or the declared/streamed body exceeds the cap.
            return .too_large;
        }

        var rv: [1][]u8 = .{buf[total..]};
        const n = conn.read(io, &rv) catch break;
        if (n == 0) break; // EOF / peer closed
        total += n;
    }

    if (total == 0) return .empty;
    return .{ .request = buf[0..total] };
}

/// Parse the `Content-Length` request header (case-insensitive) from the raw
/// header block. Returns null when absent or unparseable.
fn parseContentLength(header_block: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, header_block, "\r\n");
    _ = lines.next(); // skip the request line
    while (lines.next()) |line| {
        if (line.len == 0) break; // end of headers
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        if (!std.ascii.eqlIgnoreCase(key, "Content-Length")) continue;
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        return std.fmt.parseInt(usize, value, 10) catch null;
    }
    return null;
}

fn handleHttpConnection(allocator: std.mem.Allocator, io: std.Io, conn: std.Io.net.Stream) !void {
    try handleHttpConnectionWithAuth(allocator, io, conn, null);
}

fn handleHttpConnectionWithAuth(allocator: std.mem.Allocator, io: std.Io, conn: std.Io.net.Stream, bearer_token: ?[]const u8) !void {
    defer conn.close(io);

    var read_buf: [MAX_REQUEST_SIZE]u8 = undefined;
    const raw_request = switch (readHttpRequest(io, conn, &read_buf)) {
        .empty => return,
        .too_large => {
            const err_resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}";
            try writeHttpAll(io, conn, err_resp);
            return;
        },
        .request => |req| req,
    };

    var line_end: usize = 0;
    while (line_end < raw_request.len and raw_request[line_end] != '\n') : (line_end += 1) {}
    const request_line = std.mem.trimEnd(u8, raw_request[0..line_end], "\r");

    var req_it = std.mem.splitScalar(u8, request_line, ' ');
    const http_method = req_it.next() orelse return;
    const path = req_it.next() orelse return;

    if (bearer_token) |token| {
        if (!hasBearerToken(raw_request, token)) {
            try writeUnauthorized(io, conn);
            return;
        }
    }

    if (std.mem.eql(u8, http_method, "GET") and std.mem.eql(u8, path, "/sse")) {
        const sse_response =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n" ++
            "event: endpoint\r\ndata: /message\r\n\r\n";
        try writeHttpAll(io, conn, sse_response);
        return;
    }

    if (std.mem.eql(u8, http_method, "POST") and std.mem.eql(u8, path, "/message")) {
        const body = findHttpBody(raw_request) orelse {
            const err_resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}";
            try writeHttpAll(io, conn, err_resp);
            return;
        };

        if (body.len > MAX_REQUEST_SIZE) {
            const err_resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}";
            try writeHttpAll(io, conn, err_resp);
            return;
        }

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const result_json = processJsonRpc(arena.allocator(), body) catch |err| {
            const err_body = try std.fmt.allocPrint(arena.allocator(), "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":-32603,\"message\":\"{s}\"}}}}", .{handlers.errorMessage(err)});
            const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{err_body.len});
            try writeHttpAll(io, conn, header);
            try writeHttpAll(io, conn, err_body);
            return;
        };

        const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{result_json.len});
        try writeHttpAll(io, conn, header);
        try writeHttpAll(io, conn, result_json);
        return;
    }

    const resp = "HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n";
    try writeHttpAll(io, conn, resp);
}

fn headerValue(raw: []const u8, name: []const u8) ?[]const u8 {
    const header_block = if (std.mem.indexOf(u8, raw, "\r\n\r\n")) |idx| raw[0..idx] else raw;
    var lines = std.mem.splitSequence(u8, header_block, "\r\n");
    _ = lines.next(); // request line
    while (lines.next()) |line| {
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        if (!std.ascii.eqlIgnoreCase(key, name)) continue;
        return std.mem.trim(u8, line[colon + 1 ..], " \t");
    }
    return null;
}

fn hasBearerToken(raw: []const u8, token: []const u8) bool {
    const value = headerValue(raw, "Authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, value, prefix)) return false;
    return std.mem.eql(u8, value[prefix.len..], token);
}

fn writeUnauthorized(io: std.Io, conn: std.Io.net.Stream) !void {
    const body = "{\"error\":\"unauthorized\"}";
    var buffer: [256]u8 = undefined;
    const resp = try std.fmt.bufPrint(
        &buffer,
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Bearer\r\nConnection: close\r\n\r\n{s}",
        .{ body.len, body },
    );
    try writeHttpAll(io, conn, resp);
}

fn writeHttpAll(io: std.Io, conn: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stream_writer = conn.writer(io, &buffer);
    const writer = &stream_writer.interface;
    try writer.writeAll(bytes);
    try writer.flush();
}

fn findHttpBody(raw: []const u8) ?[]const u8 {
    const needle = "\r\n\r\n";
    const idx = std.mem.indexOf(u8, raw, needle) orelse return null;
    const body_start = idx + needle.len;
    if (body_start >= raw.len) return null;
    return raw[body_start..];
}

test "MCP HTTP port parser accepts valid user ports" {
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort("18080"));
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort(" 18080\n"));
}

// Keep invalid overrides non-fatal so stdio mode still works when users typo the environment.
test "MCP HTTP port parser rejects invalid overrides" {
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort(""));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("0"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("65536"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("not-a-port"));
}

test "MCP HTTP token parser trims and rejects empty overrides" {
    try std.testing.expectEqualStrings("local-token", parseHttpToken(" local-token\n") orelse return error.MissingToken);
    try std.testing.expect(parseHttpToken("") == null);
    try std.testing.expect(parseHttpToken(" \t\r\n") == null);
}

test "MCP HTTP Content-Length header parser" {
    try std.testing.expectEqual(
        @as(?usize, 42),
        parseContentLength("POST /message HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
    );
    // Header name match is case-insensitive and tolerates surrounding whitespace.
    try std.testing.expectEqual(
        @as(?usize, 7),
        parseContentLength("POST /message HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
    );
    // Absent header -> null (caller treats the body as whatever was buffered).
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /message HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    // Garbage value -> null rather than a wrong length.
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /message HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
    );
}

test "MCP HTTP Authorization bearer parser" {
    const raw =
        "POST /message HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST /message HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST /message HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
}

extern fn getsockname(sockfd: std.posix.fd_t, addr: *std.posix.sockaddr, addrlen: *std.posix.socklen_t) c_int;

// Bind a 127.0.0.1 listener on an ephemeral port and return both it and the
// kernel-assigned port (mirrors src/testing/test_helpers.zig's loopback helper).
fn bindLoopback(io: std.Io) !struct { server: std.Io.net.Server, port: u16 } {
    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var srv = try address.listen(io, .{ .mode = .stream, .reuse_address = true });
    errdefer srv.deinit(io);

    var addr: std.posix.sockaddr = undefined;
    var addrlen: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    if (getsockname(srv.socket.handle, &addr, &addrlen) != 0) return error.GetSockNameFailed;
    const addr_in: *const std.posix.sockaddr.in = @ptrCast(@alignCast(&addr));
    const port = std.mem.toNative(u16, addr_in.port, .big);
    if (port == 0) return error.PortIsZero;

    return .{ .server = srv, .port = port };
}

fn readHttpResponse(io: std.Io, conn: std.Io.net.Stream, buf: []u8) ![]const u8 {
    var total: usize = 0;
    while (total < buf.len) {
        var rv: [1][]u8 = .{buf[total..]};
        const n = conn.read(io, &rv) catch break;
        if (n == 0) break;
        total += n;
    }
    return buf[0..total];
}

// Regression: a POST whose headers and body arrive in separate TCP segments must
// still be parsed in full. Before the read loop, handleHttpConnection performed a
// single read and treated everything after the first CRLFCRLF as the complete
// body, truncating multi-segment requests into invalid JSON.
test "MCP HTTP transport reassembles a multi-segment request body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    // A well-formed JSON-RPC ping whose body is intentionally split from its
    // headers across two writes.
    const body = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}";
    const headers = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n",
        .{body.len},
    );
    defer allocator.free(headers);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    // Segment 1: headers only (flushed), then segment 2: body only.
    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(headers);
        try sw.interface.flush();
        try sw.interface.writeAll(body);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnection(allocator, io, conn);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);

    const resp_body = findHttpBody(resp) orelse return error.NoResponseBody;
    // ping returns an empty-object result, never a parse error.
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"result\":{}") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"id\":7") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "error") == null);
}

// Happy path: the common single-write request must behave exactly as before.
test "MCP HTTP transport handles a single-write request" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body.len, body },
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnection(allocator, io, conn);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);

    const resp_body = findHttpBody(resp) orelse return error.NoResponseBody;
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"result\":{}") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"id\":1") != null);
}

test "MCP HTTP transport requires bearer token when configured" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body.len, body },
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnectionWithAuth(allocator, io, conn, "local-token");

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "401 Unauthorized") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "WWW-Authenticate: Bearer") != null);
}

test "MCP HTTP transport accepts configured bearer token" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nAuthorization: Bearer local-token\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body.len, body },
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnectionWithAuth(allocator, io, conn, "local-token");

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    const resp_body = findHttpBody(resp) orelse return error.NoResponseBody;
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"result\":{}") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"id\":1") != null);
}
