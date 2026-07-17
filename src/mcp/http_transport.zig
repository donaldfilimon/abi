const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const rpc = @import("rpc.zig");
const shutdown = @import("shutdown.zig");
const parse = @import("http_parse.zig");
const foundation_http = @import("abi").foundation.http;

const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const isShutdownRequested = shutdown.isRequested;
const processJsonRpc = rpc.processJsonRpc;

const DEFAULT_HTTP_PORT: u16 = 8080;
pub const HTTP_PORT_ENV = "ABI_MCP_HTTP_PORT";
pub const HTTP_TOKEN_ENV = "ABI_MCP_HTTP_TOKEN";

var configured_http_port: u16 = DEFAULT_HTTP_PORT;
var configured_http_token: ?[]const u8 = null;

// --- HTTP/SSE Transport ---

pub fn runHttpServer(allocator: std.mem.Allocator, io: std.Io) void {
    const net = std.Io.net;
    const port = configuredHttpPort();
    const address = net.IpAddress.parseIp4("127.0.0.1", port) catch {
        std.log.warn("failed to parse HTTP listen address", .{});
        return;
    };

    var server = address.listen(io, .{ .reuse_address = true }) catch |err| {
        std.log.warn("failed to listen on {d}: {s}", .{ port, @errorName(err) });
        return;
    };
    defer server.deinit(io);

    std.log.info("MCP HTTP+SSE transport listening on http://127.0.0.1:{d}/sse and /message", .{port});

    while (true) {
        if (isShutdownRequested()) {
            std.log.info("MCP HTTP transport: shutdown requested, stopping accept loop", .{});
            break;
        }

        const conn = server.accept(io) catch |err| {
            if (isShutdownRequested()) break;
            std.log.warn("MCP HTTP accept failed: {s}", .{@errorName(err)});
            continue;
        };

        const bearer_token = configured_http_token;

        if (bearer_token) |token| {
            handleHttpConnectionWithAuth(allocator, io, conn, token) catch |err| {
                std.log.warn("MCP HTTP request failed: {s}", .{@errorName(err)});
            };
        } else {
            handleHttpConnection(allocator, io, conn) catch |err| {
                std.log.warn("MCP HTTP request failed: {s}", .{@errorName(err)});
            };
        }
    }
}

pub fn setHttpPort(raw: ?[]const u8) void {
    configured_http_port = if (raw) |r| (parse.parseHttpPort(r) orelse DEFAULT_HTTP_PORT) else DEFAULT_HTTP_PORT;
}

pub fn setHttpToken(raw: ?[]const u8) void {
    configured_http_token = if (raw) |r| parse.parseHttpToken(r) else null;
}

fn configuredHttpPort() u16 {
    return configured_http_port;
}

// --- HTTP Request Parsing (loopback-only HTTP/1.1, not a general-purpose parser) ---

const HttpReadResult = union(enum) {
    request: []const u8,
    empty,
    too_large,
};

fn readHttpRequest(io: std.Io, conn: std.Io.net.Stream, buf: []u8) HttpReadResult {
    var total: usize = 0;
    var header_end: ?usize = null;
    var want_total: ?usize = null;

    while (true) {
        if (header_end == null) {
            if (std.mem.indexOf(u8, buf[0..total], "\r\n\r\n")) |idx| {
                const end = idx + 4;
                header_end = end;
                const declared = parse.parseContentLength(buf[0..end]) orelse 0;
                want_total = parse.requestTargetWithinBuffer(end, declared, buf.len) orelse return .too_large;
            }
        }

        if (want_total) |want| {
            if (total >= want) break;
        }

        if (total >= buf.len) {
            return .too_large;
        }

        var rv: [1][]u8 = .{buf[total..]};
        const n = conn.read(io, &rv) catch break;
        if (n == 0) break;
        total += n;
    }

    if (total == 0) return .empty;
    return .{ .request = buf[0..total] };
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
            try foundation_http.writeHttpAll(io, conn, err_resp);
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
        if (!parse.hasBearerToken(raw_request, token)) {
            try writeUnauthorized(io, conn);
            return;
        }
    }

    if (std.mem.eql(u8, http_method, "GET") and std.mem.eql(u8, path, "/sse")) {
        const sse_response =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n" ++
            "event: endpoint\r\ndata: /message\r\n\r\n";
        try foundation_http.writeHttpAll(io, conn, sse_response);
        return;
    }

    if (std.mem.eql(u8, http_method, "POST") and std.mem.eql(u8, path, "/message")) {
        const body = foundation_http.findHttpBody(raw_request) orelse {
            const err_resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}";
            try foundation_http.writeHttpAll(io, conn, err_resp);
            return;
        };

        if (body.len > MAX_REQUEST_SIZE) {
            const err_resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}";
            try foundation_http.writeHttpAll(io, conn, err_resp);
            return;
        }

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const result_json = processJsonRpc(arena.allocator(), body) catch |err| {
            const err_body = try std.fmt.allocPrint(arena.allocator(), "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":-32603,\"message\":\"{s}\"}}}}", .{handlers.errorMessage(err)});
            const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{err_body.len});
            try foundation_http.writeHttpAll(io, conn, header);
            try foundation_http.writeHttpAll(io, conn, err_body);
            return;
        };

        const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{result_json.len});
        try foundation_http.writeHttpAll(io, conn, header);
        try foundation_http.writeHttpAll(io, conn, result_json);
        return;
    }

    const resp = "HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n";
    try foundation_http.writeHttpAll(io, conn, resp);
}

fn writeUnauthorized(io: std.Io, conn: std.Io.net.Stream) !void {
    const body = "{\"error\":\"unauthorized\"}";
    var buffer: [256]u8 = undefined;
    const resp = try std.fmt.bufPrint(
        &buffer,
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Bearer\r\nConnection: close\r\n\r\n{s}",
        .{ body.len, body },
    );
    try foundation_http.writeHttpAll(io, conn, resp);
}

extern fn getsockname(sockfd: std.posix.fd_t, addr: *std.posix.sockaddr, addrlen: *std.posix.socklen_t) c_int;

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

const readHttpResponse = foundation_http.readHttpResponse;

// Regression: a POST whose headers and body arrive in separate TCP segments must
// still be parsed in full. Before the read loop, handleHttpConnection performed a
// single read and treated everything after the first CRLFCRLF as the complete
// body, truncating multi-segment requests into invalid JSON.
test "MCP HTTP transport reassembles a multi-segment request body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

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

    const resp_body = foundation_http.findHttpBody(resp) orelse return error.NoResponseBody;
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

    const resp_body = foundation_http.findHttpBody(resp) orelse return error.NoResponseBody;
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"result\":{}") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"id\":1") != null);
}

test "MCP HTTP transport returns 413 for oversized POST body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nContent-Length: {d}\r\n\r\n",
        .{MAX_REQUEST_SIZE + 1},
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [256]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnection(allocator, io, conn);

    var resp_buf: [512]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "413 Payload Too Large") != null);
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

test "MCP HTTP transport rejects the wrong bearer token" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nAuthorization: Bearer wrong-token\r\nContent-Length: {d}\r\n\r\n{s}",
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
    const resp_body = foundation_http.findHttpBody(resp) orelse return error.NoResponseBody;
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"result\":{}") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_body, "\"id\":1") != null);
}

test {
    std.testing.refAllDecls(@This());
}
