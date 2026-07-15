//! REST listener for WDBX (Transport/API Layer).
//!
//! A real HTTP/1.1 request router over the in-process Store, exposing the
//! north-star endpoints: POST /insert, POST /query, POST /verify, GET /health,
//! GET /stats. The routing core (`route`) is a pure function (method, path,
//! body) -> Response and is fully unit-tested without binding a socket; `serve`
//! wraps it on a 127.0.0.1 loopback listener using the same std.Io.net pattern
//! as the MCP HTTP transport. This is a local single-node listener, not a
//! hardened public-facing service.
//!
//! ## TLS
//!
//! Optional TLS configuration is read from `ABI_WDBX_TLS_CERT` /
//! `ABI_WDBX_TLS_KEY` environment variables. When both are set, the server
//! validates the cert/key files and logs that TLS is intended for proxy-based
//! deployment. **Native TLS termination is not linked in this build** — the
//! server always serves plain HTTP. Deploy behind nginx, Caddy, or haproxy for
//! HTTPS. See `tls_config.zig` for details.

const std = @import("std");
const wdbx = @import("mod.zig");
const parse = @import("rest_parse.zig");
const handlers = @import("rest_handlers.zig");
const RateLimiter = @import("rate_limiter.zig").RateLimiter;
const TlsConfig = @import("tls_config.zig").TlsConfig;

pub const MAX_REQUEST_SIZE: usize = 64 * 1024;

pub const Response = parse.Response;
pub const json = parse.json;
pub const REST_TOKEN_ENV = parse.REST_TOKEN_ENV;
pub const VectorParseError = parse.VectorParseError;
pub const parseVectorField = parse.parseVectorField;
pub const vectorParseErrorResponse = parse.vectorParseErrorResponse;
pub const escapeJsonString = parse.escapeJsonString;
pub const strField = parse.strField;
pub const reasonPhrase = parse.reasonPhrase;
pub const findBody = parse.findBody;
pub const HttpReadResult = parse.HttpReadResult;
pub const readHttpRequest = parse.readHttpRequest;
pub const parseContentLength = parse.parseContentLength;
pub const headerValue = parse.headerValue;
pub const hasBearerToken = parse.hasBearerToken;
pub const loadBearerToken = parse.loadBearerToken;

pub const route = handlers.route;

const AuthConfig = struct {
    bearer_token: ?[]const u8 = null,
};

pub fn serve(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, port: u16) !void {
    const bearer_token = loadBearerToken();
    var rate_limiter = RateLimiter.initFromEnv();

    // Check for TLS configuration (disclosed partial — no native TLS server).
    const tls_config = TlsConfig.fromEnv(io);
    if (tls_config) |tls| {
        std.log.info("WDBX REST TLS configured: cert={s}, key={s}", .{ tls.cert_path, tls.key_path });
        std.log.warn("WDBX REST: native TLS termination is NOT linked in this build", .{});
        std.log.warn("WDBX REST: deploy behind a TLS-terminating reverse proxy (nginx, Caddy, haproxy)", .{});
        std.log.warn("WDBX REST: cert/key = {s} / {s} (validated)", .{ tls.cert_path, tls.key_path });
    }

    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.log.info("WDBX REST listening on http://127.0.0.1:{d} (/insert /query /verify /health /stats), auth={s}, rate_limit_cap={d}/s, tls={s}", .{
        port,
        if (bearer_token == null) "off" else "bearer",
        rate_limiter.refill_rate,
        if (tls_config != null) "configured" else "off",
    });

    while (true) {
        const conn = server.accept(io) catch |err| {
            std.log.warn("WDBX REST accept failed: {s}", .{@errorName(err)});
            continue;
        };
        handleConnectionWithAuth(allocator, io, store, conn, .{ .bearer_token = bearer_token }, &rate_limiter) catch |err| {
            std.log.warn("WDBX REST request failed: {s}", .{@errorName(err)});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream) !void {
    try handleConnectionWithAuth(allocator, io, store, conn, .{}, null);
}

fn handleConnectionWithAuth(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream, auth: AuthConfig, rate_limiter: ?*RateLimiter) !void {
    defer conn.close(io);
    var read_buf: [MAX_REQUEST_SIZE]u8 = undefined;
    const raw = switch (readHttpRequest(io, conn, &read_buf)) {
        .empty => return,
        .too_large => {
            const err_resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}";
            try writeAll(io, conn, err_resp);
            return;
        },
        .request => |req| req,
    };

    var line_end: usize = 0;
    while (line_end < raw.len and raw[line_end] != '\n') : (line_end += 1) {}
    const request_line = std.mem.trimEnd(u8, raw[0..line_end], "\r");
    var it = std.mem.splitScalar(u8, request_line, ' ');
    const method = it.next() orelse return;
    const path = it.next() orelse return;
    const body = findBody(raw);

    if (auth.bearer_token) |token| {
        if (!hasBearerToken(raw, token)) {
            const body_unauthorized = "{\"error\":\"unauthorized\"}";
            const err_resp = try std.fmt.allocPrint(
                allocator,
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Bearer\r\nConnection: close\r\n\r\n{s}",
                .{ body_unauthorized.len, body_unauthorized },
            );
            defer allocator.free(err_resp);
            try writeAll(io, conn, err_resp);
            return;
        }
    }

    // Rate-limit check: acquire a token before processing the request.
    if (rate_limiter) |rl| {
        if (!rl.acquire()) {
            const retry_after: u64 = 1; // second
            const body_limited = "{\"error\":\"too many requests\"}";
            const err_resp = try std.fmt.allocPrint(
                allocator,
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ body_limited.len, retry_after, body_limited },
            );
            defer allocator.free(err_resp);
            try writeAll(io, conn, err_resp);
            return;
        }
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var resp = route(arena.allocator(), store, method, path, body) catch |err| blk: {
        std.log.warn("WDBX REST route failed: {s}", .{@errorName(err)});
        break :blk Response{ .status = 500, .body = @constCast("{\"error\":\"internal\"}") };
    };

    // Augment /stats response with rate limiter data when available.
    if (rate_limiter) |rl| {
        if (std.mem.eql(u8, path, "/stats") and resp.status == 200) {
            const rl_json = try rl.statsJson(arena.allocator());
            const trimmed = std.mem.trimEnd(u8, resp.body, "}");
            resp.body = try std.fmt.allocPrint(arena.allocator(), "{s},{s}}}", .{ trimmed, rl_json });
        }
    }

    const header = try std.fmt.allocPrint(
        arena.allocator(),
        "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
        .{ resp.status, reasonPhrase(resp.status), resp.body.len },
    );
    var write_buf: [1024]u8 = undefined;
    var sw = conn.writer(io, &write_buf);
    const w = &sw.interface;
    try w.writeAll(header);
    try w.writeAll(resp.body);
    try w.flush();
}

fn writeAll(io: std.Io, conn: std.Io.net.Stream, bytes: []const u8) !void {
    var write_buf: [1024]u8 = undefined;
    var sw = conn.writer(io, &write_buf);
    const w = &sw.interface;
    try w.writeAll(bytes);
    try w.flush();
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

test "rest: HTTP transport reassembles a multi-segment request body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"key\":\"agent:abbey\",\"value\":\"trained\"}";
    const headers = try std.fmt.allocPrint(
        allocator,
        "POST /insert HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n",
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
    try handleConnection(allocator, io, &store, conn);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);

    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "400") == null);
    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abbey\"}");
    defer q.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "trained") != null);
}

test "rest: HTTP transport handles a single-write request" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"key\":\"agent:abi\",\"value\":\"routed\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /insert HTTP/1.1\r\nContent-Length: {d}\r\n\r\n{s}",
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
    try handleConnection(allocator, io, &store, conn);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
}

test "rest: HTTP transport requires bearer token when configured" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"key\":\"agent:abi\",\"value\":\"blocked\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /insert HTTP/1.1\r\nContent-Length: {d}\r\n\r\n{s}",
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
    try handleConnectionWithAuth(allocator, io, &store, conn, .{ .bearer_token = "local-token" }, null);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "401 Unauthorized") != null);

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abi\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 404), q.status);
}

test "rest: HTTP transport accepts configured bearer token" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"key\":\"agent:abi\",\"value\":\"authorized\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /insert HTTP/1.1\r\nAuthorization: Bearer local-token\r\nContent-Length: {d}\r\n\r\n{s}",
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
    try handleConnectionWithAuth(allocator, io, &store, conn, .{ .bearer_token = "local-token" }, null);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abi\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "authorized") != null);
}

test "rest: rate limiter returns 429 when tokens exhausted" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    var rl = RateLimiter.init(1, 10); // capacity 1

    // Request 1: consumes the only token, should succeed.
    {
        var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
        const client = try caddr.connect(io, .{ .mode = .stream });
        defer client.close(io);
        {
            var wb: [512]u8 = undefined;
            var sw = client.writer(io, &wb);
            try sw.interface.writeAll("GET /health HTTP/1.1\r\n\r\n");
            try sw.interface.flush();
        }
        const conn = try bound.server.accept(io);
        try handleConnectionWithAuth(allocator, io, &store, conn, .{}, &rl);

        var rb: [2048]u8 = undefined;
        _ = try readHttpResponse(io, client, &rb);
    }

    // Request 2: no tokens left, should get 429.
    {
        var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
        const client = try caddr.connect(io, .{ .mode = .stream });
        defer client.close(io);
        {
            var wb: [512]u8 = undefined;
            var sw = client.writer(io, &wb);
            try sw.interface.writeAll("GET /health HTTP/1.1\r\n\r\n");
            try sw.interface.flush();
        }
        const conn = try bound.server.accept(io);
        try handleConnectionWithAuth(allocator, io, &store, conn, .{}, &rl);

        var resp_buf: [2048]u8 = undefined;
        const resp = try readHttpResponse(io, client, &resp_buf);
        try std.testing.expect(std.mem.indexOf(u8, resp, "429") != null);
        try std.testing.expect(std.mem.indexOf(u8, resp, "Retry-After") != null);
        try std.testing.expect(std.mem.indexOf(u8, resp, "too many requests") != null);
    }
}

test "rest: /stats includes rate limiter fields when rate_limiter is provided" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    // Capacity 100, but we expect the stats request itself consumes a token,
    // so allowed will be >= 1 (the server's acquire for the /stats call).
    var rl = RateLimiter.init(100, 10);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll("GET /stats HTTP/1.1\r\n\r\n");
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleConnectionWithAuth(allocator, io, &store, conn, .{}, &rl);

    var resp_buf: [4096]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "rate_limit") != null);
    // allowed=1 because the stats request itself consumed the first token
    try std.testing.expect(std.mem.indexOf(u8, resp, "\"allowed\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "\"capacity\":100") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "\"tokens\":99") != null);
}

test {
    std.testing.refAllDecls(@This());
}
