//! REST listener for WDBX (Transport/API Layer).
//!
//! A real HTTP/1.1 request router over the in-process Store, exposing the
//! north-star endpoints: POST /insert, POST /query, POST /verify, GET /health,
//! GET /stats. The routing core (`route`) is a pure function (method, path,
//! body) -> Response and is fully unit-tested without binding a socket; `serve`
//! wraps it on a 127.0.0.1 loopback listener using the same std.Io.net pattern
//! as the MCP HTTP transport. This is a local single-node listener, not a
//! hardened public-facing service.

const std = @import("std");
const wdbx = @import("mod.zig");
const retrieval = @import("retrieval.zig");
const temporal = @import("temporal.zig");
const foundation_time = @import("../../foundation/time.zig");

pub const MAX_REQUEST_SIZE: usize = 64 * 1024;

/// Neutral persona weight for the REST query (no conversation persona context,
/// matching the path-addressed CLI `wdbx query`). Keeps the API-layer ranking
/// honest: semantic × temporal × causal, persona held flat.
fn constPersona(id: u32) f32 {
    _ = id;
    return 0.5;
}

pub const Response = struct {
    status: u16,
    body: []u8, // owned by the caller

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
    }
};

fn json(allocator: std.mem.Allocator, status: u16, comptime fmt: []const u8, args: anytype) !Response {
    return .{ .status = status, .body = try std.fmt.allocPrint(allocator, fmt, args) };
}

/// Pure routing core. Applies the request to `store` and returns a JSON body
/// plus HTTP status. Unknown routes return 404; malformed bodies return 400.
pub fn route(allocator: std.mem.Allocator, store: *wdbx.Store, method: []const u8, path: []const u8, body: []const u8) !Response {
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        return json(allocator, 200, "{{\"status\":\"ok\"}}", .{});
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/stats")) {
        const manifest = try store.exportManifest(allocator);
        return .{ .status = 200, .body = manifest };
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/verify")) {
        const ok = store.verifyBlocks();
        return json(allocator, 200, "{{\"chain_valid\":{},\"blocks\":{d}}}", .{ ok, store.blockCount() });
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/insert")) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
            return json(allocator, 400, "{{\"error\":\"invalid json\"}}", .{});
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return json(allocator, 400, "{{\"error\":\"expected object\"}}", .{}),
        };
        // Block insert: {"profile":..,"metadata":..}
        if (obj.get("profile")) |p_node| {
            const profile = strField(p_node) orelse return json(allocator, 400, "{{\"error\":\"profile must be a string\"}}", .{});
            const metadata = if (obj.get("metadata")) |m| (strField(m) orelse "") else "";
            _ = store.appendBlock(profile, 0, 0, metadata) catch |err|
                return json(allocator, 500, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            return json(allocator, 200, "{{\"inserted\":\"block\",\"blocks\":{d}}}", .{store.blockCount()});
        }
        // KV insert: {"key":..,"value":..}
        const key = if (obj.get("key")) |k| strField(k) else null;
        const value = if (obj.get("value")) |v| strField(v) else null;
        if (key == null or value == null) return json(allocator, 400, "{{\"error\":\"need key+value or profile\"}}", .{});
        store.store(key.?, value.?) catch |err|
            return json(allocator, 500, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
        return json(allocator, 200, "{{\"inserted\":\"kv\"}}", .{});
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/query")) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
            return json(allocator, 400, "{{\"error\":\"invalid json\"}}", .{});
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return json(allocator, 400, "{{\"error\":\"expected object\"}}", .{}),
        };
        // KV lookup: {"key":..}
        if (obj.get("key")) |k| {
            const key = strField(k) orelse return json(allocator, 400, "{{\"error\":\"key must be a string\"}}", .{});
            const val = store.get(key) orelse return json(allocator, 404, "{{\"error\":\"not found\"}}", .{});
            return json(allocator, 200, "{{\"value\":\"{s}\"}}", .{val});
        }
        // Hybrid semantic search: {"vector":[..],"limit":N} — mirrors the CLI
        // `wdbx query <path> <text>` path (semantic × temporal × causal × persona)
        // but takes a pre-embedded query vector, since the REST server lives in
        // the storage layer and must not depend on the AI embedding layer above.
        if (obj.get("vector")) |vec_node| {
            const arr = switch (vec_node) {
                .array => |a| a,
                else => return json(allocator, 400, "{{\"error\":\"vector must be an array\"}}", .{}),
            };
            if (arr.items.len == 0) return json(allocator, 400, "{{\"error\":\"vector must be non-empty\"}}", .{});
            const query_vec = allocator.alloc(f32, arr.items.len) catch
                return json(allocator, 500, "{{\"error\":\"oom\"}}", .{});
            defer allocator.free(query_vec);
            for (arr.items, 0..) |item, i| {
                query_vec[i] = switch (item) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    else => return json(allocator, 400, "{{\"error\":\"vector elements must be numbers\"}}", .{}),
                };
            }
            const limit: usize = if (obj.get("limit")) |l| switch (l) {
                .integer => |n| if (n > 0 and n <= 100) @intCast(n) else 10,
                else => 10,
            } else 10;

            const stats = store.stats();
            if (stats.vectors == 0) return json(allocator, 200, "{{\"results\":[],\"vectors\":0}}", .{});

            const scorer = temporal.HybridScorer{ .now_ms = foundation_time.unixMs(), .half_life_ms = 24 * 60 * 60 * 1000 };
            const focus_id: u32 = if (stats.next_vector_id > 1) stats.next_vector_id - 1 else 1;
            const ranked = retrieval.hybridSearch(allocator, store, query_vec, limit, &store.temporal_graph, scorer, focus_id, constPersona) catch |err|
                return json(allocator, 400, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            defer allocator.free(ranked);

            var out: std.ArrayListUnmanaged(u8) = .empty;
            errdefer out.deinit(allocator);
            try out.print(allocator, "{{\"results\":[", .{});
            for (ranked, 0..) |r, i| {
                try out.print(allocator, "{s}{{\"id\":{d},\"score\":{d:.6},\"semantic\":{d:.6},\"temporal\":{d:.6},\"causal\":{d:.6},\"persona\":{d:.6}}}", .{
                    if (i == 0) "" else ",",
                    r.id,
                    r.score,
                    r.components.semantic,
                    r.components.temporal,
                    r.components.causal,
                    r.components.persona,
                });
            }
            try out.print(allocator, "],\"vectors\":{d},\"ranking\":\"hybrid\"}}", .{stats.vectors});
            return .{ .status = 200, .body = try out.toOwnedSlice(allocator) };
        }
        return json(allocator, 400, "{{\"error\":\"need key or vector\"}}", .{});
    }

    return json(allocator, 404, "{{\"error\":\"no route for {s} {s}\"}}", .{ method, path });
}

fn strField(v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn reasonPhrase(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        else => "OK",
    };
}

fn findBody(raw: []const u8) []const u8 {
    if (std.mem.indexOf(u8, raw, "\r\n\r\n")) |i| return raw[i + 4 ..];
    if (std.mem.indexOf(u8, raw, "\n\n")) |i| return raw[i + 2 ..];
    return "";
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
/// A single `conn.read` may return only part of the request: clients routinely
/// flush headers before the body. This accumulates until the `\r\n\r\n` header
/// terminator, parses any `Content-Length`, then keeps reading until the
/// declared body has arrived. Termination is bounded three ways so the loop can
/// never hang or grow without limit: `buf` is a fixed MAX_REQUEST_SIZE-sized
/// array (a full buffer yields `.too_large`), a 0-length read (EOF) stops the
/// loop, and the body target is capped at the buffer length.
///
/// This mirrors `readHttpRequest` in src/mcp/server.zig. The two transports use
/// the same `std.Io.net.Stream` conn type and the same MAX_REQUEST_SIZE-bounded
/// framing, but live in separate feature modules with no shared HTTP helper; the
/// logic is duplicated rather than forced through a fragile cross-module
/// abstraction. Keep the two in sync. (Chunked Transfer-Encoding is unsupported
/// in both.)
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
/// header block. Returns null when absent or unparseable. Mirrors the helper of
/// the same name in src/mcp/server.zig.
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

/// Bind a loopback listener and serve REST requests against `store` until the
/// process is stopped. One request per connection (Connection: close).
pub fn serve(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, port: u16) !void {
    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.log.info("WDBX REST listening on http://127.0.0.1:{d} (/insert /query /verify /health /stats)", .{port});

    while (true) {
        const conn = server.accept(io) catch |err| {
            std.log.warn("WDBX REST accept failed: {s}", .{@errorName(err)});
            continue;
        };
        handleConnection(allocator, io, store, conn) catch |err| {
            std.log.warn("WDBX REST request failed: {s}", .{@errorName(err)});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream) !void {
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

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const resp = route(arena.allocator(), store, method, path, body) catch |err| blk: {
        std.log.warn("WDBX REST route failed: {s}", .{@errorName(err)});
        break :blk Response{ .status = 500, .body = @constCast("{\"error\":\"internal\"}") };
    };

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

test "rest: health and stats" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var health = try route(allocator, &store, "GET", "/health", "");
    defer health.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), health.status);
    try std.testing.expect(std.mem.indexOf(u8, health.body, "\"status\":\"ok\"") != null);

    var stats = try route(allocator, &store, "GET", "/stats", "");
    defer stats.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), stats.status);
    try std.testing.expect(std.mem.indexOf(u8, stats.body, "\"blocks\":0") != null);
}

test "rest: insert kv then query round-trips" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var ins = try route(allocator, &store, "POST", "/insert", "{\"key\":\"agent:abbey\",\"value\":\"trained\"}");
    defer ins.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins.status);

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abbey\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "trained") != null);

    var miss = try route(allocator, &store, "POST", "/query", "{\"key\":\"nope\"}");
    defer miss.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 404), miss.status);
}

test "rest: vector query returns hybrid-ranked results" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });

    var q = try route(allocator, &store, "POST", "/query", "{\"vector\":[1.0,0.0,0.0,0.0],\"limit\":2}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"ranking\":\"hybrid\"") != null);
    // The matching vector (id 1) must appear with its scoring components.
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"semantic\":") != null);

    // Neither key nor vector -> 400.
    var bad = try route(allocator, &store, "POST", "/query", "{}");
    defer bad.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 400), bad.status);
}

test "rest: vector query over empty store reports zero vectors" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var e = try route(allocator, &store, "POST", "/query", "{\"vector\":[1.0,0.0,0.0,0.0]}");
    defer e.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), e.status);
    try std.testing.expect(std.mem.indexOf(u8, e.body, "\"vectors\":0") != null);
}

test "rest: insert block then verify" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var ins = try route(allocator, &store, "POST", "/insert", "{\"profile\":\"abbey\",\"metadata\":\"{\\\"turn\\\":1}\"}");
    defer ins.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins.status);

    var v = try route(allocator, &store, "POST", "/verify", "");
    defer v.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), v.status);
    try std.testing.expect(std.mem.indexOf(u8, v.body, "\"chain_valid\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, v.body, "\"blocks\":1") != null);
}

test "rest: bad json and unknown route" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bad = try route(allocator, &store, "POST", "/insert", "not json");
    defer bad.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 400), bad.status);

    var nf = try route(allocator, &store, "GET", "/nope", "");
    defer nf.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 404), nf.status);
}

test "rest: Content-Length header parser" {
    try std.testing.expectEqual(
        @as(?usize, 42),
        parseContentLength("POST /insert HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
    );
    // Header name match is case-insensitive and tolerates surrounding whitespace.
    try std.testing.expectEqual(
        @as(?usize, 7),
        parseContentLength("POST /insert HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
    );
    // Absent header -> null.
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /insert HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    // Garbage value -> null rather than a wrong length.
    try std.testing.expectEqual(
        @as(?usize, null),
        parseContentLength("POST /insert HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
    );
}

extern fn getsockname(sockfd: std.posix.fd_t, addr: *std.posix.sockaddr, addrlen: *std.posix.socklen_t) c_int;

// Bind a 127.0.0.1 listener on an ephemeral port and return both it and the
// kernel-assigned port (mirrors the loopback helper in src/mcp/server.zig and
// src/testing/test_helpers.zig).
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
// still be parsed in full. Before the read loop, handleConnection performed a
// single read and treated everything after the first CRLFCRLF as the complete
// body, truncating multi-segment requests into invalid JSON (a 400 parse error).
test "rest: HTTP transport reassembles a multi-segment request body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    // A well-formed insert whose body is intentionally split from its headers.
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
    try handleConnection(allocator, io, &store, conn);

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);

    // The full body parsed and inserted: 200 OK, not a 400 truncation error.
    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "400") == null);
    // The value round-tripped into the store via the loopback path.
    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abbey\"}");
    defer q.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "trained") != null);
}

// Happy path: the common single-write request must behave exactly as before.
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

test {
    std.testing.refAllDecls(@This());
}
