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
pub const REST_TOKEN_ENV = "ABI_WDBX_REST_TOKEN";

const AuthConfig = struct {
    bearer_token: ?[]const u8 = null,
};

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

const VectorParseError = error{ NotArray, Empty, NonNumber, OutOfMemory };

/// Parse a JSON `vector` field (already fetched from the request object) into an
/// owned `[]f32`. Shared by `/insert` and `/query` so the element coercion and
/// validation stay byte-identical between the write and read paths. The caller
/// frees the returned slice.
fn parseVectorField(allocator: std.mem.Allocator, vec_node: std.json.Value) VectorParseError![]f32 {
    const arr = switch (vec_node) {
        .array => |a| a,
        else => return error.NotArray,
    };
    if (arr.items.len == 0) return error.Empty;
    const out = try allocator.alloc(f32, arr.items.len);
    errdefer allocator.free(out);
    for (arr.items, 0..) |item, i| {
        out[i] = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.NonNumber,
        };
    }
    return out;
}

/// Map a `parseVectorField` failure to the same JSON error bodies the routes
/// returned inline before the helper was factored out.
fn vectorParseErrorResponse(allocator: std.mem.Allocator, err: VectorParseError) !Response {
    return switch (err) {
        error.NotArray => json(allocator, 400, "{{\"error\":\"vector must be an array\"}}", .{}),
        error.Empty => json(allocator, 400, "{{\"error\":\"vector must be non-empty\"}}", .{}),
        error.NonNumber => json(allocator, 400, "{{\"error\":\"vector elements must be numbers\"}}", .{}),
        error.OutOfMemory => json(allocator, 500, "{{\"error\":\"oom\"}}", .{}),
    };
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
        // Vector insert: {"vector":[..]} — assigns a vector id and echoes it.
        // Uses the same vector parsing as /query so a vector inserted here is
        // exactly what a subsequent /query semantic search will match against.
        if (obj.get("vector")) |vec_node| {
            const vec = parseVectorField(allocator, vec_node) catch |err|
                return vectorParseErrorResponse(allocator, err);
            defer allocator.free(vec);
            const id = store.putVector(vec) catch |err|
                return json(allocator, 500, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            return json(allocator, 200, "{{\"inserted\":\"vector\",\"id\":{d}}}", .{id});
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
            // Stored values are opaque bytes: escape them before interpolating
            // into the JSON body so a value containing `"`, `\`, or a control
            // character cannot produce malformed JSON.
            const escaped = try escapeJsonString(allocator, val);
            defer allocator.free(escaped);
            return json(allocator, 200, "{{\"value\":\"{s}\"}}", .{escaped});
        }
        // Hybrid semantic search: {"vector":[..],"limit":N} — mirrors the CLI
        // `wdbx query <path> <text>` path (semantic × temporal × causal × persona)
        // but takes a pre-embedded query vector, since the REST server lives in
        // the storage layer and must not depend on the AI embedding layer above.
        if (obj.get("vector")) |vec_node| {
            const query_vec = parseVectorField(allocator, vec_node) catch |err|
                return vectorParseErrorResponse(allocator, err);
            defer allocator.free(query_vec);
            // A present-but-invalid limit is a client error (400), consistent with
            // every other malformed field on this route — not a silent clamp that
            // returns 200 with a different page size than the caller asked for.
            // Absent limit still defaults to 10.
            const limit: usize = if (obj.get("limit")) |l| switch (l) {
                .integer => |n| if (n > 0 and n <= 100) @intCast(n) else return json(allocator, 400, "{{\"error\":\"limit must be between 1 and 100\"}}", .{}),
                else => return json(allocator, 400, "{{\"error\":\"limit must be an integer\"}}", .{}),
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

/// Escape a stored KV value's bytes for safe interpolation as the content of a
/// JSON string (the surrounding quotes are supplied by the caller's format).
/// Mirrors the structural/control-character handling of
/// `src/mcp/json_helpers.zig:appendJsonString`, kept local so the storage-layer
/// REST listener stays free of a transport-layer dependency.
fn escapeJsonString(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    return out.toOwnedSlice(allocator);
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

fn loadBearerToken(allocator: std.mem.Allocator) !?[]u8 {
    const raw_z = std.c.getenv(REST_TOKEN_ENV) orelse return null;
    const raw = std.mem.span(raw_z);

    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return null;
    return try allocator.dupe(u8, token);
}

/// Bind a loopback listener and serve REST requests against `store` until the
/// process is stopped. One request per connection (Connection: close).
pub fn serve(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, port: u16) !void {
    const bearer_token = try loadBearerToken(allocator);
    defer if (bearer_token) |token| allocator.free(token);

    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.log.info("WDBX REST listening on http://127.0.0.1:{d} (/insert /query /verify /health /stats), auth={s}", .{ port, if (bearer_token == null) "off" else "bearer" });

    while (true) {
        const conn = server.accept(io) catch |err| {
            std.log.warn("WDBX REST accept failed: {s}", .{@errorName(err)});
            continue;
        };
        handleConnectionWithAuth(allocator, io, store, conn, .{ .bearer_token = bearer_token }) catch |err| {
            std.log.warn("WDBX REST request failed: {s}", .{@errorName(err)});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream) !void {
    try handleConnectionWithAuth(allocator, io, store, conn, .{});
}

fn handleConnectionWithAuth(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream, auth: AuthConfig) !void {
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

test "rest: insert vector then query round-trips" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // Insert a vector over the REST route and confirm a JSON success body with
    // an assigned id (valid JSON, parseable, integer id field).
    var ins = try route(allocator, &store, "POST", "/insert", "{\"vector\":[1.0,0.0,0.0,0.0]}");
    defer ins.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins.status);
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, ins.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings("vector", parsed.value.object.get("inserted").?.string);
        // The id is present and integral (first assigned vector id).
        try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("id").?.integer);
    }

    // A second, orthogonal vector so the query has a real ranking to make.
    var ins2 = try route(allocator, &store, "POST", "/insert", "{\"vector\":[0.0,1.0,0.0,0.0]}");
    defer ins2.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins2.status);

    // Query the exact first vector: it must be found and surface its id.
    var q = try route(allocator, &store, "POST", "/query", "{\"vector\":[1.0,0.0,0.0,0.0],\"limit\":2}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"ranking\":\"hybrid\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"id\":1") != null);
}

test "rest: insert vector rejects malformed and empty input" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // Empty vector, non-array vector, and non-numeric element are all 400s —
    // the same client-error shapes /query returns for bad vector input.
    const bad = [_][]const u8{
        "{\"vector\":[]}",
        "{\"vector\":\"nope\"}",
        "{\"vector\":[1.0,\"x\"]}",
    };
    for (bad) |body| {
        var r = try route(allocator, &store, "POST", "/insert", body);
        defer r.deinit(allocator);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"error\"") != null);
    }
}

test "rest: query escapes a stored value containing JSON metacharacters" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // A value containing a quote, a backslash, and a newline would corrupt the
    // response body if interpolated raw.
    try store.store("agent:abbey", "he said \"hi\"\\done\nnext");

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abbey\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);

    // The response must be valid JSON and round-trip the original bytes.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, q.body, .{});
    defer parsed.deinit();
    const value = parsed.value.object.get("value").?.string;
    try std.testing.expectEqualStrings("he said \"hi\"\\done\nnext", value);
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

test "rest: vector query rejects a present-but-invalid limit" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });

    // A present-but-invalid limit is a 400 (consistent with the route's other
    // fields), not a silent clamp that returns 200 with a different page size.
    const invalid = [_][]const u8{
        "{\"vector\":[1.0,0.0,0.0,0.0],\"limit\":0}",
        "{\"vector\":[1.0,0.0,0.0,0.0],\"limit\":5000}",
        "{\"vector\":[1.0,0.0,0.0,0.0],\"limit\":\"big\"}",
    };
    for (invalid) |body| {
        var r = try route(allocator, &store, "POST", "/query", body);
        defer r.deinit(allocator);
        try std.testing.expectEqual(@as(u16, 400), r.status);
    }

    // An absent limit still defaults (no 400).
    var ok = try route(allocator, &store, "POST", "/query", "{\"vector\":[1.0,0.0,0.0,0.0]}");
    defer ok.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ok.status);
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

test "rest: Authorization bearer parser" {
    const raw =
        "POST /insert HTTP/1.1\r\n" ++
        "Host: 127.0.0.1\r\n" ++
        "authorization:   Bearer local-token  \r\n" ++
        "Content-Length: 2\r\n\r\n{}";

    try std.testing.expect(hasBearerToken(raw, "local-token"));
    try std.testing.expect(!hasBearerToken(raw, "wrong-token"));
    try std.testing.expect(!hasBearerToken("POST /insert HTTP/1.1\r\n\r\n{}", "local-token"));
    try std.testing.expect(!hasBearerToken("POST /insert HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}", "local-token"));
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
    try handleConnectionWithAuth(allocator, io, &store, conn, .{ .bearer_token = "local-token" });

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
    try handleConnectionWithAuth(allocator, io, &store, conn, .{ .bearer_token = "local-token" });

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abi\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "authorized") != null);
}

test {
    std.testing.refAllDecls(@This());
}
