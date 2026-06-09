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

pub const MAX_REQUEST_SIZE: usize = 64 * 1024;

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
        const key = if (obj.get("key")) |k| strField(k) else null;
        if (key == null) return json(allocator, 400, "{{\"error\":\"need key\"}}", .{});
        const val = store.get(key.?) orelse return json(allocator, 404, "{{\"error\":\"not found\"}}", .{});
        return json(allocator, 200, "{{\"value\":\"{s}\"}}", .{val});
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

/// Bind a loopback listener and serve REST requests against `store` until the
/// process is stopped. One request per connection (Connection: close).
pub fn serve(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, port: u16) !void {
    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.log.info("WDBX REST listening on http://127.0.0.1:{d} (/insert /query /verify /health /stats)", .{port});

    while (true) {
        const conn = server.accept(io) catch continue;
        handleConnection(allocator, io, store, conn) catch {};
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, store: *wdbx.Store, conn: std.Io.net.Stream) !void {
    defer conn.close(io);
    var read_buf: [MAX_REQUEST_SIZE]u8 = undefined;
    var read_vec: [1][]u8 = .{&read_buf};
    const n = conn.read(io, &read_vec) catch return;
    if (n == 0) return;
    const raw = read_buf[0..n];

    var line_end: usize = 0;
    while (line_end < raw.len and raw[line_end] != '\n') : (line_end += 1) {}
    const request_line = std.mem.trimEnd(u8, raw[0..line_end], "\r");
    var it = std.mem.splitScalar(u8, request_line, ' ');
    const method = it.next() orelse return;
    const path = it.next() orelse return;
    const body = findBody(raw);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const resp = route(arena.allocator(), store, method, path, body) catch
        Response{ .status = 500, .body = @constCast("{\"error\":\"internal\"}") };

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

test {
    std.testing.refAllDecls(@This());
}
