const std = @import("std");
const wdbx = @import("mod.zig");
const retrieval = @import("retrieval.zig");
const temporal = @import("temporal.zig");
const foundation_time = @import("../../foundation/time.zig");
const parse = @import("rest_parse.zig");

pub const Response = parse.Response;
pub const json = parse.json;

fn constPersona(id: u32) f32 {
    _ = id;
    return 0.5;
}

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
        if (obj.get("profile")) |p_node| {
            const profile = parse.strField(p_node) orelse return json(allocator, 400, "{{\"error\":\"profile must be a string\"}}", .{});
            const metadata = if (obj.get("metadata")) |m| (parse.strField(m) orelse "") else "";
            _ = store.appendBlock(profile, 0, 0, metadata) catch |err|
                return json(allocator, 500, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            return json(allocator, 200, "{{\"inserted\":\"block\",\"blocks\":{d}}}", .{store.blockCount()});
        }
        if (obj.get("vector")) |vec_node| {
            const vec = parse.parseVectorField(allocator, vec_node) catch |err|
                return parse.vectorParseErrorResponse(allocator, err);
            defer allocator.free(vec);
            const id = store.putVector(vec) catch |err|
                return json(allocator, 500, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            return json(allocator, 200, "{{\"inserted\":\"vector\",\"id\":{d}}}", .{id});
        }
        const key = if (obj.get("key")) |k| parse.strField(k) else null;
        const value = if (obj.get("value")) |v| parse.strField(v) else null;
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
        if (obj.get("key")) |k| {
            const key = parse.strField(k) orelse return json(allocator, 400, "{{\"error\":\"key must be a string\"}}", .{});
            const val = store.get(key) orelse return json(allocator, 404, "{{\"error\":\"not found\"}}", .{});
            const escaped = try parse.escapeJsonString(allocator, val);
            defer allocator.free(escaped);
            return json(allocator, 200, "{{\"value\":\"{s}\"}}", .{escaped});
        }
        if (obj.get("vector")) |vec_node| {
            const query_vec = parse.parseVectorField(allocator, vec_node) catch |err|
                return parse.vectorParseErrorResponse(allocator, err);
            defer allocator.free(query_vec);
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

    var ins = try route(allocator, &store, "POST", "/insert", "{\"vector\":[1.0,0.0,0.0,0.0]}");
    defer ins.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins.status);
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, ins.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings("vector", parsed.value.object.get("inserted").?.string);
        try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("id").?.integer);
    }

    var ins2 = try route(allocator, &store, "POST", "/insert", "{\"vector\":[0.0,1.0,0.0,0.0]}");
    defer ins2.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), ins2.status);

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

    try store.store("agent:abbey", "he said \"hi\"\\done\nnext");

    var q = try route(allocator, &store, "POST", "/query", "{\"key\":\"agent:abbey\"}");
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), q.status);

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
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, q.body, "\"semantic\":") != null);

    var bad = try route(allocator, &store, "POST", "/query", "{}");
    defer bad.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 400), bad.status);
}

test "rest: vector query rejects a present-but-invalid limit" {
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });

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

test {
    std.testing.refAllDecls(@This());
}
