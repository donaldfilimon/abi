const std = @import("std");
const wdbx_mod = @import("mod.zig");

pub const HEADER = "# ABI-WDBX v1";

pub const PersistenceError = error{
    InvalidHeader,
    UnknownLineType,
    MissingField,
    OutOfMemory,
    DuplicateVectorId,
    DimensionMismatch,
};

pub fn serialize(allocator: std.mem.Allocator, store: *const wdbx_mod.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try out.writer.writeAll(HEADER);
    try out.writer.writeAll("\n");

    var kv_it = store.entries.iterator();
    while (kv_it.next()) |entry| {
        var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
        try w.beginObject();
        try w.objectField("type");
        try w.write("kv");
        try w.objectField("key");
        try w.write(entry.key_ptr.*);
        try w.objectField("value");
        try w.write(entry.value_ptr.*);
        try w.endObject();
        try out.writer.writeAll("\n");
    }

    const vector_dims = store.vector_dimensions orelse 0;
    for (store.index.nodes.items) |node| {
        const stored = store.index.storage.get(node.id);
        const values = if (vector_dims > 0) stored[0..vector_dims] else stored;
        var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
        try w.beginObject();
        try w.objectField("type");
        try w.write("vector");
        try w.objectField("id");
        try w.write(node.id);
        try w.objectField("values");
        try w.beginArray();
        for (values) |v| {
            try w.write(v);
        }
        try w.endArray();
        try w.endObject();
        try out.writer.writeAll("\n");
    }

    var chain_it = store.chain.iterator();
    defer store.chain.releaseIterator();
    while (chain_it.next()) |node| {
        var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
        try w.beginObject();
        try w.objectField("type");
        try w.write("block");
        try w.objectField("hash");
        try w.write(&node.header.hash);
        try w.objectField("prev_hash");
        try w.write(&node.header.prev_hash);
        try w.objectField("timestamp_ms");
        try w.write(node.header.timestamp_ms);
        try w.objectField("sequence");
        try w.write(node.header.sequence);
        try w.objectField("profile");
        try w.write(node.data.profile);
        try w.objectField("query_id");
        try w.write(node.data.query_id);
        try w.objectField("response_id");
        try w.write(node.data.response_id);
        try w.objectField("metadata");
        try w.write(node.data.metadata);
        try w.endObject();
        try out.writer.writeAll("\n");
    }

    for (store.spatial_index.records.items) |rec| {
        var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
        try w.beginObject();
        try w.objectField("type");
        try w.write("spatial");
        try w.objectField("id");
        try w.write(rec.id);
        try w.objectField("x");
        try w.write(rec.point.x);
        try w.objectField("y");
        try w.write(rec.point.y);
        try w.objectField("z");
        try w.write(rec.point.z);
        try w.objectField("payload");
        try w.write(rec.payload);
        try w.endObject();
        try out.writer.writeAll("\n");
    }

    return out.toOwnedSlice();
}

pub fn deserialize(allocator: std.mem.Allocator, content: []const u8) !wdbx_mod.Store {
    var store = wdbx_mod.Store.init(allocator);
    errdefer store.deinit();

    var lines = std.mem.splitScalar(u8, content, '\n');
    const first = lines.next() orelse return error.InvalidHeader;
    if (!std.mem.eql(u8, first, HEADER)) return error.InvalidHeader;

    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        try applyLine(allocator, &store, line);
    }

    return store;
}

pub fn saveToPath(io: std.Io, allocator: std.mem.Allocator, store: *const wdbx_mod.Store, path: []const u8) !void {
    const bytes = try serialize(allocator, store);
    defer allocator.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn loadFromPath(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !wdbx_mod.Store {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(content);
    return deserialize(allocator, content);
}

fn applyLine(allocator: std.mem.Allocator, store: *wdbx_mod.Store, line: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.UnknownLineType,
    };

    const type_str = obj.get("type") orelse return error.MissingField;
    const type_slice = switch (type_str) {
        .string => |s| s,
        else => return error.UnknownLineType,
    };

    if (std.mem.eql(u8, type_slice, "kv")) {
        const key = obj.get("key") orelse return error.MissingField;
        const val = obj.get("value") orelse return error.MissingField;
        const key_s = switch (key) {
            .string => |s| s,
            else => return error.MissingField,
        };
        const val_s = switch (val) {
            .string => |s| s,
            else => return error.MissingField,
        };
        try store.store(key_s, val_s);
    } else if (std.mem.eql(u8, type_slice, "vector")) {
        const values_node = obj.get("values") orelse return error.MissingField;
        const values_arr = switch (values_node) {
            .array => |a| a,
            else => return error.MissingField,
        };
        if (values_arr.items.len == 0) return error.DimensionMismatch;
        if (values_arr.items.len > wdbx_mod.HNSW_DIMENSIONS) return error.DimensionMismatch;
        var values: [wdbx_mod.HNSW_DIMENSIONS]f32 = undefined;
        var n: usize = 0;
        for (values_arr.items) |v| {
            values[n] = jsonNumberAsF32(v) orelse return error.MissingField;
            n += 1;
        }
        _ = try store.putVector(values[0..n]);
    } else if (std.mem.eql(u8, type_slice, "block")) {
        const profile_node = obj.get("profile") orelse return error.MissingField;
        const query_id_node = obj.get("query_id") orelse return error.MissingField;
        const response_id_node = obj.get("response_id") orelse return error.MissingField;
        const metadata_node = obj.get("metadata") orelse return error.MissingField;
        const profile_s = switch (profile_node) {
            .string => |s| s,
            else => return error.MissingField,
        };
        const query_id = switch (query_id_node) {
            .integer => |i| @as(u32, @intCast(i)),
            else => return error.MissingField,
        };
        const response_id = switch (response_id_node) {
            .integer => |i| @as(u32, @intCast(i)),
            else => return error.MissingField,
        };
        const metadata_s = switch (metadata_node) {
            .string => |s| s,
            else => return error.MissingField,
        };
        _ = try store.appendBlock(profile_s, query_id, response_id, metadata_s);
    } else if (std.mem.eql(u8, type_slice, "spatial")) {
        const id_node = obj.get("id") orelse return error.MissingField;
        const x_node = obj.get("x") orelse return error.MissingField;
        const y_node = obj.get("y") orelse return error.MissingField;
        const z_node = obj.get("z") orelse return error.MissingField;
        const payload_node = obj.get("payload") orelse return error.MissingField;
        const id = switch (id_node) {
            .integer => |i| @as(u32, @intCast(i)),
            else => return error.MissingField,
        };
        const x = jsonNumberAsF32(x_node) orelse return error.MissingField;
        const y = jsonNumberAsF32(y_node) orelse return error.MissingField;
        const z = jsonNumberAsF32(z_node) orelse return error.MissingField;
        const payload_s = switch (payload_node) {
            .string => |s| s,
            else => return error.MissingField,
        };
        try store.putSpatial3D(id, .{ .x = x, .y = y, .z = z }, payload_s);
    } else {
        return error.UnknownLineType;
    }
}

fn jsonNumberAsF32(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => null,
    };
}

test "persistence: round-trip kv, vector, block, spatial via JSONL" {
    var src = wdbx_mod.Store.init(std.testing.allocator);
    defer src.deinit();

    try src.store("agent:abbey", "trained");
    try src.store("agent:aviva", "creative");
    try src.store("modulator:weights", "{\"w_abbey\":0.4,\"w_aviva\":0.35,\"w_abi\":0.25}");

    _ = try src.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try src.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    _ = try src.putVector(&.{ 0.0, 0.0, 1.0, 0.0 });

    _ = try src.appendBlock("abbey", 1, 2, "{\"turn\":1}");
    _ = try src.appendBlock("aviva", 3, 4, "{\"turn\":2}");

    try src.putSpatial3D(1, .{ .x = 1, .y = 2, .z = 3 }, "origin-near");
    try src.putSpatial3D(2, .{ .x = 10, .y = 10, .z = 10 }, "far");

    const bytes = try serialize(std.testing.allocator, &src);
    defer std.testing.allocator.free(bytes);

    var dst = try deserialize(std.testing.allocator, bytes);
    defer dst.deinit();

    try std.testing.expectEqual(@as(usize, 3), dst.count());
    try std.testing.expectEqualStrings("trained", dst.get("agent:abbey").?);
    try std.testing.expectEqualStrings("creative", dst.get("agent:aviva").?);

    try std.testing.expectEqual(@as(usize, 3), dst.vectorCount());
    try std.testing.expectEqual(@as(usize, 2), dst.blockCount());
    try std.testing.expectEqual(@as(usize, 2), dst.spatial_index.count());

    try std.testing.expect(dst.verifyBlocks());

    const nearest = try dst.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 1);
    defer std.testing.allocator.free(nearest);
    try std.testing.expect(nearest.len == 1);
    try std.testing.expect(nearest[0].score > 0.99);
}

test "persistence: rejects unknown header" {
    const bad = "garbage header\n";
    try std.testing.expectError(error.InvalidHeader, deserialize(std.testing.allocator, bad));
}

test "persistence: empty body after header produces empty store" {
    const ok = HEADER ++ "\n";
    var dst = try deserialize(std.testing.allocator, ok);
    defer dst.deinit();
    try std.testing.expectEqual(@as(usize, 0), dst.count());
    try std.testing.expectEqual(@as(usize, 0), dst.vectorCount());
    try std.testing.expectEqual(@as(usize, 0), dst.blockCount());
    try std.testing.expectEqual(@as(usize, 0), dst.spatial_index.count());
}

test "persistence: save and load round-trip via std.testing.io" {
    var src = wdbx_mod.Store.init(std.testing.allocator);
    defer src.deinit();

    try src.store("k1", "v1");
    _ = try src.putVector(&.{ 0.5, 0.5, 0.0, 0.0 });
    _ = try src.appendBlock("abi", 7, 8, "round-trip");
    try src.putSpatial3D(99, .{ .x = 1, .y = 1, .z = 1 }, "rt-payload");

    const path = "zig-out/wdbx-persistence-rt.jsonl";
    try saveToPath(std.testing.io, std.testing.allocator, &src, path);
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var dst = try loadFromPath(std.testing.io, std.testing.allocator, path);
    defer dst.deinit();

    try std.testing.expectEqual(@as(usize, 1), dst.count());
    try std.testing.expectEqual(@as(usize, 1), dst.vectorCount());
    try std.testing.expectEqual(@as(usize, 1), dst.blockCount());
    try std.testing.expectEqual(@as(usize, 1), dst.spatial_index.count());
    try std.testing.expectEqualStrings("v1", dst.get("k1").?);
    try std.testing.expect(dst.verifyBlocks());
}

test {
    std.testing.refAllDecls(@This());
}
