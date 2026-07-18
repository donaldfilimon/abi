const std = @import("std");
const wdbx_mod = @import("mod.zig");
const parser = @import("persistence_parse.zig");
const test_helpers = @import("../../foundation/test_helpers.zig");

pub const HEADER = "# ABI-WDBX v1";
pub const CHECKSUM_PREFIX = "# checksum:";

pub const PersistenceError = error{
    InvalidHeader,
    UnknownLineType,
    MissingField,
    OutOfMemory,
    DuplicateVectorId,
    DimensionMismatch,
    CorruptVectorId,
    ChecksumMismatch,
    FieldOutOfRange,
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
        // Every node in the index was inserted with its vector before being
        // appended to `nodes` (hnsw.insert order), so `get` is always present
        // here; `.?` asserts that invariant rather than silently dropping a
        // vector from the export.
        const stored = store.index.storage.get(node.id).?;
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

    // Serialize blocks under a length-bounded snapshot so concurrent appends
    // that race after the read lock is taken cannot extend this checkpoint view.
    const chain_ptr = @constCast(&store.chain);
    const snapshot = chain_ptr.getSnapshot();
    defer chain_ptr.releaseSnapshot();
    var chain_it = snapshot.iterator();
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

    var temporal_nodes = store.temporal_graph.timestamps.iterator();
    while (temporal_nodes.next()) |entry| {
        var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
        try w.beginObject();
        try w.objectField("type");
        try w.write("temporal_node");
        try w.objectField("id");
        try w.write(entry.key_ptr.*);
        try w.objectField("timestamp_ms");
        try w.write(entry.value_ptr.*);
        try w.endObject();
        try out.writer.writeAll("\n");
    }

    var temporal_edges = store.temporal_graph.adjacency.iterator();
    while (temporal_edges.next()) |entry| {
        const cause = entry.key_ptr.*;
        for (entry.value_ptr.items) |effect| {
            if (cause > effect) continue;
            var w = std.json.Stringify{ .writer = &out.writer, .options = .{ .whitespace = .minified } };
            try w.beginObject();
            try w.objectField("type");
            try w.write("temporal_edge");
            try w.objectField("cause");
            try w.write(cause);
            try w.objectField("effect");
            try w.write(effect);
            try w.endObject();
            try out.writer.writeAll("\n");
        }
    }

    // Trailing integrity line: SHA-256 over the record body (everything after the
    // header line). Covers kv/spatial/vector/block records uniformly so a
    // truncated or tampered snapshot is rejected on load rather than silently
    // restoring partial state. Verification on read is optional for backward
    // compatibility with checksum-less snapshots.
    const body = out.written()[HEADER.len + 1 ..];
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body, &digest, .{});
    try out.writer.writeAll(CHECKSUM_PREFIX);
    try out.writer.writeAll(&std.fmt.bytesToHex(digest, .lower));
    try out.writer.writeAll("\n");

    return out.toOwnedSlice();
}

pub fn deserialize(allocator: std.mem.Allocator, content: []const u8) !wdbx_mod.Store {
    var store = wdbx_mod.Store.init(allocator);
    errdefer store.deinit();

    var lines = std.mem.splitScalar(u8, content, '\n');
    const first = lines.next() orelse return error.InvalidHeader;
    if (!std.mem.eql(u8, first, HEADER)) return error.InvalidHeader;

    // Optional integrity check: if a trailing checksum line is present, the
    // record body must hash to it. Older checksum-less snapshots skip this.
    if (std.mem.lastIndexOf(u8, content, "\n" ++ CHECKSUM_PREFIX)) |nl_pos| {
        const expected = std.mem.trim(u8, content[nl_pos + 1 + CHECKSUM_PREFIX.len ..], " \t\r\n");
        const body = content[HEADER.len + 1 .. nl_pos + 1];
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(body, &digest, .{});
        if (!std.mem.eql(u8, expected, &std.fmt.bytesToHex(digest, .lower))) {
            return error.ChecksumMismatch;
        }
    }

    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (line[0] == '#') continue; // header/checksum/comment line
        try parser.applyLine(allocator, &store, line);
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
    try src.addTemporalNode(1, 1000);
    try src.addTemporalNode(2, 2000);
    try src.addTemporalEdge(1, 2);

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
    try std.testing.expectEqual(@as(usize, 2), dst.temporalNodeCount());
    try std.testing.expectEqual(@as(usize, 1), dst.temporalEdgeCount());
    try std.testing.expectEqual(@as(?i64, 1000), dst.temporalTimestamp(1));

    try std.testing.expect(dst.verifyBlocks());

    const nearest = try dst.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 1);
    defer std.testing.allocator.free(nearest);
    try std.testing.expect(nearest.len == 1);
    try std.testing.expect(nearest[0].score > 0.99);
}

test "persistence: block timestamps and hashes survive round-trip exactly" {
    var src = wdbx_mod.Store.init(std.testing.allocator);
    defer src.deinit();

    _ = try src.appendBlock("abbey", 1, 2, "{\"turn\":1}");
    _ = try src.appendBlock("aviva", 3, 4, "{\"turn\":2}");
    const src_last = src.lastBlock().?;

    const bytes = try serialize(std.testing.allocator, &src);
    defer std.testing.allocator.free(bytes);

    var dst = try deserialize(std.testing.allocator, bytes);
    defer dst.deinit();

    const dst_last = dst.lastBlock().?;
    // Faithful restore: identical timestamp reproduces an identical SHA-256 hash.
    try std.testing.expectEqual(src_last.timestamp_ms, dst_last.timestamp_ms);
    try std.testing.expect(std.mem.eql(u8, &src_last.id, &dst_last.id));
    try std.testing.expect(std.mem.eql(u8, &src_last.prev_id, &dst_last.prev_id));
    try std.testing.expect(dst.verifyBlocks());
}

test "persistence: detects body corruption via trailing checksum" {
    var src = wdbx_mod.Store.init(std.testing.allocator);
    defer src.deinit();
    try src.store("k1", "v1");
    try src.putSpatial3D(1, .{ .x = 1, .y = 2, .z = 3 }, "p");

    const bytes = try serialize(std.testing.allocator, &src);
    defer std.testing.allocator.free(bytes);

    // A clean snapshot round-trips; flipping any body byte must be rejected.
    // Target the quoted kv value so the header version "v1" is not disturbed.
    const idx = std.mem.indexOf(u8, bytes, "\"v1\"").?;
    bytes[idx + 1] = 'X';
    try std.testing.expectError(error.ChecksumMismatch, deserialize(std.testing.allocator, bytes));
}

test "persistence: rejects tampered vector id" {
    const tampered = HEADER ++ "\n" ++
        "{\"type\":\"vector\",\"id\":42,\"values\":[1.0,0.0,0.0,0.0]}\n";
    try std.testing.expectError(error.CorruptVectorId, deserialize(std.testing.allocator, tampered));
}

test "persistence: rejects out-of-range integer field without panicking" {
    // id exceeds u32; a corrupt/tampered snapshot must fail cleanly, not panic.
    const overflow = HEADER ++ "\n" ++
        "{\"type\":\"vector\",\"id\":4294967296,\"values\":[1.0,0.0,0.0,0.0]}\n";
    try std.testing.expectError(error.FieldOutOfRange, deserialize(std.testing.allocator, overflow));

    const negative = HEADER ++ "\n" ++
        "{\"type\":\"spatial\",\"id\":-1,\"x\":0,\"y\":0,\"z\":0,\"payload\":\"p\"}\n";
    try std.testing.expectError(error.FieldOutOfRange, deserialize(std.testing.allocator, negative));
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
    defer deleteTestFileIfExists(path);

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
    _ = @import("persistence_parse.zig");
    std.testing.refAllDecls(@This());
}

const deleteTestFileIfExists = test_helpers.deleteTestFileIfExists;
