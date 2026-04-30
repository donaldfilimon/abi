//! WDBX Binary Persistence
//!
//! Serializes and deserializes the complete engine state (vectors, metadata,
//! HNSW graph structure) to a compact binary format.
//!
//! ## Format
//! ```
//! [4B] magic "WDBX"
//! [4B] version (1)
//! [1B] metric enum
//! [4B] dimensions
//! [4B] vector_count
//! per vector:
//!   [4B] id_len  [id_bytes]
//!   [dims×4B] float32 vector
//!   [4B] meta_text_len [meta_text_bytes]
//!   [4B] meta_cat_len  [meta_cat_bytes]   (0 = null)
//!   [4B] meta_extra_len [meta_extra_bytes] (0 = null)
//!   [4B] meta_tags_count
//!     per tag: [4B] tag_len [tag_bytes]
//!   [4B] meta_score (f32 as u32 bits)
//! HNSW state:
//!   [4B] entry_point (0xFFFFFFFF = null)
//!   [4B] max_level
//!   per node:
//!     [4B] node_level
//!     per layer:
//!       [4B] neighbor_count [neighbor_ids as u32...]
//! ```

const std = @import("std");
// Gate using runtime environment (ABI_JWT_SECRET) to avoid cross-module imports
const Engine = @import("engine.zig").Engine;
const Metadata = @import("engine.zig").Metadata;
const config = @import("config.zig");
const Cache = @import("cache.zig").Cache;
const HNSW = @import("hnsw/mod.zig").HnswIndex;

const MAGIC = [4]u8{ 'W', 'D', 'B', 'X' };
const VERSION: u32 = 1;

pub const PersistenceError = error{
    InvalidMagic,
    UnsupportedVersion,
    CorruptData,
    EndOfStream,
};

const MemWriter = struct {
    allocator: std.mem.Allocator,
    buf: *std.ArrayListUnmanaged(u8),
    pub fn writeAll(self: MemWriter, bytes: []const u8) !void {
        try self.buf.appendSlice(self.allocator, bytes);
    }
    pub fn writeInt(self: MemWriter, comptime T: type, val: T, endian: std.builtin.Endian) !void {
        var tmp: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &tmp, val, endian);
        try self.buf.appendSlice(self.allocator, &tmp);
    }
    pub fn writeByte(self: MemWriter, b: u8) !void {
        try self.buf.append(self.allocator, b);
    }
};

const MemReader = struct {
    data: []const u8,
    pos: usize = 0,
    pub fn readAll(self: *MemReader, out: []u8) !void {
        if (self.pos + out.len > self.data.len) return PersistenceError.EndOfStream;
        @memcpy(out, self.data[self.pos .. self.pos + out.len]);
        self.pos += out.len;
    }
    pub fn readInt(self: *MemReader, comptime T: type, endian: std.builtin.Endian) !T {
        if (self.pos + @sizeOf(T) > self.data.len) return PersistenceError.EndOfStream;
        const result = std.mem.readInt(T, self.data[self.pos .. self.pos + @sizeOf(T)][0..@sizeOf(T)], endian);
        self.pos += @sizeOf(T);
        return result;
    }
    pub fn readByte(self: *MemReader) !u8 {
        if (self.pos >= self.data.len) return PersistenceError.EndOfStream;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }
};

/// Save the entire engine state to a file.
pub fn save(engine: *Engine, path: []const u8) !void {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(engine.allocator);
    const writer = MemWriter{ .allocator = engine.allocator, .buf = &buf };

    // Header.
    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, VERSION, .little);
    try writer.writeByte(@intFromEnum(engine.config.metric));
    try writer.writeInt(u32, @intCast(engine.config.dimensions), .little);

    const count: u32 = @intCast(engine.vectors_array.items.len);
    try writer.writeInt(u32, count, .little);

    // Vectors + metadata.
    for (engine.vectors_array.items) |item| {
        // ID.
        try writer.writeInt(u32, @intCast(item.id.len), .little);
        try writer.writeAll(item.id);

        // Vector data.
        const vec_bytes = std.mem.sliceAsBytes(item.vec);
        try writer.writeAll(vec_bytes);

        // Metadata.
        try writeString(writer, item.metadata.text);
        try writeOptionalString(writer, item.metadata.category);
        try writeOptionalString(writer, item.metadata.extra);

        // Tags.
        try writer.writeInt(u32, @intCast(item.metadata.tags.len), .little);
        for (item.metadata.tags) |tag| {
            try writeString(writer, tag);
        }

        // Score.
        try writer.writeInt(u32, @bitCast(item.metadata.score), .little);
    }

    // HNSW graph state.
    const ep: u32 = engine.hnsw_index.entry_point orelse 0xFFFFFFFF;
    try writer.writeInt(u32, ep, .little);
    try writer.writeInt(u32, @intCast(@max(0, engine.hnsw_index.max_layer)), .little);

    for (0..count) |i| {
        if (i < engine.hnsw_index.nodes.len) {
            // Full graph mode: nodes[] is populated.
            const node = engine.hnsw_index.nodes[i];
            const node_level = @as(u32, @intCast(@max(0, @as(i32, @intCast(node.layers.len)) - 1)));
            try writer.writeInt(u32, node_level, .little);

            const layers = node.layers;
            for (0..node_level + 1) |l| {
                const nbrs = if (l < layers.len) layers[l].nodes else &[_]u32{};
                try writer.writeInt(u32, @intCast(nbrs.len), .little);
                for (nbrs) |n| {
                    try writer.writeInt(u32, n, .little);
                }
            }
        } else {
            // Shim / compatibility mode: no graph nodes built.
            // Write a single layer with zero neighbors.
            try writer.writeInt(u32, 0, .little); // node_level = 0
            try writer.writeInt(u32, 0, .little); // 0 neighbors in layer 0
        }
    }

    // Write file using std.Io backend
    var io_backend = std.Io.Threaded.init(engine.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, buf.items);
}

/// Load an engine from a file.
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Engine {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Read entire file up to 2GB.
    const max_size = 2 * 1024 * 1024 * 1024;
    const file_data = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_size));
    defer allocator.free(file_data);

    var reader = MemReader{ .data = file_data };

    // Header.
    var magic: [4]u8 = undefined;
    _ = try reader.readAll(&magic);
    if (!std.mem.eql(u8, &magic, &MAGIC)) return PersistenceError.InvalidMagic;

    const version = try reader.readInt(u32, .little);
    if (version != VERSION) return PersistenceError.UnsupportedVersion;

    const metric_byte = try reader.readByte();
    const metric: config.DistanceMetric = @enumFromInt(metric_byte);
    const dimensions = try reader.readInt(u32, .little);
    const count = try reader.readInt(u32, .little);

    const cfg = config.Config{
        .metric = metric,
        .dimensions = dimensions,
    };

    // Build a minimal engine (skip validateRuntime for loaded configs).
    var cache = try Cache.init(allocator, cfg.cache.capacity, cfg.cache.segments);
    errdefer cache.deinit();

    var hnsw = try HNSW.init(allocator, cfg, metric);
    errdefer hnsw.deinit(allocator);

    var engine = Engine{
        .allocator = allocator,
        .config = cfg,
        .cache = cache,
        .hnsw_index = hnsw,
    };

    // Read vectors + metadata.
    for (0..count) |_| {
        const id_len = try reader.readInt(u32, .little);
        const id_buf = try allocator.alloc(u8, id_len);
        errdefer allocator.free(id_buf);
        _ = try reader.readAll(id_buf);

        const vec = try allocator.alloc(f32, dimensions);
        errdefer allocator.free(vec);
        const vec_bytes = std.mem.sliceAsBytes(vec);
        _ = try reader.readAll(vec_bytes);

        const meta_text = try readString(allocator, &reader);
        errdefer allocator.free(meta_text);

        const meta_cat = try readOptionalString(allocator, &reader);
        errdefer if (meta_cat) |c| allocator.free(c);

        const meta_extra = try readOptionalString(allocator, &reader);
        errdefer if (meta_extra) |e| allocator.free(e);

        const tags_count = try reader.readInt(u32, .little);
        const tags = try allocator.alloc([]const u8, tags_count);
        var tags_read: usize = 0;
        errdefer {
            for (tags[0..tags_read]) |t| allocator.free(t);
            allocator.free(tags);
        }
        for (0..tags_count) |t| {
            tags[t] = try readString(allocator, &reader);
            tags_read += 1;
        }

        const score_bits = try reader.readInt(u32, .little);
        const score: f32 = @bitCast(score_bits);

        const EngineVector = @import("engine.zig").EngineVector;
        try engine.vectors_array.append(allocator, EngineVector{
            .id = id_buf,
            .vec = vec,
            .metadata = Metadata{
                .text = meta_text,
                .category = meta_cat,
                .tags = tags,
                .score = score,
                .extra = meta_extra,
            },
        });

        // Also insert vector into HNSW (raw, don't rebuild graph yet).
        const hnsw_vec = try allocator.dupe(f32, vec);
        try engine.hnsw_index.vectors.append(allocator, hnsw_vec);
    }

    // Read HNSW graph state.
    const ep_val = try reader.readInt(u32, .little);
    engine.hnsw_index.entry_point = if (ep_val == 0xFFFFFFFF) null else ep_val;
    engine.hnsw_index.max_layer = @intCast(try reader.readInt(u32, .little));

    for (0..count) |_| {
        const node_level = try reader.readInt(u32, .little);
        try engine.hnsw_index.node_levels.append(allocator, node_level);

        const num_layers = node_level + 1;
        const layers = try allocator.alloc([]u32, num_layers);
        for (0..num_layers) |l| {
            const nbr_count = try reader.readInt(u32, .little);
            const nbrs = try allocator.alloc(u32, nbr_count);
            for (0..nbr_count) |ni| {
                nbrs[ni] = try reader.readInt(u32, .little);
            }
            layers[l] = nbrs;
        }
        try engine.hnsw_index.neighbors.append(allocator, layers);
    }

    return engine;
}

// ─── String helpers ────────────────────────────────────────────────────

fn writeString(writer: MemWriter, s: []const u8) !void {
    try writer.writeInt(u32, @intCast(s.len), .little);
    try writer.writeAll(s);
}

fn writeOptionalString(writer: MemWriter, s: ?[]const u8) !void {
    if (s) |val| {
        try writer.writeInt(u32, @intCast(val.len), .little);
        try writer.writeAll(val);
    } else {
        try writer.writeInt(u32, 0, .little);
    }
}

fn readString(allocator: std.mem.Allocator, reader: *MemReader) ![]u8 {
    const len = try reader.readInt(u32, .little);
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    _ = try reader.readAll(buf);
    return buf;
}

fn readOptionalString(allocator: std.mem.Allocator, reader: *MemReader) !?[]u8 {
    const len = try reader.readInt(u32, .little);
    if (len == 0) return null;
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    _ = try reader.readAll(buf);
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "Persistence round-trip save and load" {
    const env_gate = @import("common");
    if (!env_gate.canRunTest()) return;
    const allocator = std.testing.allocator;

    // Build engine with some data.
    var engine = try Engine.init(allocator, .{ .metric = .euclidean, .dimensions = 3 });
    defer engine.deinit();

    // Insert vectors via cache bypass (indexByVector).
    try engine.indexByVector("doc-1", &[_]f32{ 1.0, 0.0, 0.0 }, .{
        .text = "hello",
        .category = "greetings",
    });
    try engine.indexByVector("doc-2", &[_]f32{ 0.0, 1.0, 0.0 }, .{
        .text = "world",
        .extra = "bonus",
    });

    // Save.
    const path = "/tmp/wdbx_test_roundtrip.bin";
    try save(&engine, path);

    defer {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        const io = io_backend.io();
        // Deletion failure is benign: this is test cleanup of a temp file.
        // The OS will reclaim /tmp on reboot regardless.
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
        io_backend.deinit();
    }

    // Load.
    var loaded = try load(allocator, path);
    defer loaded.deinit();

    // Verify.
    try std.testing.expectEqual(@as(usize, 2), loaded.vectors_array.items.len);
    try std.testing.expectEqualStrings("doc-1", loaded.vectors_array.items[0].id);
    try std.testing.expectEqualStrings("doc-2", loaded.vectors_array.items[1].id);
    try std.testing.expectEqualStrings("hello", loaded.vectors_array.items[0].metadata.text);
    try std.testing.expectEqualStrings("greetings", loaded.vectors_array.items[0].metadata.category.?);
    try std.testing.expectEqualStrings("bonus", loaded.vectors_array.items[1].metadata.extra.?);

    // Search should work on loaded engine.
    const results = try loaded.searchByVector(&[_]f32{ 0.9, 0.1, 0.0 }, .{ .k = 1 });
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("doc-1", results[0].id);
}

test "Persistence invalid magic" {
    const env_gate = @import("common");
    if (!env_gate.canRunTest()) return;
    const secretPtr = std.c.getenv("ABI_JWT_SECRET");
    if (secretPtr == null) return;
    const allocator = std.testing.allocator;
    const path = "/tmp/wdbx_test_bad_magic.bin";

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Write garbage.
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    try file.writeStreamingAll(io, "NOPE\x01\x00\x00\x00");
    file.close(io);

    // Deletion failure is benign: this is test cleanup of a temp file.
    // The OS will reclaim /tmp on reboot regardless.
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    try std.testing.expectError(PersistenceError.InvalidMagic, load(allocator, path));
}

test {
    std.testing.refAllDecls(@This());
}
