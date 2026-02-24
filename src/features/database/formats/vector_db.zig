//! Vector Database Integration for Unified Format
//!
//! Provides seamless integration between the unified storage format
//! and the WDBX vector database for fast similarity search.

const std = @import("std");
const unified = @import("unified.zig");
const simd = @import("../../../services/shared/simd/mod.zig");

pub const VectorDbError = error{
    InvalidDimension,
    DuplicateId,
    VectorNotFound,
    IndexCorrupted,
    OutOfMemory,
    IoError,
};

/// Vector record for database storage
pub const VectorRecord = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
};

/// Vector database stored in unified format
pub const VectorDatabase = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    dimension: usize,
    vectors: std.ArrayListUnmanaged(StoredVector),
    id_index: std.AutoHashMapUnmanaged(u64, usize),

    const StoredVector = struct {
        id: u64,
        data: []f32,
        metadata: ?[]u8,
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, dimension: usize) VectorDatabase {
        return .{
            .allocator = allocator,
            .name = name,
            .dimension = dimension,
            .vectors = std.ArrayListUnmanaged(StoredVector).empty,
            .id_index = std.AutoHashMapUnmanaged(u64, usize).empty,
        };
    }

    pub fn deinit(self: *VectorDatabase) void {
        for (self.vectors.items) |*v| {
            self.allocator.free(v.data);
            if (v.metadata) |m| self.allocator.free(m);
        }
        self.vectors.deinit(self.allocator);
        self.id_index.deinit(self.allocator);
    }

    /// Insert a vector
    pub fn insert(self: *VectorDatabase, id: u64, vector: []const f32, metadata: ?[]const u8) VectorDbError!void {
        if (vector.len != self.dimension) return error.InvalidDimension;
        if (self.id_index.contains(id)) return error.DuplicateId;

        const data = self.allocator.alloc(f32, vector.len) catch return error.OutOfMemory;
        @memcpy(data, vector);

        var meta_copy: ?[]u8 = null;
        if (metadata) |m| {
            meta_copy = self.allocator.alloc(u8, m.len) catch {
                self.allocator.free(data);
                return error.OutOfMemory;
            };
            @memcpy(meta_copy.?, m);
        }

        const index = self.vectors.items.len;
        self.vectors.append(self.allocator, .{
            .id = id,
            .data = data,
            .metadata = meta_copy,
        }) catch {
            self.allocator.free(data);
            if (meta_copy) |m| self.allocator.free(m);
            return error.OutOfMemory;
        };

        self.id_index.put(self.allocator, id, index) catch return error.OutOfMemory;
    }

    /// Search for similar vectors using cosine similarity
    pub fn search(self: *VectorDatabase, query: []const f32, top_k: usize) VectorDbError![]SearchResult {
        if (query.len != self.dimension) return error.InvalidDimension;

        var results = std.ArrayListUnmanaged(SearchResult).empty;
        defer results.deinit(self.allocator);

        for (self.vectors.items) |v| {
            const score = simd.cosineSimilarity(query, v.data);
            results.append(self.allocator, .{ .id = v.id, .score = score }) catch return error.OutOfMemory;
        }

        // Sort by score descending
        std.mem.sort(SearchResult, results.items, {}, struct {
            fn cmp(_: void, a: SearchResult, b: SearchResult) bool {
                return a.score > b.score;
            }
        }.cmp);

        const k = @min(top_k, results.items.len);
        const output = self.allocator.alloc(SearchResult, k) catch return error.OutOfMemory;
        @memcpy(output, results.items[0..k]);
        return output;
    }

    /// Get vector by ID
    pub fn get(self: *VectorDatabase, id: u64) ?VectorRecord {
        const index = self.id_index.get(id) orelse return null;
        const v = self.vectors.items[index];
        return .{ .id = v.id, .vector = v.data, .metadata = v.metadata };
    }

    /// Delete vector by ID
    pub fn delete(self: *VectorDatabase, id: u64) bool {
        const index = self.id_index.get(id) orelse return false;
        const v = &self.vectors.items[index];
        self.allocator.free(v.data);
        if (v.metadata) |m| self.allocator.free(m);
        _ = self.id_index.remove(id);

        // Swap with last element
        if (index < self.vectors.items.len - 1) {
            self.vectors.items[index] = self.vectors.items[self.vectors.items.len - 1];
            if (self.id_index.getPtr(self.vectors.items[index].id)) |ptr| {
                ptr.* = index;
            }
        }
        _ = self.vectors.pop();
        return true;
    }

    /// Save to unified format
    pub fn save(self: *VectorDatabase) VectorDbError![]u8 {
        var builder = unified.UnifiedFormatBuilder.init(self.allocator);
        defer builder.deinit();

        _ = builder.setCompression(.lz4);

        // Save vectors as tensors
        for (self.vectors.items, 0..) |v, i| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vector_{d}", .{i}) catch |err| {
                std.log.debug("Failed to format vector name {d}: {t}", .{ i, err });
                continue;
            };
            const bytes = std.mem.sliceAsBytes(v.data);
            _ = builder.addTensor(name, bytes, .vec_f32, &.{v.data.len}) catch |err| {
                std.log.debug("Failed to add vector tensor {d}: {t}", .{ i, err });
                continue;
            };
        }

        // Save metadata
        var meta_buf: [256]u8 = undefined;
        const meta = std.fmt.bufPrint(&meta_buf, "{{\"name\":\"{s}\",\"dimension\":{d},\"count\":{d}}}", .{
            self.name,
            self.dimension,
            self.vectors.items.len,
        }) catch return error.IoError;
        _ = builder.addMetadata("database_info", meta) catch return error.OutOfMemory;

        // Save ID mapping
        var ids = self.allocator.alloc(u64, self.vectors.items.len) catch return error.OutOfMemory;
        defer self.allocator.free(ids);
        for (self.vectors.items, 0..) |v, i| {
            ids[i] = v.id;
        }
        _ = builder.addTensor("__ids__", std.mem.sliceAsBytes(ids), .u64, &.{ids.len}) catch return error.OutOfMemory;

        return builder.build() catch return error.OutOfMemory;
    }

    /// Load from unified format
    pub fn load(allocator: std.mem.Allocator, data: []const u8) VectorDbError!VectorDatabase {
        var format = unified.UnifiedFormat.fromMemory(allocator, data) catch return error.IoError;
        defer format.deinit();

        // Parse metadata to get dimension from JSON: {"name":"...","dimension":N,"count":M}
        var dimension: usize = 3; // Default fallback
        var db_name: []const u8 = "loaded";
        if (format.getMetadata("database_info")) |meta_json| {
            // Simple JSON parsing for dimension field
            if (std.mem.indexOf(u8, meta_json, "\"dimension\":")) |dim_start| {
                const num_start = dim_start + "\"dimension\":".len;
                var num_end = num_start;
                while (num_end < meta_json.len and (meta_json[num_end] >= '0' and meta_json[num_end] <= '9')) {
                    num_end += 1;
                }
                if (num_end > num_start) {
                    dimension = std.fmt.parseInt(usize, meta_json[num_start..num_end], 10) catch 3;
                }
            }
            // Parse name field
            if (std.mem.indexOf(u8, meta_json, "\"name\":\"")) |name_start| {
                const str_start = name_start + "\"name\":\"".len;
                if (std.mem.indexOfPos(u8, meta_json, str_start, "\"")) |str_end| {
                    db_name = meta_json[str_start..str_end];
                }
            }
        }

        var db = VectorDatabase.init(allocator, db_name, dimension);
        errdefer db.deinit();

        // Load ID mapping
        const ids_desc = format.getTensor("__ids__") orelse return error.IndexCorrupted;
        const ids_data = format.getTensorData(allocator, "__ids__") catch return error.IoError;
        defer if (ids_desc.compressed_size > 0) allocator.free(ids_data);

        const aligned_ids: []align(@alignOf(u64)) const u8 = @alignCast(ids_data);
        const ids = std.mem.bytesAsSlice(u64, aligned_ids);

        // Load vectors
        var i: usize = 0;
        var name_buf: [64]u8 = undefined;
        while (i < ids.len) : (i += 1) {
            const name = std.fmt.bufPrint(&name_buf, "vector_{d}", .{i}) catch |err| {
                std.log.debug("Failed to format vector name during load {d}: {t}", .{ i, err });
                continue;
            };
            const desc = format.getTensor(name) orelse continue;
            const vec_data = format.getTensorData(allocator, name) catch |err| {
                std.log.debug("Failed to get tensor data for vector {d}: {t}", .{ i, err });
                continue;
            };
            defer if (desc.compressed_size > 0) allocator.free(vec_data);

            const aligned_vec: []align(@alignOf(f32)) const u8 = @alignCast(vec_data);
            const vector = std.mem.bytesAsSlice(f32, aligned_vec);
            db.insert(ids[i], vector, null) catch |err| {
                std.log.debug("Failed to insert vector {d}: {t}", .{ i, err });
                continue;
            };
        }

        return db;
    }

    pub fn count(self: *const VectorDatabase) usize {
        return self.vectors.items.len;
    }
};

pub const SearchResult = struct {
    id: u64,
    score: f32,
};

// Cosine similarity now uses shared SIMD implementation via simd.cosineSimilarity

test "vector database basic operations" {
    const allocator = std.testing.allocator;

    var db = VectorDatabase.init(allocator, "test", 3);
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0, 0.0 }, "first");
    try db.insert(2, &.{ 0.0, 1.0, 0.0 }, "second");
    try db.insert(3, &.{ 1.0, 1.0, 0.0 }, null);

    try std.testing.expectEqual(@as(usize, 3), db.count());

    const results = try db.search(&.{ 1.0, 0.5, 0.0 }, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 3), results[0].id); // Most similar

    const record = db.get(1).?;
    try std.testing.expectEqual(@as(u64, 1), record.id);
    try std.testing.expectEqualStrings("first", record.metadata.?);

    try std.testing.expect(db.delete(2));
    try std.testing.expectEqual(@as(usize, 2), db.count());
}

test "vector database save/load" {
    const allocator = std.testing.allocator;

    // Create and populate
    var db = VectorDatabase.init(allocator, "test", 3);
    defer db.deinit();

    try db.insert(100, &.{ 0.1, 0.2, 0.3 }, null);
    try db.insert(200, &.{ 0.4, 0.5, 0.6 }, null);

    // Save
    const saved = try db.save();
    defer allocator.free(saved);

    // Load
    var loaded = try VectorDatabase.load(allocator, saved);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.count());
}

test {
    std.testing.refAllDecls(@This());
}
