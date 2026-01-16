//! In-memory vector database with persistence helpers.
const std = @import("std");
const simd = @import("../../shared/simd.zig");

pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
};

pub const VectorRecord = struct {
    id: u64,
    vector: []f32,
    metadata: ?[]const u8,
};

pub const VectorView = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
};

pub const SearchResult = struct {
    id: u64,
    score: f32,
};

pub const Stats = struct {
    count: usize,
    dimension: usize,
};

pub const Database = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    records: std.ArrayListUnmanaged(VectorRecord),
    /// O(1) lookup index: id -> array index for fast findIndex operations
    id_index: std.AutoHashMapUnmanaged(u64, usize),

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !Database {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .records = std.ArrayListUnmanaged(VectorRecord).empty,
            .id_index = std.AutoHashMapUnmanaged(u64, usize){},
        };
    }

    pub fn deinit(self: *Database) void {
        for (self.records.items) |record| {
            self.allocator.free(record.vector);
            if (record.metadata) |meta| {
                self.allocator.free(meta);
            }
        }
        self.records.deinit(self.allocator);
        self.id_index.deinit(self.allocator);
        self.allocator.free(self.name);
        self.* = undefined;
    }

    pub fn insert(self: *Database, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        if (self.findIndex(id) != null) return DatabaseError.DuplicateId;

        // Validate vector dimensions against existing records
        if (self.records.items.len > 0 and vector.len != self.records.items[0].vector.len) {
            return DatabaseError.InvalidDimension;
        }

        const vector_copy = try self.cloneVector(vector);
        errdefer self.allocator.free(vector_copy);
        const metadata_copy = if (metadata) |meta|
            try self.allocator.dupe(u8, meta)
        else
            null;
        const new_index = self.records.items.len;
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector_copy,
            .metadata = metadata_copy,
        });
        // Maintain O(1) lookup index
        try self.id_index.put(self.allocator, id, new_index);
    }

    pub fn update(self: *Database, id: u64, vector: []const f32) !bool {
        const index = self.findIndex(id) orelse return false;
        const vector_copy = try self.cloneVector(vector);
        self.allocator.free(self.records.items[index].vector);
        self.records.items[index].vector = vector_copy;
        return true;
    }

    pub fn delete(self: *Database, id: u64) bool {
        const index = self.findIndex(id) orelse return false;
        const record = self.records.swapRemove(index);
        self.allocator.free(record.vector);
        if (record.metadata) |meta| {
            self.allocator.free(meta);
        }
        // Remove from O(1) index
        _ = self.id_index.remove(id);
        // If swapRemove moved the last element to fill the gap, update its index
        if (index < self.records.items.len) {
            const moved_id = self.records.items[index].id;
            // Use getPtr to safely update existing entry without allocation
            if (self.id_index.getPtr(moved_id)) |idx_ptr| {
                idx_ptr.* = index;
            }
        }
        return true;
    }

    pub fn get(self: *Database, id: u64) ?VectorView {
        const index = self.findIndex(id) orelse return null;
        const record = self.records.items[index];
        return VectorView{
            .id = record.id,
            .vector = record.vector,
            .metadata = record.metadata,
        };
    }

    pub fn list(self: *Database, allocator: std.mem.Allocator, limit: usize) ![]VectorView {
        const count = @min(limit, self.records.items.len);
        const output = try allocator.alloc(VectorView, count);
        for (output, 0..) |*view, i| {
            const record = self.records.items[i];
            view.* = .{
                .id = record.id,
                .vector = record.vector,
                .metadata = record.metadata,
            };
        }
        return output;
    }

    /// Optimized search using single-pass algorithm with heap-based top-k selection
    pub fn search(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]SearchResult {
        const qlen = query.len;
        if (qlen == 0 or self.records.items.len == 0 or top_k == 0) {
            return allocator.alloc(SearchResult, 0);
        }

        // Single allocation: results buffer sized to top_k
        var results = try std.ArrayListUnmanaged(SearchResult).initCapacity(allocator, top_k);
        errdefer results.deinit(allocator);

        // Track minimum score in results for early rejection
        var min_score: f32 = -std.math.inf(f32);
        var min_idx: usize = 0;

        // Single-pass: compute similarity and maintain top-k in-place
        for (self.records.items) |record| {
            if (record.vector.len != qlen) continue;

            const score = simd.cosineSimilarity(query, record.vector);

            if (results.items.len < top_k) {
                // Still filling results, always add
                try results.append(allocator, .{ .id = record.id, .score = score });
                // Track new minimum
                if (score < min_score or results.items.len == 1) {
                    min_score = score;
                    min_idx = results.items.len - 1;
                }
            } else if (score > min_score) {
                // Replace minimum with this better result
                results.items[min_idx] = .{ .id = record.id, .score = score };
                // Find new minimum
                min_score = results.items[0].score;
                min_idx = 0;
                for (results.items, 0..) |r, i| {
                    if (r.score < min_score) {
                        min_score = r.score;
                        min_idx = i;
                    }
                }
            }
        }

        // Final sort for output ordering (only top_k elements, not full dataset)
        sortResults(results.items);
        return results.toOwnedSlice(allocator);
    }

    pub fn stats(self: *Database) Stats {
        if (self.records.items.len == 0) {
            return .{ .count = 0, .dimension = 0 };
        }
        return .{
            .count = self.records.items.len,
            .dimension = self.records.items[0].vector.len,
        };
    }

    pub fn optimize(self: *Database) void {
        self.records.shrinkAndFree(self.allocator, self.records.items.len);
    }

    /// O(1) lookup using hash index instead of O(n) linear scan
    fn findIndex(self: *Database, id: u64) ?usize {
        return self.id_index.get(id);
    }

    fn cloneVector(self: *Database, vector: []const f32) ![]f32 {
        const copy = try self.allocator.alloc(f32, vector.len);
        std.mem.copyForwards(f32, copy, vector);
        return copy;
    }

    pub fn saveToFile(self: *const Database, path: []const u8) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);

        var buf: [4096]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        // `Io.File.Writer` in Zig 0.16 does not expose an `any()` method.
        // We just need a writer that implements `WriteSeek`/`WriteByte`, which
        // `file_writer` already is. We pass its address to `Stringify`.
        var any_writer = &file_writer;

        var stringify: std.json.Stringify = .{
            // `any_writer` is an alias to the underlying file writer; cast
            // to the generic `std.io.Writer` required by `Stringify`.
            .writer = @ptrCast(&any_writer),
            .options = .{ .whitespace = .indent_4 },
        };
        try stringify.beginArray();
        for (self.records.items) |record| {
            try stringify.write(record);
        }
        try stringify.endArray();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Database {
        var io_backend = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024)); // 1GB limit
        defer allocator.free(content);

        var parsed = try std.json.parseFromSlice([]VectorRecord, allocator, content, .{});
        defer parsed.deinit();

        var records = std.ArrayListUnmanaged(VectorRecord).empty;
        try records.ensureTotalCapacity(allocator, parsed.value.len);

        // Build O(1) lookup index during load
        var id_index = std.AutoHashMapUnmanaged(u64, usize){};
        try id_index.ensureTotalCapacity(allocator, @intCast(parsed.value.len));

        for (parsed.value, 0..) |record, idx| {
            const vector_copy = try allocator.dupe(f32, record.vector);
            const metadata_copy = if (record.metadata) |m| try allocator.dupe(u8, m) else null;

            try records.append(allocator, .{
                .id = record.id,
                .vector = vector_copy,
                .metadata = metadata_copy,
            });
            id_index.putAssumeCapacity(record.id, idx);
        }

        return Database{
            .allocator = allocator,
            .name = try allocator.dupe(u8, std.fs.path.basename(path)),
            .records = records,
            .id_index = id_index,
        };
    }

    pub fn insertOwned(self: *Database, id: u64, vector: []f32, metadata: ?[]u8) !void {
        if (self.findIndex(id) != null) {
            self.allocator.free(vector);
            if (metadata) |meta| self.allocator.free(meta);
            return DatabaseError.DuplicateId;
        }
        if (self.records.items.len > 0 and vector.len != self.records.items[0].vector.len) {
            self.allocator.free(vector);
            if (metadata) |meta| self.allocator.free(meta);
            return DatabaseError.InvalidDimension;
        }
        errdefer {
            self.allocator.free(vector);
            if (metadata) |meta| self.allocator.free(meta);
        }
        const new_index = self.records.items.len;
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector,
            .metadata = metadata,
        });
        // Maintain O(1) lookup index
        try self.id_index.put(self.allocator, id, new_index);
    }
};

fn sortResults(results: []SearchResult) void {
    std.sort.pdq(SearchResult, results, {}, struct {
        fn lessThan(_: void, lhs: SearchResult, rhs: SearchResult) bool {
            return lhs.score > rhs.score;
        }
    }.lessThan);
}

test "search sorts by descending similarity and truncates" {
    var db = try Database.init(std.testing.allocator, "search-test");
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 1.0, 1.0 }, null);

    const results = try db.search(std.testing.allocator, &.{ 1.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expectEqual(@as(u64, 3), results[1].id);
}
