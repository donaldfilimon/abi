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

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !Database {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .records = std.ArrayListUnmanaged(VectorRecord).empty,
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
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector_copy,
            .metadata = metadata_copy,
        });
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

    pub fn search(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]SearchResult {
        // Count matching vectors to size allocations more accurately
        const qlen = query.len;
        var matches: usize = 0;
        for (self.records.items) |record| {
            if (record.vector.len == qlen) matches += 1;
        }
        // Pre-allocate capacity based on matches
        const capacity = @min(matches, top_k * 2);
        var results = try std.ArrayListUnmanaged(SearchResult).initCapacity(allocator, capacity);
        errdefer results.deinit(allocator);

        // Collect vectors that match the query dimensions for batch processing
        var valid_vectors = std.ArrayListUnmanaged([]const f32).empty;
        var valid_ids = std.ArrayListUnmanaged(u64).empty;
        defer valid_vectors.deinit(allocator);
        defer valid_ids.deinit(allocator);

        for (self.records.items) |record| {
            if (record.vector.len == qlen) {
                try valid_vectors.append(allocator, record.vector);
                try valid_ids.append(allocator, record.id);
            }
        }

        // Use batch cosine similarity for better performance
        const scores = try allocator.alloc(f32, valid_vectors.items.len);
        defer allocator.free(scores);
        simd.batchCosineSimilarity(query, valid_vectors.items, scores);

        // Create search results
        for (valid_ids.items, scores) |id, score| {
            try results.append(allocator, .{ .id = id, .score = score });
        }
        sortResults(results.items);
        if (top_k < results.items.len) {
            results.shrinkRetainingCapacity(top_k);
        }
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

    fn findIndex(self: *Database, id: u64) ?usize {
        for (self.records.items, 0..) |record, i| {
            if (record.id == id) return i;
        }
        return null;
    }

    fn cloneVector(self: *Database, vector: []const f32) ![]f32 {
        const copy = try self.allocator.alloc(f32, vector.len);
        std.mem.copyForwards(f32, copy, vector);
        return copy;
    }

    pub fn saveToFile(self: *const Database, path: []const u8) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);

        var buf: [4096]u8 = undefined;
        var file_writer = file.writer(io, &buf);

        var stringify: std.json.Stringify = .{
            .writer = &file_writer,
            .options = .{ .whitespace = .indent_4 },
        };
        try stringify.beginArray();
        for (self.records.items) |record| {
            try stringify.write(record);
        }
        try stringify.endArray();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Database {
        var io_backend = std.Io.Threaded.init(allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024)); // 1GB limit
        defer allocator.free(content);

        var parsed = try std.json.parseFromSlice([]VectorRecord, allocator, content, .{});
        defer parsed.deinit();

        var records = std.ArrayListUnmanaged(VectorRecord).empty;
        try records.ensureTotalCapacity(allocator, parsed.value.len);

        for (parsed.value) |record| {
            const vector_copy = try allocator.dupe(f32, record.vector);
            const metadata_copy = if (record.metadata) |m| try allocator.dupe(u8, m) else null;

            try records.append(allocator, .{
                .id = record.id,
                .vector = vector_copy,
                .metadata = metadata_copy,
            });
        }

        return Database{
            .allocator = allocator,
            .name = try allocator.dupe(u8, std.fs.path.basename(path)),
            .records = records,
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
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector,
            .metadata = metadata,
        });
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
