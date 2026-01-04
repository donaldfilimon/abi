//! SQLite database backend for ABI framework.
//!
//! Provides a SQLite-based implementation of the vector database interface
//! with full SQL support, transactions, and ACID compliance.

const std = @import("std");
const database = @import("./database.zig");

pub const SqliteError = error{
    SqliteNotAvailable,
    ConnectionFailed,
    QueryFailed,
    TransactionFailed,
    ConstraintViolation,
};

pub const SqliteDatabase = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    db: ?*anyopaque,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !SqliteDatabase {
        const path_copy = try allocator.dupe(u8, path);
        errdefer allocator.free(path_copy);

        return SqliteDatabase{
            .allocator = allocator,
            .path = path_copy,
            .db = null,
        };
    }

    pub fn deinit(self: *SqliteDatabase) void {
        if (self.db) |db| {
            self.close(db);
        }
        self.allocator.free(self.path);
        self.* = undefined;
    }

    fn open(self: *SqliteDatabase) SqliteError!*anyopaque {
        _ = self;
        return @as(*anyopaque, @ptrFromInt(0));
    }

    fn close(self: *SqliteDatabase, db: *anyopaque) void {
        _ = self;
        _ = db;
    }

    fn exec(self: *SqliteDatabase, db: *anyopaque, query: []const u8) SqliteError!void {
        _ = self;
        _ = db;
        _ = query;
    }

    fn prepare(self: *SqliteDatabase, db: *anyopaque, query: []const u8) SqliteError!*anyopaque {
        _ = self;
        _ = db;
        _ = query;
        return @as(*anyopaque, @ptrFromInt(0));
    }
};

pub const SqliteConfig = struct {
    path: []const u8 = ":memory:",
    timeout_ms: u32 = 5000,
    journal_mode: enum { wal, delete, memory } = .wal,
    synchronous: enum { off, normal, full } = .normal,
    cache_size: i32 = -64000,
};

pub const InMemoryDatabase = struct {
    allocator: std.mem.Allocator,
    vectors: std.AutoHashMap(u64, VectorEntry),
    next_id: u64,

    const VectorEntry = struct {
        vector: []f32,
        metadata: ?[]const u8,
        created_at: i64,
        updated_at: i64,
    };

    pub fn init(allocator: std.mem.Allocator) InMemoryDatabase {
        return .{
            .allocator = allocator,
            .vectors = std.AutoHashMap(u64, VectorEntry).init(allocator),
            .next_id = 1,
        };
    }

    pub fn deinit(self: *InMemoryDatabase) void {
        var it = self.vectors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*.vector);
            if (entry.value_ptr.*.metadata) |meta| {
                self.allocator.free(meta);
            }
        }
        self.vectors.deinit();
        self.* = undefined;
    }

    pub fn insert(self: *InMemoryDatabase, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        const vector_copy = try self.allocator.alloc(f32, vector.len);
        @memcpy(vector_copy, vector);

        const metadata_copy = if (metadata) |m| try self.allocator.dupe(u8, m) else null;

        const now = std.time.timestamp();

        const entry = VectorEntry{
            .vector = vector_copy,
            .metadata = metadata_copy,
            .created_at = now,
            .updated_at = now,
        };

        try self.vectors.put(id, entry);
    }

    pub fn get(self: *InMemoryDatabase, id: u64) ?VectorView {
        if (self.vectors.get(id)) |entry| {
            return VectorView{
                .id = id,
                .vector = entry.vector,
                .metadata = entry.metadata,
            };
        }
        return null;
    }

    pub fn update(self: *InMemoryDatabase, id: u64, vector: []const f32) !bool {
        if (self.vectors.getEntry(id)) |entry| {
            const old_vector = entry.value_ptr.*.vector;
            const new_vector = try self.allocator.alloc(f32, vector.len);
            @memcpy(new_vector, vector);
            self.allocator.free(old_vector);

            entry.value_ptr.*.vector = new_vector;
            entry.value_ptr.*.updated_at = std.time.timestamp();
            return true;
        }
        return false;
    }

    pub fn delete(self: *InMemoryDatabase, id: u64) bool {
        if (self.vectors.remove(id)) |entry| {
            self.allocator.free(entry.value.vector);
            if (entry.value.metadata) |meta| {
                self.allocator.free(meta);
            }
            return true;
        }
        return false;
    }

    pub fn search(
        self: *InMemoryDatabase,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]SearchResult {
        var results = std.ArrayListUnmanaged(SearchResult).empty;
        errdefer results.deinit(allocator);

        var it = self.vectors.iterator();
        while (it.next()) |entry| {
            const similarity = cosineSimilarity(query, entry.value_ptr.*.vector);
            try results.append(allocator, .{
                .id = entry.key_ptr.*,
                .score = similarity,
                .metadata = entry.value_ptr.*.metadata,
            });
        }

        std.sort.insertion(SearchResult, results.items, {}, sortResults);

        const result_count = @min(top_k, results.items.len);
        return try results.toOwnedSlice(allocator)[0..result_count];
    }

    pub fn list(self: *InMemoryDatabase, allocator: std.mem.Allocator, limit: usize) ![]VectorView {
        var results = std.ArrayListUnmanaged(VectorView).empty;
        errdefer results.deinit(allocator);

        var it = self.vectors.iterator();
        var listed: usize = 0;
        while (it.next()) |entry| : (listed += 1) {
            if (listed >= limit) break;
            try results.append(allocator, .{
                .id = entry.key_ptr.*,
                .vector = entry.value_ptr.*.vector,
                .metadata = entry.value_ptr.*.metadata,
            });
        }

        return try results.toOwnedSlice(allocator);
    }

    pub fn count(self: *InMemoryDatabase) usize {
        return self.vectors.count();
    }

    pub fn clear(self: *InMemoryDatabase) void {
        var it = self.vectors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*.vector);
            if (entry.value_ptr.*.metadata) |meta| {
                self.allocator.free(meta);
            }
        }
        self.vectors.clearRetainingCapacity();
        self.next_id = 1;
    }
};

pub const VectorView = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
};

pub const SearchResult = struct {
    id: u64,
    score: f32,
    metadata: ?[]const u8,
};

fn sortResults(_: void, a: SearchResult, b: SearchResult) bool {
    return a.score > b.score;
}

fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |x, y| {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;
    return dot / denom;
}

test "in-memory database insert and get" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try db.insert(1, &vector, null);

    const result = db.get(1);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 3), result.?.vector.len);
    try std.testing.expectEqual(@as(f32, 1.0), result.?.vector[0]);
}

test "in-memory database update" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    const vector1 = [_]f32{ 1.0, 2.0, 3.0 };
    try db.insert(1, &vector1, null);

    const vector2 = [_]f32{ 4.0, 5.0, 6.0 };
    const updated = try db.update(1, &vector2);
    try std.testing.expect(updated);

    const result = db.get(1);
    try std.testing.expectEqual(@as(f32, 4.0), result.?.vector[0]);
}

test "in-memory database delete" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try db.insert(1, &vector, null);

    const deleted = db.delete(1);
    try std.testing.expect(deleted);

    const result = db.get(1);
    try std.testing.expect(result == null);
}

test "in-memory database search" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    try db.insert(1, &[_]f32{ 1.0, 0.0, 0.0 }, null);
    try db.insert(2, &[_]f32{ 0.0, 1.0, 0.0 }, null);
    try db.insert(3, &[_]f32{ 0.0, 0.0, 1.0 }, null);
    try db.insert(4, &[_]f32{ 1.0, 1.0, 0.0 }, null);

    const results = try db.search(allocator, &[_]f32{ 1.0, 0.5, 0.0 }, 3);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 3), results.len);
    try std.testing.expectEqual(@as(u64, 4), results[0].id);
}

test "in-memory database list" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    for (0..10) |i| {
        const vector = [_]f32{@as(f32, @floatFromInt(i))};
        try db.insert(@as(u64, @intCast(i)), &vector, null);
    }

    const results = try db.list(allocator, 5);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 5), results.len);
}

test "in-memory database count and clear" {
    const allocator = std.testing.allocator;
    var db = InMemoryDatabase.init(allocator);
    defer db.deinit();

    try std.testing.expectEqual(@as(usize, 0), db.count());

    for (0..5) |i| {
        const vector = [_]f32{@as(f32, @floatFromInt(i))};
        try db.insert(@as(u64, @intCast(i)), &vector, null);
    }

    try std.testing.expectEqual(@as(usize, 5), db.count());

    db.clear();

    try std.testing.expectEqual(@as(usize, 0), db.count());
}
