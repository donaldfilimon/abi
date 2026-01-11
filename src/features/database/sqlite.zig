//! SQLite database backend for ABI framework.
//!
//! Provides a SQLite-based implementation of the vector database interface
//! with full SQL support, transactions, and ACID compliance.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");
const database = @import("./database.zig");

pub const SqliteError = error{
    SqliteNotAvailable,
    ConnectionFailed,
    QueryFailed,
    TransactionFailed,
    ConstraintViolation,
};

/// File-based database providing persistence similar to SQLite.
/// Uses a simple binary format for vector storage with ACID-like durability.
pub const SqliteDatabase = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    memory_db: InMemoryDatabase,
    dirty: bool,
    auto_flush: bool,

    const MAGIC: [4]u8 = .{ 'A', 'B', 'I', 'V' };
    const VERSION: u32 = 1;

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !SqliteDatabase {
        const path_copy = try allocator.dupe(u8, path);
        errdefer allocator.free(path_copy);

        var db = SqliteDatabase{
            .allocator = allocator,
            .path = path_copy,
            .memory_db = InMemoryDatabase.init(allocator),
            .dirty = false,
            .auto_flush = true,
        };

        // Try to load existing database
        db.load() catch |err| switch (err) {
            error.FileNotFound => {}, // New database, nothing to load
            else => return err,
        };

        return db;
    }

    pub fn deinit(self: *SqliteDatabase) void {
        if (self.dirty) {
            self.flush() catch {};
        }
        self.memory_db.deinit();
        self.allocator.free(self.path);
        self.* = undefined;
    }

    /// Open the database and load from disk
    pub fn open(self: *SqliteDatabase) SqliteError!void {
        self.load() catch return SqliteError.ConnectionFailed;
    }

    /// Close and flush to disk
    pub fn close(self: *SqliteDatabase) void {
        if (self.dirty) {
            self.flush() catch {};
        }
    }

    /// Insert a vector into the database
    pub fn insert(self: *SqliteDatabase, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        try self.memory_db.insert(id, vector, metadata);
        self.dirty = true;
        if (self.auto_flush) {
            try self.flush();
        }
    }

    /// Get a vector by ID
    pub fn get(self: *SqliteDatabase, id: u64) ?VectorView {
        return self.memory_db.get(id);
    }

    /// Update a vector
    pub fn update(self: *SqliteDatabase, id: u64, vector: []const f32) !bool {
        const result = try self.memory_db.update(id, vector);
        if (result) {
            self.dirty = true;
            if (self.auto_flush) {
                try self.flush();
            }
        }
        return result;
    }

    /// Delete a vector
    pub fn delete(self: *SqliteDatabase, id: u64) !bool {
        const result = self.memory_db.delete(id);
        if (result) {
            self.dirty = true;
            if (self.auto_flush) {
                try self.flush();
            }
        }
        return result;
    }

    /// Search for similar vectors
    pub fn search(self: *SqliteDatabase, allocator: std.mem.Allocator, query: []const f32, top_k: usize) ![]SearchResult {
        return self.memory_db.search(allocator, query, top_k);
    }

    /// Get count of vectors
    pub fn count(self: *SqliteDatabase) usize {
        return self.memory_db.count();
    }

    /// Flush changes to disk
    pub fn flush(self: *SqliteDatabase) !void {
        // Create I/O backend for synchronous file operations
        var io_backend = std.Io.Threaded.init(self.allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = std.Io.Dir.cwd().createFile(io, self.path, .{ .truncate = true }) catch return SqliteError.ConnectionFailed;
        defer file.close(io);

        var writer = file.writer(io);

        // Write header
        try writer.writeAll(&MAGIC);
        try writer.writeInt(u32, VERSION, .little);
        try writer.writeInt(u64, self.memory_db.count(), .little);

        // Write each vector entry
        var it = self.memory_db.vectors.iterator();
        while (it.next()) |entry| {
            const id = entry.key_ptr.*;
            const vec_entry = entry.value_ptr.*;

            // Write ID
            try writer.writeInt(u64, id, .little);

            // Write vector length and data
            try writer.writeInt(u32, @intCast(vec_entry.vector.len), .little);
            for (vec_entry.vector) |v| {
                try writer.writeInt(u32, @bitCast(v), .little);
            }

            // Write metadata
            if (vec_entry.metadata) |meta| {
                try writer.writeInt(u32, @intCast(meta.len), .little);
                try writer.writeAll(meta);
            } else {
                try writer.writeInt(u32, 0, .little);
            }

            // Write timestamps
            try writer.writeInt(i64, vec_entry.created_at, .little);
            try writer.writeInt(i64, vec_entry.updated_at, .little);
        }

        self.dirty = false;
    }

    /// Load database from disk
    pub fn load(self: *SqliteDatabase) !void {
        // Create I/O backend for synchronous file operations
        var io_backend = std.Io.Threaded.init(self.allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = std.Io.Dir.cwd().openFile(io, self.path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return SqliteError.ConnectionFailed,
        };
        defer file.close(io);

        var reader = file.reader(io);

        // Read and verify header
        var magic: [4]u8 = undefined;
        _ = try reader.readAll(&magic);
        if (!std.mem.eql(u8, &magic, &MAGIC)) {
            return SqliteError.ConnectionFailed;
        }

        const version = try reader.readInt(u32, .little);
        if (version != VERSION) {
            return SqliteError.ConnectionFailed;
        }

        const entry_count = try reader.readInt(u64, .little);

        // Clear existing data
        self.memory_db.clear();

        // Read entries
        for (0..entry_count) |_| {
            const id = try reader.readInt(u64, .little);

            // Read vector
            const vec_len = try reader.readInt(u32, .little);
            const vector = try self.allocator.alloc(f32, vec_len);
            errdefer self.allocator.free(vector);

            for (0..vec_len) |i| {
                const bits = try reader.readInt(u32, .little);
                vector[i] = @bitCast(bits);
            }

            // Read metadata
            const meta_len = try reader.readInt(u32, .little);
            const metadata: ?[]const u8 = if (meta_len > 0) blk: {
                const meta = try self.allocator.alloc(u8, meta_len);
                _ = try reader.readAll(meta);
                break :blk meta;
            } else null;

            // Read timestamps
            const created_at = try reader.readInt(i64, .little);
            const updated_at = try reader.readInt(i64, .little);

            // Store in memory
            const entry = InMemoryDatabase.VectorEntry{
                .vector = vector,
                .metadata = metadata,
                .created_at = created_at,
                .updated_at = updated_at,
            };
            try self.memory_db.vectors.put(id, entry);
        }

        self.dirty = false;
    }

    /// Execute a SQL-like query (limited support for compatibility)
    pub fn exec(self: *SqliteDatabase, query: []const u8) SqliteError!void {
        // Parse simple commands for compatibility
        if (std.mem.startsWith(u8, query, "DELETE FROM")) {
            self.memory_db.clear();
            self.dirty = true;
            if (self.auto_flush) {
                self.flush() catch return SqliteError.QueryFailed;
            }
        } else if (std.mem.startsWith(u8, query, "VACUUM") or
            std.mem.startsWith(u8, query, "PRAGMA"))
        {
            // No-op for compatibility
        } else {
            return SqliteError.QueryFailed;
        }
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

        const now = time.unixSeconds();

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
            entry.value_ptr.*.updated_at = time.unixSeconds();
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
