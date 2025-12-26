//! In-memory vector database with persistence helpers.
const std = @import("std");
const simd = @import("../../shared/simd.zig");

pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
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

pub const StorageError = error{
    InvalidFormat,
    UnsupportedVersion,
    TruncatedData,
    PayloadTooLarge,
};

pub const SaveError =
    std.fs.File.OpenError ||
    std.fs.File.WriteError ||
    std.mem.Allocator.Error ||
    StorageError;
pub const LoadError =
    std.fs.Dir.ReadFileAllocError ||
    std.mem.Allocator.Error ||
    StorageError;

const storage_magic = "ABID";
const storage_version: u16 = 1;

pub const Database = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    records: std.ArrayList(VectorRecord),

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !Database {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .records = std.ArrayList(VectorRecord).empty,
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
        var results = std.ArrayList(SearchResult).empty;
        errdefer results.deinit(allocator);
        for (self.records.items) |record| {
            if (record.vector.len != query.len) continue;
            const score = simd.VectorOps.cosineSimilarity(query, record.vector);
            try results.append(allocator, .{ .id = record.id, .score = score });
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

    pub fn saveToFile(self: *Database, path: []const u8) SaveError!void {
        var buffer = std.ArrayList(u8).empty;
        defer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, storage_magic);
        try appendInt(&buffer, self.allocator, u16, storage_version);
        if (self.name.len > std.math.maxInt(u16)) return StorageError.PayloadTooLarge;
        try appendInt(&buffer, self.allocator, u16, @intCast(self.name.len));
        try buffer.appendSlice(self.allocator, self.name);
        try appendInt(&buffer, self.allocator, u64, @intCast(self.records.items.len));

        for (self.records.items) |record| {
            if (record.vector.len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;
            const meta_len: usize = if (record.metadata) |meta| meta.len else 0;
            if (meta_len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;

            try appendInt(&buffer, self.allocator, u64, record.id);
            try appendInt(&buffer, self.allocator, u32, @intCast(record.vector.len));
            try appendInt(&buffer, self.allocator, u32, @intCast(meta_len));

            for (record.vector) |value| {
                const bits: u32 = @bitCast(value);
                try appendInt(&buffer, self.allocator, u32, bits);
            }
            if (record.metadata) |meta| {
                try buffer.appendSlice(self.allocator, meta);
            }
        }

        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(buffer.items);
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) LoadError!Database {
        const data = try std.fs.cwd().readFileAlloc(
            path,
            allocator,
            .limited(64 * 1024 * 1024),
        );
        defer allocator.free(data);

        var cursor = Cursor{ .data = data };
        const magic = try cursor.readBytes(storage_magic.len);
        if (!std.mem.eql(u8, magic, storage_magic)) return StorageError.InvalidFormat;

        const version = try cursor.readInt(u16);
        if (version != storage_version) return StorageError.UnsupportedVersion;

        const name_len = try cursor.readInt(u16);
        const name_bytes = try cursor.readBytes(name_len);
        var db = try Database.init(allocator, name_bytes);
        errdefer db.deinit();

        const record_count = try cursor.readInt(u64);
        var index: u64 = 0;
        while (index < record_count) : (index += 1) {
            const id = try cursor.readInt(u64);
            const vec_len: usize = @intCast(try cursor.readInt(u32));
            const meta_len: usize = @intCast(try cursor.readInt(u32));

            const vector = try allocator.alloc(f32, vec_len);
            var metadata: ?[]u8 = null;
            var owns_buffers = true;
            errdefer {
                if (owns_buffers) {
                    allocator.free(vector);
                    if (metadata) |meta| allocator.free(meta);
                }
            }
            var i: usize = 0;
            while (i < vec_len) : (i += 1) {
                const bits = try cursor.readInt(u32);
                vector[i] = @bitCast(bits);
            }
            if (meta_len > 0) {
                metadata = try allocator.alloc(u8, meta_len);
                const meta_bytes = try cursor.readBytes(meta_len);
                std.mem.copyForwards(u8, metadata.?, meta_bytes);
            }

            owns_buffers = false;
            db.insertOwned(id, vector, metadata) catch |err| switch (err) {
                error.DuplicateId, error.VectorNotFound => return StorageError.InvalidFormat,
                error.OutOfMemory => return error.OutOfMemory,
            };
        }

        return db;
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

    fn insertOwned(self: *Database, id: u64, vector: []f32, metadata: ?[]u8) !void {
        if (self.findIndex(id) != null) {
            self.allocator.free(vector);
            if (metadata) |meta| self.allocator.free(meta);
            return DatabaseError.DuplicateId;
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

const Cursor = struct {
    data: []const u8,
    index: usize = 0,

    fn readBytes(self: *Cursor, len: usize) StorageError![]const u8 {
        if (self.index + len > self.data.len) return StorageError.TruncatedData;
        const slice = self.data[self.index .. self.index + len];
        self.index += len;
        return slice;
    }

    fn readInt(self: *Cursor, comptime T: type) StorageError!T {
        const bytes = try self.readBytes(@sizeOf(T));
        return std.mem.readInt(
            T,
            @as(*const [@sizeOf(T)]u8, @ptrCast(bytes.ptr)),
            .little,
        );
    }
};

fn appendInt(
    buffer: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    comptime T: type,
    value: T,
) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try buffer.appendSlice(allocator, &bytes);
}

test "database backup and restore" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.testing.allocator;
    const path = try tmp.dir.realpathAlloc(allocator, "db.bin");
    defer allocator.free(path);

    var db = try Database.init(allocator, "backup-test");
    defer db.deinit();

    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta");
    try db.insert(2, &.{ 0.4, 0.5, 0.6 }, null);

    try db.saveToFile(path);

    var restored = try Database.loadFromFile(allocator, path);
    defer restored.deinit();

    const view = restored.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 1), view.id);
    try std.testing.expectEqual(@as(usize, 3), view.vector.len);
    try std.testing.expectEqualStrings("meta", view.metadata.?);
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
