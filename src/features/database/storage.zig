//! Persistence helpers for the WDBX vector database.
const std = @import("std");
const database = @import("database.zig");

const storage_magic = "ABID";
const storage_version: u16 = 1;

pub const StorageError = error{
    InvalidFormat,
    UnsupportedVersion,
    TruncatedData,
    PayloadTooLarge,
};

pub const SaveError =
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.mem.Allocator.Error ||
    StorageError;
pub const LoadError =
    std.Io.Dir.ReadFileAllocError ||
    std.mem.Allocator.Error ||
    StorageError;

pub fn saveDatabase(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
) SaveError!void {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, storage_magic);
    try appendInt(&buffer, allocator, u16, storage_version);
    if (db.name.len > std.math.maxInt(u16)) return StorageError.PayloadTooLarge;
    try appendInt(&buffer, allocator, u16, @intCast(db.name.len));
    try buffer.appendSlice(allocator, db.name);
    try appendInt(&buffer, allocator, u64, @intCast(db.records.items.len));

    for (db.records.items) |record| {
        if (record.vector.len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;
        const meta_len: usize = if (record.metadata) |meta| meta.len else 0;
        if (meta_len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;

        try appendInt(&buffer, allocator, u64, record.id);
        try appendInt(&buffer, allocator, u32, @intCast(record.vector.len));
        try appendInt(&buffer, allocator, u32, @intCast(meta_len));

        for (record.vector) |value| {
            const bits: u32 = @bitCast(value);
            try appendInt(&buffer, allocator, u32, bits);
        }
        if (record.metadata) |meta| {
            try buffer.appendSlice(allocator, meta);
        }
    }

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, buffer.items);
}

pub fn loadDatabase(allocator: std.mem.Allocator, path: []const u8) LoadError!database.Database {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    const data = try std.Io.Dir.cwd().readFileAlloc(
        io,
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
    var db = try database.Database.init(allocator, name_bytes);
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
            database.DatabaseError.DuplicateId,
            database.DatabaseError.VectorNotFound,
            database.DatabaseError.InvalidDimension,
            => return StorageError.InvalidFormat,
            error.OutOfMemory => return error.OutOfMemory,
        };
    }

    return db;
}

fn appendInt(
    buffer: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    comptime T: type,
    value: T,
) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try buffer.appendSlice(allocator, &bytes);
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

test "storage backup and restore" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.testing.allocator;
    const path = try tmp.dir.realpathAlloc(allocator, "db.bin");
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "backup-test");
    defer db.deinit();

    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta");
    try db.insert(2, &.{ 0.4, 0.5, 0.6 }, null);

    try saveDatabase(allocator, &db, path);

    var restored = try loadDatabase(allocator, path);
    defer restored.deinit();

    const view = restored.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 1), view.id);
    try std.testing.expectEqual(@as(usize, 3), view.vector.len);
    try std.testing.expectEqualStrings("meta", view.metadata.?);
}
