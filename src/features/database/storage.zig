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

/// Configuration for storage operations
pub const StorageConfig = struct {
    /// Write buffer size for buffered I/O (default 128KB)
    write_buffer_size: usize = 128 * 1024,
    /// Read buffer size for buffered I/O (default 128KB)
    read_buffer_size: usize = 128 * 1024,
    /// Whether to use streaming writes (reduces memory for large databases)
    use_streaming: bool = true,
};

/// Default storage configuration
pub const default_config = StorageConfig{};

pub const SaveError =
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.mem.Allocator.Error ||
    StorageError;
pub const LoadError =
    std.Io.Dir.ReadFileAllocError ||
    std.mem.Allocator.Error ||
    StorageError;

/// Save database using default configuration
pub fn saveDatabase(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
) SaveError!void {
    return saveDatabaseWithConfig(allocator, db, path, default_config);
}

/// Save database with custom configuration
pub fn saveDatabaseWithConfig(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageConfig,
) SaveError!void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    if (config.use_streaming) {
        // Streaming mode: write directly to file with buffer
        try saveDatabaseStreaming(allocator, db, &file, io, config.write_buffer_size);
    } else {
        // Legacy mode: buffer entire database in memory first
        try saveDatabaseBuffered(allocator, db, &file, io);
    }
}

/// Streaming save - writes directly to file with small buffer (memory efficient for large DBs)
fn saveDatabaseStreaming(
    allocator: std.mem.Allocator,
    db: *database.Database,
    file: *std.Io.File,
    io: anytype,
    buffer_size: usize,
) SaveError!void {
    // Use a write buffer for batching small writes
    const write_buffer = try allocator.alloc(u8, buffer_size);
    defer allocator.free(write_buffer);
    var buffer_pos: usize = 0;

    // Flush helper
    const flush = struct {
        fn f(buf: []u8, pos: *usize, f_file: *std.Io.File, f_io: anytype) !void {
            if (pos.* > 0) {
                try f_file.writeStreamingAll(f_io, buf[0..pos.*]);
                pos.* = 0;
            }
        }
    }.f;

    // Write helper that buffers small writes
    const writeBytes = struct {
        fn f(buf: []u8, pos: *usize, data: []const u8, w_file: *std.Io.File, w_io: anytype) !void {
            var remaining = data;
            while (remaining.len > 0) {
                const space = buf.len - pos.*;
                if (space == 0) {
                    try w_file.writeStreamingAll(w_io, buf[0..pos.*]);
                    pos.* = 0;
                    continue;
                }
                const to_copy = @min(remaining.len, space);
                @memcpy(buf[pos.*..][0..to_copy], remaining[0..to_copy]);
                pos.* += to_copy;
                remaining = remaining[to_copy..];
            }
        }
    }.f;

    // Write header
    try writeBytes(write_buffer, &buffer_pos, storage_magic, file, io);

    var version_bytes: [2]u8 = undefined;
    std.mem.writeInt(u16, &version_bytes, storage_version, .little);
    try writeBytes(write_buffer, &buffer_pos, &version_bytes, file, io);

    if (db.name.len > std.math.maxInt(u16)) return StorageError.PayloadTooLarge;
    var name_len_bytes: [2]u8 = undefined;
    std.mem.writeInt(u16, &name_len_bytes, @intCast(db.name.len), .little);
    try writeBytes(write_buffer, &buffer_pos, &name_len_bytes, file, io);
    try writeBytes(write_buffer, &buffer_pos, db.name, file, io);

    var record_count_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &record_count_bytes, @intCast(db.records.items.len), .little);
    try writeBytes(write_buffer, &buffer_pos, &record_count_bytes, file, io);

    // Write records
    for (db.records.items) |record| {
        if (record.vector.len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;
        const meta_len: usize = if (record.metadata) |meta| meta.len else 0;
        if (meta_len > std.math.maxInt(u32)) return StorageError.PayloadTooLarge;

        // Record header: id (8) + vec_len (4) + meta_len (4)
        var header: [16]u8 = undefined;
        std.mem.writeInt(u64, header[0..8], record.id, .little);
        std.mem.writeInt(u32, header[8..12], @intCast(record.vector.len), .little);
        std.mem.writeInt(u32, header[12..16], @intCast(meta_len), .little);
        try writeBytes(write_buffer, &buffer_pos, &header, file, io);

        // Write vector data directly as bytes (f32 array to byte slice)
        const vector_bytes = std.mem.sliceAsBytes(record.vector);
        try writeBytes(write_buffer, &buffer_pos, vector_bytes, file, io);

        // Write metadata if present
        if (record.metadata) |meta| {
            try writeBytes(write_buffer, &buffer_pos, meta, file, io);
        }
    }

    // Final flush
    try flush(write_buffer, &buffer_pos, file, io);
}

/// Buffered save - builds entire database in memory (faster for small DBs)
fn saveDatabaseBuffered(
    allocator: std.mem.Allocator,
    db: *database.Database,
    file: *std.Io.File,
    io: anytype,
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

        // Write vector bytes directly instead of iterating
        const vector_bytes = std.mem.sliceAsBytes(record.vector);
        try buffer.appendSlice(allocator, vector_bytes);

        if (record.metadata) |meta| {
            try buffer.appendSlice(allocator, meta);
        }
    }

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
        // Read vector bytes directly (optimized bulk read)
        const vector_bytes = try cursor.readBytes(vec_len * @sizeOf(f32));
        @memcpy(std.mem.sliceAsBytes(vector), vector_bytes);
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
            database.DatabaseError.PoolExhausted,
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

test "storage streaming vs buffered save" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.testing.allocator;

    var db = try database.Database.init(allocator, "stream-test");
    defer db.deinit();

    // Add multiple records
    try db.insert(1, &.{ 0.1, 0.2, 0.3, 0.4 }, "first");
    try db.insert(2, &.{ 0.5, 0.6, 0.7, 0.8 }, "second");
    try db.insert(3, &.{ 0.9, 1.0, 1.1, 1.2 }, null);

    // Test streaming mode (small buffer to exercise buffer flushing)
    const stream_path = try tmp.dir.realpathAlloc(allocator, "stream.bin");
    defer allocator.free(stream_path);
    try saveDatabaseWithConfig(allocator, &db, stream_path, .{
        .use_streaming = true,
        .write_buffer_size = 64, // Small buffer to test flushing
    });

    var restored_stream = try loadDatabase(allocator, stream_path);
    defer restored_stream.deinit();
    try std.testing.expectEqual(@as(usize, 3), restored_stream.records.items.len);

    // Test buffered mode
    const buffer_path = try tmp.dir.realpathAlloc(allocator, "buffer.bin");
    defer allocator.free(buffer_path);
    try saveDatabaseWithConfig(allocator, &db, buffer_path, .{
        .use_streaming = false,
    });

    var restored_buffer = try loadDatabase(allocator, buffer_path);
    defer restored_buffer.deinit();
    try std.testing.expectEqual(@as(usize, 3), restored_buffer.records.items.len);

    // Verify data integrity
    const view1 = restored_stream.get(1) orelse return error.TestUnexpectedResult;
    const view2 = restored_buffer.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(f32, view1.vector, view2.vector);
}
