//! Memory-Mapped File Support for Unified Format
//!
//! Provides zero-copy file access using memory mapping.
//! Supports both read-only and read-write modes.

const std = @import("std");
const unified = @import("unified.zig");

pub const MmapError = error{
    FileNotFound,
    AccessDenied,
    MmapFailed,
    InvalidFile,
    OutOfMemory,
};

/// Memory-mapped file handle
pub const MappedFile = struct {
    io_backend: std.Io.Threaded,
    mmap: std.Io.File.MemoryMap,
    size: usize,
    read_only: bool,

    /// Open file with memory mapping (read-only)
    pub fn open(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        return openInternal(allocator, path, true);
    }

    /// Open file with memory mapping (read-write)
    pub fn openReadWrite(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        return openInternal(allocator, path, false);
    }

    fn openInternal(allocator: std.mem.Allocator, path: []const u8, read_only: bool) MmapError!MappedFile {
        if (path.len >= std.fs.max_path_bytes) return error.InvalidFile;

        var io_backend = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        errdefer io_backend.deinit();
        const io = io_backend.io();

        var file = std.Io.Dir.cwd().openFile(io, path, .{
            .mode = if (read_only) .read_only else .read_write,
            .allow_directory = false,
        }) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
            error.IsDir, error.NotDir => return error.InvalidFile,
            else => return error.MmapFailed,
        };
        errdefer file.close(io);

        const stat = file.stat(io) catch |err| switch (err) {
            error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
            else => return error.InvalidFile,
        };
        if (stat.kind != .file) return error.InvalidFile;

        const size = std.math.cast(usize, stat.size) orelse return error.InvalidFile;
        if (size == 0) return error.InvalidFile;

        const mmap = file.createMemoryMap(io, .{
            .len = size,
            .protection = if (read_only) .{ .read = true } else .{ .read = true, .write = true },
        }) catch |err| switch (err) {
            error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.MmapFailed,
        };

        return .{
            .io_backend = io_backend,
            .mmap = mmap,
            .size = size,
            .read_only = read_only,
        };
    }

    /// Close memory mapping
    pub fn close(self: *MappedFile) void {
        self.sync();
        const io = self.io_backend.io();
        const file = self.mmap.file;
        file.close(io);
        self.mmap.destroy(io);
        self.io_backend.deinit();
        self.* = undefined;
    }

    /// Get data as bytes
    pub fn bytes(self: *const MappedFile) []const u8 {
        return self.mmap.memory[0..self.size];
    }

    /// Get mutable data (fails if read-only)
    pub fn bytesMut(self: *MappedFile) MmapError![]u8 {
        if (self.read_only) return error.AccessDenied;
        return self.mmap.memory[0..self.size];
    }

    /// Sync changes to disk
    pub fn sync(self: *MappedFile) void {
        if (self.read_only) return;
        const io = self.io_backend.io();
        self.mmap.write(io) catch |err| {
            std.log.warn("Memory-mapped file sync failed: {t}", .{err});
        };
    }

    /// Load as unified format (zero-copy)
    pub fn asUnifiedFormat(self: *const MappedFile, allocator: std.mem.Allocator) unified.UnifiedError!unified.UnifiedFormat {
        return unified.UnifiedFormat.fromMemory(allocator, self.bytes());
    }
};

/// Create a memory-mapped file of given size
pub fn createMapped(allocator: std.mem.Allocator, path: []const u8, size: usize) MmapError!MappedFile {
    if (size == 0) return error.InvalidFile;
    if (path.len >= std.fs.max_path_bytes) return error.InvalidFile;

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    errdefer io_backend.deinit();
    const io = io_backend.io();

    var file = std.Io.Dir.cwd().createFile(io, path, .{
        .truncate = true,
        .read = true,
    }) catch |err| switch (err) {
        error.FileNotFound => return error.FileNotFound,
        error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
        error.IsDir, error.NotDir => return error.InvalidFile,
        else => return error.MmapFailed,
    };
    errdefer file.close(io);

    file.setLength(io, @intCast(size)) catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
        else => return error.MmapFailed,
    };

    const mmap = file.createMemoryMap(io, .{
        .len = size,
        .protection = .{ .read = true, .write = true },
    }) catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.MmapFailed,
    };

    return .{
        .io_backend = io_backend,
        .mmap = mmap,
        .size = size,
        .read_only = false,
    };
}

/// Memory cursor for parsing mapped data
pub const MemoryCursor = struct {
    data: []const u8,
    position: usize,

    pub fn init(data: []const u8) MemoryCursor {
        return .{ .data = data, .position = 0 };
    }

    pub fn fromMapped(mapped: *const MappedFile) MemoryCursor {
        return init(mapped.bytes());
    }

    pub fn read(self: *MemoryCursor, comptime T: type) ?T {
        if (self.position + @sizeOf(T) > self.data.len) return null;
        const bytes = self.data[self.position..][0..@sizeOf(T)];
        var result: T = undefined;
        @memcpy(std.mem.asBytes(&result), bytes);
        self.position += @sizeOf(T);
        return result;
    }

    pub fn readBytes(self: *MemoryCursor, len: usize) ?[]const u8 {
        if (self.position + len > self.data.len) return null;
        const result = self.data[self.position..][0..len];
        self.position += len;
        return result;
    }

    pub fn readString(self: *MemoryCursor) ?[]const u8 {
        const len = self.read(u64) orelse return null;
        return self.readBytes(@intCast(len));
    }

    pub fn alignTo(self: *MemoryCursor, alignment: usize) void {
        self.position = (self.position + alignment - 1) & ~(alignment - 1);
    }

    pub fn remaining(self: *const MemoryCursor) usize {
        return self.data.len - self.position;
    }

    pub fn skip(self: *MemoryCursor, len: usize) bool {
        if (self.position + len > self.data.len) return false;
        self.position += len;
        return true;
    }
};

test "memory cursor basic" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    var cursor = MemoryCursor.init(&data);

    const val1 = cursor.read(u32).?;
    try std.testing.expectEqual(@as(u32, 0x04030201), val1);

    const val2 = cursor.read(u16).?;
    try std.testing.expectEqual(@as(u16, 0x0605), val2);

    try std.testing.expectEqual(@as(usize, 2), cursor.remaining());
}
