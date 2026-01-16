//! Memory-Mapped File Support for Unified Format
//!
//! Provides zero-copy file access using memory mapping.
//! Supports both read-only and read-write modes.

const std = @import("std");
const builtin = @import("builtin");
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
    data: []align(std.mem.page_size) u8,
    size: usize,
    read_only: bool,

    /// Open file with memory mapping (read-only)
    pub fn open(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        _ = allocator;
        return openInternal(path, true);
    }

    /// Open file with memory mapping (read-write)
    pub fn openReadWrite(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        _ = allocator;
        return openInternal(path, false);
    }

    fn openInternal(path: []const u8, read_only: bool) MmapError!MappedFile {
        if (builtin.os.tag == .windows) {
            return openWindows(path, read_only);
        } else {
            return openPosix(path, read_only);
        }
    }

    fn openWindows(path: []const u8, read_only: bool) MmapError!MappedFile {
        // Windows implementation using kernel32
        const windows = std.os.windows;

        // Convert path to null-terminated
        var path_buf: [std.fs.max_path_bytes:0]u8 = undefined;
        if (path.len >= path_buf.len) return error.InvalidFile;
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;

        // Open file
        const access = if (read_only)
            windows.GENERIC_READ
        else
            windows.GENERIC_READ | windows.GENERIC_WRITE;

        const share = windows.FILE_SHARE_READ;
        const creation = windows.OPEN_EXISTING;

        const file_handle = windows.kernel32.CreateFileA(
            @ptrCast(&path_buf),
            access,
            share,
            null,
            creation,
            windows.FILE_ATTRIBUTE_NORMAL,
            null,
        );

        if (file_handle == windows.INVALID_HANDLE_VALUE) {
            return error.FileNotFound;
        }
        errdefer _ = windows.kernel32.CloseHandle(file_handle);

        // Get file size
        var file_size: windows.LARGE_INTEGER = undefined;
        if (windows.kernel32.GetFileSizeEx(file_handle, &file_size) == 0) {
            return error.InvalidFile;
        }
        const size: usize = @intCast(file_size);

        if (size == 0) {
            _ = windows.kernel32.CloseHandle(file_handle);
            return error.InvalidFile;
        }

        // Create file mapping
        const protect: windows.DWORD = if (read_only) windows.PAGE_READONLY else windows.PAGE_READWRITE;
        const mapping_handle = windows.kernel32.CreateFileMappingA(
            file_handle,
            null,
            protect,
            0,
            0,
            null,
        );

        if (mapping_handle == null) {
            return error.MmapFailed;
        }
        errdefer _ = windows.kernel32.CloseHandle(mapping_handle);

        // Map view
        const map_access: windows.DWORD = if (read_only) windows.FILE_MAP_READ else windows.FILE_MAP_ALL_ACCESS;
        const ptr = windows.kernel32.MapViewOfFile(
            mapping_handle,
            map_access,
            0,
            0,
            0,
        );

        if (ptr == null) {
            return error.MmapFailed;
        }

        // Close handles (mapping stays valid until UnmapViewOfFile)
        _ = windows.kernel32.CloseHandle(mapping_handle);
        _ = windows.kernel32.CloseHandle(file_handle);

        const data: []align(std.mem.page_size) u8 = @alignCast(@as([*]u8, @ptrCast(ptr))[0..size]);

        return .{
            .data = data,
            .size = size,
            .read_only = read_only,
        };
    }

    fn openPosix(path: []const u8, read_only: bool) MmapError!MappedFile {
        // POSIX implementation
        const flags: std.posix.O = if (read_only) .{ .ACCMODE = .RDONLY } else .{ .ACCMODE = .RDWR };
        const fd = std.posix.open(path, flags, 0) catch return error.FileNotFound;
        errdefer std.posix.close(fd);

        const stat = std.posix.fstat(fd) catch return error.InvalidFile;
        const size: usize = @intCast(stat.size);

        if (size == 0) {
            std.posix.close(fd);
            return error.InvalidFile;
        }

        const prot: u32 = if (read_only) std.posix.PROT.READ else std.posix.PROT.READ | std.posix.PROT.WRITE;

        const ptr = std.posix.mmap(null, size, prot, .{ .TYPE = .SHARED }, fd, 0) catch return error.MmapFailed;

        std.posix.close(fd);

        return .{
            .data = ptr,
            .size = size,
            .read_only = read_only,
        };
    }

    /// Close memory mapping
    pub fn close(self: *MappedFile) void {
        if (builtin.os.tag == .windows) {
            _ = std.os.windows.kernel32.UnmapViewOfFile(@ptrCast(self.data.ptr));
        } else {
            std.posix.munmap(self.data);
        }
        self.* = undefined;
    }

    /// Get data as bytes
    pub fn bytes(self: *const MappedFile) []const u8 {
        return self.data[0..self.size];
    }

    /// Get mutable data (fails if read-only)
    pub fn bytesMut(self: *MappedFile) MmapError![]u8 {
        if (self.read_only) return error.AccessDenied;
        return self.data[0..self.size];
    }

    /// Sync changes to disk
    pub fn sync(self: *MappedFile) void {
        if (self.read_only) return;

        if (builtin.os.tag == .windows) {
            _ = std.os.windows.kernel32.FlushViewOfFile(@ptrCast(self.data.ptr), self.size);
        } else {
            std.posix.msync(self.data, .{ .SYNC = true }) catch {};
        }
    }

    /// Load as unified format (zero-copy)
    pub fn asUnifiedFormat(self: *const MappedFile, allocator: std.mem.Allocator) unified.UnifiedError!unified.UnifiedFormat {
        return unified.UnifiedFormat.fromMemory(allocator, self.bytes());
    }
};

/// Create a memory-mapped file of given size
pub fn createMapped(allocator: std.mem.Allocator, path: []const u8, size: usize) MmapError!MappedFile {
    _ = allocator;

    if (builtin.os.tag == .windows) {
        return createWindows(path, size);
    } else {
        return createPosix(path, size);
    }
}

fn createWindows(path: []const u8, size: usize) MmapError!MappedFile {
    const windows = std.os.windows;

    var path_buf: [std.fs.max_path_bytes:0]u8 = undefined;
    if (path.len >= path_buf.len) return error.InvalidFile;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;

    // Create file
    const file_handle = windows.kernel32.CreateFileA(
        @ptrCast(&path_buf),
        windows.GENERIC_READ | windows.GENERIC_WRITE,
        0,
        null,
        windows.CREATE_ALWAYS,
        windows.FILE_ATTRIBUTE_NORMAL,
        null,
    );

    if (file_handle == windows.INVALID_HANDLE_VALUE) {
        return error.AccessDenied;
    }
    errdefer _ = windows.kernel32.CloseHandle(file_handle);

    // Set file size
    const distance: windows.LARGE_INTEGER = @intCast(size);
    if (windows.kernel32.SetFilePointerEx(file_handle, distance, null, windows.FILE_BEGIN) == 0) {
        return error.MmapFailed;
    }
    if (windows.kernel32.SetEndOfFile(file_handle) == 0) {
        return error.MmapFailed;
    }

    // Create mapping
    const mapping_handle = windows.kernel32.CreateFileMappingA(
        file_handle,
        null,
        windows.PAGE_READWRITE,
        0,
        0,
        null,
    );

    if (mapping_handle == null) {
        return error.MmapFailed;
    }
    errdefer _ = windows.kernel32.CloseHandle(mapping_handle);

    const ptr = windows.kernel32.MapViewOfFile(
        mapping_handle,
        windows.FILE_MAP_ALL_ACCESS,
        0,
        0,
        0,
    );

    if (ptr == null) {
        return error.MmapFailed;
    }

    _ = windows.kernel32.CloseHandle(mapping_handle);
    _ = windows.kernel32.CloseHandle(file_handle);

    const data: []align(std.mem.page_size) u8 = @alignCast(@as([*]u8, @ptrCast(ptr))[0..size]);

    return .{
        .data = data,
        .size = size,
        .read_only = false,
    };
}

fn createPosix(path: []const u8, size: usize) MmapError!MappedFile {
    const fd = std.posix.open(path, .{
        .ACCMODE = .RDWR,
        .CREAT = true,
        .TRUNC = true,
    }, 0o644) catch return error.AccessDenied;
    errdefer std.posix.close(fd);

    std.posix.ftruncate(fd, @intCast(size)) catch return error.MmapFailed;

    const ptr = std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .SHARED },
        fd,
        0,
    ) catch return error.MmapFailed;

    std.posix.close(fd);

    return .{
        .data = ptr,
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
        const result: *const T = @ptrCast(@alignCast(self.data.ptr + self.position));
        self.position += @sizeOf(T);
        return result.*;
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
