//! Memory-mapped file abstraction for efficient model loading.
//!
//! Provides cross-platform memory mapping using OS-native APIs:
//! - Linux/macOS: mmap via std.posix
//! - Windows: Falls back to reading file into memory
//!
//! Memory mapping allows loading multi-gigabyte model files without
//! copying into heap memory, enabling efficient random access.

const std = @import("std");
const builtin = @import("builtin");

// libc imports for Zig 0.16 compatibility
const c = @cImport({
    @cInclude("sys/stat.h");
    @cInclude("fcntl.h");
    @cInclude("unistd.h");
    @cInclude("sys/mman.h");
});

/// Page size constant (4KB on most platforms)
const page_size = 4096;

/// Windows kernel32 file I/O functions
const WindowsFile = struct {
    const HANDLE = std.os.windows.HANDLE;
    const INVALID_HANDLE_VALUE = std.os.windows.INVALID_HANDLE_VALUE;
    const GENERIC_READ = 0x80000000;
    const FILE_SHARE_READ = 0x00000001;
    const OPEN_EXISTING = 3;
    const FILE_ATTRIBUTE_NORMAL = 0x00000080;
    const DWORD = std.os.windows.DWORD;

    extern "kernel32" fn CreateFileA(
        lpFileName: [*:0]const u8,
        dwDesiredAccess: DWORD,
        dwShareMode: DWORD,
        lpSecurityAttributes: ?*anyopaque,
        dwCreationDisposition: DWORD,
        dwFlagsAndAttributes: DWORD,
        hTemplateFile: ?HANDLE,
    ) callconv(.winapi) HANDLE;

    extern "kernel32" fn GetFileSizeEx(
        hFile: HANDLE,
        lpFileSize: *i64,
    ) callconv(.winapi) std.os.windows.BOOL;

    extern "kernel32" fn ReadFile(
        hFile: HANDLE,
        lpBuffer: [*]u8,
        nNumberOfBytesToRead: DWORD,
        lpNumberOfBytesRead: ?*DWORD,
        lpOverlapped: ?*anyopaque,
    ) callconv(.winapi) std.os.windows.BOOL;

    extern "kernel32" fn CloseHandle(hObject: HANDLE) callconv(.winapi) std.os.windows.BOOL;
};

pub const MmapError = error{
    FileNotFound,
    AccessDenied,
    InvalidHandle,
    InvalidPath,
    MappingFailed,
    UnmapFailed,
    FileTooLarge,
    IoError,
    OutOfMemory,
};

/// Memory-mapped file providing zero-copy access to file contents.
pub const MappedFile = struct {
    /// Pointer to mapped memory region
    data: []align(page_size) u8,
    /// File size in bytes
    size: usize,
    /// Platform-specific handle for cleanup
    handle: Handle,

    const Handle = switch (builtin.os.tag) {
        .windows => struct {
            allocator: std.mem.Allocator,
        },
        else => struct {
            fd: std.posix.fd_t,
        },
    };

    /// Open and memory-map a file.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        return openAdvanced(allocator, path, .{});
    }

    /// Options for opening a mapped file.
    pub const OpenOptions = struct {
        /// Request huge pages (2MB) for better TLB performance
        huge_pages: bool = false,
        /// Hint that file will be accessed sequentially
        sequential: bool = false,
        /// Hint that file will be accessed randomly
        random_access: bool = false,
        /// Pre-populate page tables (may increase open time)
        populate: bool = false,
        /// Hint that the data will be needed soon (triggers readahead)
        will_need: bool = false,
    };

    /// Open with advanced options.
    pub fn openAdvanced(allocator: std.mem.Allocator, path: []const u8, options: OpenOptions) MmapError!MappedFile {
        // Platform-specific file opening and mapping
        var mapped = switch (builtin.os.tag) {
            .windows => try openWindows(allocator, path),
            else => try openPosix(path, options),
        };

        // Apply madvise hints based on options
        if (builtin.os.tag == .linux) {
            if (options.sequential) {
                mapped.advise(.sequential);
            } else if (options.random_access) {
                mapped.advise(.random);
            }

            if (options.will_need) {
                mapped.advise(.will_need);
            }
        }

        return mapped;
    }

    fn openPosix(path: []const u8, options: OpenOptions) MmapError!MappedFile {
        // Zig 0.16 with libc: Use libc functions directly for reliability
        _ = options; // Options like populate/huge_pages require more complex handling

        // Create null-terminated path
        var path_buf: [std.fs.max_path_bytes:0]u8 = undefined;
        if (path.len >= path_buf.len) return MmapError.InvalidPath;
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;

        // Open file using libc
        const fd = c.open(&path_buf, c.O_RDONLY);
        if (fd < 0) {
            return MmapError.FileNotFound;
        }
        errdefer _ = c.close(fd);

        // Get file size via fstat
        var stat_buf: c.struct_stat = undefined;
        if (c.fstat(fd, &stat_buf) != 0) {
            return MmapError.IoError;
        }
        const size: u64 = @intCast(stat_buf.st_size);

        if (size == 0) {
            return MmapError.FileTooLarge; // Empty files can't be mapped
        }

        // Memory map the file using libc mmap
        const ptr = c.mmap(
            null,
            size,
            c.PROT_READ,
            c.MAP_PRIVATE,
            fd,
            0,
        );
        if (ptr == c.MAP_FAILED) {
            return MmapError.MappingFailed;
        }

        // mmap returns page-aligned memory, cast with proper alignment
        const aligned_ptr: [*]align(page_size) u8 = @alignCast(@ptrCast(ptr));
        return MappedFile{
            .data = aligned_ptr[0..size],
            .size = size,
            .handle = .{ .fd = fd },
        };
    }

    fn openWindows(allocator: std.mem.Allocator, path: []const u8) MmapError!MappedFile {
        // On Windows, we read the file into page-aligned memory.
        // This provides the same API as mmap but without zero-copy benefits.
        // Uses kernel32 APIs directly for Zig 0.16 compatibility.

        // Create null-terminated path for Windows API
        var path_buf: [std.fs.max_path_bytes:0]u8 = undefined;
        if (path.len >= path_buf.len) {
            return MmapError.IoError;
        }
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;

        // Open file using kernel32
        const handle = WindowsFile.CreateFileA(
            path_buf[0..path.len :0],
            WindowsFile.GENERIC_READ,
            WindowsFile.FILE_SHARE_READ,
            null,
            WindowsFile.OPEN_EXISTING,
            WindowsFile.FILE_ATTRIBUTE_NORMAL,
            null,
        );

        if (handle == WindowsFile.INVALID_HANDLE_VALUE) {
            // Check GetLastError for specific error
            const err = std.os.windows.GetLastError();
            return switch (err) {
                .FILE_NOT_FOUND, .PATH_NOT_FOUND => MmapError.FileNotFound,
                .ACCESS_DENIED => MmapError.AccessDenied,
                else => MmapError.IoError,
            };
        }
        defer _ = WindowsFile.CloseHandle(handle);

        // Get file size
        var file_size_large: i64 = 0;
        if (WindowsFile.GetFileSizeEx(handle, &file_size_large) == 0) {
            return MmapError.IoError;
        }

        if (file_size_large <= 0) {
            return MmapError.FileTooLarge; // Empty files can't be mapped
        }

        // Check for files too large (> 4GB on 32-bit, or exceeding available memory)
        if (file_size_large > std.math.maxInt(usize)) {
            return MmapError.FileTooLarge;
        }

        const file_size: usize = @intCast(file_size_large);

        // Allocate page-aligned memory (4KB alignment = log2(4096) = 12)
        const data = allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(page_size), file_size) catch {
            return MmapError.OutOfMemory;
        };
        errdefer allocator.free(data);

        // Read entire file into memory in chunks (ReadFile has DWORD size limit)
        var total_read: usize = 0;
        while (total_read < file_size) {
            const remaining = file_size - total_read;
            const chunk_size: WindowsFile.DWORD = @intCast(@min(remaining, 0x7FFFFFFF));
            var bytes_read: WindowsFile.DWORD = 0;

            if (WindowsFile.ReadFile(
                handle,
                data.ptr + total_read,
                chunk_size,
                &bytes_read,
                null,
            ) == 0) {
                return MmapError.IoError;
            }

            if (bytes_read == 0) {
                return MmapError.IoError; // Unexpected EOF
            }

            total_read += bytes_read;
        }

        return MappedFile{
            .data = data,
            .size = file_size,
            .handle = .{ .allocator = allocator },
        };
    }

    /// Close the mapped file and release resources.
    pub fn close(self: *MappedFile) void {
        switch (builtin.os.tag) {
            .windows => {
                self.handle.allocator.free(self.data);
            },
            else => {
                // Use libc munmap and close for Zig 0.16 compatibility
                _ = c.munmap(@ptrCast(self.data.ptr), self.data.len);
                _ = c.close(self.handle.fd);
            },
        }
        self.* = undefined;
    }

    /// Get a slice view into the mapped region.
    pub fn slice(self: *const MappedFile, offset: usize, length: usize) ?[]const u8 {
        if (offset + length > self.size) return null;
        return self.data[offset .. offset + length];
    }

    /// Get a typed view at an offset.
    pub fn viewAs(self: *const MappedFile, comptime T: type, offset: usize) ?*const T {
        if (offset + @sizeOf(T) > self.size) return null;
        const ptr: *const T = @ptrCast(@alignCast(&self.data[offset]));
        return ptr;
    }

    /// Get a slice of typed values at an offset.
    pub fn viewSliceAs(self: *const MappedFile, comptime T: type, offset: usize, count: usize) ?[]const T {
        const byte_len = count * @sizeOf(T);
        if (offset + byte_len > self.size) return null;
        const ptr: [*]const T = @ptrCast(@alignCast(&self.data[offset]));
        return ptr[0..count];
    }

    /// Read bytes into a buffer.
    pub fn readBytes(self: *const MappedFile, offset: usize, buffer: []u8) ?usize {
        const available = if (offset >= self.size) 0 else self.size - offset;
        const to_read = @min(buffer.len, available);
        if (to_read == 0) return null;
        @memcpy(buffer[0..to_read], self.data[offset .. offset + to_read]);
        return to_read;
    }

    /// Advise the kernel about access patterns (hint only).
    pub fn advise(self: *MappedFile, advice: Advice) void {
        switch (builtin.os.tag) {
            .linux => {
                // Use libc madvise for Zig 0.16 compatibility
                const posix_advice: c_int = switch (advice) {
                    .normal => c.MADV_NORMAL,
                    .sequential => c.MADV_SEQUENTIAL,
                    .random => c.MADV_RANDOM,
                    .will_need => c.MADV_WILLNEED,
                    .dont_need => c.MADV_DONTNEED,
                };
                _ = c.madvise(@ptrCast(self.data.ptr), self.data.len, posix_advice);
            },
            else => {}, // Hints not available on all platforms
        }
    }

    pub const Advice = enum {
        normal,
        sequential,
        random,
        will_need,
        dont_need,
    };

    /// Get total mapped size.
    pub fn len(self: *const MappedFile) usize {
        return self.size;
    }

    /// Check if the mapping is valid.
    pub fn isValid(self: *const MappedFile) bool {
        return self.data.len > 0 and self.size > 0;
    }
};

/// Memory cursor for sequential reading from mapped memory.
pub const MemoryCursor = struct {
    data: []const u8,
    position: usize,

    pub fn init(data: []const u8) MemoryCursor {
        return .{ .data = data, .position = 0 };
    }

    pub fn fromMapped(file: *const MappedFile) MemoryCursor {
        return init(file.data);
    }

    pub fn read(self: *MemoryCursor, comptime T: type) ?T {
        if (self.position + @sizeOf(T) > self.data.len) return null;
        const ptr: *const T = @ptrCast(@alignCast(&self.data[self.position]));
        self.position += @sizeOf(T);
        return ptr.*;
    }

    pub fn readBytes(self: *MemoryCursor, count: usize) ?[]const u8 {
        if (self.position + count > self.data.len) return null;
        const result = self.data[self.position .. self.position + count];
        self.position += count;
        return result;
    }

    pub fn readString(self: *MemoryCursor) ?[]const u8 {
        const str_len = self.read(u64) orelse return null;
        return self.readBytes(@intCast(str_len));
    }

    pub fn skip(self: *MemoryCursor, count: usize) bool {
        if (self.position + count > self.data.len) return false;
        self.position += count;
        return true;
    }

    pub fn seek(self: *MemoryCursor, pos: usize) bool {
        if (pos > self.data.len) return false;
        self.position = pos;
        return true;
    }

    pub fn remaining(self: *const MemoryCursor) usize {
        return self.data.len - self.position;
    }

    pub fn isEof(self: *const MemoryCursor) bool {
        return self.position >= self.data.len;
    }

    /// Align position to boundary.
    pub fn alignTo(self: *MemoryCursor, alignment: usize) void {
        const aligned = (self.position + alignment - 1) & ~(alignment - 1);
        if (aligned <= self.data.len) {
            self.position = aligned;
        }
    }
};

test "memory cursor basic operations" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    var cursor = MemoryCursor.init(&data);

    try std.testing.expectEqual(@as(u8, 0x01), cursor.read(u8).?);
    try std.testing.expectEqual(@as(u8, 0x02), cursor.read(u8).?);
    try std.testing.expectEqual(@as(usize, 6), cursor.remaining());

    const bytes = cursor.readBytes(4).?;
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x03, 0x04, 0x05, 0x06 }, bytes);
}

test "memory cursor alignment" {
    const data = [_]u8{0} ** 16;
    var cursor = MemoryCursor.init(&data);

    _ = cursor.read(u8);
    try std.testing.expectEqual(@as(usize, 1), cursor.position);

    cursor.alignTo(4);
    try std.testing.expectEqual(@as(usize, 4), cursor.position);

    cursor.alignTo(8);
    try std.testing.expectEqual(@as(usize, 8), cursor.position);
}
