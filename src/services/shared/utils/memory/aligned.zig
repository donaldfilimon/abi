//! Aligned allocator utilities for SIMD and cache-line alignment.
//!
//! Provides wrappers for allocating memory with specific alignment:
//! - SIMD alignment (16, 32, 64 bytes for SSE/AVX/AVX-512)
//! - Cache-line alignment (typically 64 bytes)
//! - Page alignment (typically 4096 bytes)
//!
//! Usage:
//! ```zig
//! // Allocate SIMD-aligned memory
//! const simd_data = try aligned.simdAlloc(f32, allocator, 1024);
//! defer allocator.free(simd_data);
//!
//! // Use the aligned allocator wrapper
//! var aligned_alloc = AlignedAllocator.init(allocator, .cache_line);
//! const data = try aligned_alloc.allocator().alloc(u8, 256);
//! ```

const std = @import("std");

/// Common alignment values (power of 2 exponent).
pub const Alignment = enum(u6) {
    /// Standard alignment (8 bytes on 64-bit).
    default = 3, // 2^3 = 8
    /// SSE alignment (16 bytes).
    simd_16 = 4, // 2^4 = 16
    /// AVX alignment (32 bytes).
    simd_32 = 5, // 2^5 = 32
    /// AVX-512/cache line alignment (64 bytes).
    cache_line = 6, // 2^6 = 64 (also used for SIMD-64)
    /// Page alignment (4096 bytes).
    page = 12, // 2^12 = 4096

    /// Convert to std.mem.Alignment.
    pub fn toMemAlignment(self: Alignment) std.mem.Alignment {
        return switch (self) {
            .default => .@"8",
            .simd_16 => .@"16",
            .simd_32 => .@"32",
            .cache_line => .@"64",
            // Page alignment (4096 = 2^12) - use fromByteUnits for safety
            .page => std.mem.Alignment.fromByteUnits(4096),
        };
    }

    /// Get byte units.
    pub fn bytes(self: Alignment) usize {
        return @as(usize, 1) << @intFromEnum(self);
    }
};

/// Allocator wrapper that enforces specific alignment.
pub const AlignedAllocator = struct {
    backing: std.mem.Allocator,
    alignment: Alignment,

    const Self = @This();

    /// Initialize an aligned allocator.
    pub fn init(backing: std.mem.Allocator, alignment: Alignment) Self {
        return .{
            .backing = backing,
            .alignment = alignment,
        };
    }

    /// Get an allocator interface.
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const final_align = @max(ptr_align, @intFromEnum(self.alignment));
        return self.backing.rawAlloc(len, final_align, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, ptr_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const final_align = @max(ptr_align, @intFromEnum(self.alignment));
        return self.backing.rawResize(memory, final_align, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, ptr_align: u8, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const final_align = @max(ptr_align, @intFromEnum(self.alignment));
        self.backing.rawFree(memory, final_align, ret_addr);
    }
};

/// Allocate SIMD-aligned array (16 bytes).
pub fn simdAlloc(comptime T: type, alloc: std.mem.Allocator, count: usize) ![]align(16) T {
    return alloc.alignedAlloc(T, .@"16", count);
}

/// Allocate AVX-aligned array (32 bytes).
pub fn avxAlloc(comptime T: type, alloc: std.mem.Allocator, count: usize) ![]align(32) T {
    return alloc.alignedAlloc(T, .@"32", count);
}

/// Allocate AVX-512 aligned array (64 bytes).
pub fn avx512Alloc(comptime T: type, alloc: std.mem.Allocator, count: usize) ![]align(64) T {
    return alloc.alignedAlloc(T, .@"64", count);
}

/// Allocate cache-line aligned array (64 bytes).
pub fn cacheLineAlloc(comptime T: type, alloc: std.mem.Allocator, count: usize) ![]align(64) T {
    return alloc.alignedAlloc(T, .@"64", count);
}

/// Allocate page-aligned array.
pub fn pageAlloc(comptime T: type, alloc: std.mem.Allocator, count: usize) ![]align(4096) T {
    return alloc.alignedAlloc(T, .@"4096", count);
}

/// Check if pointer is aligned to given alignment.
pub fn isAligned(ptr: anytype, comptime alignment: Alignment) bool {
    const addr = @intFromPtr(ptr);
    return addr & (@as(usize, 1) << @intFromEnum(alignment) - 1) == 0;
}

/// Align a size up to the given alignment.
pub fn alignUp(size: usize, comptime alignment: Alignment) usize {
    const align_val = @as(usize, 1) << @intFromEnum(alignment);
    return (size + align_val - 1) & ~(align_val - 1);
}

/// Align a size down to the given alignment.
pub fn alignDown(size: usize, comptime alignment: Alignment) usize {
    const align_val = @as(usize, 1) << @intFromEnum(alignment);
    return size & ~(align_val - 1);
}

/// Padded struct for cache-line isolation.
pub fn CacheLinePadded(comptime T: type) type {
    return struct {
        value: T,
        _padding: [64 - @sizeOf(T) % 64]u8 = undefined,

        pub fn init(value: T) @This() {
            return .{ .value = value };
        }

        pub fn get(self: *@This()) *T {
            return &self.value;
        }

        pub fn getConst(self: *const @This()) *const T {
            return &self.value;
        }
    };
}

/// Aligned buffer for SIMD operations.
pub fn AlignedBuffer(comptime T: type, comptime alignment: comptime_int) type {
    return struct {
        data: []align(alignment) T,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator, size: usize) !Self {
            // Convert alignment byte value to log2 for std.mem.Alignment
            const data = try alloc.alignedAlloc(T, comptime std.mem.Alignment.fromByteUnits(alignment), size);
            return .{
                .data = data,
                .allocator = alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        pub fn slice(self: *Self) []align(alignment) T {
            return self.data;
        }

        pub fn constSlice(self: *const Self) []align(alignment) const T {
            return self.data;
        }

        pub fn len(self: *const Self) usize {
            return self.data.len;
        }

        pub fn ptr(self: *Self) [*]align(alignment) T {
            return self.data.ptr;
        }
    };
}

/// SIMD-optimized buffer (16-byte aligned).
pub const SimdBuffer = AlignedBuffer(f32, 16);

/// AVX-optimized buffer (32-byte aligned).
pub const AvxBuffer = AlignedBuffer(f32, 32);

test "aligned allocator" {
    const allocator = std.testing.allocator;
    var aligned = AlignedAllocator.init(allocator, .cache_line);
    const alloc = aligned.allocator();

    const data = try alloc.alloc(u8, 256);
    defer alloc.free(data);

    // Check alignment
    const addr = @intFromPtr(data.ptr);
    try std.testing.expect(addr % 64 == 0);
}

test "simd alloc" {
    const allocator = std.testing.allocator;

    const simd = try simdAlloc(f32, allocator, 64);
    defer allocator.free(simd);

    const addr = @intFromPtr(simd.ptr);
    try std.testing.expect(addr % 16 == 0);
}

test "cache line alloc" {
    const allocator = std.testing.allocator;

    const cl = try cacheLineAlloc(u64, allocator, 32);
    defer allocator.free(cl);

    const addr = @intFromPtr(cl.ptr);
    try std.testing.expect(addr % 64 == 0);
}

test "aligned buffer" {
    const allocator = std.testing.allocator;

    var buf = try SimdBuffer.init(allocator, 128);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 128), buf.len());

    const addr = @intFromPtr(buf.ptr());
    try std.testing.expect(addr % 16 == 0);
}

test "cache line padded" {
    const Padded = CacheLinePadded(u32);
    var padded = Padded.init(42);

    try std.testing.expectEqual(@as(u32, 42), padded.get().*);
    try std.testing.expect(@sizeOf(Padded) >= 64);
}

test "align up/down" {
    try std.testing.expectEqual(@as(usize, 64), alignUp(33, .cache_line));
    try std.testing.expectEqual(@as(usize, 64), alignUp(64, .cache_line));
    try std.testing.expectEqual(@as(usize, 128), alignUp(65, .cache_line));

    try std.testing.expectEqual(@as(usize, 0), alignDown(33, .cache_line));
    try std.testing.expectEqual(@as(usize, 64), alignDown(64, .cache_line));
    try std.testing.expectEqual(@as(usize, 64), alignDown(127, .cache_line));
}
