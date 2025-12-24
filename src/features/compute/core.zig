const std = @import("std");
const builtin = @import("builtin");

pub const CpuInfo = struct {
    core_count: u32,
    cache_line_size: u32,
    simd_width: u32,
    page_size: usize,
};

pub fn detectCpuInfo() CpuInfo {
    return .{
        .core_count = @as(u32, @intCast(std.Thread.getCpuCount() catch 1)),
        .cache_line_size = detectCacheLineSize(),
        .simd_width = detectSimdWidth(),
        .page_size = std.mem.page_size,
    };
}

pub fn alignedAlloc(
    allocator: std.mem.Allocator,
    comptime T: type,
    count: usize,
    alignment: u29,
) ![]T {
    return allocator.alignedAlloc(T, alignment, count);
}

pub fn alignedFree(allocator: std.mem.Allocator, slice: anytype) void {
    allocator.free(slice);
}

pub fn CacheAlignedBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, count: usize) !Self {
            const buffer = try allocator.alignedAlloc(T, 64, count);
            return .{ .data = buffer, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}

fn detectCacheLineSize() u32 {
    return switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => 64,
        .x86 => 32,
        else => 64,
    };
}

fn detectSimdWidth() u32 {
    return switch (builtin.cpu.arch) {
        .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 16 else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
        .aarch64 => 4,
        else => 1,
    };
}

test "cpu info detection" {
    const info = detectCpuInfo();
    try std.testing.expect(info.core_count >= 1);
    try std.testing.expect(info.cache_line_size > 0);
    try std.testing.expect(info.page_size >= 4096);
}
