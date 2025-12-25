//! Cache-aligned buffer wrapper
//!
//! Provides memory-aligned buffers optimized for CPU cache operations.

const std = @import("std");

pub fn CacheAlignedBuffer(comptime T: type, comptime alignment: ?usize) type {
    return struct {
        const Self = @This();
        const actual_alignment = alignment orelse std.mem.page_size;
        const alignment_mask = actual_alignment - 1;

        data: []T,
        allocator: std.mem.Allocator,
        raw_ptr: [*]u8,
        raw_len: usize,

        pub fn init(allocator: std.mem.Allocator, count: usize) !Self {
            const byte_len = count * @sizeOf(T);
            const aligned_byte_len = (byte_len + alignment_mask) & ~alignment_mask;

            const raw = try allocator.alignedAlloc(u8, actual_alignment, aligned_byte_len);
            errdefer allocator.free(raw);

            const aligned_ptr = std.mem.alignForward(usize, @intFromPtr(raw), actual_alignment);

            return .{
                .data = @as([*]T, @ptrFromInt(aligned_ptr))[0..count],
                .allocator = allocator,
                .raw_ptr = raw.ptr,
                .raw_len = aligned_byte_len,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.rawFree(self.raw_ptr, self.raw_len);
            self.* = undefined;
        }
    };
}

test "CacheAlignedBuffer basic usage" {
    var buf = try CacheAlignedBuffer(u8, null).init(std.testing.allocator, 16);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 16), buf.data.len);
}
