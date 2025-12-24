//! Pool allocator for fixed-size allocations
//!
//! Provides size-class-based pooling for high-frequency allocations.

const std = @import("std");

pub const PoolAllocator = struct {
    backing_allocator: std.mem.Allocator,
    pools: std.ArrayList(Pool),

    const Pool = struct {
        size_class: usize,
        free_list: std.ArrayList(*void),
        allocated_count: usize,
    };

    pub fn init(backing_allocator: std.mem.Allocator, size_classes: []const usize) !PoolAllocator {
        var pools = try std.ArrayList(Pool).initCapacity(backing_allocator, size_classes.len);
        errdefer pools.deinit();

        for (size_classes) |size| {
            try pools.append(.{
                .size_class = size,
                .free_list = std.ArrayList(*void).init(backing_allocator),
                .allocated_count = 0,
            });
        }

        return .{
            .backing_allocator = backing_allocator,
            .pools = pools,
        };
    }

    pub fn deinit(self: *PoolAllocator) void {
        for (self.pools.items) |*pool| {
            pool.free_list.deinit();
        }
        self.pools.deinit();
    }

    pub fn allocator(self: *PoolAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: ?usize) ?[*]u8 {
        const self: *PoolAllocator = @ptrCast(@alignCast(ctx));

        for (self.pools.items) |*pool| {
            if (pool.size_class == len and pool.free_list.items.len > 0) {
                const ptr = pool.free_list.pop();
                pool.allocated_count += 1;
                return @ptrCast(ptr);
            }
        }

        return self.backing_allocator.rawAlloc(len, log2_ptr_align, ret_addr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, new_len: usize, log2_buf_align: u8, ret_addr: ?usize) bool {
        _ = ctx;
        _ = new_len;
        _ = log2_buf_align;
        _ = ret_addr;
        _ = buf.len;

        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: ?usize) void {
        const self: *PoolAllocator = @ptrCast(@alignCast(ctx));

        for (self.pools.items) |*pool| {
            if (pool.size_class == buf.len) {
                pool.free_list.append(@ptrCast(buf.ptr)) catch {};
                pool.allocated_count -= 1;
                return;
            }
        }

        self.backing_allocator.rawFree(buf.ptr, buf.len, log2_buf_align, ret_addr);
    }
};
