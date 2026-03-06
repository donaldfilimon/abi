//! Allocator helpers and memory diagnostics.

const std = @import("std");

pub const TrackingAllocator = struct {
    parent_allocator: std.mem.Allocator,
    total_allocated: usize = 0,
    current_allocated: usize = 0,
    peak_allocated: usize = 0,
    count: usize = 0,

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
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
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            self.total_allocated += len;
            self.current_allocated += len;
            self.count += 1;
            if (self.current_allocated > self.peak_allocated) {
                self.peak_allocated = self.current_allocated;
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.parent_allocator.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                self.total_allocated += diff;
                self.current_allocated += diff;
                if (self.current_allocated > self.peak_allocated) {
                    self.peak_allocated = self.current_allocated;
                }
            } else {
                self.current_allocated -= (buf.len - new_len);
            }
            return true;
        }
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.parent_allocator.rawFree(buf, buf_align, ret_addr);
        self.current_allocated -= buf.len;
    }
};

pub fn getDiagnosticsAllocator(base: std.mem.Allocator) std.mem.Allocator {
    // In a real implementation, we might want to store this somewhere persistent
    // For now, we return the base, but a static/global one could be used.
    _ = base;
    return base;
}
