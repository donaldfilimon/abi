//! Allocator helpers and memory diagnostics.

const std = @import("std");

/// Snapshot of allocator stats for telemetry (TUI, logging).
pub const Stats = struct {
    total_allocations: u64 = 0,
    total_frees: u64 = 0,
    active_allocations: u64 = 0,
    current_bytes: u64 = 0,
    peak_bytes: u64 = 0,
    total_bytes_allocated: u64 = 0,
    total_bytes_freed: u64 = 0,
    failed_allocations: u64 = 0,
    double_frees: u64 = 0,
    invalid_frees: u64 = 0,
};

pub const TrackingAllocator = struct {
    parent_allocator: std.mem.Allocator,
    total_allocated: usize = 0,
    current_allocated: usize = 0,
    peak_allocated: usize = 0,
    count: usize = 0,
    free_count: usize = 0,

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

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent_allocator.rawAlloc(len, alignment, ret_addr);
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

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.parent_allocator.rawResize(buf, alignment, new_len, ret_addr)) {
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

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.parent_allocator.rawFree(buf, alignment, ret_addr);
        self.current_allocated -= buf.len;
        self.free_count += 1;
    }

    /// Return a snapshot for telemetry (TUI, logs). Thread-safe if caller synchronizes.
    pub fn getStats(self: *const TrackingAllocator) Stats {
        return .{
            .total_allocations = @intCast(self.count),
            .total_frees = @intCast(self.free_count),
            .active_allocations = @intCast(self.count -| self.free_count),
            .current_bytes = @intCast(self.current_allocated),
            .peak_bytes = @intCast(self.peak_allocated),
            .total_bytes_allocated = @intCast(self.total_allocated),
            .total_bytes_freed = @intCast(self.total_allocated -| self.current_allocated),
            .failed_allocations = 0,
            .double_frees = 0,
            .invalid_frees = 0,
        };
    }
};

pub fn getDiagnosticsAllocator(base: std.mem.Allocator) std.mem.Allocator {
    // In a real implementation, we might want to store this somewhere persistent.
    return base;
}
