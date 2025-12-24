//! Chase-Lev work-stealing deque
//!
//! Lock-free deque for work-stealing scheduler.
//! Owner pushes/pops from bottom, stealers pop from top.

const std = @import("std");

pub const ChaseLevDeque = struct {
    buffer: []std.atomic.Value(u64),
    top: std.atomic.Value(isize),
    bottom: std.atomic.Value(isize),

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ChaseLevDeque {
        const buffer = try allocator.alloc(std.atomic.Value(u64), capacity);
        @memset(buffer, @as(std.atomic.Value(u64), @splat(0)));

        return .{
            .buffer = buffer,
            .top = std.atomic.Value(isize).init(0),
            .bottom = std.atomic.Value(isize).init(0),
        };
    }

    pub fn deinit(self: *ChaseLevDeque, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        self.* = undefined;
    }
};
