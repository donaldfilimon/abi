//! Chase-Lev work-stealing deque
//!
//! Lock-free deque for work-stealing scheduler.
//! Owner pushes/pops from bottom, stealers pop from top.

const std = @import("std");

pub const ChaseLevDeque = struct {
    buffer: []std.atomic.Value(u64),
    capacity: usize,
    top: std.atomic.Value(isize),
    bottom: std.atomic.Value(isize),

    const EMPTY: u64 = 0;

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ChaseLevDeque {
        const buffer = try allocator.alloc(std.atomic.Value(u64), capacity);
        for (buffer) |*elem| {
            elem.* = std.atomic.Value(u64).init(EMPTY);
        }

        return .{
            .buffer = buffer,
            .capacity = capacity,
            .top = std.atomic.Value(isize).init(0),
            .bottom = std.atomic.Value(isize).init(0),
        };
    }

    pub fn deinit(self: *ChaseLevDeque, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        self.* = undefined;
    }

    fn grow(self: *ChaseLevDeque, allocator: std.mem.Allocator, new_size: usize) !void {
        const old_buffer = self.buffer;
        const old_capacity = self.capacity;

        const new_buffer = try allocator.alloc(std.atomic.Value(u64), new_size);
        for (new_buffer) |*elem| {
            elem.* = std.atomic.Value(u64).init(EMPTY);
        }

        const old_bottom = self.bottom.load(.monotonic);
        const old_top = self.top.load(.acquire);

        var i: isize = old_top;
        while (i < old_bottom) : (i += 1) {
            const idx = @as(usize, @intCast(@mod(i, @as(isize, @intCast(old_capacity)))));
            const new_idx = @as(usize, @intCast(@mod(i, @as(isize, @intCast(new_size)))));
            new_buffer[new_idx].store(old_buffer[idx].load(.monotonic), .monotonic);
        }

        allocator.free(old_buffer);
        self.buffer = new_buffer;
        self.capacity = new_size;
    }

    pub fn pushBottom(self: *ChaseLevDeque, allocator: std.mem.Allocator, value: u64) !void {
        const b = self.bottom.load(.monotonic);
        const t = self.top.load(.acquire);
        const size: isize = b - t;

        if (size >= @as(isize, @intCast(self.capacity))) {
            try self.grow(allocator, self.capacity * 2);
        }

        const idx = @as(usize, @intCast(@mod(b, self.capacity)));
        self.buffer[idx].store(value, .release);
        self.bottom.store(b + 1, .release);
    }

    pub fn popBottom(self: *ChaseLevDeque) ?u64 {
        const b = self.bottom.load(.monotonic);
        self.bottom.store(b - 1, .monotonic);
        const t = self.top.load(.acquire);

        if (b - t <= 0) {
            self.bottom.store(t, .monotonic);
            return null;
        }

        const idx = @as(usize, @intCast(@mod(b - 1, @as(isize, @intCast(self.capacity)))));
        const value = self.buffer[idx].load(.monotonic);

        if (b - t > 1) {
            return value;
        }

        if (self.top.cmpxchgStrong(t, t + 1, .acq_rel, .monotonic) == null) {
            return value;
        }

        self.bottom.store(t + 1, .monotonic);
        return null;
    }

    pub fn steal(self: *ChaseLevDeque) ?u64 {
        const t = self.top.load(.acquire);
        const b = self.bottom.load(.acquire);

        if (b <= t) {
            return null;
        }

        const idx = @as(usize, @intCast(@mod(t, @as(isize, @intCast(self.capacity)))));
        const value = self.buffer[idx].load(.acquire);

        if (self.top.cmpxchgStrong(t, t + 1, .acq_rel, .monotonic) == null) {
            return value;
        }

        return null;
    }
};
