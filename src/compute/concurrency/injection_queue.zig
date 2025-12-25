//! MPSC injection queue for external submissions
//!
//! Single-producer, multi-consumer queue for submitting work
//! from outside the worker pool.

const std = @import("std");

pub const InjectionQueue = struct {
    buffer: std.ArrayList(u64),

    pub fn init(allocator: std.mem.Allocator) InjectionQueue {
        return .{
            .buffer = std.ArrayList(u64).init(allocator),
        };
    }

    pub fn deinit(self: *InjectionQueue) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    pub fn push(self: *InjectionQueue, value: u64) !void {
        try self.buffer.append(value);
    }

    pub fn pop(self: *InjectionQueue) ?u64 {
        return self.buffer.popOrNull();
    }
};
