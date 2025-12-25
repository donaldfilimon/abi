const std = @import("std");

pub fn WorkQueue(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        items: std.ArrayList(T),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .allocator = allocator,
                .items = std.ArrayList(T).empty,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn enqueue(self: *@This(), item: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.items.append(self.allocator, item);
        }

        pub fn dequeue(self: *@This()) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.items.items.len == 0) return null;
            return self.items.pop();
        }
    };
}

pub const Backoff = struct {
    spins: usize = 0,

    pub fn reset(self: *Backoff) void {
        self.spins = 0;
    }

    pub fn spin(self: *Backoff) void {
        self.spins += 1;
        std.atomic.spinLoopHint();
    }
};
