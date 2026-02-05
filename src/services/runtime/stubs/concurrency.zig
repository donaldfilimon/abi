const std = @import("std");
const types = @import("types.zig");

pub fn WorkStealingQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }

        pub fn steal(_: *Self) ?T {
            return null;
        }
    };
}

pub fn WorkQueue(comptime T: type) type {
    return WorkStealingQueue(T);
}

pub fn LockFreeQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn LockFreeStack(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn ShardedMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn put(_: *Self, _: K, _: V) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn get(_: *Self, _: K) ?V {
            return null;
        }
    };
}

pub fn PriorityQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub const Backoff = struct {
    pub fn init() Backoff {
        return .{};
    }

    pub fn spin(_: *Backoff) void {}

    pub fn reset(_: *Backoff) void {}
};

// New lock-free concurrency primitives
pub fn ChaseLevDeque(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }

        pub fn steal(_: *Self) ?T {
            return null;
        }
    };
}

pub const WorkStealingScheduler = struct {
    pub fn init(_: std.mem.Allocator, _: usize) types.Error!WorkStealingScheduler {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *WorkStealingScheduler) void {}

    pub fn schedule(_: *WorkStealingScheduler, _: anytype) types.Error!void {
        return error.RuntimeDisabled;
    }
};

pub fn EpochReclamation(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn retire(_: *Self, _: *T) void {}

        pub fn collect(_: *Self) void {}
    };
}

pub fn LockFreeStackEBR(comptime T: type) type {
    return LockFreeStack(T);
}

pub fn MpmcQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) types.Error!Self {
            return error.RuntimeDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn push(_: *Self, _: T) types.Error!void {
            return error.RuntimeDisabled;
        }

        pub fn pop(_: *Self) ?T {
            return null;
        }
    };
}

pub fn BlockingMpmcQueue(comptime T: type) type {
    return MpmcQueue(T);
}
