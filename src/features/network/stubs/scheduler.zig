const std = @import("std");

pub const TaskScheduler = struct {
    pub fn init(_: std.mem.Allocator, _: SchedulerConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const SchedulerConfig = struct {
    max_tasks: usize = 1024,
    max_workers: usize = 8,
};

pub const SchedulerError = error{
    NetworkDisabled,
    TaskQueueFull,
    NoWorkersAvailable,
};

pub const TaskPriority = enum { low, normal, high, critical };

pub const TaskState = enum { pending, running, completed, failed, cancelled };

pub const ComputeNode = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    capacity: usize = 0,
};

pub const LoadBalancingStrategy = enum { round_robin, least_loaded, random, hash_based };

pub const SchedulerStats = struct {
    pending_tasks: usize = 0,
    running_tasks: usize = 0,
    completed_tasks: usize = 0,
    failed_tasks: usize = 0,
};

test {
    std.testing.refAllDecls(@This());
}
