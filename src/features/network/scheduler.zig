//! Distributed task scheduling
//!
//! Provides task scheduling, load balancing, and work distribution
//! across multiple compute nodes.

const std = @import("std");

pub const SchedulerError = error{
    NodeUnavailable,
    TaskRejected,
    NoAvailableNodes,
    SchedulingFailed,
};

pub const TaskPriority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    critical = 3,
};

pub const TaskState = enum {
    pending,
    scheduled,
    running,
    completed,
    failed,
    cancelled,
};

pub const NodeStatus = enum {
    online,
    offline,
    busy,
    degraded,
};

pub const ComputeNode = struct {
    id: []const u8,
    address: []const u8,
    port: u16,
    status: NodeStatus,
    cpu_count: u32,
    active_tasks: u32,
    last_heartbeat: i64,

    pub fn isAvailable(self: *const ComputeNode) bool {
        return self.status == .online or self.status == .busy;
    }
};

pub const ScheduledTask = struct {
    id: u64,
    priority: TaskPriority,
    state: TaskState,
    node_id: ?[]const u8,
    submit_time: i64,
    start_time: ?i64,
    end_time: ?i64,

    pub fn durationMs(self: *const ScheduledTask) ?i64 {
        if (self.start_time == null or self.end_time == null) {
            return null;
        }
        return self.end_time.? - self.start_time.?;
    }
};

pub const SchedulerConfig = struct {
    max_retries: u8 = 3,
    timeout_ms: u64 = 30_000,
    heartbeat_interval_ms: u64 = 5_000,
    load_balancing: LoadBalancingStrategy = .round_robin,
};

pub const LoadBalancingStrategy = enum {
    round_robin,
    least_loaded,
    random,
    affinity_based,
};

pub const TaskScheduler = struct {
    allocator: std.mem.Allocator,
    config: SchedulerConfig,
    nodes: std.StringHashMap(ComputeNode),
    tasks: std.AutoHashMap(u64, ScheduledTask),
    next_task_id: u64 = 1,
    current_rr_node: usize = 0,
    running_tasks: std.AutoHashMap(u64, u64),

    pub fn init(allocator: std.mem.Allocator, config: SchedulerConfig) !TaskScheduler {
        const scheduler = TaskScheduler{
            .allocator = allocator,
            .config = config,
            .nodes = std.StringHashMap(ComputeNode).init(allocator),
            .tasks = std.AutoHashMap(u64, ScheduledTask).init(allocator),
            .running_tasks = std.AutoHashMap(u64, u64).init(allocator),
        };
        return scheduler;
    }

    pub fn deinit(self: *TaskScheduler) void {
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*.id);
            self.allocator.free(entry.value_ptr.*.address);
        }
        self.nodes.deinit();

        var task_iter = self.tasks.iterator();
        while (task_iter.next()) |entry| {
            if (entry.value_ptr.*.node_id) |node_id| {
                self.allocator.free(node_id);
            }
        }
        self.tasks.deinit();
        self.running_tasks.deinit();

        self.* = undefined;
    }

    pub fn addNode(self: *TaskScheduler, node: ComputeNode) !void {
        const id_copy = try self.allocator.dupe(u8, node.id);
        errdefer self.allocator.free(id_copy);

        const addr_copy = try self.allocator.dupe(u8, node.address);
        errdefer self.allocator.free(addr_copy);

        const node_copy = ComputeNode{
            .id = id_copy,
            .address = addr_copy,
            .port = node.port,
            .status = node.status,
            .cpu_count = node.cpu_count,
            .active_tasks = node.active_tasks,
            .last_heartbeat = node.last_heartbeat,
        };

        try self.nodes.put(node_copy.id, node_copy);
    }

    pub fn removeNode(self: *TaskScheduler, node_id: []const u8) void {
        if (self.nodes.fetchRemove(node_id)) |entry| {
            self.allocator.free(entry.key);
            self.allocator.free(entry.value.id);
            self.allocator.free(entry.value.address);
        }
    }

    pub fn selectNode(self: *TaskScheduler) !?ComputeNode {
        var online_nodes = std.ArrayList(*ComputeNode).init(self.allocator);
        defer online_nodes.deinit();

        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.isAvailable()) {
                try online_nodes.append(entry.value_ptr);
            }
        }

        if (online_nodes.items.len == 0) {
            return SchedulerError.NoAvailableNodes;
        }

        return switch (self.config.load_balancing) {
            .round_robin => self.selectRoundRobin(online_nodes.items),
            .least_loaded => self.selectLeastLoaded(online_nodes.items),
            .random => self.selectRandom(online_nodes.items),
            .affinity_based => self.selectAffinityBased(online_nodes.items),
        };
    }

    pub fn submitTask(self: *TaskScheduler, priority: TaskPriority) !u64 {
        const node = try self.selectNode() orelse return SchedulerError.NoAvailableNodes;

        const task_id = self.next_task_id;
        self.next_task_id += 1;

        const node_id_copy = try self.allocator.dupe(u8, node.id);
        errdefer self.allocator.free(node_id_copy);

        const task = ScheduledTask{
            .id = task_id,
            .priority = priority,
            .state = .pending,
            .node_id = node_id_copy,
            .submit_time = std.time.milliTimestamp(),
            .start_time = null,
            .end_time = null,
        };

        try self.tasks.put(task_id, task);
        return task_id;
    }

    pub fn getTaskState(self: *TaskScheduler, task_id: u64) ?TaskState {
        const task = self.tasks.get(task_id) orelse return null;
        return task.state;
    }

    pub fn getTaskStatus(self: *TaskScheduler, task_id: u64) !ScheduledTask {
        const task = self.tasks.get(task_id) orelse return SchedulerError.TaskRejected;
        return task.*;
    }

    pub fn getStats(self: *const TaskScheduler) SchedulerStats {
        var pending: usize = 0;
        var running: usize = 0;
        var completed: usize = 0;
        var failed: usize = 0;

        var iter = self.tasks.valueIterator();
        while (iter.next()) |task| {
            switch (task.state) {
                .pending => pending += 1,
                .scheduled => pending += 1,
                .running => running += 1,
                .completed => completed += 1,
                .failed => failed += 1,
                .cancelled => failed += 1,
            }
        }

        return .{
            .total_tasks = self.tasks.count(),
            .pending_tasks = pending,
            .running_tasks = running,
            .completed_tasks = completed,
            .failed_tasks = failed,
            .node_count = self.nodes.count(),
            .online_nodes = self.countOnlineNodes(),
        };
    }

    fn selectRoundRobin(self: *TaskScheduler, nodes: []const *ComputeNode) ?*ComputeNode {
        if (nodes.len == 0) return null;
        const node = nodes[self.current_rr_node % nodes.len];
        self.current_rr_node += 1;
        return node;
    }

    fn selectLeastLoaded(self: *TaskScheduler, nodes: []const *ComputeNode) ?*ComputeNode {
        _ = self;
        if (nodes.len == 0) return null;

        var best_node: ?*ComputeNode = null;
        var min_tasks: u32 = std.math.maxInt(u32);

        for (nodes) |node| {
            if (node.active_tasks < min_tasks) {
                min_tasks = node.active_tasks;
                best_node = node;
            }
        }

        return best_node;
    }

    fn selectRandom(self: *TaskScheduler, nodes: []const *ComputeNode) ?*ComputeNode {
        _ = self;
        if (nodes.len == 0) return null;

        const index = std.crypto.random.intRangeLessThan(usize, nodes.len);
        return nodes[index];
    }

    fn selectAffinityBased(self: *TaskScheduler, nodes: []const *ComputeNode) ?*ComputeNode {
        _ = self;
        _ = nodes;
        return null;
    }

    fn countOnlineNodes(self: *const TaskScheduler) usize {
        var count: usize = 0;
        var iter = self.nodes.valueIterator();
        while (iter.next()) |node| {
            if (node.status == .online or node.status == .busy) {
                count += 1;
            }
        }
        return count;
    }
};

pub const SchedulerStats = struct {
    total_tasks: usize,
    pending_tasks: usize,
    running_tasks: usize,
    completed_tasks: usize,
    failed_tasks: usize,
    node_count: usize,
    online_nodes: usize,
};

test "scheduler adds and removes nodes" {
    const allocator = std.testing.allocator;

    var scheduler = try TaskScheduler.init(allocator, .{});
    defer scheduler.deinit();

    const node = ComputeNode{
        .id = "node-1",
        .address = "127.0.0.1",
        .port = 8080,
        .status = .online,
        .cpu_count = 8,
        .active_tasks = 0,
        .last_heartbeat = std.time.milliTimestamp(),
    };

    try scheduler.addNode(node);
    try std.testing.expectEqual(@as(usize, 1), scheduler.nodes.count());

    scheduler.removeNode("node-1");
    try std.testing.expectEqual(@as(usize, 0), scheduler.nodes.count());
}

test "scheduler round robin selection" {
    const allocator = std.testing.allocator;

    var scheduler = try TaskScheduler.init(allocator, .{
        .load_balancing = .round_robin,
    });
    defer scheduler.deinit();

    const nodes = [_]ComputeNode{
        .{ .id = "n1", .address = "127.0.0.1", .port = 8080, .status = .online, .cpu_count = 8, .active_tasks = 0, .last_heartbeat = 0 },
        .{ .id = "n2", .address = "127.0.0.2", .port = 8080, .status = .online, .cpu_count = 8, .active_tasks = 0, .last_heartbeat = 0 },
        .{ .id = "n3", .address = "127.0.0.3", .port = 8080, .status = .online, .cpu_count = 8, .active_tasks = 0, .last_heartbeat = 0 },
    };

    for (nodes) |node| {
        try scheduler.addNode(node);
    }

    const n1 = scheduler.nodes.get("n1").?;
    const n2 = scheduler.nodes.get("n2").?;
    const n3 = scheduler.nodes.get("n3").?;

    var expected = [_]*const ComputeNode{ &n1, &n2, &n3 };
    for (0..3) |i| {
        const selected = try scheduler.selectNode() orelse return error.TestUnexpected;
        try std.testing.expectEqual(expected[i], selected.?);
    }
}
