//! Priority Workload Queue for GPU Scheduling
//!
//! Provides priority-based workload queuing with fair scheduling,
//! deadline support, and retry policies.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const coordinator = @import("coordinator.zig");

/// Workload priority levels.
pub const Priority = enum(u8) {
    critical = 0,
    high = 1,
    normal = 2,
    low = 3,
    background = 4,

    /// Get weight for priority (higher = more important).
    pub fn weight(self: Priority) u32 {
        return switch (self) {
            .critical => 16,
            .high => 8,
            .normal => 4,
            .low => 2,
            .background => 1,
        };
    }
};

/// Status of a queued workload.
pub const WorkloadStatus = enum {
    pending,
    processing,
    completed,
    failed,
    cancelled,
    timeout,
};

/// A workload in the queue.
pub const QueuedWorkload = struct {
    id: u64,
    profile: coordinator.WorkloadProfile,
    priority: Priority,
    deadline_ms: i64,
    submitted_at: i64,
    callback_data: ?*anyopaque,
    status: WorkloadStatus,
    retry_count: u32,
    max_retries: u32,
    tag: ?[]const u8,

    /// Check if the workload has exceeded its deadline.
    pub fn isExpired(self: QueuedWorkload) bool {
        if (self.deadline_ms == 0) return false;
        return std.time.milliTimestamp() > self.deadline_ms;
    }

    /// Get time spent waiting in queue.
    pub fn waitTime(self: QueuedWorkload) i64 {
        return std.time.milliTimestamp() - self.submitted_at;
    }

    /// Check if workload can be retried.
    pub fn canRetry(self: QueuedWorkload) bool {
        return self.retry_count < self.max_retries;
    }
};

/// Options for enqueueing a workload.
pub const EnqueueOptions = struct {
    profile: coordinator.WorkloadProfile,
    priority: Priority = .normal,
    deadline_offset_ms: i64 = 0,
    max_retries: u32 = 3,
    callback_data: ?*anyopaque = null,
    tag: ?[]const u8 = null,
};

/// Queue configuration.
pub const QueueConfig = struct {
    max_capacity: usize = 10000,
    fair_scheduling: bool = true,
    starvation_threshold_ms: i64 = 5000,
    enable_batching: bool = false,
    min_batch_size: usize = 4,
    max_batch_size: usize = 32,
    batch_timeout_ms: i64 = 100,
};

/// Statistics about the queue.
pub const QueueStats = struct {
    total_enqueued: u64 = 0,
    total_dequeued: u64 = 0,
    total_completed: u64 = 0,
    total_failed: u64 = 0,
    total_cancelled: u64 = 0,
    total_timeout: u64 = 0,
    current_depth: usize = 0,
    priority_counts: [5]u64 = [_]u64{0} ** 5,
    avg_wait_time_ms: f32 = 0,
    max_wait_time_ms: i64 = 0,
};

/// Priority-based workload queue.
pub const WorkloadQueue = struct {
    allocator: std.mem.Allocator,
    config: QueueConfig,
    queues: [5]std.ArrayListUnmanaged(QueuedWorkload),
    stats: QueueStats,
    next_id: u64,
    mutex: sync.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: QueueConfig) !*WorkloadQueue {
        const self = try allocator.create(WorkloadQueue);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .queues = [_]std.ArrayListUnmanaged(QueuedWorkload){.{}} ** 5,
            .stats = .{},
            .next_id = 1,
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *WorkloadQueue) void {
        for (&self.queues) |*q| {
            q.deinit(self.allocator);
        }
        self.allocator.destroy(self);
    }

    /// Add a workload to the queue.
    pub fn enqueue(self: *WorkloadQueue, options: EnqueueOptions) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.totalPending() >= self.config.max_capacity) {
            return error.QueueFull;
        }

        const id = self.next_id;
        self.next_id += 1;
        const now = std.time.milliTimestamp();
        const deadline = if (options.deadline_offset_ms > 0) now + options.deadline_offset_ms else 0;

        const workload = QueuedWorkload{
            .id = id,
            .profile = options.profile,
            .priority = options.priority,
            .deadline_ms = deadline,
            .submitted_at = now,
            .callback_data = options.callback_data,
            .status = .pending,
            .retry_count = 0,
            .max_retries = options.max_retries,
            .tag = options.tag,
        };

        const priority_idx = @intFromEnum(options.priority);
        try self.queues[priority_idx].append(self.allocator, workload);
        self.stats.total_enqueued += 1;
        self.stats.current_depth += 1;
        self.stats.priority_counts[priority_idx] += 1;

        return id;
    }

    /// Remove and return the highest priority pending workload.
    pub fn dequeue(self: *WorkloadQueue) ?QueuedWorkload {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check for starvation and promote if needed
        if (self.config.fair_scheduling) {
            self.checkStarvation();
        }

        // Return first available from highest priority
        for (&self.queues, 0..) |*queue, priority_idx| {
            if (queue.items.len > 0) {
                var workload = queue.orderedRemove(0);
                workload.status = .processing;
                self.stats.total_dequeued += 1;
                self.stats.current_depth -|= 1;
                self.stats.priority_counts[priority_idx] -|= 1;

                const wait = workload.waitTime();
                self.stats.max_wait_time_ms = @max(self.stats.max_wait_time_ms, wait);

                return workload;
            }
        }
        return null;
    }

    /// Mark a workload as completed.
    pub fn complete(self: *WorkloadQueue, id: u64, success: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        _ = id; // ID tracking for future enhancement
        if (success) {
            self.stats.total_completed += 1;
        } else {
            self.stats.total_failed += 1;
        }
    }

    /// Cancel a pending workload.
    pub fn cancel(self: *WorkloadQueue, id: u64) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (&self.queues, 0..) |*queue, priority_idx| {
            for (queue.items, 0..) |*item, i| {
                if (item.id == id and item.status == .pending) {
                    _ = queue.orderedRemove(i);
                    self.stats.total_cancelled += 1;
                    self.stats.current_depth -|= 1;
                    self.stats.priority_counts[priority_idx] -|= 1;
                    return true;
                }
            }
        }
        return false;
    }

    /// Get current queue statistics.
    pub fn getStats(self: *WorkloadQueue) QueueStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get total queue depth.
    pub fn depth(self: *WorkloadQueue) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.totalPending();
    }

    /// Check if queue is empty.
    pub fn isEmpty(self: *WorkloadQueue) bool {
        return self.depth() == 0;
    }

    /// Get queue depth for a specific priority.
    pub fn depthForPriority(self: *WorkloadQueue, priority: Priority) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.queues[@intFromEnum(priority)].items.len;
    }

    /// Clear all pending workloads.
    pub fn clear(self: *WorkloadQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (&self.queues, 0..) |*queue, priority_idx| {
            self.stats.total_cancelled += queue.items.len;
            self.stats.priority_counts[priority_idx] = 0;
            queue.clearRetainingCapacity();
        }
        self.stats.current_depth = 0;
    }

    fn totalPending(self: *WorkloadQueue) usize {
        var total: usize = 0;
        for (self.queues) |queue| {
            total += queue.items.len;
        }
        return total;
    }

    fn checkStarvation(self: *WorkloadQueue) void {
        const now = std.time.milliTimestamp();
        const threshold = self.config.starvation_threshold_ms;

        // Check lower priority queues for starving workloads
        var priority_idx: usize = 4; // Start from background
        while (priority_idx > 0) : (priority_idx -= 1) {
            const queue = &self.queues[priority_idx];
            if (queue.items.len > 0) {
                const oldest = &queue.items[0];
                if (now - oldest.submitted_at > threshold) {
                    // Promote to next higher priority
                    const workload = queue.orderedRemove(0);
                    self.stats.priority_counts[priority_idx] -|= 1;
                    const new_priority = priority_idx - 1;
                    self.queues[new_priority].append(self.allocator, workload) catch {
                        // Re-enqueue at original priority if promotion fails
                        self.queues[priority_idx].append(self.allocator, workload) catch {};
                        self.stats.priority_counts[priority_idx] += 1;
                        continue;
                    };
                    self.stats.priority_counts[new_priority] += 1;
                }
            }
        }
    }
};

test "queue basic operations" {
    const allocator = std.testing.allocator;
    const queue = try WorkloadQueue.init(allocator, .{});
    defer queue.deinit();

    // Enqueue
    const id = try queue.enqueue(.{ .profile = .{}, .priority = .normal });
    try std.testing.expect(id > 0);
    try std.testing.expectEqual(@as(usize, 1), queue.depth());

    // Dequeue
    const item = queue.dequeue();
    try std.testing.expect(item != null);
    try std.testing.expect(queue.isEmpty());
}

test "queue priority ordering" {
    const allocator = std.testing.allocator;
    const queue = try WorkloadQueue.init(allocator, .{ .fair_scheduling = false });
    defer queue.deinit();

    // Enqueue in reverse priority order
    _ = try queue.enqueue(.{ .profile = .{}, .priority = .low });
    _ = try queue.enqueue(.{ .profile = .{}, .priority = .high });
    _ = try queue.enqueue(.{ .profile = .{}, .priority = .critical });

    // Should dequeue in priority order
    const first = queue.dequeue();
    try std.testing.expect(first != null);
    try std.testing.expectEqual(Priority.critical, first.?.priority);

    const second = queue.dequeue();
    try std.testing.expect(second != null);
    try std.testing.expectEqual(Priority.high, second.?.priority);
}

test "queue cancel" {
    const allocator = std.testing.allocator;
    const queue = try WorkloadQueue.init(allocator, .{});
    defer queue.deinit();

    const id = try queue.enqueue(.{ .profile = .{}, .priority = .normal });
    try std.testing.expectEqual(@as(usize, 1), queue.depth());

    const cancelled = queue.cancel(id);
    try std.testing.expect(cancelled);
    try std.testing.expect(queue.isEmpty());
}
