//! NUMA-Aware Work Stealing Policy
//!
//! Provides intelligent victim selection for work-stealing schedulers
//! that respects NUMA topology to minimize cross-node memory access.
//!
//! ## Design
//!
//! - **Local-first**: Prefer stealing from workers on the same NUMA node
//! - **Hierarchical**: Falls back to cross-node stealing when local queues empty
//! - **Adaptive**: Adjusts stealing aggression based on load imbalance
//! - **Statistics**: Tracks local vs remote steals for tuning
//!
//! ## Usage
//!
//! ```zig
//! var policy = try NumaStealPolicy.init(allocator, topology, worker_count);
//! defer policy.deinit();
//!
//! // Get victim for stealing
//! const victim = policy.selectVictim(worker_id, &rng);
//! if (victim) |v| {
//!     if (workers[v].deque.steal()) |task| {
//!         // Got work
//!     }
//! }
//! ```

const std = @import("std");
const sync = @import("../../shared/sync.zig");
const numa = @import("numa.zig");

/// Configuration for steal policy.
pub const StealPolicyConfig = struct {
    /// Probability of attempting local steal first (0-100)
    local_steal_probability: u8 = 80,
    /// Maximum attempts before giving up
    max_steal_attempts: u32 = 8,
    /// Whether to prefer workers with more work
    prefer_loaded_victims: bool = true,
    /// Enable adaptive stealing based on load
    adaptive: bool = true,
    /// Backoff multiplier when steals fail
    backoff_factor: u32 = 2,
};

/// Statistics for steal operations.
pub const StealStats = struct {
    /// Successful steals from same NUMA node
    local_steals: u64 = 0,
    /// Successful steals from different NUMA node
    remote_steals: u64 = 0,
    /// Failed steal attempts
    failed_steals: u64 = 0,
    /// Total steal attempts
    total_attempts: u64 = 0,

    /// Get local steal percentage.
    pub fn localRate(self: StealStats) f64 {
        const total = self.local_steals + self.remote_steals;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.local_steals)) / @as(f64, @floatFromInt(total)) * 100.0;
    }

    /// Get overall success rate.
    pub fn successRate(self: StealStats) f64 {
        if (self.total_attempts == 0) return 0;
        const successes = self.local_steals + self.remote_steals;
        return @as(f64, @floatFromInt(successes)) / @as(f64, @floatFromInt(self.total_attempts)) * 100.0;
    }
};

/// Per-worker NUMA information.
const WorkerInfo = struct {
    /// NUMA node this worker is assigned to
    numa_node: u32,
    /// CPU ID this worker is pinned to (if any)
    cpu_id: ?u32,
    /// Other workers on the same NUMA node
    local_peers: []u32,
    /// Workers on different NUMA nodes
    remote_peers: []u32,
};

/// NUMA-aware work stealing policy.
pub const NumaStealPolicy = struct {
    allocator: std.mem.Allocator,
    config: StealPolicyConfig,
    /// Per-worker information
    workers: []WorkerInfo,
    /// Number of workers
    worker_count: usize,
    /// Number of NUMA nodes
    node_count: usize,
    /// Statistics
    stats: StealStats = .{},
    /// Mutex for stats updates
    stats_mutex: sync.Mutex = .{},

    /// Initialize with NUMA topology.
    pub fn init(
        allocator: std.mem.Allocator,
        topology: ?*const numa.CpuTopology,
        worker_count: usize,
        config: StealPolicyConfig,
    ) !NumaStealPolicy {
        const workers = try allocator.alloc(WorkerInfo, worker_count);
        errdefer allocator.free(workers);

        const node_count = if (topology) |t| t.node_count else 1;

        // Initialize worker info
        for (workers, 0..) |*w, i| {
            // Assign workers to NUMA nodes round-robin
            const node_id = if (topology) |t|
                @as(u32, @intCast(i % t.node_count))
            else
                0;

            w.numa_node = node_id;
            w.cpu_id = if (topology) |t|
                if (i < t.cpu_count) @as(u32, @intCast(i)) else null
            else
                null;

            w.local_peers = &.{};
            w.remote_peers = &.{};
        }

        // Build peer lists
        for (workers, 0..) |*w, i| {
            var local_peers = std.ArrayListUnmanaged(u32).empty;
            var remote_peers = std.ArrayListUnmanaged(u32).empty;

            for (workers, 0..) |other, j| {
                if (i == j) continue; // Skip self

                if (other.numa_node == w.numa_node) {
                    try local_peers.append(allocator, @intCast(j));
                } else {
                    try remote_peers.append(allocator, @intCast(j));
                }
            }

            w.local_peers = try local_peers.toOwnedSlice(allocator);
            w.remote_peers = try remote_peers.toOwnedSlice(allocator);
        }

        return NumaStealPolicy{
            .allocator = allocator,
            .config = config,
            .workers = workers,
            .worker_count = worker_count,
            .node_count = node_count,
        };
    }

    pub fn deinit(self: *NumaStealPolicy) void {
        for (self.workers) |*w| {
            if (w.local_peers.len > 0) {
                self.allocator.free(w.local_peers);
            }
            if (w.remote_peers.len > 0) {
                self.allocator.free(w.remote_peers);
            }
        }
        self.allocator.free(self.workers);
        self.* = undefined;
    }

    /// Select a victim worker to steal from.
    /// Returns null if no suitable victim found.
    pub fn selectVictim(
        self: *NumaStealPolicy,
        worker_id: usize,
        rng: *std.Random.DefaultPrng,
    ) ?usize {
        if (worker_id >= self.worker_count) return null;

        const worker = &self.workers[worker_id];
        const random = rng.random();

        self.stats_mutex.lock();
        self.stats.total_attempts += 1;
        self.stats_mutex.unlock();

        // Decide whether to try local first
        const try_local_first = random.intRangeLessThan(u8, 0, 100) < self.config.local_steal_probability;

        if (try_local_first and worker.local_peers.len > 0) {
            // Try local peers first
            const local_victim = worker.local_peers[random.intRangeLessThan(usize, 0, worker.local_peers.len)];
            return local_victim;
        } else if (worker.remote_peers.len > 0) {
            // Try remote peers
            const remote_victim = worker.remote_peers[random.intRangeLessThan(usize, 0, worker.remote_peers.len)];
            return remote_victim;
        } else if (worker.local_peers.len > 0) {
            // Fallback to local if no remote available
            const local_victim = worker.local_peers[random.intRangeLessThan(usize, 0, worker.local_peers.len)];
            return local_victim;
        }

        return null;
    }

    /// Select multiple potential victims in priority order.
    /// Returns victims ordered by NUMA locality preference.
    pub fn selectVictims(
        self: *NumaStealPolicy,
        worker_id: usize,
        rng: *std.Random.DefaultPrng,
        out_victims: []usize,
    ) usize {
        if (worker_id >= self.worker_count) return 0;

        const worker = &self.workers[worker_id];
        const random = rng.random();
        var count: usize = 0;

        // Shuffle and add local peers first
        if (worker.local_peers.len > 0) {
            const local_copy = self.allocator.dupe(u32, worker.local_peers) catch return 0;
            defer self.allocator.free(local_copy);

            random.shuffle(u32, local_copy);

            for (local_copy) |peer| {
                if (count >= out_victims.len) break;
                out_victims[count] = peer;
                count += 1;
            }
        }

        // Then add remote peers
        if (worker.remote_peers.len > 0) {
            const remote_copy = self.allocator.dupe(u32, worker.remote_peers) catch return count;
            defer self.allocator.free(remote_copy);

            random.shuffle(u32, remote_copy);

            for (remote_copy) |peer| {
                if (count >= out_victims.len) break;
                out_victims[count] = peer;
                count += 1;
            }
        }

        return count;
    }

    /// Record a successful steal.
    pub fn recordSteal(self: *NumaStealPolicy, thief_id: usize, victim_id: usize) void {
        self.stats_mutex.lock();
        defer self.stats_mutex.unlock();

        if (thief_id < self.worker_count and victim_id < self.worker_count) {
            if (self.workers[thief_id].numa_node == self.workers[victim_id].numa_node) {
                self.stats.local_steals += 1;
            } else {
                self.stats.remote_steals += 1;
            }
        }
    }

    /// Record a failed steal attempt.
    pub fn recordFailedSteal(self: *NumaStealPolicy) void {
        self.stats_mutex.lock();
        self.stats.failed_steals += 1;
        self.stats_mutex.unlock();
    }

    /// Get current statistics.
    pub fn getStats(self: *NumaStealPolicy) StealStats {
        self.stats_mutex.lock();
        defer self.stats_mutex.unlock();
        return self.stats;
    }

    /// Get the NUMA node for a worker.
    pub fn getWorkerNode(self: *const NumaStealPolicy, worker_id: usize) ?u32 {
        if (worker_id >= self.worker_count) return null;
        return self.workers[worker_id].numa_node;
    }

    /// Get all workers on a specific NUMA node.
    pub fn getWorkersOnNode(self: *const NumaStealPolicy, node_id: u32) []const u32 {
        // Find first worker on this node to get its local_peers
        for (self.workers) |*w| {
            if (w.numa_node == node_id) {
                return w.local_peers;
            }
        }
        return &.{};
    }
};

/// Simple round-robin steal policy (NUMA-unaware baseline).
pub const RoundRobinStealPolicy = struct {
    worker_count: usize,
    /// Per-worker next victim index
    next_victim: []std.atomic.Value(usize),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, worker_count: usize) !RoundRobinStealPolicy {
        const next_victim = try allocator.alloc(std.atomic.Value(usize), worker_count);
        for (next_victim, 0..) |*nv, i| {
            nv.* = std.atomic.Value(usize).init((i + 1) % worker_count);
        }

        return RoundRobinStealPolicy{
            .worker_count = worker_count,
            .next_victim = next_victim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RoundRobinStealPolicy) void {
        self.allocator.free(self.next_victim);
        self.* = undefined;
    }

    pub fn selectVictim(self: *RoundRobinStealPolicy, worker_id: usize) ?usize {
        if (worker_id >= self.worker_count or self.worker_count <= 1) return null;

        const current = self.next_victim[worker_id].load(.monotonic);
        var next = (current + 1) % self.worker_count;
        if (next == worker_id) {
            next = (next + 1) % self.worker_count;
        }
        self.next_victim[worker_id].store(next, .monotonic);

        return if (current != worker_id) current else null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "numa steal policy basic" {
    var policy = try NumaStealPolicy.init(
        std.testing.allocator,
        null, // No topology
        4,
        .{},
    );
    defer policy.deinit();

    var rng = std.Random.DefaultPrng.init(12345);

    // Should be able to select victims
    for (0..4) |i| {
        const victim = policy.selectVictim(i, &rng);
        if (victim) |v| {
            try std.testing.expect(v != i); // Should not select self
        }
    }
}

test "numa steal policy stats" {
    var policy = try NumaStealPolicy.init(
        std.testing.allocator,
        null,
        4,
        .{},
    );
    defer policy.deinit();

    // Record some steals
    policy.recordSteal(0, 1);
    policy.recordSteal(0, 2);
    policy.recordFailedSteal();

    const stats = policy.getStats();
    try std.testing.expect(stats.local_steals + stats.remote_steals == 2);
    try std.testing.expect(stats.failed_steals == 1);
}

test "round robin steal policy" {
    var policy = try RoundRobinStealPolicy.init(std.testing.allocator, 4);
    defer policy.deinit();

    // Worker 0 should cycle through 1, 2, 3
    const v1 = policy.selectVictim(0);
    const v2 = policy.selectVictim(0);
    const v3 = policy.selectVictim(0);

    try std.testing.expect(v1 != null);
    try std.testing.expect(v2 != null);
    try std.testing.expect(v3 != null);
    try std.testing.expect(v1.? != v2.?);
}

test "numa steal policy victim ordering" {
    var policy = try NumaStealPolicy.init(
        std.testing.allocator,
        null,
        8,
        .{ .local_steal_probability = 100 },
    );
    defer policy.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var victims: [4]usize = undefined;

    const count = policy.selectVictims(0, &rng, &victims);
    try std.testing.expect(count > 0);

    // All victims should be different from worker 0
    for (victims[0..count]) |v| {
        try std.testing.expect(v != 0);
    }
}

test {
    std.testing.refAllDecls(@This());
}
