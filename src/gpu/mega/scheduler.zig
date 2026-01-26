//! Learning-Based GPU Scheduler
//!
//! Uses reinforcement learning principles to optimize workload scheduling
//! across multiple GPU backends based on observed performance.
//!
//! ## Features
//!
//! - **Q-Learning**: Tabular Q-learning for discrete state-action spaces
//! - **Experience Replay**: Stores and samples past experiences for training
//! - **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
//! - **Adaptive Learning**: Decays exploration rate over time
//!
//! ## Usage
//!
//! ```zig
//! const mega = @import("mega/mod.zig");
//!
//! var coordinator = try mega.Coordinator.init(allocator);
//! defer coordinator.deinit();
//!
//! var scheduler = try mega.LearningScheduler.init(allocator, coordinator);
//! defer scheduler.deinit();
//!
//! // Schedule workload using learned policy
//! const profile = mega.WorkloadProfile{
//!     .compute_intensity = 0.8,
//!     .memory_requirement_mb = 2048,
//! };
//! const decision = scheduler.schedule(profile);
//!
//! // Execute workload...
//! const actual_time_ms: u64 = 150;
//! const success = true;
//!
//! // Record outcome for learning
//! try scheduler.recordAndLearn(decision, actual_time_ms, success);
//! ```

const std = @import("std");
const coordinator = @import("coordinator.zig");

/// Experience for replay buffer - stores a single state transition.
pub const Experience = struct {
    state: SchedulerState,
    action: u8, // Backend index
    reward: f32,
    next_state: SchedulerState,
    done: bool,
};

/// Compressed state representation for the scheduler.
/// Captures essential information about backend loads and system status.
pub const SchedulerState = struct {
    /// Normalized load for each backend (0-1), max 8 backends
    backend_loads: [8]f32,
    /// Memory pressure for each backend (0-1), max 8 backends
    memory_pressures: [8]f32,
    /// Number of pending workloads in queue
    pending_workloads: u32,
    /// Current system throughput
    current_throughput: f32,

    /// Create a state snapshot from the coordinator.
    pub fn fromCoordinator(coord: *coordinator.Coordinator) SchedulerState {
        var state = SchedulerState{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        };

        // Use thread-safe accessor
        const backends = coord.getBackendsSummary();
        for (backends, 0..) |backend, i| {
            if (i >= 8) break;
            // Calculate memory pressure: higher value means more memory used
            state.memory_pressures[i] = 1.0 - (@as(f32, @floatFromInt(backend.available_memory_mb)) /
                @as(f32, @floatFromInt(backend.total_memory_mb + 1)));
        }

        return state;
    }
};

/// Q-value approximation for backend selection using tabular Q-learning.
pub const QTable = struct {
    /// Q-values: 8 backends x 16 discretized states
    values: [8][16]f32,
    /// Learning rate (alpha) for Q-updates
    learning_rate: f32,
    /// Discount factor (gamma) for future rewards
    discount_factor: f32,
    /// Exploration rate (epsilon) for epsilon-greedy
    exploration_rate: f32,

    /// Initialize Q-table with default hyperparameters.
    pub fn init() QTable {
        return .{
            .values = [_][16]f32{[_]f32{0} ** 16} ** 8,
            .learning_rate = 0.1,
            .discount_factor = 0.95,
            .exploration_rate = 0.1,
        };
    }

    /// Discretize continuous state into 4-bit index (0-15).
    pub fn discretizeState(state: SchedulerState) u4 {
        // Simple discretization based on average memory pressure
        var avg_pressure: f32 = 0;
        for (state.memory_pressures) |p| {
            avg_pressure += p;
        }
        avg_pressure /= 8.0;

        return @intFromFloat(@min(15.0, avg_pressure * 16.0));
    }

    /// Select action using epsilon-greedy policy.
    pub fn selectAction(self: *QTable, state: SchedulerState, num_backends: usize) u8 {
        const state_idx = discretizeState(state);

        // Epsilon-greedy exploration - use Timer for Zig 0.16 compatibility
        const seed: u64 = blk: {
            var timer = std.time.Timer.start() catch break :blk 0;
            break :blk timer.read();
        };
        var prng = std.Random.DefaultPrng.init(seed);
        if (prng.random().float(f32) < self.exploration_rate) {
            return @intCast(prng.random().uintLessThan(usize, num_backends));
        }

        // Greedy selection - pick action with highest Q-value
        var best_action: u8 = 0;
        var best_value: f32 = self.values[0][state_idx];
        for (0..num_backends) |i| {
            if (self.values[i][state_idx] > best_value) {
                best_value = self.values[i][state_idx];
                best_action = @intCast(i);
            }
        }

        return best_action;
    }

    /// Update Q-value using the Q-learning update rule.
    /// Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
    pub fn update(self: *QTable, exp: Experience, num_backends: usize) void {
        const state_idx = discretizeState(exp.state);
        const next_state_idx = discretizeState(exp.next_state);

        // Find max Q for next state
        var max_next_q: f32 = self.values[0][next_state_idx];
        for (0..num_backends) |i| {
            if (self.values[i][next_state_idx] > max_next_q) {
                max_next_q = self.values[i][next_state_idx];
            }
        }

        // Q-learning update
        const target = if (exp.done) exp.reward else exp.reward + self.discount_factor * max_next_q;
        const current = self.values[exp.action][state_idx];
        self.values[exp.action][state_idx] = current + self.learning_rate * (target - current);
    }

    /// Decay exploration rate with a minimum floor.
    pub fn decayExploration(self: *QTable, min_rate: f32) void {
        self.exploration_rate = @max(min_rate, self.exploration_rate * 0.995);
    }
};

/// Experience replay buffer for storing and sampling past experiences.
pub const ReplayBuffer = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(Experience),
    capacity: usize,

    /// Initialize replay buffer with given capacity.
    pub fn init(allocator: std.mem.Allocator, capacity: usize) ReplayBuffer {
        return .{
            .allocator = allocator,
            .buffer = .{},
            .capacity = capacity,
        };
    }

    /// Deinitialize and free buffer memory.
    pub fn deinit(self: *ReplayBuffer) void {
        self.buffer.deinit(self.allocator);
    }

    /// Add experience to buffer, removing oldest if at capacity.
    pub fn add(self: *ReplayBuffer, exp: Experience) !void {
        if (self.buffer.items.len >= self.capacity) {
            // Efficient O(n) removal: shift all items left and shrink
            const items = self.buffer.items;
            std.mem.copyForwards(Experience, items[0 .. self.capacity - 1], items[1..]);
            self.buffer.shrinkRetainingCapacity(self.capacity - 1);
        }
        try self.buffer.append(self.allocator, exp);
    }

    /// Sample a batch of experiences (returns most recent batch_size items).
    pub fn sample(self: *ReplayBuffer, batch_size: usize) []Experience {
        if (self.buffer.items.len <= batch_size) {
            return self.buffer.items;
        }
        // Return last batch_size items for simplicity
        return self.buffer.items[self.buffer.items.len - batch_size ..];
    }

    /// Get current buffer size.
    pub fn size(self: *ReplayBuffer) usize {
        return self.buffer.items.len;
    }
};

/// Learning statistics for monitoring training progress.
pub const LearningStats = struct {
    /// Number of completed episodes
    episodes: usize,
    /// Average reward per episode
    avg_episode_reward: f32,
    /// Current exploration rate
    exploration_rate: f32,
    /// Current replay buffer size
    replay_buffer_size: usize,
};

/// Learning-based scheduler using Q-learning and experience replay.
pub const LearningScheduler = struct {
    allocator: std.mem.Allocator,
    q_table: QTable,
    replay_buffer: ReplayBuffer,
    coord: *coordinator.Coordinator,
    episode_rewards: std.ArrayListUnmanaged(f32),
    current_episode_reward: f32,

    /// Initialize learning scheduler with coordinator.
    pub fn init(allocator: std.mem.Allocator, coord: *coordinator.Coordinator) !*LearningScheduler {
        const self = try allocator.create(LearningScheduler);
        self.* = .{
            .allocator = allocator,
            .q_table = QTable.init(),
            .replay_buffer = ReplayBuffer.init(allocator, 10000),
            .coord = coord,
            .episode_rewards = .{},
            .current_episode_reward = 0,
        };
        return self;
    }

    /// Deinitialize scheduler and free resources.
    pub fn deinit(self: *LearningScheduler) void {
        self.replay_buffer.deinit();
        self.episode_rewards.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Schedule workload using learned policy.
    pub fn schedule(self: *LearningScheduler, profile: coordinator.WorkloadProfile) coordinator.ScheduleDecision {
        const state = SchedulerState.fromCoordinator(self.coord);
        const backends = self.coord.getBackendsSummary();
        const num_backends = backends.len;

        if (num_backends == 0) {
            return .{
                .backend_type = .stdgpu,
                .device_id = 0,
                .estimated_time_ms = 100,
                .confidence = 0.0,
                .reason = "No backends available",
                .decision_id = 0,
            };
        }

        const action = self.q_table.selectAction(state, num_backends);
        const backend_idx = @min(action, @as(u8, @intCast(num_backends - 1)));
        const backend = backends[backend_idx];

        // Adjust confidence based on memory requirements
        var confidence = self.q_table.values[action][QTable.discretizeState(state)];
        if (backend.available_memory_mb < profile.memory_requirement_mb) {
            confidence *= 0.5;
        }

        return .{
            .backend_type = backend.backend_type,
            .device_id = 0,
            .estimated_time_ms = 100,
            .confidence = @max(0.0, @min(1.0, confidence)),
            .reason = "Selected by learning scheduler",
            .decision_id = 0,
        };
    }

    /// Record outcome and learn from experience.
    pub fn recordAndLearn(
        self: *LearningScheduler,
        decision: coordinator.ScheduleDecision,
        actual_time_ms: u64,
        success: bool,
    ) !void {
        const state = SchedulerState.fromCoordinator(self.coord);
        const backends = self.coord.getBackendsSummary();

        // Calculate reward based on time and success
        const time_factor = 1000.0 / @as(f32, @floatFromInt(actual_time_ms + 1));
        const success_factor: f32 = if (success) 1.0 else -1.0;
        const reward = time_factor * success_factor;

        self.current_episode_reward += reward;

        // Find action index from decision
        var action: u8 = 0;
        for (backends, 0..) |b, i| {
            if (b.backend_type == decision.backend_type) {
                action = @intCast(i);
                break;
            }
        }

        // Store experience
        const exp = Experience{
            .state = state,
            .action = action,
            .reward = reward,
            .next_state = state, // Will be updated on next call
            .done = false,
        };
        try self.replay_buffer.add(exp);

        // Learn from replay buffer when we have enough experiences
        if (self.replay_buffer.size() >= 32) {
            const batch = self.replay_buffer.sample(32);
            for (batch) |e| {
                self.q_table.update(e, backends.len);
            }
        }

        // Decay exploration rate
        self.q_table.decayExploration(0.01);

        // Record in coordinator too for its statistics
        try self.coord.recordOutcome(decision, actual_time_ms, success);
    }

    /// End current episode and record total reward.
    pub fn endEpisode(self: *LearningScheduler) !void {
        try self.episode_rewards.append(self.allocator, self.current_episode_reward);
        self.current_episode_reward = 0;
    }

    /// Get learning statistics.
    pub fn getStats(self: *LearningScheduler) LearningStats {
        var avg_reward: f32 = 0;
        if (self.episode_rewards.items.len > 0) {
            for (self.episode_rewards.items) |r| {
                avg_reward += r;
            }
            avg_reward /= @floatFromInt(self.episode_rewards.items.len);
        }

        return .{
            .episodes = self.episode_rewards.items.len,
            .avg_episode_reward = avg_reward,
            .exploration_rate = self.q_table.exploration_rate,
            .replay_buffer_size = self.replay_buffer.size(),
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "learning scheduler init" {
    const allocator = std.testing.allocator;
    const coord = try coordinator.Coordinator.init(allocator);
    defer coord.deinit();

    const sched = try LearningScheduler.init(allocator, coord);
    defer sched.deinit();

    try std.testing.expect(sched.q_table.exploration_rate > 0);
}

test "q-table update" {
    var q = QTable.init();

    const exp = Experience{
        .state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0.5} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .action = 0,
        .reward = 1.0,
        .next_state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0.5} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .done = false,
    };

    q.update(exp, 4);
    // State with 0.5 average pressure -> discretizes to index 8
    try std.testing.expect(q.values[0][8] > 0);
}

test "replay buffer" {
    const allocator = std.testing.allocator;
    var buffer = ReplayBuffer.init(allocator, 100);
    defer buffer.deinit();

    const exp = Experience{
        .state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .action = 0,
        .reward = 1.0,
        .next_state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .done = false,
    };

    // Add experiences
    for (0..50) |_| {
        try buffer.add(exp);
    }
    try std.testing.expectEqual(@as(usize, 50), buffer.size());

    // Sample batch
    const batch = buffer.sample(10);
    try std.testing.expectEqual(@as(usize, 10), batch.len);
}

test "replay buffer capacity" {
    const allocator = std.testing.allocator;
    var buffer = ReplayBuffer.init(allocator, 10);
    defer buffer.deinit();

    const exp = Experience{
        .state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .action = 0,
        .reward = 1.0,
        .next_state = .{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        },
        .done = false,
    };

    // Add more than capacity
    for (0..20) |_| {
        try buffer.add(exp);
    }

    // Should be at capacity
    try std.testing.expectEqual(@as(usize, 10), buffer.size());
}

test "q-table discretization" {
    const state_low = SchedulerState{
        .backend_loads = [_]f32{0} ** 8,
        .memory_pressures = [_]f32{0} ** 8,
        .pending_workloads = 0,
        .current_throughput = 0,
    };
    try std.testing.expectEqual(@as(u4, 0), QTable.discretizeState(state_low));

    const state_high = SchedulerState{
        .backend_loads = [_]f32{1} ** 8,
        .memory_pressures = [_]f32{1} ** 8,
        .pending_workloads = 100,
        .current_throughput = 1000,
    };
    try std.testing.expectEqual(@as(u4, 15), QTable.discretizeState(state_high));

    const state_mid = SchedulerState{
        .backend_loads = [_]f32{0.5} ** 8,
        .memory_pressures = [_]f32{0.5} ** 8,
        .pending_workloads = 50,
        .current_throughput = 500,
    };
    try std.testing.expectEqual(@as(u4, 8), QTable.discretizeState(state_mid));
}

test "q-table exploration decay" {
    var q = QTable.init();
    const initial_rate = q.exploration_rate;

    // Decay multiple times
    for (0..100) |_| {
        q.decayExploration(0.01);
    }

    // Should have decayed
    try std.testing.expect(q.exploration_rate < initial_rate);
    // Should not go below minimum
    try std.testing.expect(q.exploration_rate >= 0.01);
}

test "scheduler schedule" {
    const allocator = std.testing.allocator;
    const coord = try coordinator.Coordinator.init(allocator);
    defer coord.deinit();

    const sched = try LearningScheduler.init(allocator, coord);
    defer sched.deinit();

    const profile = coordinator.WorkloadProfile{
        .compute_intensity = 0.8,
        .memory_requirement_mb = 1024,
        .is_training = true,
    };

    const decision = sched.schedule(profile);

    // Should get a valid decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
}

test "scheduler learning stats" {
    const allocator = std.testing.allocator;
    const coord = try coordinator.Coordinator.init(allocator);
    defer coord.deinit();

    const sched = try LearningScheduler.init(allocator, coord);
    defer sched.deinit();

    const stats = sched.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.episodes);
    try std.testing.expectEqual(@as(f32, 0.0), stats.avg_episode_reward);
    try std.testing.expect(stats.exploration_rate > 0);
}
