//! Abbey Meta-Learning System
//!
//! Learning to learn - adapts learning strategies based on task characteristics:
//! - Task difficulty estimation
//! - Adaptive learning rate scheduling
//! - Learning strategy selection
//! - Knowledge transfer between domains
//! - Few-shot learning optimization

const std = @import("std");
const neural = @import("../neural/mod.zig");
const types = @import("../core/types.zig");

// ============================================================================
// Meta-Learning Types
// ============================================================================

/// Task characteristics for meta-learning
pub const TaskProfile = struct {
    complexity: f32, // 0.0 = simple, 1.0 = highly complex
    novelty: f32, // 0.0 = familiar, 1.0 = completely new
    ambiguity: f32, // 0.0 = clear, 1.0 = highly ambiguous
    time_sensitivity: f32, // 0.0 = timeless, 1.0 = highly time-sensitive
    domain: TaskDomain,
    required_reasoning: ReasoningType,

    pub const TaskDomain = enum {
        factual,
        procedural,
        creative,
        analytical,
        social,
        technical,
        unknown,
    };

    pub const ReasoningType = enum {
        deductive,
        inductive,
        abductive,
        analogical,
        causal,
        counterfactual,
        none,
    };

    pub fn estimateDifficulty(self: TaskProfile) f32 {
        return (self.complexity * 0.3 +
            self.novelty * 0.25 +
            self.ambiguity * 0.25 +
            self.time_sensitivity * 0.2);
    }
};

/// Learning strategy configuration
pub const LearningStrategy = struct {
    base_learning_rate: f32,
    momentum: f32,
    exploration_rate: f32,
    batch_priority: BatchPriority,
    update_frequency: UpdateFrequency,

    pub const BatchPriority = enum {
        recent_first,
        difficult_first,
        diverse_sampling,
        curiosity_driven,
    };

    pub const UpdateFrequency = enum {
        every_step,
        every_episode,
        on_significant_change,
        adaptive,
    };
};

/// Meta-learning statistics
pub const MetaStats = struct {
    total_tasks: usize,
    successful_adaptations: usize,
    failed_adaptations: usize,
    avg_adaptation_time_ms: f32,
    domain_proficiency: [7]f32, // One per TaskDomain
    current_strategy: LearningStrategy,
};

// ============================================================================
// Meta-Learner (MAML-inspired)
// ============================================================================

/// Model-Agnostic Meta-Learner
/// Learns optimal initialization and adaptation strategies
pub const MetaLearner = struct {
    allocator: std.mem.Allocator,

    // Meta-parameters
    meta_learning_rate: f32,
    inner_learning_rate: f32,
    adaptation_steps: usize,

    // Task history for meta-learning
    task_history: std.ArrayListUnmanaged(TaskRecord),
    domain_statistics: [7]DomainStats,

    // Strategy selection network (simplified)
    strategy_weights: neural.F32Tensor,

    // Performance tracking
    cumulative_reward: f32 = 0,
    task_count: usize = 0,

    const Self = @This();

    pub const TaskRecord = struct {
        profile: TaskProfile,
        strategy_used: LearningStrategy,
        success_score: f32,
        adaptation_time_ms: f32,
        timestamp: i64,
    };

    pub const DomainStats = struct {
        task_count: usize = 0,
        avg_success: f32 = 0.5,
        avg_complexity: f32 = 0.5,
        best_strategy_idx: usize = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: MetaConfig) !Self {
        var strategy_weights = try neural.F32Tensor.random(allocator, &.{ 7, 4 }, -0.1, 0.1);
        errdefer strategy_weights.deinit();

        return Self{
            .allocator = allocator,
            .meta_learning_rate = config.meta_learning_rate,
            .inner_learning_rate = config.inner_learning_rate,
            .adaptation_steps = config.adaptation_steps,
            .task_history = .{},
            .domain_statistics = [_]DomainStats{.{}} ** 7,
            .strategy_weights = strategy_weights,
        };
    }

    pub fn deinit(self: *Self) void {
        self.task_history.deinit(self.allocator);
        self.strategy_weights.deinit();
    }

    pub const MetaConfig = struct {
        meta_learning_rate: f32 = 0.001,
        inner_learning_rate: f32 = 0.01,
        adaptation_steps: usize = 5,
    };

    /// Analyze a task and select optimal learning strategy
    pub fn selectStrategy(self: *Self, profile: TaskProfile) LearningStrategy {
        const domain_idx = @intFromEnum(profile.domain);
        const domain_stats = self.domain_statistics[domain_idx];

        // Base strategy from domain statistics
        var strategy = self.getBaseStrategy(domain_stats.best_strategy_idx);

        // Adapt based on task characteristics
        strategy.base_learning_rate *= self.computeLRMultiplier(profile);
        strategy.exploration_rate = self.computeExplorationRate(profile);

        return strategy;
    }

    fn getBaseStrategy(self: *Self, idx: usize) LearningStrategy {
        _ = self;
        return switch (idx % 4) {
            0 => .{
                .base_learning_rate = 0.01,
                .momentum = 0.9,
                .exploration_rate = 0.1,
                .batch_priority = .recent_first,
                .update_frequency = .every_step,
            },
            1 => .{
                .base_learning_rate = 0.001,
                .momentum = 0.95,
                .exploration_rate = 0.2,
                .batch_priority = .difficult_first,
                .update_frequency = .every_episode,
            },
            2 => .{
                .base_learning_rate = 0.005,
                .momentum = 0.85,
                .exploration_rate = 0.3,
                .batch_priority = .diverse_sampling,
                .update_frequency = .on_significant_change,
            },
            else => .{
                .base_learning_rate = 0.002,
                .momentum = 0.9,
                .exploration_rate = 0.25,
                .batch_priority = .curiosity_driven,
                .update_frequency = .adaptive,
            },
        };
    }

    fn computeLRMultiplier(self: *Self, profile: TaskProfile) f32 {
        _ = self;
        // Higher novelty = lower LR (more careful)
        // Higher complexity = lower LR
        const novelty_factor = 1.0 - (profile.novelty * 0.5);
        const complexity_factor = 1.0 - (profile.complexity * 0.3);
        return novelty_factor * complexity_factor;
    }

    fn computeExplorationRate(self: *Self, profile: TaskProfile) f32 {
        _ = self;
        // Higher novelty = higher exploration
        // Higher ambiguity = higher exploration
        return @min(0.5, 0.1 + profile.novelty * 0.2 + profile.ambiguity * 0.2);
    }

    /// Record task outcome for meta-learning
    pub fn recordOutcome(
        self: *Self,
        profile: TaskProfile,
        strategy: LearningStrategy,
        success_score: f32,
        adaptation_time_ms: f32,
    ) !void {
        // Record in history
        try self.task_history.append(self.allocator, .{
            .profile = profile,
            .strategy_used = strategy,
            .success_score = success_score,
            .adaptation_time_ms = adaptation_time_ms,
            .timestamp = types.getTimestampSec(),
        });

        // Update domain statistics
        const domain_idx = @intFromEnum(profile.domain);
        var stats = &self.domain_statistics[domain_idx];
        stats.task_count += 1;
        stats.avg_success = stats.avg_success * 0.9 + success_score * 0.1;
        stats.avg_complexity = stats.avg_complexity * 0.9 + profile.complexity * 0.1;

        // Update cumulative stats
        self.cumulative_reward += success_score;
        self.task_count += 1;

        // Meta-update if enough history
        if (self.task_history.items.len >= 10) {
            try self.metaUpdate();
        }
    }

    /// Perform meta-learning update (learns from task history)
    fn metaUpdate(self: *Self) !void {
        // Compute gradients for strategy selection based on recent performance
        const recent_count = @min(50, self.task_history.items.len);
        const recent = self.task_history.items[self.task_history.items.len - recent_count ..];

        // Aggregate performance by domain
        var domain_rewards: [7]f32 = [_]f32{0} ** 7;
        var domain_counts: [7]usize = [_]usize{0} ** 7;

        for (recent) |record| {
            const idx = @intFromEnum(record.profile.domain);
            domain_rewards[idx] += record.success_score;
            domain_counts[idx] += 1;
        }

        // Update strategy weights using simplified policy gradient
        for (0..7) |i| {
            if (domain_counts[i] > 0) {
                const avg_reward = domain_rewards[i] / @as(f32, @floatFromInt(domain_counts[i]));
                const advantage = avg_reward - 0.5; // Baseline

                // Update weights for this domain
                for (0..4) |j| {
                    const idx = i * 4 + j;
                    if (idx < self.strategy_weights.data.len) {
                        self.strategy_weights.data[idx] += self.meta_learning_rate * advantage;
                    }
                }
            }
        }
    }

    /// Get meta-learning statistics
    pub fn getStats(self: *const Self) MetaStats {
        var successful: usize = 0;
        var failed: usize = 0;
        var total_time: f32 = 0;

        for (self.task_history.items) |record| {
            if (record.success_score >= 0.5) {
                successful += 1;
            } else {
                failed += 1;
            }
            total_time += record.adaptation_time_ms;
        }

        var domain_prof: [7]f32 = undefined;
        for (0..7) |i| {
            domain_prof[i] = self.domain_statistics[i].avg_success;
        }

        return MetaStats{
            .total_tasks = self.task_count,
            .successful_adaptations = successful,
            .failed_adaptations = failed,
            .avg_adaptation_time_ms = if (self.task_count > 0)
                total_time / @as(f32, @floatFromInt(self.task_count))
            else
                0,
            .domain_proficiency = domain_prof,
            .current_strategy = self.getBaseStrategy(0),
        };
    }
};

// ============================================================================
// Few-Shot Learning Module
// ============================================================================

/// Few-shot learning for rapid adaptation
pub const FewShotLearner = struct {
    allocator: std.mem.Allocator,
    support_set: std.ArrayListUnmanaged(Example),
    prototype_dim: usize,
    prototypes: std.StringHashMapUnmanaged(neural.F32Tensor),

    const Self = @This();

    pub const Example = struct {
        embedding: neural.F32Tensor,
        label: []const u8,
        confidence: f32,
    };

    pub fn init(allocator: std.mem.Allocator, dim: usize) Self {
        return Self{
            .allocator = allocator,
            .support_set = .{},
            .prototype_dim = dim,
            .prototypes = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.support_set.items) |*ex| {
            ex.embedding.deinit();
        }
        self.support_set.deinit(self.allocator);

        var it = self.prototypes.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.prototypes.deinit(self.allocator);
    }

    /// Add a support example
    pub fn addExample(self: *Self, embedding: []const f32, label: []const u8) !void {
        const emb = try neural.F32Tensor.fromSlice(self.allocator, embedding, &.{embedding.len});

        try self.support_set.append(self.allocator, .{
            .embedding = emb,
            .label = label,
            .confidence = 1.0,
        });

        // Update prototype for this label
        try self.updatePrototype(label);
    }

    /// Update prototype (mean of embeddings for label)
    fn updatePrototype(self: *Self, label: []const u8) !void {
        var sum = try neural.F32Tensor.zeros(self.allocator, &.{self.prototype_dim});
        defer sum.deinit();

        var count: usize = 0;
        for (self.support_set.items) |ex| {
            if (std.mem.eql(u8, ex.label, label)) {
                for (0..self.prototype_dim) |i| {
                    if (i < sum.data.len and i < ex.embedding.data.len) {
                        sum.data[i] += ex.embedding.data[i];
                    }
                }
                count += 1;
            }
        }

        if (count > 0) {
            const count_f = @as(f32, @floatFromInt(count));
            for (sum.data) |*v| {
                v.* /= count_f;
            }

            // Store prototype
            if (self.prototypes.get(label)) |*old| {
                old.deinit();
            }
            const proto = try sum.clone();
            try self.prototypes.put(self.allocator, label, proto);
        }
    }

    /// Classify using nearest prototype
    pub fn classify(self: *Self, query_embedding: []const f32) ?Classification {
        if (self.prototypes.count() == 0) return null;

        var best_label: ?[]const u8 = null;
        var best_distance: f32 = std.math.inf(f32);

        var it = self.prototypes.iterator();
        while (it.next()) |entry| {
            const dist = self.euclideanDistance(query_embedding, entry.value_ptr.data);
            if (dist < best_distance) {
                best_distance = dist;
                best_label = entry.key_ptr.*;
            }
        }

        if (best_label) |label| {
            // Convert distance to confidence (smaller distance = higher confidence)
            const confidence = 1.0 / (1.0 + best_distance);
            return Classification{
                .label = label,
                .confidence = confidence,
                .distance = best_distance,
            };
        }

        return null;
    }

    fn euclideanDistance(self: *Self, a: []const f32, b: []const f32) f32 {
        _ = self;
        var sum: f32 = 0;
        const len = @min(a.len, b.len);
        for (0..len) |i| {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return @sqrt(sum);
    }

    pub const Classification = struct {
        label: []const u8,
        confidence: f32,
        distance: f32,
    };
};

// ============================================================================
// Curriculum Learning
// ============================================================================

/// Curriculum learning - presents tasks in optimal order
pub const CurriculumScheduler = struct {
    allocator: std.mem.Allocator,
    current_difficulty: f32,
    progression_rate: f32,
    task_queue: std.ArrayListUnmanaged(ScheduledTask),
    completed_count: usize = 0,

    const Self = @This();

    pub const ScheduledTask = struct {
        id: u64,
        difficulty: f32,
        domain: TaskProfile.TaskDomain,
        priority: f32,
    };

    pub fn init(allocator: std.mem.Allocator, initial_difficulty: f32) Self {
        return Self{
            .allocator = allocator,
            .current_difficulty = initial_difficulty,
            .progression_rate = 0.05,
            .task_queue = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.task_queue.deinit(self.allocator);
    }

    /// Schedule a task based on current curriculum position
    pub fn scheduleTask(self: *Self, task: ScheduledTask) !void {
        // Insert in priority order (easier tasks first when early in curriculum)
        var insert_idx: usize = 0;
        for (self.task_queue.items, 0..) |existing, i| {
            const existing_score = self.computeScheduleScore(existing);
            const new_score = self.computeScheduleScore(task);
            if (new_score < existing_score) {
                insert_idx = i;
                break;
            }
            insert_idx = i + 1;
        }

        try self.task_queue.insert(self.allocator, insert_idx, task);
    }

    fn computeScheduleScore(self: *Self, task: ScheduledTask) f32 {
        // Score based on how appropriate task is for current difficulty
        const difficulty_gap = @abs(task.difficulty - self.current_difficulty);
        const priority_boost = task.priority * 0.2;
        return difficulty_gap - priority_boost;
    }

    /// Get next task from curriculum
    pub fn getNextTask(self: *Self) ?ScheduledTask {
        if (self.task_queue.items.len == 0) return null;
        return self.task_queue.orderedRemove(0);
    }

    /// Report task completion to update curriculum
    pub fn completeTask(self: *Self, success: bool) void {
        self.completed_count += 1;

        if (success) {
            // Increase difficulty on success
            self.current_difficulty = @min(1.0, self.current_difficulty + self.progression_rate);
        } else {
            // Decrease difficulty on failure
            self.current_difficulty = @max(0.0, self.current_difficulty - self.progression_rate * 0.5);
        }
    }

    /// Get current curriculum position
    pub fn getProgress(self: *const Self) CurriculumProgress {
        return CurriculumProgress{
            .current_difficulty = self.current_difficulty,
            .tasks_completed = self.completed_count,
            .tasks_pending = self.task_queue.items.len,
        };
    }

    pub const CurriculumProgress = struct {
        current_difficulty: f32,
        tasks_completed: usize,
        tasks_pending: usize,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "meta learner initialization" {
    const allocator = std.testing.allocator;

    var meta = try MetaLearner.init(allocator, .{});
    defer meta.deinit();

    const stats = meta.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_tasks);
}

test "meta learner strategy selection" {
    const allocator = std.testing.allocator;

    var meta = try MetaLearner.init(allocator, .{});
    defer meta.deinit();

    const profile = TaskProfile{
        .complexity = 0.7,
        .novelty = 0.3,
        .ambiguity = 0.5,
        .time_sensitivity = 0.2,
        .domain = .technical,
        .required_reasoning = .deductive,
    };

    const strategy = meta.selectStrategy(profile);
    try std.testing.expect(strategy.base_learning_rate > 0);
    try std.testing.expect(strategy.exploration_rate >= 0);
}

test "few shot learner" {
    const allocator = std.testing.allocator;

    var learner = FewShotLearner.init(allocator, 4);
    defer learner.deinit();

    // Add examples
    try learner.addExample(&[_]f32{ 1.0, 0.0, 0.0, 0.0 }, "class_a");
    try learner.addExample(&[_]f32{ 0.0, 1.0, 0.0, 0.0 }, "class_b");

    // Classify
    const result = learner.classify(&[_]f32{ 0.9, 0.1, 0.0, 0.0 });
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("class_a", result.?.label);
}

test "curriculum scheduler" {
    const allocator = std.testing.allocator;

    var scheduler = CurriculumScheduler.init(allocator, 0.3);
    defer scheduler.deinit();

    try scheduler.scheduleTask(.{ .id = 1, .difficulty = 0.2, .domain = .factual, .priority = 0.5 });
    try scheduler.scheduleTask(.{ .id = 2, .difficulty = 0.8, .domain = .analytical, .priority = 0.3 });

    const next = scheduler.getNextTask();
    try std.testing.expect(next != null);

    scheduler.completeTask(true);
    try std.testing.expect(scheduler.current_difficulty > 0.3);
}
