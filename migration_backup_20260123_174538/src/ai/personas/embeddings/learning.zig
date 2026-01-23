//! Adaptive Learning Module for Personas
//!
//! Tracks routing decisions and interaction outcomes to dynamically adjust
//! persona selection weights over time. This enables the system to "learn"
//! which personas perform best for certain users or domains.
//!
//! Features:
//! - Exponential moving average for smooth weight updates
//! - Domain-specific weight tracking
//! - User preference learning
//! - Trend analysis for routing optimization

const std = @import("std");
const types = @import("../types.zig");

/// Result of a single interaction with a persona.
pub const InteractionResult = struct {
    /// The persona that was used.
    persona: types.PersonaType,
    /// Success score from user feedback or automated evaluation (0.0 to 1.0).
    success_score: f32,
    /// Processing latency in milliseconds.
    latency_ms: u64 = 0,
    /// Unix timestamp of the interaction.
    timestamp: i64,
    /// Optional domain/category for domain-specific learning.
    domain: ?[]const u8 = null,
    /// Optional user/session ID for personalized learning.
    user_id: ?[]const u8 = null,
    /// Whether this was an explicit user correction.
    is_correction: bool = false,
    /// The persona that should have been used (for corrections).
    corrected_to: ?types.PersonaType = null,
};

/// Configuration for the adaptive learner.
pub const LearnerConfig = struct {
    /// Maximum number of interactions to retain in memory.
    max_history: usize = 1000,
    /// How quickly the model adapts to new feedback (0.0 to 1.0).
    learning_rate: f32 = 0.1,
    /// How much weight is decayed over time toward 1.0 (baseline).
    decay_rate: f32 = 0.95,
    /// Minimum weight to prevent complete suppression of a persona.
    min_weight: f32 = 0.5,
    /// Maximum weight to prevent over-reliance on a single persona.
    max_weight: f32 = 1.5,
    /// Whether to enable domain-specific learning.
    enable_domain_learning: bool = true,
    /// Whether to enable user-specific learning.
    enable_user_learning: bool = true,
    /// How often to apply decay (in interactions).
    decay_interval: usize = 100,
};

/// Domain-specific weight tracking.
pub const DomainWeights = struct {
    domain: []const u8,
    weights: std.AutoHashMapUnmanaged(types.PersonaType, f32),
    interaction_count: usize = 0,
};

/// Logic for adjusting persona weights based on historical success.
pub const AdaptiveLearner = struct {
    allocator: std.mem.Allocator,
    /// Configuration.
    config: LearnerConfig,
    /// Map of persona types to their current success weights (centered at 1.0).
    weights: std.AutoHashMapUnmanaged(types.PersonaType, f32),
    /// Interaction history for trend analysis.
    history: std.ArrayListUnmanaged(InteractionResult),
    /// Domain-specific weight maps.
    domain_weights: std.StringHashMapUnmanaged(DomainWeights),
    /// User-specific weight maps.
    user_weights: std.StringHashMapUnmanaged(std.AutoHashMapUnmanaged(types.PersonaType, f32)),
    /// Correction tracking for learning from mistakes.
    correction_counts: std.AutoHashMapUnmanaged(types.PersonaType, CorrectionStats),
    /// Total interactions processed.
    total_interactions: usize = 0,

    const Self = @This();

    /// Statistics about routing corrections.
    pub const CorrectionStats = struct {
        /// How many times this persona was incorrectly chosen.
        incorrect_count: u32 = 0,
        /// How many times this persona was the correct choice.
        correct_count: u32 = 0,
    };

    /// Initialize a new adaptive learner instance.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: LearnerConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .weights = .{},
            .history = .{},
            .domain_weights = .{},
            .user_weights = .{},
            .correction_counts = .{},
        };
    }

    /// Deinitialize the learner and free all memory.
    pub fn deinit(self: *Self) void {
        self.weights.deinit(self.allocator);
        self.history.deinit(self.allocator);

        // Free domain weights
        var domain_it = self.domain_weights.valueIterator();
        while (domain_it.next()) |dw| {
            dw.weights.deinit(self.allocator);
        }
        self.domain_weights.deinit(self.allocator);

        // Free user weights
        var user_it = self.user_weights.valueIterator();
        while (user_it.next()) |uw| {
            uw.deinit(self.allocator);
        }
        self.user_weights.deinit(self.allocator);

        self.correction_counts.deinit(self.allocator);
    }

    /// Record a new interaction result and update internal weights.
    pub fn recordInteraction(self: *Self, result: InteractionResult) !void {
        self.total_interactions += 1;

        // Maintain sliding window history
        if (self.history.items.len >= self.config.max_history) {
            _ = self.history.orderedRemove(0);
        }
        try self.history.append(self.allocator, result);

        // Update global weights
        try self.updateWeight(result.persona, result.success_score);

        // Update domain-specific weights
        if (self.config.enable_domain_learning) {
            if (result.domain) |domain| {
                try self.updateDomainWeight(domain, result.persona, result.success_score);
            }
        }

        // Update user-specific weights
        if (self.config.enable_user_learning) {
            if (result.user_id) |user| {
                try self.updateUserWeight(user, result.persona, result.success_score);
            }
        }

        // Track corrections
        if (result.is_correction) {
            try self.recordCorrection(result);
        }

        // Periodically apply global weight decay to prevent extreme biases
        if (self.total_interactions % self.config.decay_interval == 0) {
            self.applyDecay();
        }
    }

    /// Update individual persona weight.
    fn updateWeight(self: *Self, persona: types.PersonaType, success_score: f32) !void {
        const current = self.weights.get(persona) orelse 1.0;

        // Adjust weight toward the outcome: new = old + lr * (target - old)
        // We normalize the target weight so that a score of 0.5 is "neutral" (1.0 weight)
        // 0.0 score -> 0.5 weight, 1.0 score -> 1.5 weight
        const target_weight = 0.5 + success_score;
        var updated = current + self.config.learning_rate * (target_weight - current);

        // Clamp to configured bounds
        updated = std.math.clamp(updated, self.config.min_weight, self.config.max_weight);

        try self.weights.put(self.allocator, persona, updated);
    }

    /// Update domain-specific weight.
    fn updateDomainWeight(self: *Self, domain: []const u8, persona: types.PersonaType, score: f32) !void {
        const entry = try self.domain_weights.getOrPut(self.allocator, domain);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{
                .domain = domain,
                .weights = .{},
            };
        }

        const current = entry.value_ptr.weights.get(persona) orelse 1.0;
        const target = 0.5 + score;
        var updated = current + self.config.learning_rate * (target - current);
        updated = std.math.clamp(updated, self.config.min_weight, self.config.max_weight);

        try entry.value_ptr.weights.put(self.allocator, persona, updated);
        entry.value_ptr.interaction_count += 1;
    }

    /// Update user-specific weight.
    fn updateUserWeight(self: *Self, user_id: []const u8, persona: types.PersonaType, score: f32) !void {
        const entry = try self.user_weights.getOrPut(self.allocator, user_id);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{};
        }

        const current = entry.value_ptr.get(persona) orelse 1.0;
        const target = 0.5 + score;
        var updated = current + self.config.learning_rate * (target - current);
        updated = std.math.clamp(updated, self.config.min_weight, self.config.max_weight);

        try entry.value_ptr.put(self.allocator, persona, updated);
    }

    /// Record a correction for learning.
    fn recordCorrection(self: *Self, result: InteractionResult) !void {
        // Track that this persona was wrong
        const incorrect = try self.correction_counts.getOrPut(self.allocator, result.persona);
        if (!incorrect.found_existing) {
            incorrect.value_ptr.* = .{};
        }
        incorrect.value_ptr.incorrect_count += 1;

        // Track that the corrected persona was right
        if (result.corrected_to) |correct_persona| {
            const correct = try self.correction_counts.getOrPut(self.allocator, correct_persona);
            if (!correct.found_existing) {
                correct.value_ptr.* = .{};
            }
            correct.value_ptr.correct_count += 1;

            // Apply stronger weight adjustment for explicit corrections
            const correction_boost = self.config.learning_rate * 2.0;

            // Penalize the incorrectly chosen persona
            if (self.weights.get(result.persona)) |w| {
                const new_w = std.math.clamp(w - correction_boost, self.config.min_weight, self.config.max_weight);
                try self.weights.put(self.allocator, result.persona, new_w);
            }

            // Boost the correct persona
            if (self.weights.get(correct_persona)) |w| {
                const new_w = std.math.clamp(w + correction_boost, self.config.min_weight, self.config.max_weight);
                try self.weights.put(self.allocator, correct_persona, new_w);
            }
        }
    }

    /// Retrieve the current weight boost for a specific persona.
    /// Returns 1.0 if no learning data exists.
    pub fn getWeight(self: *const Self, persona: types.PersonaType) f32 {
        return self.weights.get(persona) orelse 1.0;
    }

    /// Get weight considering domain context.
    pub fn getWeightForDomain(self: *const Self, persona: types.PersonaType, domain: ?[]const u8) f32 {
        var weight = self.weights.get(persona) orelse 1.0;

        if (domain) |d| {
            if (self.domain_weights.get(d)) |dw| {
                const domain_weight = dw.weights.get(persona) orelse 1.0;
                // Blend global and domain weights (60% domain, 40% global)
                weight = domain_weight * 0.6 + weight * 0.4;
            }
        }

        return weight;
    }

    /// Get weight considering user context.
    pub fn getWeightForUser(self: *const Self, persona: types.PersonaType, user_id: ?[]const u8) f32 {
        var weight = self.weights.get(persona) orelse 1.0;

        if (user_id) |uid| {
            if (self.user_weights.get(uid)) |uw| {
                const user_weight = uw.get(persona) orelse 1.0;
                // Blend global and user weights (70% user, 30% global)
                weight = user_weight * 0.7 + weight * 0.3;
            }
        }

        return weight;
    }

    /// Get combined weight considering all context.
    pub fn getCombinedWeight(
        self: *const Self,
        persona: types.PersonaType,
        domain: ?[]const u8,
        user_id: ?[]const u8,
    ) f32 {
        const global = self.weights.get(persona) orelse 1.0;

        var domain_contrib: f32 = global;
        var user_contrib: f32 = global;

        if (domain) |d| {
            if (self.domain_weights.get(d)) |dw| {
                domain_contrib = dw.weights.get(persona) orelse 1.0;
            }
        }

        if (user_id) |uid| {
            if (self.user_weights.get(uid)) |uw| {
                user_contrib = uw.get(persona) orelse 1.0;
            }
        }

        // Weighted combination: 40% user, 30% domain, 30% global
        return user_contrib * 0.4 + domain_contrib * 0.3 + global * 0.3;
    }

    /// Update all weights to move closer back to the default 1.0 baseline.
    fn applyDecay(self: *Self) void {
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            const current = entry.value_ptr.*;
            // Move weight toward 1.0 by the decay rate
            entry.value_ptr.* = 1.0 + (current - 1.0) * self.config.decay_rate;
        }

        // Also decay domain weights
        var domain_it = self.domain_weights.valueIterator();
        while (domain_it.next()) |dw| {
            var wit = dw.weights.iterator();
            while (wit.next()) |entry| {
                const current = entry.value_ptr.*;
                entry.value_ptr.* = 1.0 + (current - 1.0) * self.config.decay_rate;
            }
        }
    }

    /// Get current average success rate for a specific persona from recent history.
    pub fn getAverageSuccess(self: *const Self, persona: types.PersonaType) f32 {
        var count: usize = 0;
        var sum: f32 = 0;

        for (self.history.items) |item| {
            if (item.persona == persona) {
                count += 1;
                sum += item.success_score;
            }
        }

        if (count == 0) return 0.5;
        return sum / @as(f32, @floatFromInt(count));
    }

    /// Get correction statistics for a persona.
    pub fn getCorrectionStats(self: *const Self, persona: types.PersonaType) ?CorrectionStats {
        return self.correction_counts.get(persona);
    }

    /// Get accuracy rate based on corrections.
    pub fn getAccuracyRate(self: *const Self, persona: types.PersonaType) f32 {
        const stats = self.correction_counts.get(persona) orelse return 1.0;
        const total = stats.correct_count + stats.incorrect_count;
        if (total == 0) return 1.0;
        return @as(f32, @floatFromInt(stats.correct_count)) / @as(f32, @floatFromInt(total));
    }

    /// Analyze trend for a persona over recent interactions.
    pub fn analyzeTrend(self: *const Self, persona: types.PersonaType, window: usize) TrendAnalysis {
        const items = self.history.items;
        if (items.len == 0) return .{ .trend = .stable, .change = 0.0 };

        const actual_window = @min(window, items.len);
        const start = items.len - actual_window;

        var first_half_sum: f32 = 0;
        var first_half_count: usize = 0;
        var second_half_sum: f32 = 0;
        var second_half_count: usize = 0;

        const midpoint = start + actual_window / 2;

        for (items[start..], start..) |item, i| {
            if (item.persona != persona) continue;

            if (i < midpoint) {
                first_half_sum += item.success_score;
                first_half_count += 1;
            } else {
                second_half_sum += item.success_score;
                second_half_count += 1;
            }
        }

        if (first_half_count == 0 or second_half_count == 0) {
            return .{ .trend = .stable, .change = 0.0 };
        }

        const first_avg = first_half_sum / @as(f32, @floatFromInt(first_half_count));
        const second_avg = second_half_sum / @as(f32, @floatFromInt(second_half_count));
        const change = second_avg - first_avg;

        return .{
            .trend = if (change > 0.1) .improving else if (change < -0.1) .declining else .stable,
            .change = change,
        };
    }

    /// Clear all learned data and reset history.
    pub fn reset(self: *Self) void {
        self.weights.clearRetainingCapacity();
        self.history.clearRetainingCapacity();

        var domain_it = self.domain_weights.valueIterator();
        while (domain_it.next()) |dw| {
            dw.weights.clearRetainingCapacity();
        }

        var user_it = self.user_weights.valueIterator();
        while (user_it.next()) |uw| {
            uw.clearRetainingCapacity();
        }

        self.correction_counts.clearRetainingCapacity();
        self.total_interactions = 0;
    }

    /// Get statistics about the learner.
    pub fn getStats(self: *const Self) LearnerStats {
        return .{
            .total_interactions = self.total_interactions,
            .history_size = self.history.items.len,
            .domain_count = self.domain_weights.count(),
            .user_count = self.user_weights.count(),
        };
    }
};

/// Trend direction.
pub const TrendDirection = enum {
    improving,
    stable,
    declining,
};

/// Result of trend analysis.
pub const TrendAnalysis = struct {
    trend: TrendDirection,
    change: f32,
};

/// Statistics about the learner.
pub const LearnerStats = struct {
    total_interactions: usize,
    history_size: usize,
    domain_count: usize,
    user_count: usize,
};

// Tests

test "AdaptiveLearner weight updates" {
    const allocator = std.testing.allocator;
    var learner = AdaptiveLearner.init(allocator);
    defer learner.deinit();

    // Record a successful interaction
    try learner.recordInteraction(.{
        .persona = .abbey,
        .success_score = 0.9,
        .timestamp = 1000,
    });

    const weight = learner.getWeight(.abbey);
    // Weight should be above 1.0 for successful interaction
    try std.testing.expect(weight > 1.0);
}

test "AdaptiveLearner domain weights" {
    const allocator = std.testing.allocator;
    var learner = AdaptiveLearner.init(allocator);
    defer learner.deinit();

    // Record interactions in "code" domain favoring Aviva
    try learner.recordInteraction(.{
        .persona = .aviva,
        .success_score = 0.95,
        .timestamp = 1000,
        .domain = "code",
    });

    const weight = learner.getWeightForDomain(.aviva, "code");
    try std.testing.expect(weight > 1.0);
}

test "AdaptiveLearner correction learning" {
    const allocator = std.testing.allocator;
    var learner = AdaptiveLearner.init(allocator);
    defer learner.deinit();

    // Initial weight
    try learner.recordInteraction(.{
        .persona = .aviva,
        .success_score = 0.5,
        .timestamp = 1000,
    });

    const before = learner.getWeight(.aviva);

    // Record a correction
    try learner.recordInteraction(.{
        .persona = .aviva,
        .success_score = 0.0,
        .timestamp = 2000,
        .is_correction = true,
        .corrected_to = .abbey,
    });

    const after = learner.getWeight(.aviva);

    // Weight should decrease after correction
    try std.testing.expect(after < before);
}

test "AdaptiveLearner trend analysis" {
    const allocator = std.testing.allocator;
    var learner = AdaptiveLearner.init(allocator);
    defer learner.deinit();

    // Record improving trend
    for (0..10) |i| {
        try learner.recordInteraction(.{
            .persona = .abbey,
            .success_score = 0.5 + @as(f32, @floatFromInt(i)) * 0.05, // 0.5 -> 0.95
            .timestamp = @intCast(i * 1000),
        });
    }

    const trend = learner.analyzeTrend(.abbey, 10);
    try std.testing.expect(trend.trend == .improving);
}
