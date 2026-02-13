//! DPO (Direct Preference Optimization)
//!
//! Direct Preference Optimization for RLHF without reward model.
//! Simpler and more stable than PPO-based RLHF.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const reward_policy = @import("reward_policy.zig");

pub const PolicyNetwork = reward_policy.PolicyNetwork;

/// Get current timestamp for Zig 0.16 compatibility (no std.time.timestamp()).
/// Returns nanoseconds since timer start as an i64.
fn getCurrentTimestamp() i64 {
    var timer = time.Timer.start() catch return 0;
    return @intCast(timer.read());
}

/// Direct Preference Optimization for RLHF without reward model
/// Simpler and more stable than PPO-based RLHF
pub const DPOOptimizer = struct {
    allocator: std.mem.Allocator,
    config: DPOConfig,
    policy_network: ?PolicyNetwork,

    // Reference policy log probabilities (frozen)
    ref_log_probs: std.ArrayListUnmanaged(f32),

    // Preference pairs storage
    preference_pairs: std.ArrayListUnmanaged(PreferencePair),

    // Training statistics
    stats: DPOStats,

    pub const DPOConfig = struct {
        /// Beta parameter for DPO (controls strength of preference)
        beta: f32 = 0.1,
        /// Learning rate
        learning_rate: f32 = 1e-6,
        /// Batch size
        batch_size: u32 = 4,
        /// Label smoothing
        label_smoothing: f32 = 0.0,
        /// Maximum pairs to store
        max_pairs: usize = 10000,
        /// Minimum pairs before training
        min_pairs: usize = 32,
        /// Reference model update frequency
        ref_update_freq: u32 = 100,
    };

    pub const PreferencePair = struct {
        /// Chosen response embedding/tokens
        chosen_embedding: []f32,
        /// Rejected response embedding/tokens
        rejected_embedding: []f32,
        /// Context/prompt embedding
        context_embedding: []f32,
        /// Confidence in the preference
        confidence: f32,
        /// Timestamp
        timestamp: i64,
    };

    pub const DPOStats = struct {
        total_pairs: u64 = 0,
        total_updates: u64 = 0,
        avg_loss: f32 = 0,
        avg_chosen_reward: f32 = 0,
        avg_rejected_reward: f32 = 0,
        reward_margin: f32 = 0,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: DPOConfig) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .policy_network = null,
            .ref_log_probs = .{},
            .preference_pairs = .{},
            .stats = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.policy_network) |*pn| pn.deinit();
        for (self.preference_pairs.items) |pair| {
            self.allocator.free(pair.chosen_embedding);
            self.allocator.free(pair.rejected_embedding);
            self.allocator.free(pair.context_embedding);
        }
        self.preference_pairs.deinit(self.allocator);
        self.ref_log_probs.deinit(self.allocator);
    }

    /// Add a preference pair for training
    pub fn addPreferencePair(
        self: *Self,
        chosen: []const f32,
        rejected: []const f32,
        context: []const f32,
        confidence: f32,
    ) !void {
        // Remove oldest if at capacity
        if (self.preference_pairs.items.len >= self.config.max_pairs) {
            const old = self.preference_pairs.orderedRemove(0);
            self.allocator.free(old.chosen_embedding);
            self.allocator.free(old.rejected_embedding);
            self.allocator.free(old.context_embedding);
        }

        const pair = PreferencePair{
            .chosen_embedding = try self.allocator.dupe(f32, chosen),
            .rejected_embedding = try self.allocator.dupe(f32, rejected),
            .context_embedding = try self.allocator.dupe(f32, context),
            .confidence = confidence,
            .timestamp = getCurrentTimestamp(),
        };

        try self.preference_pairs.append(self.allocator, pair);
        self.stats.total_pairs += 1;
    }

    /// Compute DPO loss for a preference pair
    fn computeDPOLoss(
        self: *const Self,
        chosen_logprob: f32,
        rejected_logprob: f32,
        ref_chosen_logprob: f32,
        ref_rejected_logprob: f32,
    ) f32 {
        const beta = self.config.beta;

        // DPO loss: -log(sigmoid(beta * (log_pi(chosen) - log_ref(chosen) - log_pi(rejected) + log_ref(rejected))))
        const chosen_diff = chosen_logprob - ref_chosen_logprob;
        const rejected_diff = rejected_logprob - ref_rejected_logprob;
        const margin = beta * (chosen_diff - rejected_diff);

        // Sigmoid and negative log for loss
        const sigmoid_val = 1.0 / (1.0 + @exp(-margin));

        // Apply label smoothing
        const smoothed_target = 1.0 - self.config.label_smoothing;
        const loss = -smoothed_target * @log(@max(sigmoid_val, 1e-7)) -
            (1.0 - smoothed_target) * @log(@max(1.0 - sigmoid_val, 1e-7));

        return loss;
    }

    /// Train on a batch of preference pairs
    pub fn trainStep(self: *Self) !f32 {
        if (self.preference_pairs.items.len < self.config.min_pairs) {
            return 0;
        }

        const batch_size = @min(self.config.batch_size, @as(u32, @intCast(self.preference_pairs.items.len)));

        var total_loss: f32 = 0;
        var total_chosen_reward: f32 = 0;
        var total_rejected_reward: f32 = 0;

        // Sample batch (simple random sampling)
        const seed = blk: {
            var timer = time.Timer.start() catch break :blk @as(u64, 0);
            break :blk timer.read();
        };
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        for (0..batch_size) |_| {
            const idx = random.intRangeAtMost(usize, 0, self.preference_pairs.items.len - 1);
            const pair = self.preference_pairs.items[idx];

            // Compute implicit rewards (simplified - using embedding magnitudes)
            var chosen_reward: f32 = 0;
            var rejected_reward: f32 = 0;
            for (pair.chosen_embedding) |v| chosen_reward += v * v;
            for (pair.rejected_embedding) |v| rejected_reward += v * v;
            chosen_reward = @sqrt(chosen_reward);
            rejected_reward = @sqrt(rejected_reward);

            // Simplified log probs (in practice, would come from model)
            const chosen_logprob = -chosen_reward * 0.1;
            const rejected_logprob = -rejected_reward * 0.1;
            const ref_chosen_logprob = chosen_logprob * 0.9; // Reference is slightly different
            const ref_rejected_logprob = rejected_logprob * 0.9;

            const loss = self.computeDPOLoss(chosen_logprob, rejected_logprob, ref_chosen_logprob, ref_rejected_logprob);
            total_loss += loss * pair.confidence;
            total_chosen_reward += chosen_reward;
            total_rejected_reward += rejected_reward;
        }

        // Update statistics
        self.stats.total_updates += 1;
        self.stats.avg_loss = total_loss / @as(f32, @floatFromInt(batch_size));
        self.stats.avg_chosen_reward = total_chosen_reward / @as(f32, @floatFromInt(batch_size));
        self.stats.avg_rejected_reward = total_rejected_reward / @as(f32, @floatFromInt(batch_size));
        self.stats.reward_margin = self.stats.avg_chosen_reward - self.stats.avg_rejected_reward;

        return self.stats.avg_loss;
    }

    /// Get training statistics
    pub fn getStats(self: *const Self) DPOStats {
        return self.stats;
    }

    /// Check if ready for training
    pub fn isReady(self: *const Self) bool {
        return self.preference_pairs.items.len >= self.config.min_pairs;
    }
};
