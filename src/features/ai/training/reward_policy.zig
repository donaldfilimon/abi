//! Reward Model and Policy Network
//!
//! Core RLHF components:
//! - RewardModel: learns reward signals from preference pairs
//! - PolicyNetwork: actor-critic architecture for policy gradient updates

const std = @import("std");
const time = @import("../../../services/shared/time.zig");

// ============================================================================
// Reward Model
// ============================================================================

/// Reward model for RLHF
pub const RewardModel = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    bias: f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, input_dim: usize) !Self {
        const weights = try allocator.alloc(f32, input_dim);
        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_dim)));
        const seed = blk: {
            var timer = time.Timer.start() catch break :blk @as(u64, 42);
            break :blk timer.read();
        };
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (weights) |*w| {
            w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        return .{
            .allocator = allocator,
            .weights = weights,
            .bias = 0.0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.weights);
    }

    /// Compute reward for a response embedding
    pub fn computeReward(self: *const Self, embedding: []const f32) f32 {
        var sum: f32 = self.bias;
        const min_len = @min(embedding.len, self.weights.len);
        for (0..min_len) |i| {
            sum += embedding[i] * self.weights[i];
        }
        // Tanh activation to bound reward
        return std.math.tanh(sum);
    }

    /// Update reward model with preference pairs
    pub fn updateFromPreferences(
        self: *Self,
        chosen: []const f32,
        rejected: []const f32,
        learning_rate: f32,
    ) void {
        const chosen_reward = self.computeReward(chosen);
        const rejected_reward = self.computeReward(rejected);

        // Bradley-Terry loss gradient
        const sigmoid = 1.0 / (1.0 + @exp(-(chosen_reward - rejected_reward)));
        const grad_scale = (1.0 - sigmoid) * learning_rate;

        const min_len = @min(chosen.len, @min(rejected.len, self.weights.len));
        for (0..min_len) |i| {
            self.weights[i] += grad_scale * (chosen[i] - rejected[i]);
        }
        self.bias += grad_scale;
    }
};

// ============================================================================
// Policy Network (Actor-Critic)
// ============================================================================

/// Neural network policy for action selection and value estimation
/// Implements actor-critic architecture for RLHF
pub const PolicyNetwork = struct {
    allocator: std.mem.Allocator,
    config: PolicyConfig,

    // Actor (policy) network weights
    actor_w1: []f32, // [hidden_dim, input_dim]
    actor_b1: []f32, // [hidden_dim]
    actor_w2: []f32, // [hidden_dim, hidden_dim]
    actor_b2: []f32, // [hidden_dim]
    actor_out: []f32, // [output_dim, hidden_dim]
    actor_out_b: []f32, // [output_dim]

    // Critic (value) network weights
    critic_w1: []f32, // [hidden_dim, input_dim]
    critic_b1: []f32, // [hidden_dim]
    critic_w2: []f32, // [hidden_dim, hidden_dim]
    critic_b2: []f32, // [hidden_dim]
    critic_out: []f32, // [1, hidden_dim]
    critic_out_b: f32,

    // Optimizer state (Adam)
    adam_m: ?[]f32, // First moment
    adam_v: ?[]f32, // Second moment
    adam_t: u64, // Timestep

    pub const PolicyConfig = struct {
        input_dim: u32 = 768,
        hidden_dim: u32 = 256,
        output_dim: u32 = 768, // Same as embedding dim for residual connection
        learning_rate: f32 = 3e-4,
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        epsilon: f32 = 1e-8,
        weight_decay: f32 = 0.01,
        max_grad_norm: f32 = 1.0,
        use_layer_norm: bool = true,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: PolicyConfig) !Self {
        const in_dim = config.input_dim;
        const hid_dim = config.hidden_dim;
        const out_dim = config.output_dim;

        var self = Self{
            .allocator = allocator,
            .config = config,
            .actor_w1 = try allocator.alloc(f32, hid_dim * in_dim),
            .actor_b1 = try allocator.alloc(f32, hid_dim),
            .actor_w2 = try allocator.alloc(f32, hid_dim * hid_dim),
            .actor_b2 = try allocator.alloc(f32, hid_dim),
            .actor_out = try allocator.alloc(f32, out_dim * hid_dim),
            .actor_out_b = try allocator.alloc(f32, out_dim),
            .critic_w1 = try allocator.alloc(f32, hid_dim * in_dim),
            .critic_b1 = try allocator.alloc(f32, hid_dim),
            .critic_w2 = try allocator.alloc(f32, hid_dim * hid_dim),
            .critic_b2 = try allocator.alloc(f32, hid_dim),
            .critic_out = try allocator.alloc(f32, hid_dim),
            .critic_out_b = 0.0,
            .adam_m = null,
            .adam_v = null,
            .adam_t = 0,
        };

        // Initialize weights with Xavier/He initialization
        self.initializeWeights();

        return self;
    }

    fn initializeWeights(self: *Self) void {
        const seed = blk: {
            var timer = time.Timer.start() catch break :blk @as(u64, 12345);
            break :blk timer.read();
        };
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        // He initialization for ReLU/GELU
        const scale1 = @sqrt(2.0 / @as(f32, @floatFromInt(self.config.input_dim)));
        const scale2 = @sqrt(2.0 / @as(f32, @floatFromInt(self.config.hidden_dim)));

        for (self.actor_w1) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale1;
        for (self.actor_w2) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale2;
        for (self.actor_out) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale2 * 0.01;
        for (self.critic_w1) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale1;
        for (self.critic_w2) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale2;
        for (self.critic_out) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale2 * 0.01;

        @memset(self.actor_b1, 0.0);
        @memset(self.actor_b2, 0.0);
        @memset(self.actor_out_b, 0.0);
        @memset(self.critic_b1, 0.0);
        @memset(self.critic_b2, 0.0);
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.actor_w1);
        self.allocator.free(self.actor_b1);
        self.allocator.free(self.actor_w2);
        self.allocator.free(self.actor_b2);
        self.allocator.free(self.actor_out);
        self.allocator.free(self.actor_out_b);
        self.allocator.free(self.critic_w1);
        self.allocator.free(self.critic_b1);
        self.allocator.free(self.critic_w2);
        self.allocator.free(self.critic_b2);
        self.allocator.free(self.critic_out);
        if (self.adam_m) |m| self.allocator.free(m);
        if (self.adam_v) |v| self.allocator.free(v);
    }

    /// GELU activation
    fn gelu(x: f32) f32 {
        const sqrt_2_over_pi: f32 = 0.7978845608;
        const coeff: f32 = 0.044715;
        const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        return 0.5 * x * (1.0 + std.math.tanh(inner));
    }

    /// Forward pass through actor network
    /// Returns policy logits/embedding adjustment
    pub fn actorForward(self: *const Self, state: []const f32) ![]f32 {
        const hid_dim = self.config.hidden_dim;
        const out_dim = self.config.output_dim;
        const in_dim = self.config.input_dim;

        // First layer
        const h1 = try self.allocator.alloc(f32, hid_dim);
        defer self.allocator.free(h1);

        for (0..hid_dim) |i| {
            var sum: f32 = self.actor_b1[i];
            for (0..@min(in_dim, state.len)) |j| {
                sum += state[j] * self.actor_w1[i * in_dim + j];
            }
            h1[i] = gelu(sum);
        }

        // Second layer
        const h2 = try self.allocator.alloc(f32, hid_dim);
        defer self.allocator.free(h2);

        for (0..hid_dim) |i| {
            var sum: f32 = self.actor_b2[i];
            for (0..hid_dim) |j| {
                sum += h1[j] * self.actor_w2[i * hid_dim + j];
            }
            h2[i] = gelu(sum);
        }

        // Output layer
        const output = try self.allocator.alloc(f32, out_dim);
        for (0..out_dim) |i| {
            var sum: f32 = self.actor_out_b[i];
            for (0..hid_dim) |j| {
                sum += h2[j] * self.actor_out[i * hid_dim + j];
            }
            output[i] = sum; // No activation - used as logit adjustment
        }

        return output;
    }

    /// Forward pass through critic network
    /// Returns value estimate
    pub fn criticForward(self: *const Self, state: []const f32) !f32 {
        const hid_dim = self.config.hidden_dim;
        const in_dim = self.config.input_dim;

        // First layer
        const h1 = try self.allocator.alloc(f32, hid_dim);
        defer self.allocator.free(h1);

        for (0..hid_dim) |i| {
            var sum: f32 = self.critic_b1[i];
            for (0..@min(in_dim, state.len)) |j| {
                sum += state[j] * self.critic_w1[i * in_dim + j];
            }
            h1[i] = gelu(sum);
        }

        // Second layer
        const h2 = try self.allocator.alloc(f32, hid_dim);
        defer self.allocator.free(h2);

        for (0..hid_dim) |i| {
            var sum: f32 = self.critic_b2[i];
            for (0..hid_dim) |j| {
                sum += h1[j] * self.critic_w2[i * hid_dim + j];
            }
            h2[i] = gelu(sum);
        }

        // Output layer (single value)
        var value: f32 = self.critic_out_b;
        for (0..hid_dim) |j| {
            value += h2[j] * self.critic_out[j];
        }

        return value;
    }

    /// Compute advantage using GAE (Generalized Advantage Estimation)
    pub fn computeGAE(
        self: *const Self,
        rewards: []const f32,
        values: []const f32,
        dones: []const bool,
        gamma: f32,
        gae_lambda: f32,
    ) ![]f32 {
        const n = rewards.len;
        if (n == 0) return &[_]f32{};

        const advantages = try self.allocator.alloc(f32, n);
        var last_gae: f32 = 0;

        var i: usize = n;
        while (i > 0) {
            i -= 1;
            const next_value = if (i + 1 < n) values[i + 1] else 0;
            const next_non_terminal: f32 = if (i + 1 < n and !dones[i + 1]) 1.0 else 0.0;

            const delta = rewards[i] + gamma * next_value * next_non_terminal - values[i];
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
            advantages[i] = last_gae;
        }

        return advantages;
    }

    /// Get total parameter count
    pub fn numParams(self: *const Self) usize {
        const in_dim = self.config.input_dim;
        const hid_dim = self.config.hidden_dim;
        const out_dim = self.config.output_dim;

        const actor_params = hid_dim * in_dim + hid_dim + // w1, b1
            hid_dim * hid_dim + hid_dim + // w2, b2
            out_dim * hid_dim + out_dim; // out, out_b

        const critic_params = hid_dim * in_dim + hid_dim + // w1, b1
            hid_dim * hid_dim + hid_dim + // w2, b2
            hid_dim + 1; // out, out_b

        return actor_params + critic_params;
    }
};
