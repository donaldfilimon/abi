//! Self-Learning Module for Ava
//!
//! Enables autonomous learning and improvement through:
//! - Reinforcement Learning from Human Feedback (RLHF)
//! - Vision and document understanding training
//! - Continuous experience replay
//! - Self-evaluation and correction
//! - Multi-modal learning (text, images, documents)
//!
//! Architecture:
//! ```
//!  ┌─────────────────────────────────────────────────────────────┐
//!  │                    Self-Learning System                      │
//!  ├─────────────────────────────────────────────────────────────┤
//!  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
//!  │  │   Feedback    │  │   Vision      │  │   Document    │   │
//!  │  │   Collector   │  │   Trainer     │  │   Trainer     │   │
//!  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘   │
//!  │          │                  │                  │           │
//!  │          ▼                  ▼                  ▼           │
//!  │  ┌─────────────────────────────────────────────────────┐   │
//!  │  │              Experience Replay Buffer                │   │
//!  │  └─────────────────────────┬───────────────────────────┘   │
//!  │                            │                               │
//!  │                            ▼                               │
//!  │  ┌─────────────────────────────────────────────────────┐   │
//!  │  │              Policy Gradient Optimizer               │   │
//!  │  └─────────────────────────┬───────────────────────────┘   │
//!  │                            │                               │
//!  │                            ▼                               │
//!  │  ┌─────────────────────────────────────────────────────┐   │
//!  │  │              Self-Improvement Loop                   │   │
//!  │  └─────────────────────────────────────────────────────┘   │
//!  └─────────────────────────────────────────────────────────────┘
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const trainable_model = @import("trainable_model.zig");
const loss_mod = @import("loss.zig");
const gradient = @import("gradient.zig");
const logging = @import("logging.zig");

/// Get current timestamp for Zig 0.16 compatibility (no std.time.timestamp()).
/// Returns nanoseconds since timer start as an i64.
fn getCurrentTimestamp() i64 {
    var timer = time.Timer.start() catch return 0;
    return @intCast(timer.read());
}

// ============================================================================
// Self-Learning Configuration
// ============================================================================

/// Configuration for self-learning training
pub const SelfLearningConfig = struct {
    /// Enable RLHF training
    enable_rlhf: bool = true,
    /// Enable vision training
    enable_vision: bool = true,
    /// Enable document training
    enable_documents: bool = true,
    /// Experience replay buffer size
    replay_buffer_size: usize = 10000,
    /// Batch size for training
    batch_size: u32 = 16,
    /// Learning rate for policy updates
    learning_rate: f32 = 1e-6,
    /// Discount factor for rewards
    gamma: f32 = 0.99,
    /// PPO clipping parameter
    ppo_clip: f32 = 0.2,
    /// Value function coefficient
    value_coef: f32 = 0.5,
    /// Entropy bonus coefficient
    entropy_coef: f32 = 0.01,
    /// KL divergence target
    kl_target: f32 = 0.01,
    /// Maximum gradient norm
    max_grad_norm: f32 = 0.5,
    /// Number of PPO epochs per update
    ppo_epochs: u32 = 4,
    /// Minimum buffer size before training
    min_buffer_size: usize = 100,
    /// Update frequency (experiences between updates)
    update_frequency: usize = 64,
    /// Enable reward shaping
    reward_shaping: bool = true,
    /// Self-evaluation threshold
    self_eval_threshold: f32 = 0.7,
    /// Checkpoint interval (updates)
    checkpoint_interval: u32 = 100,
    /// Enable continuous learning
    continuous_learning: bool = true,
};

// ============================================================================
// Learning Experience Types
// ============================================================================

/// Type of learning experience
pub const ExperienceType = enum {
    /// Text-based conversation
    text_conversation,
    /// Image understanding
    vision,
    /// Document parsing
    document,
    /// Code generation
    code,
    /// Reasoning task
    reasoning,
    /// Multi-modal (combined)
    multi_modal,
};

/// Feedback type from user or self-evaluation
pub const FeedbackType = enum {
    /// Explicit positive feedback
    positive,
    /// Explicit negative feedback
    negative,
    /// Implicit acceptance (no correction)
    implicit_accept,
    /// Implicit rejection (correction provided)
    implicit_reject,
    /// Self-evaluation rating
    self_eval,
    /// No feedback available
    none,
};

/// A learning experience for replay
pub const LearningExperience = struct {
    /// Unique experience ID
    id: u64,
    /// Type of experience
    exp_type: ExperienceType,
    /// Input tokens or embedding
    input: []const u32,
    /// Output tokens generated
    output: []const u32,
    /// Reward signal (-1 to 1)
    reward: f32,
    /// Confidence in the response
    confidence: f32,
    /// Feedback type
    feedback: FeedbackType,
    /// Timestamp
    timestamp: i64,
    /// Token probabilities (for PPO)
    log_probs: ?[]const f32,
    /// Value estimate
    value: f32,
    /// Advantage estimate
    advantage: f32,
    /// Is terminal state
    done: bool,
    /// Optional image data (for vision)
    image_data: ?[]const u8,
    /// Optional document content
    document_content: ?[]const u8,
    /// Metadata
    metadata: ExperienceMetadata,

    pub const ExperienceMetadata = struct {
        topic: []const u8 = "",
        user_id: u64 = 0,
        session_id: u64 = 0,
        latency_ms: u64 = 0,
        model_version: u32 = 1,
    };

    pub fn deinit(self: *LearningExperience, allocator: std.mem.Allocator) void {
        allocator.free(self.input);
        allocator.free(self.output);
        if (self.log_probs) |lp| allocator.free(lp);
        if (self.image_data) |img| allocator.free(img);
        if (self.document_content) |doc| allocator.free(doc);
    }
};

// ============================================================================
// Experience Replay Buffer
// ============================================================================

/// Priority experience replay buffer with importance sampling
pub const ExperienceBuffer = struct {
    allocator: std.mem.Allocator,
    experiences: std.ArrayListUnmanaged(LearningExperience),
    priorities: std.ArrayListUnmanaged(f32),
    capacity: usize,
    total_priority: f64,
    alpha: f32, // Priority exponent
    beta: f32, // Importance sampling exponent
    beta_increment: f32,
    next_id: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: usize) Self {
        return .{
            .allocator = allocator,
            .experiences = .{},
            .priorities = .{},
            .capacity = capacity,
            .total_priority = 0,
            .alpha = 0.6,
            .beta = 0.4,
            .beta_increment = 0.001,
            .next_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.experiences.items) |*exp| {
            exp.deinit(self.allocator);
        }
        self.experiences.deinit(self.allocator);
        self.priorities.deinit(self.allocator);
    }

    /// Add experience with initial priority
    pub fn add(self: *Self, experience: LearningExperience) !void {
        const max_priority: f32 = if (self.priorities.items.len > 0) blk: {
            var max: f32 = 0;
            for (self.priorities.items) |p| {
                if (p > max) max = p;
            }
            break :blk max;
        } else 1.0;

        if (self.experiences.items.len >= self.capacity) {
            // Remove lowest priority experience
            var min_idx: usize = 0;
            var min_priority: f32 = self.priorities.items[0];
            for (self.priorities.items, 0..) |p, i| {
                if (p < min_priority) {
                    min_priority = p;
                    min_idx = i;
                }
            }
            self.total_priority -= std.math.pow(f64, min_priority, self.alpha);
            self.experiences.items[min_idx].deinit(self.allocator);
            _ = self.experiences.orderedRemove(min_idx);
            _ = self.priorities.orderedRemove(min_idx);
        }

        var exp_copy = experience;
        exp_copy.id = self.next_id;
        self.next_id += 1;

        try self.experiences.append(self.allocator, exp_copy);
        try self.priorities.append(self.allocator, max_priority);
        self.total_priority += std.math.pow(f64, max_priority, self.alpha);
    }

    /// Sample a batch with priority weighting
    pub fn sample(self: *Self, batch_size: usize) !SampledBatch {
        if (self.experiences.items.len < batch_size) {
            return error.InsufficientExperiences;
        }

        var indices = try self.allocator.alloc(usize, batch_size);
        errdefer self.allocator.free(indices);
        var weights = try self.allocator.alloc(f32, batch_size);
        errdefer self.allocator.free(weights);

        const n = self.experiences.items.len;
        const segment = self.total_priority / @as(f64, @floatFromInt(batch_size));

        // Get time-based seed
        const seed = blk: {
            var timer = time.Timer.start() catch break :blk @as(u64, 0);
            break :blk timer.read();
        };
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        for (0..batch_size) |i| {
            const lower = segment * @as(f64, @floatFromInt(i));
            const upper = segment * @as(f64, @floatFromInt(i + 1));
            const sample_val = lower + random.float(f64) * (upper - lower);

            var cumsum: f64 = 0;
            var idx: usize = 0;
            for (self.priorities.items, 0..) |p, j| {
                cumsum += std.math.pow(f64, p, self.alpha);
                if (cumsum >= sample_val) {
                    idx = j;
                    break;
                }
            }
            indices[i] = idx;

            // Importance sampling weight
            const prob = std.math.pow(f64, self.priorities.items[idx], self.alpha) / self.total_priority;
            const weight = std.math.pow(f32, @floatCast(1.0 / (@as(f64, @floatFromInt(n)) * prob)), self.beta);
            weights[i] = weight;
        }

        // Normalize weights
        var max_weight: f32 = 0;
        for (weights) |w| {
            if (w > max_weight) max_weight = w;
        }
        if (max_weight > 0) {
            for (weights) |*w| {
                w.* /= max_weight;
            }
        }

        // Increment beta towards 1
        self.beta = @min(1.0, self.beta + self.beta_increment);

        return .{
            .indices = indices,
            .weights = weights,
            .experiences = self.experiences.items,
            .allocator = self.allocator,
        };
    }

    /// Update priorities after training
    pub fn updatePriorities(self: *Self, indices: []const usize, td_errors: []const f32) void {
        const epsilon: f32 = 1e-6;
        for (indices, td_errors) |idx, td_err| {
            const old_priority = self.priorities.items[idx];
            const new_priority = @abs(td_err) + epsilon;
            self.total_priority -= std.math.pow(f64, old_priority, self.alpha);
            self.priorities.items[idx] = new_priority;
            self.total_priority += std.math.pow(f64, new_priority, self.alpha);
        }
    }

    pub fn len(self: *const Self) usize {
        return self.experiences.items.len;
    }
};

pub const SampledBatch = struct {
    indices: []usize,
    weights: []f32,
    experiences: []LearningExperience,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SampledBatch) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.weights);
    }
};

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

// ============================================================================
// DPO (Direct Preference Optimization)
// ============================================================================

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

// ============================================================================
// Continuous Learning Integration
// ============================================================================

/// Feedback integrator that connects agent execution to learning
pub const FeedbackIntegrator = struct {
    allocator: std.mem.Allocator,
    self_learning: ?*SelfLearningSystem,
    dpo_optimizer: ?*DPOOptimizer,
    policy_network: ?*PolicyNetwork,

    /// Pending feedback entries
    pending_feedback: std.ArrayListUnmanaged(PendingFeedback),

    /// Session tracking
    session_count: u64,
    current_session_id: u64,

    pub const PendingFeedback = struct {
        input_embedding: []f32,
        output_embedding: []f32,
        timestamp: i64,
        latency_ms: u64,
        awaiting_feedback: bool,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .self_learning = null,
            .dpo_optimizer = null,
            .policy_network = null,
            .pending_feedback = .{},
            .session_count = 0,
            .current_session_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.pending_feedback.items) |entry| {
            self.allocator.free(entry.input_embedding);
            self.allocator.free(entry.output_embedding);
        }
        self.pending_feedback.deinit(self.allocator);
    }

    /// Connect to a self-learning system
    pub fn connectSelfLearning(self: *Self, sl: *SelfLearningSystem) void {
        self.self_learning = sl;
    }

    /// Connect to a DPO optimizer
    pub fn connectDPO(self: *Self, dpo: *DPOOptimizer) void {
        self.dpo_optimizer = dpo;
    }

    /// Connect to a policy network
    pub fn connectPolicy(self: *Self, policy: *PolicyNetwork) void {
        self.policy_network = policy;
    }

    /// Record a response for pending feedback
    pub fn recordResponse(
        self: *Self,
        input_embedding: []const f32,
        output_embedding: []const f32,
        latency_ms: u64,
    ) !void {
        const entry = PendingFeedback{
            .input_embedding = try self.allocator.dupe(f32, input_embedding),
            .output_embedding = try self.allocator.dupe(f32, output_embedding),
            .timestamp = getCurrentTimestamp(),
            .latency_ms = latency_ms,
            .awaiting_feedback = true,
        };
        try self.pending_feedback.append(self.allocator, entry);
    }

    /// Process user feedback for the most recent response
    pub fn processFeedback(
        self: *Self,
        feedback_type: FeedbackType,
        confidence: f32,
    ) !void {
        if (self.pending_feedback.items.len == 0) return;

        // Get most recent pending entry
        var idx = self.pending_feedback.items.len - 1;
        while (idx > 0 and !self.pending_feedback.items[idx].awaiting_feedback) {
            idx -= 1;
        }

        const entry = &self.pending_feedback.items[idx];
        entry.awaiting_feedback = false;

        // If negative feedback and we have a previous good response, create preference pair
        if (feedback_type == .negative and self.dpo_optimizer != null) {
            // Find a previous positive response as the chosen example
            for (self.pending_feedback.items) |*prev| {
                if (!prev.awaiting_feedback and prev.timestamp < entry.timestamp) {
                    // Use previous as chosen, current as rejected
                    try self.dpo_optimizer.?.addPreferencePair(
                        prev.output_embedding,
                        entry.output_embedding,
                        entry.input_embedding,
                        confidence,
                    );
                    break;
                }
            }
        }

        // Trigger learning update if connected
        if (self.dpo_optimizer) |dpo| {
            if (dpo.isReady()) {
                _ = try dpo.trainStep();
            }
        }

        // Clean up old entries (keep last 100)
        while (self.pending_feedback.items.len > 100) {
            const old = self.pending_feedback.orderedRemove(0);
            self.allocator.free(old.input_embedding);
            self.allocator.free(old.output_embedding);
        }
    }

    /// Start a new session
    pub fn startSession(self: *Self) u64 {
        self.session_count += 1;
        self.current_session_id = self.session_count;
        return self.current_session_id;
    }
};

// ============================================================================
// Vision Trainer
// ============================================================================

/// Image understanding and vision training
pub const VisionTrainer = struct {
    allocator: std.mem.Allocator,
    config: VisionConfig,
    encoder_weights: []f32,
    patch_size: u32,
    hidden_dim: u32,

    pub const VisionConfig = struct {
        image_size: u32 = 224,
        patch_size: u32 = 16,
        hidden_dim: u32 = 768,
        num_heads: u32 = 12,
        num_layers: u32 = 12,
        learning_rate: f32 = 1e-4,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: VisionConfig) !Self {
        const num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
        const encoder_size = num_patches * config.hidden_dim;
        const encoder_weights = try allocator.alloc(f32, encoder_size);
        @memset(encoder_weights, 0);

        return .{
            .allocator = allocator,
            .config = config,
            .encoder_weights = encoder_weights,
            .patch_size = config.patch_size,
            .hidden_dim = config.hidden_dim,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.encoder_weights);
    }

    /// Encode image to embedding
    pub fn encodeImage(self: *const Self, image_data: []const u8) ![]f32 {
        const embedding = try self.allocator.alloc(f32, self.hidden_dim);

        // Simple patch embedding (production would use ViT)
        const patch_count = image_data.len / (self.patch_size * self.patch_size * 3);
        var idx: usize = 0;
        for (0..self.hidden_dim) |i| {
            var sum: f32 = 0;
            const patch_idx = i % @max(patch_count, 1);
            const start = patch_idx * self.patch_size * self.patch_size * 3;
            const end = @min(start + self.patch_size * 3, image_data.len);
            for (start..end) |j| {
                sum += @as(f32, @floatFromInt(image_data[j])) / 255.0;
            }
            embedding[idx] = sum / @as(f32, @floatFromInt(@max(end - start, 1)));
            idx += 1;
        }

        return embedding;
    }

    /// Train on image-text pairs
    pub fn trainStep(
        self: *Self,
        image_data: []const u8,
        text_embedding: []const f32,
        reward: f32,
    ) !f32 {
        const image_embedding = try self.encodeImage(image_data);
        defer self.allocator.free(image_embedding);

        // Contrastive loss
        var similarity: f32 = 0;
        const min_len = @min(image_embedding.len, text_embedding.len);
        for (0..min_len) |i| {
            similarity += image_embedding[i] * text_embedding[i];
        }

        // Scale by reward
        const loss = -reward * std.math.log(@max(1e-7, (similarity + 1.0) / 2.0));

        // Gradient update (simplified)
        for (0..@min(self.encoder_weights.len, min_len)) |i| {
            self.encoder_weights[i] -= self.config.learning_rate * reward * text_embedding[i];
        }

        return loss;
    }
};

// ============================================================================
// Document Trainer
// ============================================================================

/// Document understanding and parsing training
pub const DocumentTrainer = struct {
    allocator: std.mem.Allocator,
    config: DocumentConfig,
    layout_weights: []f32,
    structure_weights: []f32,

    pub const DocumentConfig = struct {
        max_pages: u32 = 100,
        hidden_dim: u32 = 512,
        max_elements: u32 = 1000,
        learning_rate: f32 = 1e-4,
        /// Supported document types
        doc_types: []const DocumentType = &.{ .pdf, .html, .markdown, .code },
    };

    pub const DocumentType = enum {
        pdf,
        html,
        markdown,
        code,
        plain_text,
        json,
        xml,
    };

    pub const DocumentElement = struct {
        element_type: ElementType,
        content: []const u8,
        position: struct { x: f32, y: f32, w: f32, h: f32 },
        confidence: f32,

        pub const ElementType = enum {
            title,
            heading,
            paragraph,
            list_item,
            table,
            figure,
            code_block,
            formula,
            footer,
            header,
        };
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: DocumentConfig) !Self {
        const layout_weights = try allocator.alloc(f32, config.hidden_dim * config.max_elements);
        @memset(layout_weights, 0);
        const structure_weights = try allocator.alloc(f32, config.hidden_dim);
        @memset(structure_weights, 0);

        return .{
            .allocator = allocator,
            .config = config,
            .layout_weights = layout_weights,
            .structure_weights = structure_weights,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.layout_weights);
        self.allocator.free(self.structure_weights);
    }

    /// Parse document structure
    pub fn parseDocument(self: *const Self, content: []const u8, doc_type: DocumentType) ![]DocumentElement {
        var elements: std.ArrayListUnmanaged(DocumentElement) = .{};
        errdefer elements.deinit(self.allocator);

        // Simplified parsing - production would use proper parsers
        switch (doc_type) {
            .markdown, .plain_text => {
                var lines = std.mem.splitScalar(u8, content, '\n');
                var y: f32 = 0;
                while (lines.next()) |line| {
                    if (line.len == 0) continue;

                    const elem_type: DocumentElement.ElementType = blk: {
                        if (std.mem.startsWith(u8, line, "# ")) break :blk .title;
                        if (std.mem.startsWith(u8, line, "## ") or std.mem.startsWith(u8, line, "### ")) break :blk .heading;
                        if (std.mem.startsWith(u8, line, "- ") or std.mem.startsWith(u8, line, "* ")) break :blk .list_item;
                        if (std.mem.startsWith(u8, line, "```")) break :blk .code_block;
                        break :blk .paragraph;
                    };

                    try elements.append(self.allocator, .{
                        .element_type = elem_type,
                        .content = line,
                        .position = .{ .x = 0, .y = y, .w = 1, .h = 0.05 },
                        .confidence = 0.9,
                    });
                    y += 0.05;
                }
            },
            else => {
                // Generic text extraction
                try elements.append(self.allocator, .{
                    .element_type = .paragraph,
                    .content = content,
                    .position = .{ .x = 0, .y = 0, .w = 1, .h = 1 },
                    .confidence = 0.5,
                });
            },
        }

        return elements.toOwnedSlice(self.allocator);
    }

    /// Train on document understanding task
    pub fn trainStep(
        self: *Self,
        document: []const u8,
        expected_elements: []const DocumentElement,
        reward: f32,
    ) !f32 {
        const parsed = try self.parseDocument(document, .plain_text);
        defer self.allocator.free(parsed);

        // Compute element matching loss
        var total_loss: f32 = 0;
        for (expected_elements) |expected| {
            var best_match: f32 = 0;
            for (parsed) |actual| {
                if (actual.element_type == expected.element_type) {
                    const overlap = @min(actual.content.len, expected.content.len);
                    const match = @as(f32, @floatFromInt(overlap)) /
                        @as(f32, @floatFromInt(@max(actual.content.len, expected.content.len)));
                    if (match > best_match) best_match = match;
                }
            }
            total_loss += (1.0 - best_match);
        }

        const avg_loss = if (expected_elements.len > 0)
            total_loss / @as(f32, @floatFromInt(expected_elements.len))
        else
            0;

        // Update weights based on reward
        for (self.structure_weights) |*w| {
            w.* -= self.config.learning_rate * avg_loss * reward;
        }

        return avg_loss;
    }
};

// ============================================================================
// Self-Learning System
// ============================================================================

/// Main self-learning system integrating all components
pub const SelfLearningSystem = struct {
    allocator: std.mem.Allocator,
    config: SelfLearningConfig,
    experience_buffer: ExperienceBuffer,
    reward_model: RewardModel,
    vision_trainer: ?VisionTrainer,
    document_trainer: ?DocumentTrainer,
    stats: LearningStats,
    update_count: u64,

    pub const LearningStats = struct {
        total_experiences: u64 = 0,
        total_updates: u64 = 0,
        avg_reward: f32 = 0,
        avg_loss: f32 = 0,
        positive_feedback_count: u64 = 0,
        negative_feedback_count: u64 = 0,
        vision_samples: u64 = 0,
        document_samples: u64 = 0,
        improvement_rate: f32 = 0,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: SelfLearningConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .experience_buffer = ExperienceBuffer.init(allocator, config.replay_buffer_size),
            .reward_model = try RewardModel.init(allocator, 768), // Default embedding dim
            .vision_trainer = null,
            .document_trainer = null,
            .stats = .{},
            .update_count = 0,
        };

        if (config.enable_vision) {
            self.vision_trainer = try VisionTrainer.init(allocator, .{});
        }

        if (config.enable_documents) {
            self.document_trainer = try DocumentTrainer.init(allocator, .{});
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.experience_buffer.deinit();
        self.reward_model.deinit();
        if (self.vision_trainer) |*vt| vt.deinit();
        if (self.document_trainer) |*dt| dt.deinit();
    }

    /// Record a learning experience from conversation
    pub fn recordExperience(
        self: *Self,
        input: []const u32,
        output: []const u32,
        feedback: FeedbackType,
        confidence: f32,
        exp_type: ExperienceType,
    ) !void {
        const reward = self.computeReward(feedback, confidence);

        const input_copy = try self.allocator.dupe(u32, input);
        errdefer self.allocator.free(input_copy);
        const output_copy = try self.allocator.dupe(u32, output);
        errdefer self.allocator.free(output_copy);

        const experience = LearningExperience{
            .id = 0, // Will be set by buffer
            .exp_type = exp_type,
            .input = input_copy,
            .output = output_copy,
            .reward = reward,
            .confidence = confidence,
            .feedback = feedback,
            .timestamp = getCurrentTimestamp(),
            .log_probs = null,
            .value = 0,
            .advantage = 0,
            .done = true,
            .image_data = null,
            .document_content = null,
            .metadata = .{},
        };

        try self.experience_buffer.add(experience);
        self.stats.total_experiences += 1;

        switch (feedback) {
            .positive, .implicit_accept => self.stats.positive_feedback_count += 1,
            .negative, .implicit_reject => self.stats.negative_feedback_count += 1,
            else => {},
        }

        // Check if we should update
        if (self.experience_buffer.len() >= self.config.min_buffer_size and
            self.stats.total_experiences % self.config.update_frequency == 0)
        {
            try self.update();
        }
    }

    /// Record a vision learning experience
    pub fn recordVisionExperience(
        self: *Self,
        input: []const u32,
        output: []const u32,
        image_data: []const u8,
        feedback: FeedbackType,
        confidence: f32,
    ) !void {
        const reward = self.computeReward(feedback, confidence);

        const input_copy = try self.allocator.dupe(u32, input);
        errdefer self.allocator.free(input_copy);
        const output_copy = try self.allocator.dupe(u32, output);
        errdefer self.allocator.free(output_copy);
        const image_copy = try self.allocator.dupe(u8, image_data);
        errdefer self.allocator.free(image_copy);

        const experience = LearningExperience{
            .id = 0,
            .exp_type = .vision,
            .input = input_copy,
            .output = output_copy,
            .reward = reward,
            .confidence = confidence,
            .feedback = feedback,
            .timestamp = getCurrentTimestamp(),
            .log_probs = null,
            .value = 0,
            .advantage = 0,
            .done = true,
            .image_data = image_copy,
            .document_content = null,
            .metadata = .{},
        };

        try self.experience_buffer.add(experience);
        self.stats.total_experiences += 1;
        self.stats.vision_samples += 1;
    }

    /// Record a document learning experience
    pub fn recordDocumentExperience(
        self: *Self,
        input: []const u32,
        output: []const u32,
        document: []const u8,
        feedback: FeedbackType,
        confidence: f32,
    ) !void {
        const reward = self.computeReward(feedback, confidence);

        const input_copy = try self.allocator.dupe(u32, input);
        errdefer self.allocator.free(input_copy);
        const output_copy = try self.allocator.dupe(u32, output);
        errdefer self.allocator.free(output_copy);
        const doc_copy = try self.allocator.dupe(u8, document);
        errdefer self.allocator.free(doc_copy);

        const experience = LearningExperience{
            .id = 0,
            .exp_type = .document,
            .input = input_copy,
            .output = output_copy,
            .reward = reward,
            .confidence = confidence,
            .feedback = feedback,
            .timestamp = getCurrentTimestamp(),
            .log_probs = null,
            .value = 0,
            .advantage = 0,
            .done = true,
            .image_data = null,
            .document_content = doc_copy,
            .metadata = .{},
        };

        try self.experience_buffer.add(experience);
        self.stats.total_experiences += 1;
        self.stats.document_samples += 1;
    }

    /// Compute reward from feedback and confidence
    fn computeReward(self: *const Self, feedback: FeedbackType, confidence: f32) f32 {
        _ = self;
        const base_reward: f32 = switch (feedback) {
            .positive => 1.0,
            .negative => -1.0,
            .implicit_accept => 0.3,
            .implicit_reject => -0.3,
            .self_eval => confidence * 2.0 - 1.0, // Map [0,1] to [-1,1]
            .none => 0.0,
        };
        return base_reward;
    }

    /// Perform a training update
    pub fn update(self: *Self) !void {
        if (self.experience_buffer.len() < self.config.batch_size) {
            return;
        }

        var batch = try self.experience_buffer.sample(self.config.batch_size);
        defer batch.deinit();

        var total_loss: f32 = 0;
        var total_reward: f32 = 0;
        var td_errors = try self.allocator.alloc(f32, batch.indices.len);
        defer self.allocator.free(td_errors);

        for (batch.indices, 0..) |idx, i| {
            const exp = batch.experiences[idx];
            const weight = batch.weights[i];

            // Compute TD error (simplified)
            const target_value = exp.reward + if (!exp.done) self.config.gamma * exp.value else 0;
            const td_error = target_value - exp.value;
            td_errors[i] = td_error;

            // Weighted loss
            const loss = td_error * td_error * weight;
            total_loss += loss;
            total_reward += exp.reward;

            // Vision training
            if (exp.exp_type == .vision and exp.image_data != null) {
                if (self.vision_trainer) |*vt| {
                    // Create simple text embedding from output tokens
                    var text_emb = try self.allocator.alloc(f32, vt.hidden_dim);
                    defer self.allocator.free(text_emb);
                    for (0..vt.hidden_dim) |j| {
                        text_emb[j] = if (j < exp.output.len)
                            @as(f32, @floatFromInt(exp.output[j])) / 65536.0
                        else
                            0;
                    }
                    _ = try vt.trainStep(exp.image_data.?, text_emb, exp.reward);
                }
            }

            // Document training
            if (exp.exp_type == .document and exp.document_content != null) {
                if (self.document_trainer) |*dt| {
                    _ = try dt.trainStep(exp.document_content.?, &.{}, exp.reward);
                }
            }
        }

        // Update priorities
        self.experience_buffer.updatePriorities(batch.indices, td_errors);

        // Update stats
        self.update_count += 1;
        self.stats.total_updates += 1;
        self.stats.avg_loss = (self.stats.avg_loss * 0.99) + (total_loss / @as(f32, @floatFromInt(batch.indices.len))) * 0.01;
        self.stats.avg_reward = (self.stats.avg_reward * 0.99) + (total_reward / @as(f32, @floatFromInt(batch.indices.len))) * 0.01;

        // Compute improvement rate
        const positive_ratio = if (self.stats.positive_feedback_count + self.stats.negative_feedback_count > 0)
            @as(f32, @floatFromInt(self.stats.positive_feedback_count)) /
                @as(f32, @floatFromInt(self.stats.positive_feedback_count + self.stats.negative_feedback_count))
        else
            0.5;
        self.stats.improvement_rate = positive_ratio;
    }

    /// Get current learning statistics
    pub fn getStats(self: *const Self) LearningStats {
        return self.stats;
    }

    /// Self-evaluate a response
    pub fn selfEvaluate(self: *const Self, input: []const u32, output: []const u32) f32 {
        _ = self;
        // Simple heuristic evaluation
        var score: f32 = 0.5;

        // Length appropriateness
        const ratio = @as(f32, @floatFromInt(output.len)) / @as(f32, @floatFromInt(@max(input.len, 1)));
        if (ratio > 0.5 and ratio < 5.0) {
            score += 0.1;
        }

        // Non-empty response
        if (output.len > 10) {
            score += 0.1;
        }

        // Reasonable length
        if (output.len < 2048) {
            score += 0.1;
        }

        // Variety (not all same token)
        if (output.len > 1) {
            var unique: u32 = 1;
            for (1..output.len) |i| {
                if (output[i] != output[i - 1]) unique += 1;
            }
            const variety = @as(f32, @floatFromInt(unique)) / @as(f32, @floatFromInt(output.len));
            score += variety * 0.2;
        }

        return @min(1.0, score);
    }

    /// Check if model should be updated based on self-evaluation
    pub fn shouldUpdate(self: *const Self) bool {
        return self.experience_buffer.len() >= self.config.min_buffer_size and
            self.stats.avg_reward < self.config.self_eval_threshold;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ExperienceBuffer basic operations" {
    const allocator = std.testing.allocator;
    var buffer = ExperienceBuffer.init(allocator, 100);
    defer buffer.deinit();

    const input = try allocator.dupe(u32, &[_]u32{ 1, 2, 3 });
    const output = try allocator.dupe(u32, &[_]u32{ 4, 5, 6 });

    const exp = LearningExperience{
        .id = 0,
        .exp_type = .text_conversation,
        .input = input,
        .output = output,
        .reward = 0.5,
        .confidence = 0.8,
        .feedback = .positive,
        .timestamp = 0,
        .log_probs = null,
        .value = 0,
        .advantage = 0,
        .done = true,
        .image_data = null,
        .document_content = null,
        .metadata = .{},
    };

    try buffer.add(exp);
    try std.testing.expectEqual(@as(usize, 1), buffer.len());
}

test "RewardModel computation" {
    const allocator = std.testing.allocator;
    var model = try RewardModel.init(allocator, 64);
    defer model.deinit();

    var embedding: [64]f32 = undefined;
    for (&embedding) |*e| {
        e.* = 0.1;
    }

    const reward = model.computeReward(&embedding);
    try std.testing.expect(reward >= -1.0 and reward <= 1.0);
}

test "SelfLearningSystem initialization" {
    const allocator = std.testing.allocator;
    var system = try SelfLearningSystem.init(allocator, .{});
    defer system.deinit();

    try std.testing.expectEqual(@as(u64, 0), system.stats.total_experiences);
}
