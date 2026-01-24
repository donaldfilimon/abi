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
const trainable_model = @import("trainable_model.zig");
const loss_mod = @import("loss.zig");
const gradient = @import("gradient.zig");
const logging = @import("logging.zig");

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
    experiences: std.ArrayList(LearningExperience),
    priorities: std.ArrayList(f32),
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
            .experiences = std.ArrayList(LearningExperience).init(allocator),
            .priorities = std.ArrayList(f32).init(allocator),
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
        self.experiences.deinit();
        self.priorities.deinit();
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

        try self.experiences.append(exp_copy);
        try self.priorities.append(max_priority);
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
            var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
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
            var timer = std.time.Timer.start() catch break :blk @as(u64, 42);
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
        var elements = std.ArrayList(DocumentElement).init(self.allocator);
        errdefer elements.deinit();

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

                    try elements.append(.{
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
                try elements.append(.{
                    .element_type = .paragraph,
                    .content = content,
                    .position = .{ .x = 0, .y = 0, .w = 1, .h = 1 },
                    .confidence = 0.5,
                });
            },
        }

        return elements.toOwnedSlice();
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
            .timestamp = std.time.timestamp(),
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
            .timestamp = std.time.timestamp(),
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
            .timestamp = std.time.timestamp(),
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
