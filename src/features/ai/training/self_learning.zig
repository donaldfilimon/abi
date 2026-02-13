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

// Re-exports from extracted modules
const learning_types_mod = @import("learning_types.zig");
pub const SelfLearningConfig = learning_types_mod.SelfLearningConfig;
pub const ExperienceType = learning_types_mod.ExperienceType;
pub const FeedbackType = learning_types_mod.FeedbackType;
pub const LearningExperience = learning_types_mod.LearningExperience;

const experience_buffer_mod = @import("experience_buffer.zig");
pub const ExperienceBuffer = experience_buffer_mod.ExperienceBuffer;
pub const SampledBatch = experience_buffer_mod.SampledBatch;

const reward_policy_mod = @import("reward_policy.zig");
pub const RewardModel = reward_policy_mod.RewardModel;
pub const PolicyNetwork = reward_policy_mod.PolicyNetwork;

const dpo_optimizer_mod = @import("dpo_optimizer.zig");
pub const DPOOptimizer = dpo_optimizer_mod.DPOOptimizer;

// Test imports
test {
    _ = @import("self_learning_test.zig");
}

/// Get current timestamp for Zig 0.16 compatibility (no std.time.timestamp()).
/// Returns nanoseconds since timer start as an i64.
fn getCurrentTimestamp() i64 {
    var timer = time.Timer.start() catch return 0;
    return @intCast(timer.read());
}

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
