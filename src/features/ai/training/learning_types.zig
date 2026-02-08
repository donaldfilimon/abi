//! Learning Types for Self-Learning Module
//!
//! Configuration and type definitions for the self-learning system:
//! - SelfLearningConfig
//! - ExperienceType, FeedbackType
//! - LearningExperience

const std = @import("std");

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
