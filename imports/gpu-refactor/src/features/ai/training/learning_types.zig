//! Learning Types for Self-Learning Module
//!
//! Configuration and type definitions for the self-learning system. Models can be
//! trained to handle and generate all types of data: text, images, video, audio,
//! documents, and arbitrary payloads (raw_data + content_type).

const std = @import("std");

// ============================================================================
// Self-Learning Configuration
// ============================================================================

/// Configuration for self-learning training
pub const SelfLearningConfig = struct {
    /// Enable RLHF training
    enable_rlhf: bool = true,
    /// Enable vision training (images)
    enable_vision: bool = true,
    /// Enable document training
    enable_documents: bool = true,
    /// Enable video training (frames / video clips)
    enable_video: bool = true,
    /// Enable audio training
    enable_audio: bool = true,
    /// Enable training on arbitrary data types (raw_data + content_type)
    enable_all_modalities: bool = true,
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

/// Type of learning experience. Covers all data types: text, images, video, audio, and any.
pub const ExperienceType = enum {
    /// Text-based conversation
    text_conversation,
    /// Image understanding
    vision,
    /// Video (frames or clips)
    video,
    /// Audio
    audio,
    /// Document parsing
    document,
    /// Code generation
    code,
    /// Reasoning task
    reasoning,
    /// Multi-modal (combined modalities)
    multi_modal,
    /// Arbitrary / other data (use raw_data + content_type)
    any,
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

/// Data kind for arbitrary payloads (process/generate all types).
pub const DataKind = enum {
    text,
    image,
    video,
    audio,
    document,
    other,
};

/// A learning experience for replay. Supports all data types: text, images, video, audio, documents, and raw payloads.
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
    /// Optional video data (frames or encoded clip)
    video_data: ?[]const u8 = null,
    /// Optional audio data
    audio_data: ?[]const u8 = null,
    /// Optional document content
    document_content: ?[]const u8 = null,
    /// Arbitrary payload (when exp_type == .any or multi-modal)
    raw_data: ?[]const u8 = null,
    /// Content type for raw_data (e.g. "video/mp4", "image/png", "application/octet-stream")
    content_type: ?[]const u8 = null,
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
        if (self.video_data) |v| allocator.free(v);
        if (self.audio_data) |a| allocator.free(a);
        if (self.document_content) |doc| allocator.free(doc);
        if (self.raw_data) |r| allocator.free(r);
        if (self.content_type) |ct| allocator.free(ct);
    }
};

test {
    std.testing.refAllDecls(@This());
}
