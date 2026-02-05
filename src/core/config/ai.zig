//! AI Configuration
//!
//! Configuration for AI features including LLM inference, embeddings,
//! agents, and training pipelines.

const std = @import("std");
const build_options = @import("build_options");

// Import the actual personas configuration from the personas module
const personas_config = @import("../../features/ai/personas/config.zig");
pub const PersonasConfig = personas_config.MultiPersonaConfig;

/// AI configuration with independent sub-features.
pub const AiConfig = struct {
    /// LLM inference settings. Set to enable local LLM.
    llm: ?LlmConfig = null,

    /// Embeddings generation settings.
    embeddings: ?EmbeddingsConfig = null,

    /// Agent runtime settings.
    agents: ?AgentsConfig = null,

    /// Training pipeline settings.
    training: ?TrainingConfig = null,

    /// Multi-persona assistant settings.
    personas: ?PersonasConfig = null,

    /// Enable automatic model discovery from standard paths.
    auto_discover: bool = false,

    /// Custom model search paths (in addition to standard paths).
    model_paths: []const []const u8 = &.{},

    /// Enable adaptive configuration based on system capabilities.
    adaptive_config: bool = true,

    /// Run warm-up diagnostics on model load.
    warmup_diagnostics: bool = false,

    pub fn defaults() AiConfig {
        return .{
            .llm = if (build_options.enable_llm) LlmConfig.defaults() else null,
            .embeddings = EmbeddingsConfig.defaults(),
            .agents = AgentsConfig.defaults(),
            .training = null, // Training not enabled by default
            .personas = if (build_options.enable_ai) .{} else null,
            .auto_discover = true, // Enable auto-discovery by default
        };
    }

    /// Configuration with auto-discovery enabled.
    pub fn withAutoDiscovery() AiConfig {
        return .{
            .auto_discover = true,
            .adaptive_config = true,
        };
    }

    /// Enable only LLM inference.
    pub fn llmOnly(config: LlmConfig) AiConfig {
        return .{ .llm = config };
    }

    /// Enable only embeddings.
    pub fn embeddingsOnly(config: EmbeddingsConfig) AiConfig {
        return .{ .embeddings = config };
    }
};

/// LLM inference configuration.
pub const LlmConfig = struct {
    /// Path to model file (GGUF format).
    model_path: ?[]const u8 = null,

    /// Model to use from registry.
    model_name: []const u8 = "gpt2",

    /// Context window size.
    context_size: u32 = 2048,

    /// Number of threads for inference.
    threads: ?u32 = null,

    /// Use GPU acceleration if available.
    use_gpu: bool = true,

    /// Batch size for inference.
    batch_size: u32 = 512,

    pub fn defaults() LlmConfig {
        return .{};
    }
};

/// Embeddings generation configuration.
pub const EmbeddingsConfig = struct {
    /// Embedding model to use.
    model: []const u8 = "default",

    /// Output embedding dimension.
    dimension: u32 = 384,

    /// Normalize output vectors.
    normalize: bool = true,

    pub fn defaults() EmbeddingsConfig {
        return .{};
    }
};

/// Agent runtime configuration.
pub const AgentsConfig = struct {
    /// Maximum concurrent agents.
    max_agents: u32 = 16,

    /// Default agent timeout in milliseconds.
    timeout_ms: u64 = 30000,

    /// Enable agent memory/context persistence.
    persistent_memory: bool = false,

    pub fn defaults() AgentsConfig {
        return .{};
    }
};

/// Training pipeline configuration.
pub const TrainingConfig = struct {
    /// Number of training epochs.
    epochs: u32 = 10,

    /// Training batch size.
    batch_size: u32 = 32,

    /// Learning rate.
    learning_rate: f32 = 0.001,

    /// Optimizer to use.
    optimizer: Optimizer = .adamw,

    /// Checkpoint directory.
    checkpoint_dir: ?[]const u8 = null,

    /// Checkpoint frequency (epochs).
    checkpoint_frequency: u32 = 1,

    pub const Optimizer = enum {
        sgd,
        adam,
        adamw,
        rmsprop,
    };

    pub fn defaults() TrainingConfig {
        return .{};
    }
};

// PersonasConfig is re-exported from ai/personas/config.zig above
