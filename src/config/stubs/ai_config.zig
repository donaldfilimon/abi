const std = @import("std");

pub const AiConfig = struct {
    llm: ?LlmConfig = null,
    embeddings: ?EmbeddingsConfig = null,
    agents: ?AgentsConfig = null,
    training: ?TrainingConfig = null,
    personas: ?PersonasConfig = null,
    auto_discover: bool = false,
    model_paths: []const []const u8 = &.{},
    adaptive_config: bool = false,
    warmup_diagnostics: bool = false,

    pub fn defaults() AiConfig {
        return .{};
    }

    pub fn withAutoDiscovery() AiConfig {
        return .{};
    }

    pub fn llmOnly(config: LlmConfig) AiConfig {
        _ = config;
        return .{};
    }

    pub fn embeddingsOnly(config: EmbeddingsConfig) AiConfig {
        _ = config;
        return .{};
    }
};

pub const LlmConfig = struct {
    model_path: ?[]const u8 = null,
    model_name: []const u8 = "gpt2",
    context_size: u32 = 2048,
    threads: ?u32 = null,
    use_gpu: bool = false,
    batch_size: u32 = 512,

    pub fn defaults() LlmConfig {
        return .{};
    }
};

pub const EmbeddingsConfig = struct {
    model: []const u8 = "default",
    dimension: u32 = 384,
    normalize: bool = true,

    pub fn defaults() EmbeddingsConfig {
        return .{};
    }
};

pub const AgentsConfig = struct {
    max_agents: u32 = 0,
    timeout_ms: u64 = 30000,
    persistent_memory: bool = false,

    pub fn defaults() AgentsConfig {
        return .{};
    }
};

pub const TrainingConfig = struct {
    epochs: u32 = 0,
    batch_size: u32 = 0,
    learning_rate: f32 = 0.0,
    optimizer: Optimizer = .adamw,
    checkpoint_dir: ?[]const u8 = null,
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

pub const PersonasConfig = struct {
    pub fn defaults() PersonasConfig {
        return .{};
    }
};
