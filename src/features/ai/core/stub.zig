//! AI Core stub — active when AI feature is disabled.
//!
//! Provides disabled versions of fundamental types and configuration.

// ============================================================================
// Core Types
// ============================================================================

pub const InstanceId = u64;
pub const SessionId = u64;

pub const ConfidenceLevel = enum {
    very_low,
    low,
    medium,
    high,
    very_high,
};

pub const Confidence = struct {
    level: ConfidenceLevel = .medium,
    score: f32 = 0,
    reasoning: []const u8 = "",
};

pub const EmotionType = enum {
    neutral,
    happy,
    sad,
    curious,
    frustrated,
    excited,
    confused,
    thoughtful,
};

pub const EmotionalState = struct {
    detected: EmotionType = .neutral,
    intensity: f32 = 0,
    valence: f32 = 0,

    pub fn init() EmotionalState {
        return .{};
    }

    pub fn detectFromText(_: *EmotionalState, _: []const u8) void {}
};

pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

pub const Message = struct {
    role: Role = .user,
    content: []const u8 = "",
    name: ?[]const u8 = null,
    timestamp: i64 = 0,
    token_count: ?usize = null,
    metadata: ?[]const u8 = null,
};

pub const TrustLevel = enum {
    unknown,
    low,
    medium,
    high,
    verified,
};

pub const Relationship = struct {
    user_id: []const u8 = "",
    trust: TrustLevel = .unknown,
    interaction_count: u32 = 0,
    score: f32 = 0.5,
};

pub const Topic = struct {
    name: []const u8 = "",
    relevance: f32 = 0,
    mentions: u32 = 0,
};

pub const Response = struct {
    content: []const u8 = "",
    confidence: Confidence = .{},
    emotional_context: ?EmotionalState = null,
    reasoning_summary: ?[]const u8 = null,
};

pub const AbbeyError = error{
    FeatureDisabled,
    InitializationFailed,
    InferenceFailed,
    MemoryFull,
    InvalidInput,
};

// ============================================================================
// Configuration
// ============================================================================

pub const AbbeyConfig = struct {
    name: []const u8 = "Abbey",
    behavior: BehaviorConfig = .{},
    memory: MemoryConfig = .{},
    reasoning: ReasoningConfig = .{},
    emotion: EmotionConfig = .{},
    learning: LearningConfig = .{},
    llm: LLMConfig = .{},
    server: ServerConfig = .{},
    discord: DiscordConfig = .{},
};

pub const BehaviorConfig = struct {
    base_temperature: f32 = 0.7,
    research_first: bool = true,
    enable_emotions: bool = true,
    enable_reasoning_log: bool = true,
};

pub const MemoryConfig = struct {
    max_entries: usize = 1000,
    embedding_dim: usize = 384,
};

pub const ReasoningConfig = struct {
    max_steps: usize = 10,
    confidence_threshold: f32 = 0.7,
};

pub const EmotionConfig = struct {
    enabled: bool = true,
    intensity_decay: f32 = 0.1,
};

pub const LearningConfig = struct {
    enabled: bool = true,
    learning_rate: f32 = 0.01,
};

pub const LLMConfig = struct {
    backend: enum { echo, local, api } = .echo,
    model_path: ?[]const u8 = null,
    api_key: ?[]const u8 = null,
};

pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
};

pub const DiscordConfig = struct {
    token: ?[]const u8 = null,
    prefix: []const u8 = "!",
};

pub const ConfigBuilder = struct {
    config: AbbeyConfig = .{},

    pub fn init() ConfigBuilder {
        return .{};
    }

    pub fn name(self: *ConfigBuilder, n: []const u8) *ConfigBuilder {
        self.config.name = n;
        return self;
    }

    pub fn temperature(self: *ConfigBuilder, t: f32) *ConfigBuilder {
        self.config.behavior.base_temperature = t;
        return self;
    }

    pub fn researchFirst(self: *ConfigBuilder, v: bool) *ConfigBuilder {
        self.config.behavior.research_first = v;
        return self;
    }

    pub fn llmBackend(self: *ConfigBuilder, b: @TypeOf((LLMConfig{}).backend)) *ConfigBuilder {
        self.config.llm.backend = b;
        return self;
    }

    pub fn build(self: *ConfigBuilder) !AbbeyConfig {
        return self.config;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

pub fn getTimestampNs() i128 {
    return 0;
}

pub fn getTimestampMs() i64 {
    return 0;
}

pub fn getTimestampSec() i64 {
    return 0;
}

pub fn loadConfigFromEnvironment() !AbbeyConfig {
    return error.FeatureDisabled;
}
