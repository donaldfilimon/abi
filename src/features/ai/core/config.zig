//! Abbey Configuration System
//!
//! Comprehensive configuration for all Abbey subsystems.
//! Supports runtime reconfiguration and validation.

const std = @import("std");
const types = @import("types.zig");

/// Main Abbey configuration
pub const AbbeyConfig = struct {
    // Identity
    name: []const u8 = "Abbey",
    version: []const u8 = "2.0.0",

    // Core behavior
    behavior: BehaviorConfig = .{},

    // Memory subsystem
    memory: MemoryConfig = .{},

    // Reasoning subsystem
    reasoning: ReasoningConfig = .{},

    // Emotional intelligence
    emotions: EmotionConfig = .{},

    // Learning subsystem
    learning: LearningConfig = .{},

    // LLM integration
    llm: LLMConfig = .{},

    // API server
    server: ServerConfig = .{},

    // Discord integration
    discord: DiscordConfig = .{},

    pub fn validate(self: *const AbbeyConfig) !void {
        // Validate temperature range
        if (self.behavior.base_temperature < 0.0 or self.behavior.base_temperature > 2.0) {
            return error.InvalidTemperature;
        }

        // Validate memory limits
        if (self.memory.max_context_tokens < 100) {
            return error.ContextTooSmall;
        }

        // Validate confidence threshold
        if (self.behavior.confidence_threshold < 0.0 or self.behavior.confidence_threshold > 1.0) {
            return error.InvalidConfidenceThreshold;
        }
    }

    pub fn withLLMBackend(self: AbbeyConfig, backend: LLMConfig.Backend) AbbeyConfig {
        var config = self;
        config.llm.backend = backend;
        return config;
    }

    pub fn withMemoryType(self: AbbeyConfig, memory_type: MemoryConfig.MemoryType) AbbeyConfig {
        var config = self;
        config.memory.primary_type = memory_type;
        return config;
    }
};

/// Behavior configuration
pub const BehaviorConfig = struct {
    /// Base temperature for generation (adjusted by emotional state)
    base_temperature: f32 = 0.7,

    /// Top-p sampling parameter
    top_p: f32 = 0.9,

    /// Maximum tokens to generate
    max_tokens: u32 = 2048,

    /// Confidence threshold for direct answers (below triggers research)
    confidence_threshold: f32 = 0.7,

    /// Enable research-first behavior
    research_first: bool = true,

    /// Enable opinionated responses
    opinionated: bool = true,

    /// Enable emotional adaptation
    emotional_adaptation: bool = true,

    /// Enable proactive suggestions
    proactive: bool = true,

    /// Maximum reasoning steps before forcing a response
    max_reasoning_steps: usize = 10,

    /// Timeout for research operations (ms)
    research_timeout_ms: u32 = 30_000,

    /// Enable verbose reasoning output
    verbose_reasoning: bool = false,
};

/// Memory subsystem configuration
pub const MemoryConfig = struct {
    pub const MemoryType = enum {
        /// Simple short-term buffer
        short_term,
        /// Token-based sliding window
        sliding_window,
        /// Summarizing with compression
        summarizing,
        /// Vector-based long-term
        long_term,
        /// Hybrid of all types
        hybrid,
        /// Episodic memory (event-based)
        episodic,
    };

    /// Primary memory type
    primary_type: MemoryType = .hybrid,

    /// Maximum context window tokens
    max_context_tokens: usize = 8000,

    /// Short-term capacity (messages)
    short_term_capacity: usize = 50,

    /// System prompt reserve tokens
    system_reserve: usize = 500,

    /// Enable long-term storage
    enable_long_term: bool = true,

    /// Auto-store important messages
    auto_store: bool = true,

    /// Importance threshold for auto-storage
    importance_threshold: f32 = 0.6,

    /// Enable memory consolidation
    enable_consolidation: bool = true,

    /// Consolidation interval (seconds)
    consolidation_interval_sec: u32 = 300,

    /// Maximum episodic memories
    max_episodes: usize = 1000,

    /// Embedding dimension
    embedding_dim: usize = 384,

    /// Persistence path (null for in-memory only)
    persistence_path: ?[]const u8 = null,
};

/// Reasoning subsystem configuration
pub const ReasoningConfig = struct {
    /// Enable chain-of-thought reasoning
    enable_cot: bool = true,

    /// Maximum chain depth
    max_chain_depth: usize = 10,

    /// Enable step-by-step output
    show_steps: bool = false,

    /// Confidence decay per step
    confidence_decay: f32 = 0.05,

    /// Enable self-verification
    enable_verification: bool = true,

    /// Minimum confidence to proceed
    min_confidence: f32 = 0.3,

    /// Enable reasoning caching
    cache_reasoning: bool = true,

    /// Cache TTL (seconds)
    cache_ttl_sec: u32 = 3600,
};

/// Emotional intelligence configuration
pub const EmotionConfig = struct {
    /// Enable emotion detection
    enable_detection: bool = true,

    /// Enable emotional adaptation
    enable_adaptation: bool = true,

    /// Emotion history length
    history_length: usize = 8,

    /// Persistence threshold for emotional patterns
    persistence_threshold: usize = 2,

    /// Temperature adjustment range
    temperature_adjustment_range: f32 = 0.3,

    /// Enable relationship tracking
    track_relationships: bool = true,

    /// Enable communication preference learning
    learn_preferences: bool = true,
};

/// Learning subsystem configuration
pub const LearningConfig = struct {
    /// Enable online learning
    enable_online_learning: bool = true,

    /// Learning rate
    learning_rate: f32 = 0.001,

    /// Momentum for gradient updates
    momentum: f32 = 0.9,

    /// Weight decay (L2 regularization)
    weight_decay: f32 = 0.0001,

    /// Batch size for updates
    batch_size: usize = 8,

    /// Enable gradient accumulation
    accumulate_gradients: bool = true,

    /// Gradient accumulation steps
    accumulation_steps: usize = 4,

    /// Enable experience replay
    experience_replay: bool = true,

    /// Replay buffer size
    replay_buffer_size: usize = 1000,

    /// Enable meta-learning
    meta_learning: bool = false,

    /// Enable attention adaptation
    adapt_attention: bool = true,
};

/// LLM integration configuration
pub const LLMConfig = struct {
    pub const Backend = enum {
        /// Local echo (for testing)
        echo,
        /// OpenAI API
        openai,
        /// Anthropic API
        anthropic,
        /// Ollama local
        ollama,
        /// HuggingFace Inference
        huggingface,
        /// Local GGUF model
        local,
        /// Custom backend
        custom,
    };

    /// LLM backend to use
    backend: Backend = .echo,

    /// Model name/identifier
    model: []const u8 = "gpt-4",

    /// API key (null to use environment)
    api_key: ?[]const u8 = null,

    /// Base URL for API
    base_url: ?[]const u8 = null,

    /// Request timeout (ms)
    timeout_ms: u32 = 60_000,

    /// Maximum retries
    max_retries: u32 = 3,

    /// Retry delay base (ms)
    retry_delay_ms: u32 = 1000,

    /// Enable streaming responses
    streaming: bool = true,

    /// Enable function calling
    function_calling: bool = true,

    /// Local model path (for local backend)
    local_model_path: ?[]const u8 = null,
};

/// API server configuration
pub const ServerConfig = struct {
    /// Enable HTTP API server
    enabled: bool = false,

    /// Host to bind to
    host: []const u8 = "127.0.0.1",

    /// Port to listen on
    port: u16 = 8080,

    /// Enable CORS
    enable_cors: bool = true,

    /// CORS allowed origins
    cors_origins: []const []const u8 = &.{"*"},

    /// Enable rate limiting
    rate_limiting: bool = true,

    /// Requests per minute per client
    rate_limit_rpm: u32 = 60,

    /// Enable authentication
    require_auth: bool = false,

    /// API key for authentication
    api_key: ?[]const u8 = null,

    /// Enable WebSocket for streaming
    enable_websocket: bool = true,

    /// Maximum concurrent connections
    max_connections: u32 = 100,

    /// Request body size limit
    max_body_size: usize = 1024 * 1024, // 1MB
};

/// Discord integration configuration
pub const DiscordConfig = struct {
    /// Enable Discord bot
    enabled: bool = false,

    /// Bot token (null to use environment)
    bot_token: ?[]const u8 = null,

    /// Command prefix
    command_prefix: []const u8 = "!abbey",

    /// Enable slash commands
    slash_commands: bool = true,

    /// Allowed channel IDs (empty = all)
    allowed_channels: []const []const u8 = &.{},

    /// Allowed guild IDs (empty = all)
    allowed_guilds: []const []const u8 = &.{},

    /// Enable DMs
    allow_dms: bool = true,

    /// Maximum message length
    max_message_length: usize = 2000,

    /// Enable reactions
    enable_reactions: bool = true,

    /// Enable threading
    enable_threads: bool = true,
};

// ============================================================================
// Configuration Builder
// ============================================================================

pub const ConfigBuilder = struct {
    config: AbbeyConfig,

    pub fn init() ConfigBuilder {
        return .{ .config = .{} };
    }

    pub fn name(self: *ConfigBuilder, n: []const u8) *ConfigBuilder {
        self.config.name = n;
        return self;
    }

    pub fn temperature(self: *ConfigBuilder, temp: f32) *ConfigBuilder {
        self.config.behavior.base_temperature = temp;
        return self;
    }

    pub fn maxTokens(self: *ConfigBuilder, tokens: u32) *ConfigBuilder {
        self.config.behavior.max_tokens = tokens;
        return self;
    }

    pub fn researchFirst(self: *ConfigBuilder, enabled: bool) *ConfigBuilder {
        self.config.behavior.research_first = enabled;
        return self;
    }

    pub fn opinionated(self: *ConfigBuilder, enabled: bool) *ConfigBuilder {
        self.config.behavior.opinionated = enabled;
        return self;
    }

    pub fn memoryType(self: *ConfigBuilder, mt: MemoryConfig.MemoryType) *ConfigBuilder {
        self.config.memory.primary_type = mt;
        return self;
    }

    pub fn contextTokens(self: *ConfigBuilder, tokens: usize) *ConfigBuilder {
        self.config.memory.max_context_tokens = tokens;
        return self;
    }

    pub fn llmBackend(self: *ConfigBuilder, backend: LLMConfig.Backend) *ConfigBuilder {
        self.config.llm.backend = backend;
        return self;
    }

    pub fn llmModel(self: *ConfigBuilder, model: []const u8) *ConfigBuilder {
        self.config.llm.model = model;
        return self;
    }

    pub fn enableServer(self: *ConfigBuilder, port: u16) *ConfigBuilder {
        self.config.server.enabled = true;
        self.config.server.port = port;
        return self;
    }

    pub fn enableDiscord(self: *ConfigBuilder) *ConfigBuilder {
        self.config.discord.enabled = true;
        return self;
    }

    pub fn enableOnlineLearning(self: *ConfigBuilder, enabled: bool) *ConfigBuilder {
        self.config.learning.enable_online_learning = enabled;
        return self;
    }

    pub fn build(self: *ConfigBuilder) !AbbeyConfig {
        try self.config.validate();
        return self.config;
    }
};

// ============================================================================
// Environment Loading
// ============================================================================

pub fn loadFromEnvironment(allocator: std.mem.Allocator) !AbbeyConfig {
    var config = AbbeyConfig{};

    // Load LLM configuration from environment
    if (getEnv(allocator, "ABI_ABBEY_LLM_BACKEND")) |backend_str| {
        defer allocator.free(backend_str);
        config.llm.backend = parseBackend(backend_str) orelse .echo;
    }

    if (getEnv(allocator, "ABI_ABBEY_MODEL")) |model| {
        // Transfer ownership - caller is responsible for freeing via config.deinit()
        config.llm.model = model;
    }

    if (getEnv(allocator, "OPENAI_API_KEY") orelse getEnv(allocator, "ABI_OPENAI_API_KEY")) |key| {
        // Transfer ownership - caller is responsible for freeing via config.deinit()
        config.llm.api_key = key;
        if (config.llm.backend == .echo) {
            config.llm.backend = .openai;
        }
    }

    // Load server configuration
    if (getEnv(allocator, "ABI_ABBEY_SERVER_PORT")) |port_str| {
        defer allocator.free(port_str);
        config.server.port = std.fmt.parseInt(u16, port_str, 10) catch 8080;
        config.server.enabled = true;
    }

    // Load Discord configuration
    if (getEnv(allocator, "DISCORD_BOT_TOKEN") orelse getEnv(allocator, "ABI_DISCORD_TOKEN")) |token| {
        // Transfer ownership - caller is responsible for freeing via config.deinit()
        config.discord.bot_token = token;
        config.discord.enabled = true;
    }

    try config.validate();
    return config;
}

fn getEnv(allocator: std.mem.Allocator, key: []const u8) ?[]const u8 {
    const raw = std.c.getenv(key.ptr) orelse return null;
    return allocator.dupe(u8, std.mem.sliceTo(raw, 0)) catch null;
}

fn parseBackend(str: []const u8) ?LLMConfig.Backend {
    const backends = .{
        .{ "echo", LLMConfig.Backend.echo },
        .{ "openai", LLMConfig.Backend.openai },
        .{ "anthropic", LLMConfig.Backend.anthropic },
        .{ "ollama", LLMConfig.Backend.ollama },
        .{ "huggingface", LLMConfig.Backend.huggingface },
        .{ "local", LLMConfig.Backend.local },
    };

    inline for (backends) |pair| {
        if (std.mem.eql(u8, str, pair[0])) {
            return pair[1];
        }
    }
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "config validation" {
    var config = AbbeyConfig{};
    try config.validate();

    config.behavior.base_temperature = 3.0;
    try std.testing.expectError(error.InvalidTemperature, config.validate());
}

test "config builder" {
    var b = ConfigBuilder.init();
    const config = try b
        .name("TestAbbey")
        .temperature(0.8)
        .maxTokens(4096)
        .researchFirst(true)
        .llmBackend(.openai)
        .build();

    try std.testing.expectEqualStrings("TestAbbey", config.name);
    try std.testing.expectEqual(@as(f32, 0.8), config.behavior.base_temperature);
    try std.testing.expectEqual(LLMConfig.Backend.openai, config.llm.backend);
}
