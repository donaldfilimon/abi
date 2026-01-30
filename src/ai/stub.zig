//! AI Stub Module
//!
//! Stub implementation when AI is disabled at compile time.

const std = @import("std");
const config_module = @import("../config/mod.zig");
const stub_root = @This();

pub const Error = error{
    AiDisabled,
    LlmDisabled,
    EmbeddingsDisabled,
    AgentsDisabled,
    TrainingDisabled,
    ModelNotFound,
    InferenceFailed,
    InvalidConfig,
};

// Sub-module stubs
pub const core = struct {};
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const training = @import("training/stub.zig");
pub const database = @import("database/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const models = @import("models/stub.zig");

// Multi-agent re-exports
pub const MultiAgentCoordinator = multi_agent.Coordinator;

// Agent module stub (singular - for backward compatibility)
pub const agent = struct {
    pub const MIN_TEMPERATURE: f32 = 0.0;
    pub const MAX_TEMPERATURE: f32 = 2.0;
    pub const MIN_TOP_P: f32 = 0.0;
    pub const MAX_TOP_P: f32 = 1.0;
    pub const MAX_TOKENS_LIMIT: u32 = 128000;
    pub const DEFAULT_TEMPERATURE: f32 = 0.7;
    pub const DEFAULT_TOP_P: f32 = 0.9;
    pub const DEFAULT_MAX_TOKENS: u32 = 1024;

    pub const AgentError = error{
        InvalidTemperature,
        InvalidTopP,
        InvalidMaxTokens,
        NoMessages,
        EmptyResponse,
        ConnectionFailed,
        RateLimited,
        AuthenticationFailed,
        ContextLengthExceeded,
        ModelNotAvailable,
        BackendError,
        AiDisabled,
    };

    pub const AgentBackend = enum {
        openai,
        ollama,
        huggingface,
        local,
    };

    pub const AgentConfig = struct {
        name: []const u8 = "",
        backend: AgentBackend = .openai,
        model: []const u8 = "gpt-4",
        temperature: f32 = DEFAULT_TEMPERATURE,
        top_p: f32 = DEFAULT_TOP_P,
        max_tokens: u32 = DEFAULT_MAX_TOKENS,
        system_prompt: ?[]const u8 = null,
        enable_history: bool = true,
    };

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const Agent = struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: AgentConfig) AgentError!Self {
            return error.AiDisabled;
        }

        pub fn deinit(_: *Self) void {}

        pub fn chat(_: *Self, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
            return error.AiDisabled;
        }

        pub fn process(_: *Self, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
            return error.AiDisabled;
        }
    };
};

// Model registry module stub
pub const model_registry = struct {
    pub const ModelInfo = struct {};
    pub const ModelRegistry = struct {};
};

// Stub types
pub const Agent = struct {};
pub const ModelRegistry = struct {
    pub fn init(_: std.mem.Allocator) ModelRegistry {
        return .{};
    }
};
pub const ModelInfo = struct {};
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = Error;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const TrainableModel = training.TrainableModel;
pub const TokenizedDataset = training.TokenizedDataset;
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const TokenBlock = training.TokenBlock;
pub const encodeTokenBlock = training.encodeTokenBlock;
pub const decodeTokenBlock = training.decodeTokenBlock;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const exportGguf = database.exportGguf;

pub const trainable_model = struct {
    pub const TrainableModelConfig = struct {
        hidden_dim: u32 = 768,
        num_layers: u32 = 12,
        num_heads: u32 = 12,
        num_kv_heads: u32 = 12,
        intermediate_dim: u32 = 2048,
        vocab_size: u32 = 50257,
        max_seq_len: u32 = 1024,
    };
};

pub const tools = struct {
    const T = @This(); // Alias for this struct to avoid ambiguity
    pub const ParameterType = enum { string, integer, boolean, array, object, number };
    pub const Parameter = struct {
        name: []const u8,
        type: ParameterType,
        required: bool = false,
        description: []const u8 = "",
        enum_values: ?[]const []const u8 = null,
    };
    pub const Tool = struct {
        name: []const u8 = "",
        description: []const u8 = "",
        parameters: []const T.Parameter = &.{},
        execute: ?*const anyopaque = null,
    };
    pub const ToolResult = struct {
        success: bool = false,
        output: []const u8 = "",
        error_message: ?[]const u8 = null,
    };
    pub const ToolRegistry = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *@This()) void {}
        pub fn register(_: *@This(), _: *const T.Tool) !void {}
        pub fn get(_: *@This(), _: []const u8) ?*const T.Tool {
            return null;
        }
    };
    pub const TaskTool = struct {};
    pub const Subagent = struct {};
    pub const DiscordTools = struct {
        pub const send_message_tool = T.Tool{ .name = "discord_send_message" };
        pub const get_channel_tool = T.Tool{ .name = "discord_get_channel" };
        pub const list_guilds_tool = T.Tool{ .name = "discord_list_guilds" };
        pub const get_bot_info_tool = T.Tool{ .name = "discord_get_bot_info" };
        pub const execute_webhook_tool = T.Tool{ .name = "discord_execute_webhook" };
        pub const add_reaction_tool = T.Tool{ .name = "discord_add_reaction" };
        pub const get_messages_tool = T.Tool{ .name = "discord_get_messages" };
        pub fn registerAll(_: *T.ToolRegistry) void {}
    };
    pub const OsTools = struct {
        pub fn registerAll(_: *T.ToolRegistry) void {}
    };
    pub fn registerDiscordTools(registry: *T.ToolRegistry) !void {
        T.DiscordTools.registerAll(registry);
    }
    pub fn registerOsTools(registry: *T.ToolRegistry) !void {
        T.OsTools.registerAll(registry);
    }
};
// Re-exports for module-level access (e.g., ai.ToolRegistry)
pub const Tool = tools.Tool;
pub const ToolResult = tools.ToolResult;
pub const ToolRegistry = tools.ToolRegistry;
pub const TaskTool = tools.TaskTool;
pub const Subagent = tools.Subagent;
pub const DiscordTools = tools.DiscordTools;
pub const OsTools = tools.OsTools;
pub const registerDiscordTools = tools.registerDiscordTools;
pub const registerOsTools = tools.registerOsTools;

pub const transformer = struct {
    pub const TransformerConfig = struct {};
    pub const TransformerModel = struct {
        pub fn init(_: transformer.TransformerConfig) transformer.TransformerModel {
            return .{};
        }
    };
};
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

pub const streaming = struct {
    const Self = @This();

    pub const StreamingGenerator = struct {};
    pub const StreamToken = struct {};
    pub const StreamState = enum { idle, generating, done };
    pub const GenerationConfig = struct {};

    // Backend types (stub) - must be defined before ServerConfig
    pub const BackendType = enum { local, openai, ollama, anthropic };
    pub const BackendRouter = struct {};
    pub const Backend = struct {};

    // Recovery config stub
    pub const RecoveryConfig = struct {
        max_retries: u32 = 3,
        base_delay_ms: u64 = 1000,
        max_delay_ms: u64 = 30000,
    };

    // Server types (stub) - must match streaming/server.zig ServerConfig
    pub const ServerConfig = struct {
        address: []const u8 = "127.0.0.1:8080",
        auth_token: ?[]const u8 = null,
        allow_health_without_auth: bool = true,
        default_backend: Self.BackendType = .local,
        heartbeat_interval_ms: u64 = 15000,
        max_concurrent_streams: u32 = 100,
        enable_openai_compat: bool = true,
        enable_websocket: bool = true,
        default_model_path: ?[]const u8 = null,
        preload_model: bool = false,
        enable_recovery: bool = true,
        recovery_config: Self.RecoveryConfig = .{},
    };
    pub const StreamingServer = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: Self.ServerConfig) !Self.StreamingServer {
            return .{ .allocator = allocator };
        }

        pub fn deinit(_: *Self.StreamingServer) void {}

        pub fn serve(_: *Self.StreamingServer) !void {
            return error.StreamingDisabled;
        }
    };
    pub const StreamingServerError = error{
        StreamingDisabled,
        BindFailed,
        AcceptFailed,
        AuthenticationFailed,
    };

    // Recovery types (stub)
    pub const StreamRecovery = struct {};
    pub const CircuitBreaker = struct {};
    pub const SessionCache = struct {};
    pub const StreamingMetrics = struct {};
};
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;
pub const ServerConfig = streaming.ServerConfig;
pub const StreamingServer = streaming.StreamingServer;
pub const StreamingServerError = streaming.StreamingServerError;
pub const BackendType = streaming.BackendType;

pub const LlmEngine = struct {};
pub const LlmModel = struct {};
pub const LlmConfig = struct {};
pub const GgufFile = struct {};
pub const BpeTokenizer = struct {};

pub const prompts = struct {
    const Self = @This();

    pub const PersonaType = enum {
        assistant,
        coder,
        writer,
        analyst,
        companion,
        docs,
        reviewer,
        minimal,
        abbey,
        ralph,
    };

    pub const Persona = struct {
        name: []const u8 = "",
        description: []const u8 = "",
        system_prompt: []const u8 = "",
        suggested_temperature: f32 = 0.7,
        include_examples: bool = false,
    };

    pub const PromptBuilder = struct {
        const Builder = @This();

        pub fn init(_: std.mem.Allocator, _: Self.PersonaType) Builder {
            return .{};
        }
        pub fn deinit(_: *Builder) void {}

        pub fn addUserMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
            return error.AiDisabled;
        }

        pub fn addSystemMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
            return error.AiDisabled;
        }

        pub fn addAssistantMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
            return error.AiDisabled;
        }

        pub fn build(_: *Builder, _: Self.PromptFormat) error{AiDisabled}![]u8 {
            return error.AiDisabled;
        }

        pub fn exportDebug(_: *Builder) error{AiDisabled}![]u8 {
            return error.AiDisabled;
        }

        pub fn addMessage(_: *Builder, _: Self.Role, _: []const u8) error{AiDisabled}!void {
            return error.AiDisabled;
        }
    };

    pub const PromptFormat = enum { plain, chat, text };
    pub const Message = struct {};
    pub const Role = enum { system, user, assistant, tool };

    pub fn getPersona(_: Self.PersonaType) Self.Persona {
        return .{};
    }

    pub fn listPersonas() []const Self.PersonaType {
        return &[_]Self.PersonaType{
            .assistant,
            .coder,
            .writer,
            .analyst,
            .companion,
            .docs,
            .reviewer,
            .minimal,
            .abbey,
            .ralph,
        };
    }

    pub fn createBuilder(allocator: std.mem.Allocator) Self.PromptBuilder {
        return Self.PromptBuilder.init(allocator, .assistant);
    }

    pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: Self.PersonaType) Self.PromptBuilder {
        return Self.PromptBuilder.init(allocator, persona_type);
    }
};
pub const PromptBuilder = prompts.PromptBuilder;
pub const Persona = prompts.Persona;
pub const PersonaType = prompts.PersonaType;
pub const PromptFormat = prompts.PromptFormat;

pub const abbey = struct {
    pub fn createEngine(allocator: std.mem.Allocator) Error!AbbeyInstance {
        _ = allocator;
        return error.AiDisabled;
    }

    pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: AbbeyConfig) Error!AbbeyInstance {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }
};
pub const AbbeyInstance = struct {
    pub fn runRalphLoop(_: *AbbeyInstance, _: []const u8, _: usize) Error![]const u8 {
        return error.AiDisabled;
    }

    pub fn deinit(_: *AbbeyInstance) void {}
};
pub const Abbey = struct {};
pub const AbbeyConfig = struct {};
pub const AbbeyResponse = struct {};
pub const AbbeyStats = struct {};
pub const ReasoningChain = struct {};
pub const ReasoningStep = struct {};
pub const Confidence = struct {};
pub const ConfidenceLevel = enum { low, medium, high };
pub const EmotionalState = struct {};
pub const EmotionType = enum { neutral };
pub const ConversationContext = struct {};
pub const TopicTracker = struct {};

pub const explore = struct {
    const E = @This();

    pub fn isEnabled() bool {
        return false;
    }

    pub const ExploreLevel = enum { quick, medium, thorough, shallow, deep };
    pub const OutputFormat = enum { human, json, compact, yaml };
    pub const FileType = enum { zig, all };
    pub const FileFilter = struct {};
    pub const SearchScope = struct {};
    pub const SearchOptions = struct {};
    pub const ExploreResult = struct {
        matches_found: usize = 0,
        files_searched: usize = 0,
        duration_ms: u64 = 0,

        pub fn deinit(_: *@This()) void {}
        pub fn format(_: *@This(), _: E.OutputFormat, _: std.io.AnyWriter) !void {}
        pub fn formatHuman(_: *@This(), _: anytype) void {}
        pub fn formatJson(_: *@This(), _: anytype) void {}
        pub fn formatJSON(_: *@This(), _: anytype) void {}
        pub fn formatCompact(_: *@This(), _: anytype) void {}
    };
    pub const Match = struct {};
    pub const MatchType = enum { exact, fuzzy };
    pub const ExploreError = error{AiDisabled};
    pub const ExplorationStats = struct {};
    pub const ExploreAgent = struct {
        pub fn init(_: std.mem.Allocator, _: E.ExploreConfig) E.ExploreError!@This() {
            return error.AiDisabled;
        }

        pub fn deinit(_: *@This()) void {}

        pub fn explore(_: *@This(), _: []const u8, _: []const u8) E.ExploreError!E.ExploreResult {
            return error.AiDisabled;
        }
    };
    pub const QueryIntent = enum { search, understand, explain };
    pub const ParsedQuery = struct {};
    pub const QueryUnderstanding = struct {};

    pub const ExploreConfig = struct {
        level: E.ExploreLevel = .medium,
        output_format: E.OutputFormat = .human,
        max_files: usize = 10000,
        max_depth: usize = 20,
        timeout_ms: u64 = 60000,
        case_sensitive: bool = false,
        use_regex: bool = false,
        include_patterns: []const []const u8 = &.{},
        exclude_patterns: []const []const u8 = &.{},
        parallel_io: bool = true,
        worker_count: ?usize = null,

        pub fn defaultForLevel(_: E.ExploreLevel) @This() {
            return .{};
        }
    };
};
pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;

pub const memory = struct {
    pub const ShortTermMemory = struct {};
    pub const SlidingWindowMemory = struct {};
    pub const SummarizingMemory = struct {};
    pub const LongTermMemory = struct {};
    pub const MemoryManager = struct {};
    pub const MemoryConfig = struct {};
    pub const MemoryType = enum { short_term, sliding_window, summarizing, long_term };
    pub const MessageRole = enum {
        system,
        user,
        assistant,
        tool,

        pub fn toString(self: MessageRole) []const u8 {
            return switch (self) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
                .tool => "tool",
            };
        }
    };
    pub const Message = struct {
        role: MessageRole,
        content: []const u8,
        name: ?[]const u8 = null,
        timestamp: i64 = 0,
        token_count: usize = 0,
        metadata: ?[]const u8 = null,
    };
    pub const ConversationContext = struct {};
    pub const MemoryStats = struct {};
    pub const persistence = struct {
        const P = @This();
        const M = stub_root.memory; // Parent memory module

        pub const SessionConfig = struct {};

        pub const SessionData = struct {
            id: []const u8 = "",
            name: []const u8 = "",
            created_at: i64 = 0,
            updated_at: i64 = 0,
            messages: []const M.Message = &.{},
            config: P.SessionConfig = .{},

            pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
        };

        pub const SessionMeta = struct {
            id: []const u8 = "",
            name: []const u8 = "",
            created_at: i64 = 0,
            updated_at: i64 = 0,
            message_count: usize = 0,
            total_tokens: usize = 0,
        };
        pub const PersistenceError = error{AiDisabled};

        pub const SessionStore = struct {
            const Store = @This();
            allocator: std.mem.Allocator = undefined,
            base_dir: []const u8 = "",

            pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) Store {
                return .{
                    .allocator = allocator,
                    .base_dir = base_dir,
                };
            }

            pub fn deinit(_: *Store) void {}

            pub fn loadSession(_: *Store, _: []const u8) P.PersistenceError!P.SessionData {
                return error.AiDisabled;
            }

            pub fn listSessions(_: *Store) P.PersistenceError![]P.SessionMeta {
                return error.AiDisabled;
            }

            pub fn saveSession(_: *Store, _: P.SessionData) P.PersistenceError!void {
                return error.AiDisabled;
            }

            pub fn deleteSession(_: *Store, _: []const u8) P.PersistenceError!void {
                return error.AiDisabled;
            }
        };
    };
    pub const SessionStore = persistence.SessionStore;
    pub const SessionData = persistence.SessionData;
    pub const SessionMeta = persistence.SessionMeta;
    pub const SessionConfig = persistence.SessionConfig;
    pub const PersistenceError = persistence.PersistenceError;
};
pub const federated = struct {};
pub const rag = struct {};
pub const templates = struct {};
pub const eval = struct {};

// Orchestration - Multi-model coordination (stub re-exports)
pub const Orchestrator = orchestration.Orchestrator;
pub const OrchestrationConfig = orchestration.OrchestrationConfig;
pub const OrchestrationError = orchestration.OrchestrationError;
pub const RoutingStrategy = orchestration.RoutingStrategy;
pub const TaskType = orchestration.TaskType;
pub const RouteResult = orchestration.RouteResult;
pub const EnsembleMethod = orchestration.EnsembleMethod;
pub const EnsembleResult = orchestration.EnsembleResult;
pub const FallbackPolicy = orchestration.FallbackPolicy;
pub const HealthStatus = orchestration.HealthStatus;
pub const ModelBackend = orchestration.ModelBackend;
pub const ModelCapability = orchestration.Capability;
pub const OrchestrationModelConfig = orchestration.ModelConfig;

pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training };

    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) Error!*Context {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getLlm(_: *Context) Error!*llm.Context {
        return error.AiDisabled;
    }

    pub fn getEmbeddings(_: *Context) Error!*embeddings.Context {
        return error.AiDisabled;
    }

    pub fn getAgents(_: *Context) Error!*agents.Context {
        return error.AiDisabled;
    }

    pub fn getTraining(_: *Context) Error!*training.Context {
        return error.AiDisabled;
    }

    pub fn isSubFeatureEnabled(_: *Context, _: SubFeature) bool {
        return false;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isLlmEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.AiDisabled;
}

pub fn deinit() void {}

pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    _ = allocator;
    return .{};
}

pub fn train(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingReport {
    return error.AiDisabled;
}

pub fn trainWithResult(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingResult {
    return error.AiDisabled;
}

pub fn createAgent(_: std.mem.Allocator, _: []const u8) Error!Agent {
    return error.AiDisabled;
}

pub fn createTransformer(_: TransformerConfig) TransformerModel {
    return .{};
}

pub fn inferText(_: std.mem.Allocator, _: []const u8) Error![]u8 {
    return error.AiDisabled;
}

pub fn embedText(_: std.mem.Allocator, _: []const u8) Error![]f32 {
    return error.AiDisabled;
}

pub fn encodeTokens(_: std.mem.Allocator, _: []const u8) Error![]u32 {
    return error.AiDisabled;
}

pub fn decodeTokens(_: std.mem.Allocator, _: []const u32) Error![]u8 {
    return error.AiDisabled;
}

pub fn loadCheckpoint(_: std.mem.Allocator, _: []const u8) Error!Checkpoint {
    return error.AiDisabled;
}
