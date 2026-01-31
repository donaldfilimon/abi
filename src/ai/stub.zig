//! AI Stub Module
//!
//! Stub implementation when AI is disabled at compile time.

const std = @import("std");
const config_module = @import("../config/mod.zig");

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
pub const core = @import("core/mod.zig");
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const training = @import("training/stub.zig");
pub const database = @import("database/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const documents = @import("documents/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const models = @import("models/stub.zig");
pub const memory = @import("memory/stub.zig");
pub const streaming = @import("streaming/stub.zig");
pub const explore = @import("explore/stub.zig");
pub const personas = @import("personas/stub.zig");
pub const rag = @import("rag/stub.zig");
pub const templates = @import("templates/stub.zig");
pub const eval = @import("eval/stub.zig");
pub const federated = struct {};

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
    pub const ModelRegistry = struct {
        pub fn init(_: std.mem.Allocator) ModelRegistry {
            return .{};
        }
    };
};

// Stub types
pub const Agent = agent.Agent;
pub const ModelRegistry = model_registry.ModelRegistry;
pub const ModelInfo = model_registry.ModelInfo;
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = training.TrainError;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const TrainableModel = training.TrainableModel;
pub const trainable_model = training.trainable_model;
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;
pub const TrainableViTModel = training.TrainableViTModel;
pub const TrainableViTConfig = training.TrainableViTConfig;
pub const TrainableViTWeights = training.TrainableViTWeights;
pub const VisionTrainingError = training.VisionTrainingError;
pub const TrainableCLIPModel = training.TrainableCLIPModel;
pub const CLIPTrainingConfig = training.CLIPTrainingConfig;
pub const MultimodalTrainingError = training.MultimodalTrainingError;
pub const TokenizedDataset = training.TokenizedDataset;
pub const DataLoader = training.DataLoader;
pub const BatchIterator = training.BatchIterator;
pub const Batch = training.Batch;
pub const SequencePacker = training.SequencePacker;
pub const parseInstructionDataset = training.parseInstructionDataset;
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const TokenBlock = training.TokenBlock;
pub const encodeTokenBlock = training.encodeTokenBlock;
pub const decodeTokenBlock = training.decodeTokenBlock;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const exportGguf = database.exportGguf;

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

pub const gpu_agent = struct {
    pub const WorkloadType = enum {
        inference,
        training,
        embedding,
        fine_tuning,
        batch_inference,

        pub fn gpuIntensive(self: WorkloadType) bool {
            return switch (self) {
                .training, .fine_tuning => true,
                .inference, .embedding, .batch_inference => false,
            };
        }

        pub fn memoryIntensive(self: WorkloadType) bool {
            return switch (self) {
                .training, .fine_tuning, .batch_inference => true,
                .inference, .embedding => false,
            };
        }

        pub fn name(self: WorkloadType) []const u8 {
            return switch (self) {
                .inference => "Inference",
                .training => "Training",
                .embedding => "Embedding",
                .fine_tuning => "FineTuning",
                .batch_inference => "BatchInference",
            };
        }
    };

    pub const Priority = enum {
        low,
        normal,
        high,
        critical,

        pub fn weight(self: Priority) f32 {
            return switch (self) {
                .low => 0.25,
                .normal => 1.0,
                .high => 2.0,
                .critical => 4.0,
            };
        }

        pub fn name(self: Priority) []const u8 {
            return switch (self) {
                .low => "Low",
                .normal => "Normal",
                .high => "High",
                .critical => "Critical",
            };
        }
    };

    pub const GpuAwareRequest = struct {
        prompt: []const u8,
        workload_type: WorkloadType,
        priority: Priority = .normal,
        max_tokens: u32 = 1024,
        temperature: f32 = 0.7,
        memory_hint_mb: ?u64 = null,
        preferred_backend: ?[]const u8 = null,
        model_id: ?[]const u8 = null,
        stream: bool = false,
        timeout_ms: u64 = 0,
    };

    pub const GpuAwareResponse = struct {
        content: []const u8,
        tokens_generated: u32,
        latency_ms: u64,
        gpu_backend_used: []const u8,
        gpu_memory_used_mb: u64,
        scheduling_confidence: f32,
        energy_estimate_wh: ?f32 = null,
        device_id: u32 = 0,
        truncated: bool = false,
        error_message: ?[]const u8 = null,
    };

    pub const AgentStats = struct {
        total_requests: u64 = 0,
        gpu_accelerated: u64 = 0,
        cpu_fallback: u64 = 0,
        total_tokens: u64 = 0,
        total_latency_ms: u64 = 0,
        learning_episodes: u64 = 0,
        avg_scheduling_confidence: f32 = 0,
        avg_latency_ms: f32 = 0,
        failed_requests: u64 = 0,
        total_gpu_memory_mb: u64 = 0,

        pub fn updateConfidence(self: *AgentStats, new_confidence: f32) void {
            if (self.gpu_accelerated == 0) {
                self.avg_scheduling_confidence = new_confidence;
            } else {
                const n = @as(f32, @floatFromInt(self.gpu_accelerated));
                self.avg_scheduling_confidence =
                    (self.avg_scheduling_confidence * (n - 1) + new_confidence) / n;
            }
        }

        pub fn updateLatency(self: *AgentStats, latency: u64) void {
            if (self.total_requests == 0) {
                self.avg_latency_ms = @floatFromInt(latency);
            } else {
                const n = @as(f32, @floatFromInt(self.total_requests));
                self.avg_latency_ms =
                    (self.avg_latency_ms * (n - 1) + @as(f32, @floatFromInt(latency))) / n;
            }
        }

        pub fn successRate(self: AgentStats) f32 {
            if (self.total_requests == 0) return 1.0;
            const successful = self.total_requests - self.failed_requests;
            return @as(f32, @floatFromInt(successful)) / @as(f32, @floatFromInt(self.total_requests));
        }

        pub fn gpuUtilizationRate(self: AgentStats) f32 {
            if (self.total_requests == 0) return 0.0;
            return @as(f32, @floatFromInt(self.gpu_accelerated)) /
                @as(f32, @floatFromInt(self.total_requests));
        }

        pub fn avgTokensPerRequest(self: AgentStats) f32 {
            if (self.total_requests == 0) return 0.0;
            return @as(f32, @floatFromInt(self.total_tokens)) /
                @as(f32, @floatFromInt(self.total_requests));
        }
    };

    pub const BackendInfo = struct {
        name: []const u8 = "cpu",
        device_count: u32 = 0,
        total_memory_mb: u64 = 0,
        available_memory_mb: u64 = 0,
        is_healthy: bool = false,
    };

    pub const LearningStatsInfo = struct {
        episodes: usize = 0,
        avg_episode_reward: f32 = 0,
        exploration_rate: f32 = 0,
        replay_buffer_size: usize = 0,
    };

    pub const GpuAgent = struct {
        allocator: std.mem.Allocator,
        stats: AgentStats = .{},

        pub fn init(_: std.mem.Allocator) !*GpuAgent {
            return error.AiDisabled;
        }

        pub fn initWithConfig(
            _: std.mem.Allocator,
            _: struct {
                default_timeout_ms: u64 = 30000,
                enable_learning: bool = true,
            },
        ) !*GpuAgent {
            return error.AiDisabled;
        }

        pub fn deinit(_: *GpuAgent) void {}

        pub fn process(_: *GpuAgent, _: GpuAwareRequest) !GpuAwareResponse {
            return error.AiDisabled;
        }

        pub fn getStats(self: *const GpuAgent) AgentStats {
            return self.stats;
        }

        pub fn isGpuEnabled(_: *const GpuAgent) bool {
            return false;
        }

        pub fn isLearningEnabled(_: *const GpuAgent) bool {
            return false;
        }

        pub fn endEpisode(_: *GpuAgent) void {}

        pub fn resetStats(self: *GpuAgent) void {
            self.stats = .{};
        }

        pub fn getBackendsSummary(_: *GpuAgent, _: std.mem.Allocator) ![]const BackendInfo {
            return error.AiDisabled;
        }

        pub fn getLearningStats(_: *GpuAgent) ?LearningStatsInfo {
            return null;
        }
    };
};

pub const discovery = struct {
    pub const DiscoveryConfig = struct {
        custom_paths: []const []const u8 = &.{},
        recursive: bool = true,
        max_depth: u32 = 5,
        extensions: []const []const u8 = &.{ ".gguf", ".bin", ".safetensors" },
        validate_files: bool = true,
        validation_timeout_ms: u32 = 5000,
    };

    pub const ModelFormat = enum {
        gguf,
        safetensors,
        pytorch_bin,
        onnx,
        unknown,

        pub fn fromExtension(_: []const u8) ModelFormat {
            return .unknown;
        }
    };

    pub const QuantizationType = enum {
        f32,
        f16,
        q8_0,
        q8_1,
        q5_0,
        q5_1,
        q4_0,
        q4_1,
        q4_k_m,
        q4_k_s,
        q5_k_m,
        q5_k_s,
        q6_k,
        q2_k,
        q3_k_m,
        q3_k_s,
        iq2_xxs,
        iq2_xs,
        iq3_xxs,
        unknown,

        pub fn bitsPerWeight(self: QuantizationType) f32 {
            return switch (self) {
                .f32 => 32.0,
                .f16 => 16.0,
                .q8_0, .q8_1 => 8.0,
                .q6_k => 6.0,
                .q5_0, .q5_1, .q5_k_m, .q5_k_s => 5.0,
                .q4_0, .q4_1, .q4_k_m, .q4_k_s => 4.0,
                .q3_k_m, .q3_k_s => 3.0,
                .q2_k => 2.0,
                .iq2_xxs, .iq2_xs => 2.5,
                .iq3_xxs => 3.0,
                .unknown => 4.0,
            };
        }
    };

    pub const DiscoveredModel = struct {
        path: []const u8 = "",
        name: []const u8 = "",
        size_bytes: u64 = 0,
        format: ModelFormat = .unknown,
        estimated_params: ?u64 = null,
        quantization: ?QuantizationType = null,
        validated: bool = false,
        modified_time: i128 = 0,

        pub fn deinit(_: *DiscoveredModel, _: std.mem.Allocator) void {}
    };

    pub const SystemCapabilities = struct {
        cpu_cores: u32 = 1,
        total_ram_bytes: u64 = 0,
        available_ram_bytes: u64 = 0,
        gpu_available: bool = false,
        gpu_memory_bytes: u64 = 0,
        gpu_compute_capability: ?f32 = null,
        avx2_available: bool = false,
        avx512_available: bool = false,
        neon_available: bool = false,
        os: std.Target.Os.Tag = .linux,
        arch: std.Target.Cpu.Arch = .x86_64,

        pub fn maxModelSize(self: SystemCapabilities) u64 {
            return self.available_ram_bytes * 80 / 100;
        }

        pub fn recommendedThreads(self: SystemCapabilities) u32 {
            if (self.cpu_cores > 2) return self.cpu_cores - 1;
            return self.cpu_cores;
        }

        pub fn recommendedBatchSize(_: SystemCapabilities, _: u64) u32 {
            return 1;
        }
    };

    pub const AdaptiveConfig = struct {
        num_threads: u32 = 4,
        batch_size: u32 = 1,
        context_length: u32 = 2048,
        use_gpu: bool = false,
        use_mmap: bool = true,
        mlock: bool = false,
        kv_cache_type: KvCacheType = .standard,
        flash_attention: bool = false,
        tensor_parallel: u32 = 1,
        prefill_chunk_size: u32 = 512,

        pub const KvCacheType = enum {
            standard,
            sliding_window,
            paged,
        };
    };

    pub const ModelRequirements = struct {
        min_ram_bytes: u64 = 0,
        min_gpu_memory_bytes: u64 = 0,
        min_compute_capability: f32 = 0,
        requires_avx2: bool = false,
        requires_avx512: bool = false,
        recommended_context: u32 = 2048,
    };

    pub const WarmupResult = struct {
        load_time_ms: u64 = 0,
        first_inference_ms: u64 = 0,
        tokens_per_second: f32 = 0,
        memory_usage_bytes: u64 = 0,
        success: bool = false,
        error_message: ?[]const u8 = null,
        recommended_config: ?AdaptiveConfig = null,
    };

    pub const ModelDiscovery = struct {
        allocator: std.mem.Allocator,
        config: DiscoveryConfig,
        discovered_models: std.ArrayListUnmanaged(DiscoveredModel) = .empty,
        capabilities: SystemCapabilities = .{},

        pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) ModelDiscovery {
            return .{
                .allocator = allocator,
                .config = config,
                .discovered_models = .empty,
                .capabilities = detectCapabilities(),
            };
        }

        pub fn deinit(self: *ModelDiscovery) void {
            for (self.discovered_models.items) |*model| {
                model.deinit(self.allocator);
            }
            self.discovered_models.deinit(self.allocator);
        }

        pub fn scanAll(_: *ModelDiscovery) !void {
            return error.AiDisabled;
        }

        pub fn scanPath(_: *ModelDiscovery, _: []const u8) !void {
            return error.AiDisabled;
        }

        pub fn addModelPath(_: *ModelDiscovery, _: []const u8) !void {
            return error.AiDisabled;
        }

        pub fn addModelWithSize(_: *ModelDiscovery, _: []const u8, _: u64) !void {
            return error.AiDisabled;
        }

        pub fn getModels(self: *ModelDiscovery) []DiscoveredModel {
            return self.discovered_models.items;
        }

        pub fn modelCount(self: *ModelDiscovery) usize {
            return self.discovered_models.items.len;
        }

        pub fn findBestModel(_: *ModelDiscovery, _: ModelRequirements) ?*DiscoveredModel {
            return null;
        }

        pub fn generateConfig(_: *ModelDiscovery, _: *const DiscoveredModel) AdaptiveConfig {
            return .{};
        }
    };

    pub fn detectCapabilities() SystemCapabilities {
        return .{};
    }

    pub fn runWarmup(_: []const u8, _: AdaptiveConfig) WarmupResult {
        return .{};
    }
};

// streaming stub module declared above
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;
pub const ServerConfig = streaming.ServerConfig;
pub const StreamingServer = streaming.StreamingServer;
pub const StreamingServerError = streaming.StreamingServerError;
pub const BackendType = streaming.BackendType;

pub const LlmEngine = llm.Engine;
pub const LlmModel = llm.Model;
pub const LlmConfig = llm.InferenceConfig;
pub const GgufFile = llm.GgufFile;
pub const BpeTokenizer = llm.BpeTokenizer;

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
    pub const Abbey = struct {};
    pub const Stats = struct {};
    pub const ReasoningChain = struct {};
    pub const ReasoningStep = struct {};
    pub const ConversationContext = struct {};

    pub const AbbeyInstance = struct {
        pub fn runRalphLoop(_: *AbbeyInstance, _: []const u8, _: usize) Error![]const u8 {
            return error.AiDisabled;
        }

        pub fn deinit(_: *AbbeyInstance) void {}
    };

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
pub const AbbeyInstance = abbey.AbbeyInstance;
pub const Abbey = abbey.Abbey;
pub const AbbeyConfig = core.AbbeyConfig;
pub const AbbeyResponse = core.Response;
pub const AbbeyStats = abbey.Stats;
pub const ReasoningChain = abbey.ReasoningChain;
pub const ReasoningStep = abbey.ReasoningStep;
pub const Confidence = core.Confidence;
pub const ConfidenceLevel = core.ConfidenceLevel;
pub const EmotionalState = core.EmotionalState;
pub const EmotionType = core.EmotionType;
pub const ConversationContext = abbey.ConversationContext;
pub const TopicTracker = core.Topic;

// explore stub module declared above
pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;

// memory stub module declared above

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

// GPU-Aware Agent types
pub const GpuAgent = gpu_agent.GpuAgent;
pub const GpuAwareRequest = gpu_agent.GpuAwareRequest;
pub const GpuAwareResponse = gpu_agent.GpuAwareResponse;
pub const WorkloadType = gpu_agent.WorkloadType;
pub const GpuAgentPriority = gpu_agent.Priority;
pub const GpuAgentStats = gpu_agent.AgentStats;

// Model Auto-Discovery and Adaptive Configuration
pub const ModelDiscovery = discovery.ModelDiscovery;
pub const DiscoveredModel = discovery.DiscoveredModel;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const SystemCapabilities = discovery.SystemCapabilities;
pub const AdaptiveConfig = discovery.AdaptiveConfig;
pub const ModelRequirements = discovery.ModelRequirements;
pub const WarmupResult = discovery.WarmupResult;
pub const detectCapabilities = discovery.detectCapabilities;
pub const runWarmup = discovery.runWarmup;

// Document Understanding
pub const DocumentPipeline = documents.DocumentPipeline;
pub const Document = documents.Document;
pub const DocumentFormat = documents.DocumentFormat;
pub const DocumentElement = documents.DocumentElement;
pub const ElementType = documents.ElementType;
pub const TextSegment = documents.TextSegment;
pub const TextSegmenter = documents.TextSegmenter;
pub const NamedEntity = documents.NamedEntity;
pub const EntityType = documents.EntityType;
pub const EntityExtractor = documents.EntityExtractor;
pub const LayoutAnalyzer = documents.LayoutAnalyzer;
pub const PipelineConfig = documents.PipelineConfig;
pub const SegmentationConfig = documents.SegmentationConfig;

pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training, personas };

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

    pub fn getPersonas(_: *Context) Error!*personas.Context {
        return error.AiDisabled;
    }

    pub fn isSubFeatureEnabled(_: *Context, _: SubFeature) bool {
        return false;
    }

    pub fn getDiscoveredModels(_: *Context) []discovery.DiscoveredModel {
        return &.{};
    }

    pub fn discoveredModelCount(_: *Context) usize {
        return 0;
    }

    pub fn findBestModel(_: *Context, _: discovery.ModelRequirements) ?*discovery.DiscoveredModel {
        return null;
    }

    pub fn generateAdaptiveConfig(_: *Context, _: *const discovery.DiscoveredModel) discovery.AdaptiveConfig {
        return .{};
    }

    pub fn getCapabilities(_: *const Context) discovery.SystemCapabilities {
        return .{};
    }

    pub fn addModelPath(_: *Context, _: []const u8) !void {
        return error.AiDisabled;
    }

    pub fn addModelWithSize(_: *Context, _: []const u8, _: u64) !void {
        return error.AiDisabled;
    }

    pub fn clearDiscoveredModels(_: *Context) void {}
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
