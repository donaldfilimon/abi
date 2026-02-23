//! Abbey AI stub — active when AI sub-feature is disabled.

const std = @import("std");

// ── Core Modules (inline stubs) ────────────────────────────────────────────

pub const core = struct {
    pub const types = struct {
        pub const InstanceId = u64;
        pub const SessionId = u64;
        pub const ConfidenceLevel = StubConfidenceLevel;
        pub const Confidence = StubConfidence;
        pub const EmotionType = StubEmotionType;
        pub const EmotionalState = StubEmotionalState;
        pub const Role = StubRole;
        pub const Message = StubMessage;
        pub const TrustLevel = StubTrustLevel;
        pub const Relationship = StubRelationship;
        pub const Topic = StubTopic;
        pub const Response = StubResponse;
        pub const AbbeyError = StubAbbeyError;
        pub const getTimestampNs = stubTimestampNs;
        pub const getTimestampMs = stubTimestampMs;
        pub const getTimestampSec = stubTimestampSec;
    };
    pub const config = struct {
        pub const AbbeyConfig = StubAbbeyConfig;
        pub const BehaviorConfig = StubBehaviorConfig;
        pub const MemoryConfig = StubMemoryConfig;
        pub const ReasoningConfig = StubReasoningConfig;
        pub const EmotionConfig = StubEmotionConfig;
        pub const LearningConfig = StubLearningConfig;
        pub const LLMConfig = StubLLMConfig;
        pub const ServerConfig = StubServerConfig;
        pub const DiscordConfig = StubDiscordConfig;
        pub const ConfigBuilder = StubConfigBuilder;
        pub const loadFromEnvironment = stubLoadFromEnvironment;
    };
};

pub const neural = struct {
    pub const Tensor = StubTensor;
    pub const F32Tensor = StubTensor;
    pub const LinearLayer = struct {};
    pub const EmbeddingLayer = struct {};
    pub const LayerNorm = struct {};
    pub const MultiHeadAttention = struct {};
    pub const SelfAttention = struct {};
    pub const CrossAttention = struct {};
    pub const AdaptiveAttention = struct {};
    pub const OnlineLearner = struct {};
    pub const ReplayBuffer = struct {};
    pub const AdamOptimizer = struct {};
};

pub const memory = struct {
    pub const Episode = struct {};
    pub const EpisodicMemory = struct {};
    pub const Knowledge = struct {};
    pub const SemanticMemory = struct {};
    pub const WorkingItem = struct {};
    pub const WorkingMemory = StubWorkingMemory;
    pub const MemoryManager = StubMemoryManager;
};

pub const reasoning = struct {
    pub const ReasoningChain = StubReasoningChain;
    pub const ReasoningStep = StubReasoningStep;
    pub const StepType = StubStepType;
    pub const Confidence = StubConfidence;
};

pub const emotions = struct {
    pub const EmotionalState = StubEmotionalState;
    pub const EmotionType = StubEmotionType;
};

pub const context = struct {
    pub const ConversationContext = StubConversationContext;
    pub const TopicTracker = StubTopicTracker;
    pub const ContextWindow = struct {};
};

pub const calibration = struct {
    pub const Evidence = struct {};
    pub const CalibrationResult = struct {};
    pub const ConfidenceCalibrator = struct {};
    pub const QueryAnalyzer = struct {};
};

pub const client = struct {
    pub const ChatMessage = struct {};
    pub const CompletionRequest = struct {};
    pub const CompletionResponse = struct {};
    pub const StreamChunk = struct {};
    pub const LLMClient = struct {};
    pub const EchoBackend = struct {};
    pub const ClientWrapper = struct {};
    pub const RetryHandler = struct {};
    pub fn createClient(_: std.mem.Allocator, _: anytype) !@This().LLMClient {
        return error.FeatureDisabled;
    }
};

pub const engine = struct {
    pub const AbbeyEngine = StubAbbeyEngine;
    pub const Response = struct { content: []const u8 = "" };
    pub const EngineStats = struct {};
};

pub const server = struct {
    pub const ServerConfig = struct { host: []const u8 = "127.0.0.1", port: u16 = 8080 };
    pub const AbbeyServerConfig = struct {};
    pub const ServerError = error{FeatureDisabled};
    pub fn serve(_: std.mem.Allocator, _: anytype) !void {
        return error.FeatureDisabled;
    }
    pub fn serveWithConfig(_: std.mem.Allocator, _: anytype, _: anytype) !void {
        return error.FeatureDisabled;
    }
};

pub const discord_bot = struct {
    pub const AbbeyDiscordBot = struct {};
    pub const DiscordBotConfig = struct {};
    pub const DiscordBotError = error{FeatureDisabled};
    pub const SessionManager = struct {};
    pub const BotStats = struct {};
    pub const AbbeyCommands = struct {};
};

pub const custom_framework = struct {
    pub const CustomAI = struct {};
    pub const CustomAIConfig = struct {};
    pub const PersonaTemplate = struct {};
    pub const Builder = struct {};
    pub const Response = struct {};
    pub const Stats = struct {};
    pub fn create(_: std.mem.Allocator, _: anytype) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createFromPersona(_: std.mem.Allocator, _: anytype) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createWithSeedPrompt(_: std.mem.Allocator, _: anytype) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createResearcher(_: std.mem.Allocator) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createCoder(_: std.mem.Allocator) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createWriter(_: std.mem.Allocator) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createCompanion(_: std.mem.Allocator) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createOpinionated(_: std.mem.Allocator) !@This().CustomAI {
        return error.FeatureDisabled;
    }
};

pub const advanced = struct {
    pub const TaskProfile = struct { complexity: f32 = 0 };
    pub const TaskDomain = enum { general };
    pub const LearningStrategy = struct {};
    pub const MetaLearner = struct {};
    pub const FewShotLearner = struct {};
    pub const CurriculumScheduler = struct {};
    pub const MentalModel = struct { trust_level: f32 = 0 };
    pub const BeliefSystem = struct {};
    pub const KnowledgeState = struct {};
    pub const IntentionTracker = struct {};
    pub const EmotionalModel = struct {};
    pub const TheoryOfMind = StubTheoryOfMind;
    pub const ProblemDecomposition = struct {};
    pub const SubProblem = struct {};
    pub const ExecutionPlan = struct {};
    pub const ProblemDecomposer = struct {};
    pub const CounterfactualReasoner = struct {};
    pub const SelfEvaluation = struct { overall_quality: f32 = 0 };
    pub const UncertaintyArea = struct {};
    pub const DetectedBias = struct {};
    pub const ReasoningQuality = struct {};
    pub const SelfReflectionEngine = StubSelfReflectionEngine;
    pub const AdvancedCognition = StubAdvancedCognition;
    pub const CognitiveResult = struct { task_profile: @This().TaskProfile = .{}, cognitive_load: f32 = 0 };
    pub const CognitiveState = struct {};
};

// ── Type Re-exports (Core) ─────────────────────────────────────────────────

pub const InstanceId = core.types.InstanceId;
pub const SessionId = core.types.SessionId;
pub const ConfidenceLevel = StubConfidenceLevel;
pub const Confidence = StubConfidence;
pub const EmotionType = StubEmotionType;
pub const EmotionalState = StubEmotionalState;
pub const Role = StubRole;
pub const Message = StubMessage;
pub const TrustLevel = StubTrustLevel;
pub const Relationship = StubRelationship;
pub const Topic = StubTopic;
pub const Response = StubResponse;
pub const AbbeyError = StubAbbeyError;
pub const AbbeyConfig = StubAbbeyConfig;
pub const BehaviorConfig = StubBehaviorConfig;
pub const MemoryConfig = StubMemoryConfig;
pub const ReasoningConfig = StubReasoningConfig;
pub const EmotionConfig = StubEmotionConfig;
pub const LearningConfig = StubLearningConfig;
pub const LLMConfig = StubLLMConfig;
pub const ServerConfig = StubServerConfig;
pub const DiscordConfig = StubDiscordConfig;
pub const ConfigBuilder = StubConfigBuilder;

// ── Type Re-exports (Neural) ──────────────────────────────────────────────

pub const Tensor = StubTensor;
pub const F32Tensor = StubTensor;
pub const LinearLayer = neural.LinearLayer;
pub const EmbeddingLayer = neural.EmbeddingLayer;
pub const LayerNorm = neural.LayerNorm;
pub const MultiHeadAttention = neural.MultiHeadAttention;
pub const SelfAttention = neural.SelfAttention;
pub const CrossAttention = neural.CrossAttention;
pub const AdaptiveAttention = neural.AdaptiveAttention;
pub const OnlineLearner = neural.OnlineLearner;
pub const ReplayBuffer = neural.ReplayBuffer;
pub const AdamOptimizer = neural.AdamOptimizer;

// ── Type Re-exports (Memory) ──────────────────────────────────────────────

pub const Episode = memory.Episode;
pub const EpisodicMemory = memory.EpisodicMemory;
pub const Knowledge = memory.Knowledge;
pub const SemanticMemory = memory.SemanticMemory;
pub const WorkingItem = memory.WorkingItem;
pub const WorkingMemory = StubWorkingMemory;
pub const MemoryManager = StubMemoryManager;
pub const MemoryStats = StubMemoryManager.MemoryStats;

// ── Type Re-exports (Legacy) ──────────────────────────────────────────────

pub const ReasoningChain = StubReasoningChain;
pub const ReasoningStep = StubReasoningStep;
pub const StepType = StubStepType;
pub const ConversationContext = StubConversationContext;
pub const TopicTracker = StubTopicTracker;
pub const ContextWindow = context.ContextWindow;

// ── Type Re-exports (Calibration/Client/Engine) ───────────────────────────

pub const Evidence = calibration.Evidence;
pub const CalibrationResult = calibration.CalibrationResult;
pub const ConfidenceCalibrator = calibration.ConfidenceCalibrator;
pub const QueryAnalyzer = calibration.QueryAnalyzer;
pub const ChatMessage = client.ChatMessage;
pub const CompletionRequest = client.CompletionRequest;
pub const CompletionResponse = client.CompletionResponse;
pub const StreamChunk = client.StreamChunk;
pub const LLMClient = client.LLMClient;
pub const EchoBackend = client.EchoBackend;
pub const createClient = client.createClient;
pub const ClientWrapper = client.ClientWrapper;
pub const RetryHandler = client.RetryHandler;
pub const AbbeyEngine = StubAbbeyEngine;
pub const EngineResponse = engine.Response;
pub const EngineStats = engine.EngineStats;

// ── Type Re-exports (Server/Discord) ──────────────────────────────────────

pub const HttpServerConfig = server.ServerConfig;
pub const AbbeyServerConfig = server.AbbeyServerConfig;
pub const ServerError = server.ServerError;
pub const serveHttp = server.serve;
pub const serveHttpWithConfig = server.serveWithConfig;
pub const AbbeyDiscordBot = discord_bot.AbbeyDiscordBot;
pub const DiscordBotConfig = discord_bot.DiscordBotConfig;
pub const DiscordBotError = discord_bot.DiscordBotError;
pub const SessionManager = discord_bot.SessionManager;
pub const BotStats = discord_bot.BotStats;
pub const AbbeyCommands = discord_bot.AbbeyCommands;

// ── Type Re-exports (Advanced) ────────────────────────────────────────────

pub const TaskProfile = advanced.TaskProfile;
pub const TaskDomain = advanced.TaskDomain;
pub const LearningStrategy = advanced.LearningStrategy;
pub const MetaLearner = advanced.MetaLearner;
pub const FewShotLearner = advanced.FewShotLearner;
pub const CurriculumScheduler = advanced.CurriculumScheduler;
pub const MentalModel = advanced.MentalModel;
pub const BeliefSystem = advanced.BeliefSystem;
pub const KnowledgeState = advanced.KnowledgeState;
pub const IntentionTracker = advanced.IntentionTracker;
pub const EmotionalModel = advanced.EmotionalModel;
pub const TheoryOfMind = StubTheoryOfMind;
pub const ProblemDecomposition = advanced.ProblemDecomposition;
pub const SubProblem = advanced.SubProblem;
pub const ExecutionPlan = advanced.ExecutionPlan;
pub const ProblemDecomposer = advanced.ProblemDecomposer;
pub const CounterfactualReasoner = advanced.CounterfactualReasoner;
pub const SelfEvaluation = advanced.SelfEvaluation;
pub const UncertaintyArea = advanced.UncertaintyArea;
pub const DetectedBias = advanced.DetectedBias;
pub const ReasoningQuality = advanced.ReasoningQuality;
pub const SelfReflectionEngine = StubSelfReflectionEngine;
pub const AdvancedCognition = StubAdvancedCognition;
pub const CognitiveResult = advanced.CognitiveResult;
pub const CognitiveState = advanced.CognitiveState;

// ── Type Re-exports (Custom Framework) ────────────────────────────────────

pub const CustomAI = custom_framework.CustomAI;
pub const CustomAIConfig = custom_framework.CustomAIConfig;
pub const PersonaTemplate = custom_framework.PersonaTemplate;
pub const CustomAIBuilder = custom_framework.Builder;
pub const CustomAIResponse = custom_framework.Response;
pub const CustomAIStats = custom_framework.Stats;
pub const createCustomAI = custom_framework.create;
pub const createFromPersona = custom_framework.createFromPersona;
pub const createWithSeedPrompt = custom_framework.createWithSeedPrompt;
pub const createResearcher = custom_framework.createResearcher;
pub const createCoder = custom_framework.createCoder;
pub const createWriter = custom_framework.createWriter;
pub const createCompanion = custom_framework.createCompanion;
pub const createOpinionated = custom_framework.createOpinionated;
pub const Stats = custom_framework.Stats;

// ── Legacy Abbey ───────────────────────────────────────────────────────────

pub const Abbey = struct {
    pub const LegacyConfig = struct {
        name: []const u8 = "Abbey",
        enable_emotions: bool = true,
        enable_reasoning_log: bool = true,
        enable_topic_tracking: bool = true,
        base_temperature: f32 = 0.7,
        max_reasoning_steps: usize = 10,
        confidence_threshold: f32 = 0.7,
        research_first: bool = true,
    };
    allocator: std.mem.Allocator = undefined,
    turn_count: usize = 0,
    relationship_score: f32 = 0.5,
    pub fn init(_: std.mem.Allocator, _: LegacyConfig) !Abbey {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Abbey) void {}
    pub fn process(_: *Abbey, _: []const u8) !LegacyResponse {
        return error.FeatureDisabled;
    }
    pub fn getEmotionalState(_: *const Abbey) StubEmotionalState {
        return .{};
    }
    pub fn getStats(_: *const Abbey) LegacyStats {
        return .{};
    }
    pub fn clearConversation(_: *Abbey) void {}
    pub fn reset(_: *Abbey) void {}
};

pub const LegacyResponse = struct { content: []const u8 = "", confidence: StubConfidence = .{}, emotional_context: StubEmotionalState = .{}, reasoning_summary: ?[]const u8 = null, topics: []const []const u8 = &.{} };
pub const LegacyStats = struct { turn_count: usize = 0, relationship_score: f32 = 0, current_emotion: StubEmotionType = .neutral, topics_discussed: usize = 0 };

// ── Ralph Swarm (parallel multi-agent) ─────────────────────────────────────

pub const ralph_swarm = struct {
    pub const ParallelRalphContext = struct {
        allocator: std.mem.Allocator,
        bus: *ralph_multi.RalphBus,
        goals: []const []const u8,
        results: []?[]const u8,
        max_iterations: usize,
        post_result_to_bus: bool = true,
    };

    pub fn parallelRalphWorker(_: *ParallelRalphContext, _: u32) void {}
};

// ── Ralph Multi-Agent Coordination ─────────────────────────────────────────

pub const ralph_multi = struct {
    pub const max_message_content_len = 1024;

    pub const RalphMessageKind = enum(u8) {
        task_result,
        handoff,
        skill_share,
        coordination,
    };

    pub const RalphMessage = struct {
        from_id: u32 = 0,
        to_id: u32 = 0,
        kind: RalphMessageKind = .task_result,
        content_len: u16 = 0,
        content: [max_message_content_len]u8 = [_]u8{0} ** max_message_content_len,

        pub fn setContent(self: *RalphMessage, slice: []const u8) void {
            const n = @min(slice.len, max_message_content_len);
            @memcpy(self.content[0..n], slice[0..n]);
            self.content_len = @intCast(n);
        }

        pub fn getContent(self: *const RalphMessage) []const u8 {
            return self.content[0..self.content_len];
        }
    };

    pub const RalphBus = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: usize) !RalphBus {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *RalphBus) void {}
        pub fn send(_: *RalphBus, _: RalphMessage) !void {
            return error.FeatureDisabled;
        }
        pub fn trySend(_: *RalphBus, _: RalphMessage) bool {
            return false;
        }
        pub fn recv(_: *RalphBus) !RalphMessage {
            return error.FeatureDisabled;
        }
        pub fn tryRecv(_: *RalphBus) ?RalphMessage {
            return null;
        }
        pub fn recvFor(_: *RalphBus, _: u32) ?RalphMessage {
            return null;
        }
        pub fn close(_: *RalphBus) void {}
        pub fn isClosed(_: *const RalphBus) bool {
            return true;
        }
    };
};

// ── Convenience Functions ──────────────────────────────────────────────────

pub fn createEngine(_: std.mem.Allocator) !StubAbbeyEngine {
    return error.FeatureDisabled;
}
pub fn createEngineWithConfig(_: std.mem.Allocator, _: StubAbbeyConfig) !StubAbbeyEngine {
    return error.FeatureDisabled;
}
pub fn builder() StubConfigBuilder {
    return StubConfigBuilder.init();
}
pub fn createAdvancedCognition(_: std.mem.Allocator) !StubAdvancedCognition {
    return error.FeatureDisabled;
}
pub fn createAdvancedCognitionWithConfig(_: std.mem.Allocator, _: StubAdvancedCognition.Config) !StubAdvancedCognition {
    return error.FeatureDisabled;
}

// ── Internal Stub Type Definitions ─────────────────────────────────────────

const StubConfidenceLevel = enum { very_low, low, medium, high, very_high };
const StubConfidence = struct { level: StubConfidenceLevel = .medium, score: f32 = 0, reasoning: []const u8 = "" };
const StubEmotionType = enum { neutral, happy, sad, curious, frustrated, excited, confused, thoughtful };
const StubEmotionalState = struct {
    detected: StubEmotionType = .neutral,
    intensity: f32 = 0,
    valence: f32 = 0,
    pub fn init() StubEmotionalState {
        return .{};
    }
    pub fn detectFromText(_: *StubEmotionalState, _: []const u8) void {}
};
const StubRole = enum { system, user, assistant, tool };
const StubMessage = struct { role: StubRole = .user, content: []const u8 = "", name: ?[]const u8 = null, timestamp: i64 = 0, token_count: ?usize = null, metadata: ?[]const u8 = null };
const StubTrustLevel = enum { unknown, low, medium, high, verified };
const StubRelationship = struct { user_id: []const u8 = "", trust: StubTrustLevel = .unknown, interaction_count: u32 = 0, score: f32 = 0.5 };
const StubTopic = struct { name: []const u8 = "", relevance: f32 = 0, mentions: u32 = 0 };
const StubResponse = struct { content: []const u8 = "", confidence: StubConfidence = .{}, emotional_context: ?StubEmotionalState = null, reasoning_summary: ?[]const u8 = null };
const StubAbbeyError = error{ FeatureDisabled, InitializationFailed, InferenceFailed, MemoryFull, InvalidInput };
const StubAbbeyConfig = struct { name: []const u8 = "Abbey", behavior: StubBehaviorConfig = .{} };
const StubBehaviorConfig = struct { base_temperature: f32 = 0.7, research_first: bool = true, enable_emotions: bool = true, enable_reasoning_log: bool = true };
const StubMemoryConfig = struct { max_entries: usize = 1000, embedding_dim: usize = 384 };
const StubReasoningConfig = struct { max_steps: usize = 10, confidence_threshold: f32 = 0.7 };
const StubEmotionConfig = struct { enabled: bool = true, intensity_decay: f32 = 0.1 };
const StubLearningConfig = struct { enabled: bool = true, learning_rate: f32 = 0.01 };
const StubLLMConfig = struct { backend: enum { echo, local, api } = .echo, model_path: ?[]const u8 = null, api_key: ?[]const u8 = null };
const StubServerConfig = struct { host: []const u8 = "127.0.0.1", port: u16 = 8080 };
const StubDiscordConfig = struct { token: ?[]const u8 = null, prefix: []const u8 = "!" };

const StubConfigBuilder = struct {
    config: StubAbbeyConfig = .{},
    pub fn init() StubConfigBuilder {
        return .{};
    }
    pub fn name(self: *StubConfigBuilder, n: []const u8) *StubConfigBuilder {
        self.config.name = n;
        return self;
    }
    pub fn temperature(self: *StubConfigBuilder, _: f32) *StubConfigBuilder {
        return self;
    }
    pub fn researchFirst(self: *StubConfigBuilder, _: bool) *StubConfigBuilder {
        return self;
    }
    pub fn llmBackend(self: *StubConfigBuilder, _: @TypeOf((StubLLMConfig{}).backend)) *StubConfigBuilder {
        return self;
    }
    pub fn build(_: *StubConfigBuilder) !StubAbbeyConfig {
        return error.FeatureDisabled;
    }
};

const StubTensor = struct {
    pub fn zeros(_: std.mem.Allocator, _: []const usize) !StubTensor {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *StubTensor) void {}
    pub fn size(_: *const StubTensor) usize {
        return 0;
    }
};

const StubWorkingMemory = struct {
    items: struct { items: []const u8 = &.{} } = .{},
    pub fn init(_: std.mem.Allocator, _: usize, _: usize) StubWorkingMemory {
        return .{};
    }
    pub fn deinit(_: *StubWorkingMemory) void {}
    pub fn add(_: *StubWorkingMemory, _: []const u8, _: anytype, _: f32) !usize {
        return error.FeatureDisabled;
    }
};

const StubMemoryManager = struct {
    pub const MemoryStats = struct {};
    pub fn init(_: std.mem.Allocator, _: anytype) !StubMemoryManager {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *StubMemoryManager) void {}
    pub fn addMessage(_: *StubMemoryManager, _: anytype) !void {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const StubMemoryManager) @This().MemoryStats {
        return .{};
    }
    pub fn clear(_: *StubMemoryManager) void {}
};

const StubAbbeyEngine = struct {
    conversation_active: bool = false,
    pub fn init(_: std.mem.Allocator, _: StubAbbeyConfig) !StubAbbeyEngine {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *StubAbbeyEngine) void {}
    pub fn runRalphLoop(_: *StubAbbeyEngine, _: []const u8, _: usize) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn storeSkill(_: *StubAbbeyEngine, _: []const u8) !u64 {
        return error.FeatureDisabled;
    }
    pub fn extractAndStoreSkill(_: *StubAbbeyEngine, _: []const u8, _: []const u8) !bool {
        return error.FeatureDisabled;
    }
    pub fn recordRalphRun(_: *StubAbbeyEngine, _: []const u8, _: usize, _: usize, _: f32) !void {
        return error.FeatureDisabled;
    }
};

const StubReasoningChain = struct {
    allocator: std.mem.Allocator = undefined,
    pub fn init(allocator: std.mem.Allocator, _: []const u8) StubReasoningChain {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *StubReasoningChain) void {}
    pub fn addStep(_: *StubReasoningChain, _: StubStepType, _: []const u8, _: StubConfidence) !void {
        return error.FeatureDisabled;
    }
    pub fn finalize(_: *StubReasoningChain) !void {}
    pub fn getOverallConfidence(_: *const StubReasoningChain) StubConfidence {
        return .{};
    }
    pub fn getSummary(_: *const StubReasoningChain, _: std.mem.Allocator) !?[]const u8 {
        return null;
    }
};

const StubReasoningStep = struct {};
const StubStepType = enum { assessment, analysis, synthesis, conclusion };

const StubConversationContext = struct {
    pub fn init(_: std.mem.Allocator) StubConversationContext {
        return .{};
    }
    pub fn deinit(_: *StubConversationContext) void {}
    pub fn clear(_: *StubConversationContext) void {}
};

const StubTopicTracker = struct {
    pub fn init(_: std.mem.Allocator) StubTopicTracker {
        return .{};
    }
    pub fn deinit(_: *StubTopicTracker) void {}
    pub fn updateFromMessage(_: *StubTopicTracker, _: []const u8) !void {}
    pub fn getCurrentTopics(_: *const StubTopicTracker) []const []const u8 {
        return &.{};
    }
    pub fn getTopicCount(_: *const StubTopicTracker) usize {
        return 0;
    }
    pub fn clear(_: *StubTopicTracker) void {}
};

const StubTheoryOfMind = struct {
    pub fn init(_: std.mem.Allocator) StubTheoryOfMind {
        return .{};
    }
    pub fn deinit(_: *StubTheoryOfMind) void {}
    pub fn getModel(_: *StubTheoryOfMind, _: []const u8) !*advanced.MentalModel {
        return error.FeatureDisabled;
    }
};

const StubSelfReflectionEngine = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) StubSelfReflectionEngine {
        return .{};
    }
    pub fn deinit(_: *StubSelfReflectionEngine) void {}
    pub fn evaluate(_: *StubSelfReflectionEngine, _: []const u8, _: []const u8, _: anytype) !advanced.SelfEvaluation {
        return error.FeatureDisabled;
    }
};

const StubAdvancedCognition = struct {
    pub const Config = struct {};
    pub fn init(_: std.mem.Allocator, _: Config) !StubAdvancedCognition {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *StubAdvancedCognition) void {}
    pub fn process(_: *StubAdvancedCognition, _: []const u8, _: []const u8) !advanced.CognitiveResult {
        return error.FeatureDisabled;
    }
};

fn stubTimestampNs() i128 {
    return 0;
}
fn stubTimestampMs() i64 {
    return 0;
}
fn stubTimestampSec() i64 {
    return 0;
}
fn stubLoadFromEnvironment() !StubAbbeyConfig {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
