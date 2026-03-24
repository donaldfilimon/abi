//! Abbey AI stub — active when AI sub-feature is disabled.

const std = @import("std");
const abbey_types = @import("types.zig");

// ── Core Modules (inline stubs) ────────────────────────────────────────────

pub const core = struct {
    pub const types = struct {
        pub const InstanceId = abbey_types.InstanceId;
        pub const SessionId = abbey_types.SessionId;
        pub const ConfidenceLevel = abbey_types.ConfidenceLevel;
        pub const Confidence = abbey_types.Confidence;
        pub const EmotionType = abbey_types.EmotionType;
        pub const EmotionalState = abbey_types.EmotionalState;
        pub const Role = abbey_types.Role;
        pub const Message = abbey_types.Message;
        pub const TrustLevel = abbey_types.TrustLevel;
        pub const Relationship = abbey_types.Relationship;
        pub const Topic = abbey_types.Topic;
        pub const Response = abbey_types.Response;
        pub const AbbeyError = abbey_types.AbbeyError;
        pub const getTimestampNs = abbey_types.stubTimestampNs;
        pub const getTimestampMs = abbey_types.stubTimestampMs;
        pub const getTimestampSec = abbey_types.stubTimestampSec;
    };
    pub const config = struct {
        pub const AbbeyConfig = abbey_types.AbbeyConfig;
        pub const BehaviorConfig = abbey_types.BehaviorConfig;
        pub const MemoryConfig = abbey_types.MemoryConfig;
        pub const ReasoningConfig = abbey_types.ReasoningConfig;
        pub const EmotionConfig = abbey_types.EmotionConfig;
        pub const LearningConfig = abbey_types.LearningConfig;
        pub const LLMConfig = abbey_types.LLMConfig;
        pub const ServerConfig = abbey_types.ServerConfig;
        pub const DiscordConfig = abbey_types.DiscordConfig;
        pub const ConfigBuilder = abbey_types.ConfigBuilder;
        pub const loadFromEnvironment = abbey_types.stubLoadFromEnvironment;
    };
};

pub const neural = struct {
    pub const Tensor = abbey_types.Tensor;
    pub const F32Tensor = abbey_types.Tensor;
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
    pub const WorkingMemory = abbey_types.WorkingMemory;
    pub const MemoryManager = abbey_types.MemoryManager;
};

pub const reasoning = struct {
    pub const ReasoningChain = abbey_types.ReasoningChain;
    pub const ReasoningStep = abbey_types.ReasoningStep;
    pub const StepType = abbey_types.StepType;
    pub const Confidence = abbey_types.Confidence;
};

pub const emotions = struct {
    pub const EmotionalState = abbey_types.EmotionalState;
    pub const EmotionType = abbey_types.EmotionType;
};

pub const context = struct {
    pub const ConversationContext = abbey_types.ConversationContext;
    pub const TopicTracker = abbey_types.TopicTracker;
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
    pub const AbbeyEngine = abbey_types.AbbeyEngine;
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
    pub const ProfileTemplate = struct {};
    pub const Builder = struct {};
    pub const Response = struct {};
    pub const Stats = struct {};
    pub fn create(_: std.mem.Allocator, _: anytype) !@This().CustomAI {
        return error.FeatureDisabled;
    }
    pub fn createFromProfile(_: std.mem.Allocator, _: anytype) !@This().CustomAI {
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
    pub const TaskProfile = abbey_types.TaskProfile;
    pub const TaskDomain = abbey_types.TaskDomain;
    pub const LearningStrategy = abbey_types.LearningStrategy;
    pub const MetaLearner = abbey_types.MetaLearner;
    pub const FewShotLearner = abbey_types.FewShotLearner;
    pub const CurriculumScheduler = abbey_types.CurriculumScheduler;
    pub const MentalModel = abbey_types.MentalModel;
    pub const BeliefSystem = abbey_types.BeliefSystem;
    pub const KnowledgeState = abbey_types.KnowledgeState;
    pub const IntentionTracker = abbey_types.IntentionTracker;
    pub const EmotionalModel = abbey_types.EmotionalModel;
    pub const TheoryOfMind = abbey_types.TheoryOfMind;
    pub const ProblemDecomposition = abbey_types.ProblemDecomposition;
    pub const SubProblem = abbey_types.SubProblem;
    pub const ExecutionPlan = abbey_types.ExecutionPlan;
    pub const ProblemDecomposer = abbey_types.ProblemDecomposer;
    pub const CounterfactualReasoner = abbey_types.CounterfactualReasoner;
    pub const SelfEvaluation = abbey_types.SelfEvaluation;
    pub const UncertaintyArea = abbey_types.UncertaintyArea;
    pub const DetectedBias = abbey_types.DetectedBias;
    pub const ReasoningQuality = abbey_types.ReasoningQuality;
    pub const SelfReflectionEngine = abbey_types.SelfReflectionEngine;
    pub const AdvancedCognition = abbey_types.AdvancedCognition;
    pub const CognitiveResult = abbey_types.CognitiveResult;
    pub const CognitiveState = abbey_types.CognitiveState;
};

// ── Top-level type re-exports ──────────────────────────────────────────────

pub const InstanceId = abbey_types.InstanceId;
pub const SessionId = abbey_types.SessionId;
pub const ConfidenceLevel = abbey_types.ConfidenceLevel;
pub const Confidence = abbey_types.Confidence;
pub const EmotionType = abbey_types.EmotionType;
pub const EmotionalState = abbey_types.EmotionalState;
pub const Role = abbey_types.Role;
pub const Message = abbey_types.Message;
pub const TrustLevel = abbey_types.TrustLevel;
pub const Relationship = abbey_types.Relationship;
pub const Topic = abbey_types.Topic;
pub const Response = abbey_types.Response;
pub const AbbeyError = abbey_types.AbbeyError;
pub const AbbeyConfig = abbey_types.AbbeyConfig;
pub const BehaviorConfig = abbey_types.BehaviorConfig;
pub const MemoryConfig = abbey_types.MemoryConfig;
pub const ReasoningConfig = abbey_types.ReasoningConfig;
pub const EmotionConfig = abbey_types.EmotionConfig;
pub const LearningConfig = abbey_types.LearningConfig;
pub const LLMConfig = abbey_types.LLMConfig;
pub const ServerConfig = abbey_types.ServerConfig;
pub const DiscordConfig = abbey_types.DiscordConfig;
pub const ConfigBuilder = abbey_types.ConfigBuilder;
pub const Tensor = abbey_types.Tensor;
pub const F32Tensor = abbey_types.Tensor;
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
pub const Episode = memory.Episode;
pub const EpisodicMemory = memory.EpisodicMemory;
pub const Knowledge = memory.Knowledge;
pub const SemanticMemory = memory.SemanticMemory;
pub const WorkingItem = memory.WorkingItem;
pub const WorkingMemory = abbey_types.WorkingMemory;
pub const MemoryManager = abbey_types.MemoryManager;
pub const MemoryStats = abbey_types.MemoryManager.MemoryStats;
pub const ReasoningChain = abbey_types.ReasoningChain;
pub const ReasoningStep = abbey_types.ReasoningStep;
pub const StepType = abbey_types.StepType;
pub const ConversationContext = abbey_types.ConversationContext;
pub const TopicTracker = abbey_types.TopicTracker;
pub const ContextWindow = context.ContextWindow;
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
pub const AbbeyEngine = abbey_types.AbbeyEngine;
pub const EngineResponse = engine.Response;
pub const EngineStats = engine.EngineStats;
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
pub const TaskProfile = abbey_types.TaskProfile;
pub const TaskDomain = abbey_types.TaskDomain;
pub const LearningStrategy = abbey_types.LearningStrategy;
pub const MetaLearner = abbey_types.MetaLearner;
pub const FewShotLearner = abbey_types.FewShotLearner;
pub const CurriculumScheduler = abbey_types.CurriculumScheduler;
pub const MentalModel = abbey_types.MentalModel;
pub const BeliefSystem = abbey_types.BeliefSystem;
pub const KnowledgeState = abbey_types.KnowledgeState;
pub const IntentionTracker = abbey_types.IntentionTracker;
pub const EmotionalModel = abbey_types.EmotionalModel;
pub const TheoryOfMind = abbey_types.TheoryOfMind;
pub const ProblemDecomposition = abbey_types.ProblemDecomposition;
pub const SubProblem = abbey_types.SubProblem;
pub const ExecutionPlan = abbey_types.ExecutionPlan;
pub const ProblemDecomposer = abbey_types.ProblemDecomposer;
pub const CounterfactualReasoner = abbey_types.CounterfactualReasoner;
pub const SelfEvaluation = abbey_types.SelfEvaluation;
pub const UncertaintyArea = abbey_types.UncertaintyArea;
pub const DetectedBias = abbey_types.DetectedBias;
pub const ReasoningQuality = abbey_types.ReasoningQuality;
pub const SelfReflectionEngine = abbey_types.SelfReflectionEngine;
pub const AdvancedCognition = abbey_types.AdvancedCognition;
pub const CognitiveResult = abbey_types.CognitiveResult;
pub const CognitiveState = abbey_types.CognitiveState;
pub const CustomAI = custom_framework.CustomAI;
pub const CustomAIConfig = custom_framework.CustomAIConfig;
pub const ProfileTemplate = custom_framework.ProfileTemplate;
pub const CustomAIBuilder = custom_framework.Builder;
pub const CustomAIResponse = custom_framework.Response;
pub const CustomAIStats = custom_framework.Stats;
pub const createCustomAI = custom_framework.create;
pub const createFromProfile = custom_framework.createFromProfile;
pub const createWithSeedPrompt = custom_framework.createWithSeedPrompt;
pub const createResearcher = custom_framework.createResearcher;
pub const createCoder = custom_framework.createCoder;
pub const createWriter = custom_framework.createWriter;
pub const createCompanion = custom_framework.createCompanion;
pub const createOpinionated = custom_framework.createOpinionated;
pub const Stats = custom_framework.Stats;
pub const Abbey = abbey_types.Abbey;
pub const LegacyResponse = abbey_types.LegacyResponse;
pub const LegacyStats = abbey_types.LegacyStats;
pub const abbey_train = struct {
    pub const AbbyTrainConfig = struct {
        base_gguf_path: []const u8 = "",
        training_data_path: []const u8 = "",
        output_path: []const u8 = "",
    };

    pub const AbbyTrainError = error{
        NoTrainingData,
        InvalidModel,
        TokenizerLoadFailed,
        FeatureDisabled,
    };

    pub fn run(_: std.mem.Allocator, _: AbbyTrainConfig) AbbyTrainError!void {
        return error.FeatureDisabled;
    }
};
pub const ralph_swarm = struct {
    pub const ParallelRalphContext = struct {
        allocator: std.mem.Allocator,
        bus: *abbey_types.ralph_multi.RalphBus,
        goals: []const []const u8,
        results: []?[]const u8,
        max_iterations: usize,
        post_result_to_bus: bool = true,
    };
    pub fn parallelRalphWorker(_: *ParallelRalphContext, _: u32) void {}
};
pub const ralph_multi = abbey_types.ralph_multi;

// ── Convenience Functions ──────────────────────────────────────────────────

pub fn createEngine(_: std.mem.Allocator) !abbey_types.AbbeyEngine {
    return error.FeatureDisabled;
}
pub fn createEngineWithConfig(_: std.mem.Allocator, _: abbey_types.AbbeyConfig) !abbey_types.AbbeyEngine {
    return error.FeatureDisabled;
}
pub fn builder() abbey_types.ConfigBuilder {
    return abbey_types.ConfigBuilder.init();
}
pub fn createAdvancedCognition(_: std.mem.Allocator) !abbey_types.AdvancedCognition {
    return error.FeatureDisabled;
}
pub fn createAdvancedCognitionWithConfig(_: std.mem.Allocator, _: abbey_types.AdvancedCognition.Config) !abbey_types.AdvancedCognition {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
