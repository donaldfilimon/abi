//! Abbey Type Re-exports
//!
//! Centralizes all type re-exports from sub-modules so that
//! abbey/mod.zig stays thin.

// Core types
pub const InstanceId = @import("../types.zig").InstanceId;
pub const SessionId = @import("../types.zig").SessionId;
pub const ConfidenceLevel = @import("../types.zig").ConfidenceLevel;
pub const Confidence = @import("../types.zig").Confidence;
pub const EmotionType = @import("../types.zig").EmotionType;
pub const EmotionalState = @import("../types.zig").EmotionalState;
pub const Role = @import("../types.zig").Role;
pub const Message = @import("../types.zig").Message;
pub const TrustLevel = @import("../types.zig").TrustLevel;
pub const Relationship = @import("../types.zig").Relationship;
pub const Topic = @import("../types.zig").Topic;
pub const Response = @import("../types.zig").Response;
pub const AbbeyError = @import("../types.zig").AbbeyError;

// Configuration
const config = @import("../core/config.zig");
pub const AbbeyConfig = config.AbbeyConfig;
pub const BehaviorConfig = config.BehaviorConfig;
pub const MemoryConfig = config.MemoryConfig;
pub const ReasoningConfig = config.ReasoningConfig;
pub const EmotionConfig = config.EmotionConfig;
pub const LearningConfig = config.LearningConfig;
pub const LLMConfig = config.LLMConfig;
pub const ServerConfig = config.ServerConfig;
pub const DiscordConfig = config.DiscordConfig;
pub const ConfigBuilder = config.ConfigBuilder;

// Neural
pub const Tensor = @import("neural/mod.zig").Tensor;
pub const F32Tensor = @import("neural/mod.zig").F32Tensor;
pub const LinearLayer = @import("neural/mod.zig").LinearLayer;
pub const EmbeddingLayer = @import("neural/mod.zig").EmbeddingLayer;
pub const LayerNorm = @import("neural/mod.zig").LayerNorm;
pub const MultiHeadAttention = @import("neural/mod.zig").MultiHeadAttention;
pub const SelfAttention = @import("neural/mod.zig").SelfAttention;
pub const CrossAttention = @import("neural/mod.zig").CrossAttention;
pub const AdaptiveAttention = @import("neural/mod.zig").AdaptiveAttention;
pub const OnlineLearner = @import("neural/mod.zig").OnlineLearner;
pub const ReplayBuffer = @import("neural/mod.zig").ReplayBuffer;
pub const AdamOptimizer = @import("neural/mod.zig").AdamOptimizer;

// Memory
pub const Episode = @import("memory/mod.zig").Episode;
pub const EpisodicMemory = @import("memory/mod.zig").EpisodicMemory;
pub const Knowledge = @import("memory/mod.zig").Knowledge;
pub const SemanticMemory = @import("memory/mod.zig").SemanticMemory;
pub const WorkingItem = @import("memory/mod.zig").WorkingItem;
pub const WorkingMemory = @import("memory/mod.zig").WorkingMemory;
pub const MemoryManager = @import("memory/mod.zig").MemoryManager;
pub const MemoryStats = @import("memory/mod.zig").MemoryManager.MemoryStats;

// Legacy
pub const ReasoningChain = @import("reasoning.zig").ReasoningChain;
pub const ReasoningStep = @import("reasoning.zig").ReasoningStep;
pub const StepType = @import("types.zig").StepType;
pub const ConversationContext = @import("context.zig").ConversationContext;
pub const TopicTracker = @import("context.zig").TopicTracker;
pub const ContextWindow = @import("context.zig").ContextWindow;

// Calibration
pub const Evidence = @import("calibration.zig").Evidence;
pub const CalibrationResult = @import("calibration.zig").CalibrationResult;
pub const ConfidenceCalibrator = @import("calibration.zig").ConfidenceCalibrator;
pub const QueryAnalyzer = @import("calibration.zig").QueryAnalyzer;

// Client
pub const ChatMessage = @import("client.zig").ChatMessage;
pub const CompletionRequest = @import("client.zig").CompletionRequest;
pub const CompletionResponse = @import("client.zig").CompletionResponse;
pub const StreamChunk = @import("client.zig").StreamChunk;
pub const LLMClient = @import("client.zig").LLMClient;
pub const EchoBackend = @import("client.zig").EchoBackend;
pub const createClient = @import("client.zig").createClient;
pub const ClientWrapper = @import("client.zig").ClientWrapper;
pub const RetryHandler = @import("client.zig").RetryHandler;

// Engine
pub const AbbeyEngine = @import("engine.zig").AbbeyEngine;
pub const EngineResponse = @import("engine.zig").Response;
pub const EngineStats = @import("engine.zig").EngineStats;

// Server
pub const HttpServerConfig = @import("server.zig").ServerConfig;
pub const AbbeyServerConfig = @import("server.zig").AbbeyServerConfig;
pub const ServerError = @import("server.zig").ServerError;
pub const serveHttp = @import("server.zig").serve;
pub const serveHttpWithConfig = @import("server.zig").serveWithConfig;

// Discord Bot
pub const AbbeyDiscordBot = @import("discord.zig").AbbeyDiscordBot;
pub const DiscordBotConfig = @import("discord.zig").DiscordBotConfig;
pub const DiscordBotError = @import("discord.zig").DiscordBotError;
pub const SessionManager = @import("discord.zig").SessionManager;
pub const BotStats = @import("discord.zig").BotStats;
pub const GatewayBridge = @import("discord.zig").GatewayBridge;
pub const GatewayStats = @import("discord.zig").GatewayStats;
pub const AbbeyCommands = @import("discord.zig").AbbeyCommands;

// Advanced Cognitive
pub const TaskProfile = @import("advanced.zig").TaskProfile;
pub const TaskDomain = @import("advanced.zig").TaskDomain;
pub const LearningStrategy = @import("advanced.zig").LearningStrategy;
pub const MetaLearner = @import("advanced.zig").MetaLearner;
pub const FewShotLearner = @import("advanced.zig").FewShotLearner;
pub const CurriculumScheduler = @import("advanced.zig").CurriculumScheduler;
pub const MentalModel = @import("advanced.zig").MentalModel;
pub const BeliefSystem = @import("advanced.zig").BeliefSystem;
pub const KnowledgeState = @import("advanced.zig").KnowledgeState;
pub const IntentionTracker = @import("advanced.zig").IntentionTracker;
pub const EmotionalModel = @import("advanced.zig").EmotionalModel;
pub const TheoryOfMind = @import("advanced.zig").TheoryOfMind;
pub const ProblemDecomposer = @import("advanced.zig").ProblemDecomposer;
pub const SelfEvaluation = @import("advanced.zig").SelfEvaluation;
pub const UncertaintyArea = @import("advanced.zig").UncertaintyArea;
pub const DetectedBias = @import("advanced.zig").DetectedBias;
pub const ReasoningQuality = @import("advanced.zig").ReasoningQuality;
pub const SelfReflectionEngine = @import("advanced.zig").SelfReflectionEngine;
pub const AdvancedCognition = @import("advanced.zig").AdvancedCognition;
pub const CognitiveResult = @import("advanced.zig").CognitiveResult;
pub const CognitiveState = @import("advanced.zig").CognitiveState;

// Custom Framework
pub const CustomAI = @import("custom_framework.zig").CustomAI;
pub const CustomAIConfig = @import("custom_framework.zig").CustomAIConfig;
pub const ProfileTemplate = @import("custom_framework.zig").ProfileTemplate;
pub const CustomAIBuilder = @import("custom_framework.zig").Builder;
pub const CustomAIResponse = @import("custom_framework.zig").Response;
pub const CustomAIStats = @import("custom_framework.zig").Stats;
pub const Stats = @import("custom_framework.zig").Stats;
pub const createCustomAI = @import("custom_framework.zig").create;
pub const createFromProfile = @import("custom_framework.zig").createFromProfile;
pub const createWithSeedPrompt = @import("custom_framework.zig").createWithSeedPrompt;
pub const createResearcher = @import("custom_framework.zig").createResearcher;
pub const createCoder = @import("custom_framework.zig").createCoder;
pub const createWriter = @import("custom_framework.zig").createWriter;
pub const createCompanion = @import("custom_framework.zig").createCompanion;
pub const createOpinionated = @import("custom_framework.zig").createOpinionated;
