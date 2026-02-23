//! Abbey AI Module
//!
//! A comprehensive, opinionated, emotionally intelligent AI framework with:
//! - Advanced neural learning and attention mechanisms
//! - Three-tier memory architecture (episodic, semantic, working)
//! - Confidence calibration with Bayesian updating
//! - Emotional intelligence and adaptive responses
//! - Research-first approach with tool integration
//! - LLM client abstraction for multiple backends
//!
//! Abbey embodies the design philosophy of being genuinely helpful through
//! honest opinions, emotional intelligence, and research commitment.

const std = @import("std");

// ============================================================================
// Core Modules
// ============================================================================

pub const core = struct {
    pub const types = @import("../core/types.zig");
    pub const config = @import("../core/config.zig");
};

// ============================================================================
// Neural Module (Tensors, Layers, Attention, Learning)
// ============================================================================

pub const neural = @import("neural/mod.zig");

// ============================================================================
// Memory Module (Episodic, Semantic, Working)
// ============================================================================

pub const memory = @import("memory/mod.zig");

// ============================================================================
// Legacy Modules (Original Abbey components)
// ============================================================================

pub const reasoning = @import("reasoning.zig");
pub const emotions = @import("emotions.zig");
pub const context = @import("context.zig");

// ============================================================================
// New Advanced Modules
// ============================================================================

pub const calibration = @import("calibration.zig");
pub const client = @import("client.zig");
pub const engine = @import("engine.zig");
pub const server = @import("server.zig");
pub const discord_bot = @import("discord.zig");
/// Multi-Ralph coordination: lock-free message bus for communication between Ralph loops.
pub const ralph_multi = @import("ralph_multi.zig");
/// Multi-agent swarm: parallel Ralph workers via ThreadPool + RalphBus (Zig-native, fast).
pub const ralph_swarm = @import("ralph_swarm.zig");

// ============================================================================
// Fine-Tuning Pipeline (lilex JSONL → LoRA → GGUF)
// ============================================================================

pub const abbey_train = @import("abbey_train.zig");

// ============================================================================
// Customizable AI Framework
// ============================================================================

pub const custom_framework = @import("custom_framework.zig");

// ============================================================================
// Advanced Cognitive Modules
// ============================================================================

pub const advanced = @import("advanced/mod.zig");

// ============================================================================
// Type Re-exports (Core)
// ============================================================================

// Core types
pub const InstanceId = core.types.InstanceId;
pub const SessionId = core.types.SessionId;
pub const ConfidenceLevel = core.types.ConfidenceLevel;
pub const Confidence = core.types.Confidence;
pub const EmotionType = core.types.EmotionType;
pub const EmotionalState = core.types.EmotionalState;
pub const Role = core.types.Role;
pub const Message = core.types.Message;
pub const TrustLevel = core.types.TrustLevel;
pub const Relationship = core.types.Relationship;
pub const Topic = core.types.Topic;
pub const Response = core.types.Response;
pub const AbbeyError = core.types.AbbeyError;

// Configuration
pub const AbbeyConfig = core.config.AbbeyConfig;
pub const BehaviorConfig = core.config.BehaviorConfig;
pub const MemoryConfig = core.config.MemoryConfig;
pub const ReasoningConfig = core.config.ReasoningConfig;
pub const EmotionConfig = core.config.EmotionConfig;
pub const LearningConfig = core.config.LearningConfig;
pub const LLMConfig = core.config.LLMConfig;
pub const ServerConfig = core.config.ServerConfig;
pub const DiscordConfig = core.config.DiscordConfig;
pub const ConfigBuilder = core.config.ConfigBuilder;

// ============================================================================
// Type Re-exports (Neural)
// ============================================================================

pub const Tensor = neural.Tensor;
pub const F32Tensor = neural.F32Tensor;
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

// ============================================================================
// Type Re-exports (Memory)
// ============================================================================

pub const Episode = memory.Episode;
pub const EpisodicMemory = memory.EpisodicMemory;
pub const Knowledge = memory.Knowledge;
pub const SemanticMemory = memory.SemanticMemory;
pub const WorkingItem = memory.WorkingItem;
pub const WorkingMemory = memory.WorkingMemory;
pub const MemoryManager = memory.MemoryManager;
pub const MemoryStats = memory.MemoryManager.MemoryStats;

// ============================================================================
// Type Re-exports (Legacy)
// ============================================================================

pub const ReasoningChain = reasoning.ReasoningChain;
pub const ReasoningStep = reasoning.ReasoningStep;
pub const StepType = reasoning.StepType;
pub const ConversationContext = context.ConversationContext;
pub const TopicTracker = context.TopicTracker;
pub const ContextWindow = context.ContextWindow;

// ============================================================================
// Type Re-exports (Calibration)
// ============================================================================

pub const Evidence = calibration.Evidence;
pub const CalibrationResult = calibration.CalibrationResult;
pub const ConfidenceCalibrator = calibration.ConfidenceCalibrator;
pub const QueryAnalyzer = calibration.QueryAnalyzer;

// ============================================================================
// Type Re-exports (Client)
// ============================================================================

pub const ChatMessage = client.ChatMessage;
pub const CompletionRequest = client.CompletionRequest;
pub const CompletionResponse = client.CompletionResponse;
pub const StreamChunk = client.StreamChunk;
pub const LLMClient = client.LLMClient;
pub const EchoBackend = client.EchoBackend;
pub const createClient = client.createClient;
pub const ClientWrapper = client.ClientWrapper;
pub const RetryHandler = client.RetryHandler;

// ============================================================================
// Type Re-exports (Engine)
// ============================================================================

pub const AbbeyEngine = engine.AbbeyEngine;
pub const EngineResponse = engine.Response;
pub const EngineStats = engine.EngineStats;

// ============================================================================
// Type Re-exports (Server)
// ============================================================================

pub const HttpServerConfig = server.ServerConfig;
pub const AbbeyServerConfig = server.AbbeyServerConfig;
pub const ServerError = server.ServerError;
pub const serveHttp = server.serve;
pub const serveHttpWithConfig = server.serveWithConfig;

// ============================================================================
// Type Re-exports (Discord Bot)
// ============================================================================

pub const AbbeyDiscordBot = discord_bot.AbbeyDiscordBot;
pub const DiscordBotConfig = discord_bot.DiscordBotConfig;
pub const DiscordBotError = discord_bot.DiscordBotError;
pub const SessionManager = discord_bot.SessionManager;
pub const BotStats = discord_bot.BotStats;
pub const AbbeyCommands = discord_bot.AbbeyCommands;

// ============================================================================
// Type Re-exports (Advanced Cognitive)
// ============================================================================

// Meta-Learning
pub const TaskProfile = advanced.TaskProfile;
pub const TaskDomain = advanced.TaskDomain;
pub const LearningStrategy = advanced.LearningStrategy;
pub const MetaLearner = advanced.MetaLearner;
pub const FewShotLearner = advanced.FewShotLearner;
pub const CurriculumScheduler = advanced.CurriculumScheduler;

// Theory of Mind
pub const MentalModel = advanced.MentalModel;
pub const BeliefSystem = advanced.BeliefSystem;
pub const KnowledgeState = advanced.KnowledgeState;
pub const IntentionTracker = advanced.IntentionTracker;
pub const EmotionalModel = advanced.EmotionalModel;
pub const TheoryOfMind = advanced.TheoryOfMind;

// Compositional Reasoning
pub const ProblemDecomposition = advanced.ProblemDecomposition;
pub const SubProblem = advanced.SubProblem;
pub const ExecutionPlan = advanced.ExecutionPlan;
pub const ProblemDecomposer = advanced.ProblemDecomposer;
pub const CounterfactualReasoner = advanced.CounterfactualReasoner;

// Self-Reflection
pub const SelfEvaluation = advanced.SelfEvaluation;
pub const UncertaintyArea = advanced.UncertaintyArea;
pub const DetectedBias = advanced.DetectedBias;
pub const ReasoningQuality = advanced.ReasoningQuality;
pub const SelfReflectionEngine = advanced.SelfReflectionEngine;

// Integrated System
pub const AdvancedCognition = advanced.AdvancedCognition;
pub const CognitiveResult = advanced.CognitiveResult;
pub const CognitiveState = advanced.CognitiveState;

// ============================================================================
// Type Re-exports (Custom Framework)
// ============================================================================

pub const CustomAI = custom_framework.CustomAI;
pub const CustomAIConfig = custom_framework.CustomAIConfig;
pub const PersonaTemplate = custom_framework.PersonaTemplate;
pub const CustomAIBuilder = custom_framework.Builder;
pub const CustomAIResponse = custom_framework.Response;
pub const CustomAIStats = custom_framework.Stats;

// Alias used by ai/mod.zig: `abbey.Stats`
pub const Stats = custom_framework.Stats;

// Factory functions
pub const createCustomAI = custom_framework.create;
pub const createFromPersona = custom_framework.createFromPersona;
pub const createWithSeedPrompt = custom_framework.createWithSeedPrompt;
pub const createResearcher = custom_framework.createResearcher;
pub const createCoder = custom_framework.createCoder;
pub const createWriter = custom_framework.createWriter;
pub const createCompanion = custom_framework.createCompanion;
pub const createOpinionated = custom_framework.createOpinionated;

// ============================================================================
// Convenience Aliases
// ============================================================================

/// Create a new Abbey engine with default configuration
pub fn createEngine(allocator: std.mem.Allocator) !AbbeyEngine {
    return AbbeyEngine.init(allocator, .{});
}

/// Create an Abbey engine with custom configuration
pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: AbbeyConfig) !AbbeyEngine {
    return AbbeyEngine.init(allocator, config);
}

/// Create configuration using builder pattern
pub fn builder() ConfigBuilder {
    return ConfigBuilder.init();
}

/// Create an advanced cognition system with default configuration
pub fn createAdvancedCognition(allocator: std.mem.Allocator) !AdvancedCognition {
    return AdvancedCognition.init(allocator, .{});
}

/// Create an advanced cognition system with custom configuration
pub fn createAdvancedCognitionWithConfig(
    allocator: std.mem.Allocator,
    config: AdvancedCognition.Config,
) !AdvancedCognition {
    return AdvancedCognition.init(allocator, config);
}

// ============================================================================
// Legacy Abbey Implementation (Kept for backwards compatibility)
// ============================================================================

const legacy_memory = @import("../memory/mod.zig");
const tools = @import("../tools/mod.zig");

/// Legacy Abbey - original implementation for backwards compatibility
/// Use AbbeyEngine for new implementations
pub const Abbey = struct {
    allocator: std.mem.Allocator,
    config: LegacyConfig,
    memory_manager: legacy_memory.MemoryManager,
    tool_registry: tools.ToolRegistry,
    emotional_state: emotions.EmotionalState,
    current_reasoning: ?ReasoningChain,
    conversation_context: context.ConversationContext,
    topic_tracker: context.TopicTracker,
    session_id: ?[]const u8,
    turn_count: usize,
    relationship_score: f32,

    pub const LegacyConfig = struct {
        name: []const u8 = "Abbey",
        enable_emotions: bool = true,
        enable_reasoning_log: bool = true,
        enable_topic_tracking: bool = true,
        memory_config: legacy_memory.MemoryConfig = .{},
        base_temperature: f32 = 0.7,
        max_reasoning_steps: usize = 10,
        confidence_threshold: f32 = 0.7,
        research_first: bool = true,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: LegacyConfig) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .memory_manager = legacy_memory.MemoryManager.init(allocator, config.memory_config),
            .tool_registry = tools.ToolRegistry.init(allocator),
            .emotional_state = emotions.EmotionalState.init(),
            .current_reasoning = null,
            .conversation_context = context.ConversationContext.init(allocator),
            .topic_tracker = context.TopicTracker.init(allocator),
            .session_id = null,
            .turn_count = 0,
            .relationship_score = 0.5,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.current_reasoning) |*chain| {
            chain.deinit();
        }
        self.conversation_context.deinit();
        self.topic_tracker.deinit();
        self.memory_manager.deinit();
        self.tool_registry.deinit();
        if (self.session_id) |id| {
            self.allocator.free(id);
        }
    }

    pub fn process(self: *Self, message: []const u8) !LegacyResponse {
        self.turn_count += 1;

        if (self.config.enable_emotions) {
            self.emotional_state.detectFromText(message);
        }

        if (self.config.enable_topic_tracking) {
            try self.topic_tracker.updateFromMessage(message);
        }

        const user_msg = legacy_memory.Message.user(message);
        try self.memory_manager.addMessage(user_msg);

        var chain = ReasoningChain.init(self.allocator, message);
        errdefer chain.deinit();

        const initial_confidence = self.assessConfidence(message);
        try chain.addStep(.assessment, "Assessing query", initial_confidence);

        const response_text = try self.generateResponse(message);

        const assistant_msg = legacy_memory.Message.assistant(response_text);
        try self.memory_manager.addMessage(assistant_msg);

        self.updateRelationship();

        try chain.finalize();
        if (self.current_reasoning) |*old| {
            old.deinit();
        }
        self.current_reasoning = chain;

        return LegacyResponse{
            .content = response_text,
            .confidence = chain.getOverallConfidence(),
            .emotional_context = self.emotional_state,
            .reasoning_summary = try chain.getSummary(self.allocator),
            .topics = self.topic_tracker.getCurrentTopics(),
        };
    }

    fn assessConfidence(self: *Self, query: []const u8) reasoning.Confidence {
        _ = self;
        var buf: [1024]u8 = undefined;
        const len = @min(query.len, buf.len);
        for (0..len) |i| {
            buf[i] = std.ascii.toLower(query[i]);
        }
        const query_lower = buf[0..len];

        const high_patterns = [_][]const u8{ "what is", "how do i", "explain" };
        const low_patterns = [_][]const u8{ "latest", "current", "today", "2024", "2025" };

        for (low_patterns) |p| {
            if (std.mem.indexOf(u8, query_lower, p) != null) {
                return .{ .level = .low, .score = 0.3, .reasoning = "Time-sensitive query" };
            }
        }
        for (high_patterns) |p| {
            if (std.mem.indexOf(u8, query_lower, p) != null) {
                return .{ .level = .high, .score = 0.85, .reasoning = "Established knowledge" };
            }
        }
        return .{ .level = .medium, .score = 0.6, .reasoning = "Standard query" };
    }

    fn generateResponse(self: *Self, query: []const u8) ![]u8 {
        var response = std.ArrayListUnmanaged(u8).empty;
        errdefer response.deinit(self.allocator);

        try response.appendSlice(self.allocator, "[Abbey] Echo: ");
        try response.appendSlice(self.allocator, query);

        return response.toOwnedSlice(self.allocator);
    }

    fn updateRelationship(self: *Self) void {
        self.relationship_score = @min(1.0, self.relationship_score + 0.01);
    }

    pub fn getEmotionalState(self: *const Self) emotions.EmotionalState {
        return self.emotional_state;
    }

    pub fn getStats(self: *const Self) LegacyStats {
        return .{
            .turn_count = self.turn_count,
            .relationship_score = self.relationship_score,
            .current_emotion = self.emotional_state.detected,
            .topics_discussed = self.topic_tracker.getTopicCount(),
            .memory_stats = self.memory_manager.getStats(),
        };
    }

    pub fn clearConversation(self: *Self) void {
        self.memory_manager.clear();
        self.conversation_context.clear();
        if (self.current_reasoning) |*chain| {
            chain.deinit();
            self.current_reasoning = null;
        }
    }

    pub fn reset(self: *Self) void {
        self.clearConversation();
        self.topic_tracker.clear();
        self.relationship_score = 0.5;
        self.turn_count = 0;
        self.emotional_state = emotions.EmotionalState.init();
    }
};

pub const LegacyResponse = struct {
    content: []const u8,
    confidence: reasoning.Confidence,
    emotional_context: emotions.EmotionalState,
    reasoning_summary: ?[]const u8,
    topics: []const []const u8,
};

pub const LegacyStats = struct {
    turn_count: usize,
    relationship_score: f32,
    current_emotion: emotions.EmotionType,
    topics_discussed: usize,
    memory_stats: legacy_memory.MemoryStats,
};

// ============================================================================
// Tests
// ============================================================================

test "abbey engine creation" {
    const allocator = std.testing.allocator;

    var abbey_engine = try createEngine(allocator);
    defer abbey_engine.deinit();

    try std.testing.expect(!abbey_engine.conversation_active);
}

test "abbey configuration builder" {
    var b = builder();
    const config = try b
        .name("TestAbbey")
        .temperature(0.8)
        .researchFirst(true)
        .llmBackend(.echo)
        .build();

    try std.testing.expectEqualStrings("TestAbbey", config.name);
    try std.testing.expectEqual(@as(f32, 0.8), config.behavior.base_temperature);
}

test "legacy abbey compatibility" {
    const allocator = std.testing.allocator;

    var abbey = try Abbey.init(allocator, .{});
    defer abbey.deinit();

    try std.testing.expectEqual(@as(usize, 0), abbey.turn_count);
    try std.testing.expectEqual(@as(f32, 0.5), abbey.relationship_score);
}

test "neural module available" {
    const allocator = std.testing.allocator;

    var t = try F32Tensor.zeros(allocator, &.{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.size());
}

test "memory module available" {
    const allocator = std.testing.allocator;

    var working = WorkingMemory.init(allocator, 100, 4000);
    defer working.deinit();

    _ = try working.add("test", .user_input, 0.5);
    try std.testing.expectEqual(@as(usize, 1), working.items.items.len);
}

test "advanced cognition available" {
    const allocator = std.testing.allocator;

    var cognition = try createAdvancedCognition(allocator);
    defer cognition.deinit();

    // Process a query
    const result = try cognition.process("user123", "How does machine learning work?");
    try std.testing.expect(result.task_profile.complexity > 0.0);
    try std.testing.expect(result.cognitive_load >= 0.0);
}

test "theory of mind integration" {
    const allocator = std.testing.allocator;

    var tom = TheoryOfMind.init(allocator);
    defer tom.deinit();

    // Get/create mental model
    const model = try tom.getModel("user1");
    try std.testing.expectEqualStrings("user1", model.user_id);
}

test "self-reflection engine" {
    const allocator = std.testing.allocator;

    var reflection = SelfReflectionEngine.init(allocator, .{});
    defer reflection.deinit();

    const evaluation = try reflection.evaluate(
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "What is machine learning?",
        null,
    );
    try std.testing.expect(evaluation.overall_quality > 0.0);
}

test {
    _ = neural;
    _ = memory;
    _ = reasoning;
    _ = emotions;
    _ = context;
    _ = calibration;
    _ = client;
    _ = engine;
    _ = server;
    _ = discord_bot;
    _ = custom_framework;
    _ = advanced;
}
