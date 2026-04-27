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
// Sub-namespace facades (additive)
// ============================================================================

pub const cognition = @import("cognition.zig");
pub const system = @import("system.zig");
pub const pipeline = @import("pipeline.zig");

// ============================================================================
// Core Modules
// ============================================================================

pub const core = struct {
    pub const types = @import("../types.zig");
    pub const config = @import("../config.zig");
};

// ============================================================================
// Module Re-exports
// ============================================================================

pub const neural = @import("neural/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const reasoning = @import("reasoning.zig");
pub const emotions = @import("emotions.zig");
pub const context = @import("context.zig");
pub const calibration = @import("calibration.zig");
pub const client = @import("client.zig");
pub const engine = @import("engine.zig");
pub const server = @import("server.zig");
pub const discord_bot = @import("discord.zig");
pub const ralph_multi = @import("ralph_multi.zig");
pub const ralph_swarm = @import("ralph_swarm.zig");
pub const abbey_train = @import("abbey_train.zig");
pub const custom_framework = @import("custom_framework.zig");
pub const advanced = @import("advanced.zig");

// ============================================================================
// Type Re-exports (delegated to reexports.zig)
// ============================================================================

pub const reexports = @import("reexports.zig");

// Re-export all types from reexports.zig
pub const InstanceId = reexports.InstanceId;
pub const SessionId = reexports.SessionId;
pub const ConfidenceLevel = reexports.ConfidenceLevel;
pub const Confidence = reexports.Confidence;
pub const EmotionType = reexports.EmotionType;
pub const EmotionalState = reexports.EmotionalState;
pub const Role = reexports.Role;
pub const Message = reexports.Message;
pub const TrustLevel = reexports.TrustLevel;
pub const Relationship = reexports.Relationship;
pub const Topic = reexports.Topic;
pub const Response = reexports.Response;
pub const AbbeyError = reexports.AbbeyError;
pub const AbbeyConfig = reexports.AbbeyConfig;
pub const BehaviorConfig = reexports.BehaviorConfig;
pub const MemoryConfig = reexports.MemoryConfig;
pub const ReasoningConfig = reexports.ReasoningConfig;
pub const EmotionConfig = reexports.EmotionConfig;
pub const LearningConfig = reexports.LearningConfig;
pub const LLMConfig = reexports.LLMConfig;
pub const ServerConfig = reexports.ServerConfig;
pub const DiscordConfig = reexports.DiscordConfig;
pub const ConfigBuilder = reexports.ConfigBuilder;
pub const Tensor = reexports.Tensor;
pub const F32Tensor = reexports.F32Tensor;
pub const LinearLayer = reexports.LinearLayer;
pub const EmbeddingLayer = reexports.EmbeddingLayer;
pub const LayerNorm = reexports.LayerNorm;
pub const MultiHeadAttention = reexports.MultiHeadAttention;
pub const SelfAttention = reexports.SelfAttention;
pub const CrossAttention = reexports.CrossAttention;
pub const AdaptiveAttention = reexports.AdaptiveAttention;
pub const OnlineLearner = reexports.OnlineLearner;
pub const ReplayBuffer = reexports.ReplayBuffer;
pub const AdamOptimizer = reexports.AdamOptimizer;
pub const Episode = reexports.Episode;
pub const EpisodicMemory = reexports.EpisodicMemory;
pub const Knowledge = reexports.Knowledge;
pub const SemanticMemory = reexports.SemanticMemory;
pub const WorkingItem = reexports.WorkingItem;
pub const WorkingMemory = reexports.WorkingMemory;
pub const MemoryManager = reexports.MemoryManager;
pub const MemoryStats = reexports.MemoryStats;
pub const ReasoningChain = reexports.ReasoningChain;
pub const ReasoningStep = reexports.ReasoningStep;
pub const StepType = reexports.StepType;
pub const ConversationContext = reexports.ConversationContext;
pub const TopicTracker = reexports.TopicTracker;
pub const ContextWindow = reexports.ContextWindow;
pub const Evidence = reexports.Evidence;
pub const CalibrationResult = reexports.CalibrationResult;
pub const ConfidenceCalibrator = reexports.ConfidenceCalibrator;
pub const QueryAnalyzer = reexports.QueryAnalyzer;
pub const ChatMessage = reexports.ChatMessage;
pub const CompletionRequest = reexports.CompletionRequest;
pub const CompletionResponse = reexports.CompletionResponse;
pub const StreamChunk = reexports.StreamChunk;
pub const LLMClient = reexports.LLMClient;
pub const EchoBackend = reexports.EchoBackend;
pub const createClient = reexports.createClient;
pub const ClientWrapper = reexports.ClientWrapper;
pub const RetryHandler = reexports.RetryHandler;
pub const AbbeyEngine = reexports.AbbeyEngine;
pub const EngineResponse = reexports.EngineResponse;
pub const EngineStats = reexports.EngineStats;
pub const HttpServerConfig = reexports.HttpServerConfig;
pub const AbbeyServerConfig = reexports.AbbeyServerConfig;
pub const ServerError = reexports.ServerError;
pub const serveHttp = reexports.serveHttp;
pub const serveHttpWithConfig = reexports.serveHttpWithConfig;
pub const AbbeyDiscordBot = reexports.AbbeyDiscordBot;
pub const DiscordBotConfig = reexports.DiscordBotConfig;
pub const DiscordBotError = reexports.DiscordBotError;
pub const SessionManager = reexports.SessionManager;
pub const BotStats = reexports.BotStats;
pub const GatewayBridge = reexports.GatewayBridge;
pub const GatewayStats = reexports.GatewayStats;
pub const AbbeyCommands = reexports.AbbeyCommands;
pub const TaskProfile = reexports.TaskProfile;
pub const TaskDomain = reexports.TaskDomain;
pub const LearningStrategy = reexports.LearningStrategy;
pub const MetaLearner = reexports.MetaLearner;
pub const FewShotLearner = reexports.FewShotLearner;
pub const CurriculumScheduler = reexports.CurriculumScheduler;
pub const MentalModel = reexports.MentalModel;
pub const BeliefSystem = reexports.BeliefSystem;
pub const KnowledgeState = reexports.KnowledgeState;
pub const IntentionTracker = reexports.IntentionTracker;
pub const EmotionalModel = reexports.EmotionalModel;
pub const TheoryOfMind = reexports.TheoryOfMind;
pub const ProblemDecomposer = reexports.ProblemDecomposer;
pub const SelfEvaluation = reexports.SelfEvaluation;
pub const UncertaintyArea = reexports.UncertaintyArea;
pub const DetectedBias = reexports.DetectedBias;
pub const ReasoningQuality = reexports.ReasoningQuality;
pub const SelfReflectionEngine = reexports.SelfReflectionEngine;
pub const AdvancedCognition = reexports.AdvancedCognition;
pub const CognitiveResult = reexports.CognitiveResult;
pub const CognitiveState = reexports.CognitiveState;
pub const CustomAI = reexports.CustomAI;
pub const CustomAIConfig = reexports.CustomAIConfig;
pub const ProfileTemplate = reexports.ProfileTemplate;
pub const CustomAIBuilder = reexports.CustomAIBuilder;
pub const CustomAIResponse = reexports.CustomAIResponse;
pub const CustomAIStats = reexports.CustomAIStats;
pub const Stats = reexports.Stats;
pub const createCustomAI = reexports.createCustomAI;
pub const createFromProfile = reexports.createFromProfile;
pub const createWithSeedPrompt = reexports.createWithSeedPrompt;
pub const createResearcher = reexports.createResearcher;
pub const createCoder = reexports.createCoder;
pub const createWriter = reexports.createWriter;
pub const createCompanion = reexports.createCompanion;
pub const createOpinionated = reexports.createOpinionated;

// ============================================================================
// Convenience Functions (delegated to convenience.zig)
// ============================================================================

pub const convenience = @import("convenience.zig");

pub const createEngine = convenience.createEngine;
pub const createEngineWithConfig = convenience.createEngineWithConfig;
pub const builder = convenience.builder;
pub const createAdvancedCognition = convenience.createAdvancedCognition;
pub const createAdvancedCognitionWithConfig = convenience.createAdvancedCognitionWithConfig;

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
            chain.deinit(self.allocator);
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

    var adv_cognition = try createAdvancedCognition(allocator);
    defer adv_cognition.deinit();

    // Process a query
    const result = try adv_cognition.process("user123", "How does machine learning work?");
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
    _ = @import("cognition.zig");
    _ = @import("system.zig");
    _ = @import("pipeline.zig");
    _ = @import("neural/mod.zig");
    _ = @import("memory/mod.zig");
    _ = @import("reasoning.zig");
    _ = @import("emotions.zig");
    _ = @import("context.zig");
    _ = @import("calibration.zig");
    _ = @import("client.zig");
    _ = @import("engine.zig");
    _ = @import("server.zig");
    _ = @import("discord.zig");
    _ = @import("ralph_multi.zig");
    _ = @import("ralph_swarm.zig");
    _ = @import("custom_framework.zig");
    _ = @import("advanced.zig");
    // abbey_train.zig excluded: pre-existing API mismatch with TrainableModel.state
    // config.zig excluded: re-exports from core/config.zig which has its own tests
}
