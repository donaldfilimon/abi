//! Abbey Unified Engine
//!
//! The core Abbey engine that integrates all subsystems:
//! - Memory (episodic, semantic, working)
//! - Neural learning and attention
//! - Confidence calibration
//! - Emotional intelligence
//! - LLM client integration
//!
//! This is the main entry point for Abbey interactions.

const std = @import("std");
const core_types = @import("../core/types.zig");
const core_config = @import("../core/config.zig");
const memory_mod = @import("memory/mod.zig");
const neural = @import("neural/mod.zig");
const calibration = @import("calibration.zig");
const client = @import("client.zig");
const reasoning = @import("reasoning.zig");
const emotions = @import("emotions.zig");
const context = @import("context.zig");
const prompts = @import("../prompts/mod.zig");

// ============================================================================
// Abbey Engine
// ============================================================================

/// The unified Abbey AI engine
pub const AbbeyEngine = struct {
    allocator: std.mem.Allocator,
    config: core_config.AbbeyConfig,

    // Core subsystems
    memory: memory_mod.MemoryManager,
    calibrator: calibration.ConfidenceCalibrator,
    llm_client: client.ClientWrapper,

    // State
    emotional_state: emotions.EmotionalState,
    topic_tracker: context.TopicTracker,
    current_reasoning: ?reasoning.ReasoningChain = null,
    relationship: core_types.Relationship,

    // Session
    session_id: ?core_types.SessionId = null,
    turn_count: usize = 0,
    conversation_active: bool = false,

    // Statistics
    total_queries: usize = 0,
    total_tokens_used: usize = 0,
    avg_response_time_ms: f32 = 0,

    // Online learning
    learner: ?neural.OnlineLearner = null,

    const Self = @This();

    /// Initialize Abbey with configuration
    pub fn init(allocator: std.mem.Allocator, abbey_config: core_config.AbbeyConfig) !Self {
        // Validate configuration
        try abbey_config.validate();

        // Initialize memory
        var mem = try memory_mod.MemoryManager.init(allocator, abbey_config.memory);
        errdefer mem.deinit();

        // Initialize calibrator
        var cal = calibration.ConfidenceCalibrator.init(allocator);
        errdefer cal.deinit();

        // Initialize LLM client
        var llm = try client.createClient(allocator, abbey_config.llm);
        errdefer llm.deinit(allocator);

        // Initialize emotional state
        const emotional = emotions.EmotionalState.init();

        // Initialize topic tracker
        var topics = context.TopicTracker.init(allocator);
        errdefer topics.deinit();

        // Initialize online learner if enabled
        var learner: ?neural.OnlineLearner = null;
        if (abbey_config.learning.enable_online_learning) {
            learner = try neural.OnlineLearner.init(allocator, .{
                .learning_rate = abbey_config.learning.learning_rate,
                .batch_size = abbey_config.learning.batch_size,
            });
        }

        return Self{
            .allocator = allocator,
            .config = abbey_config,
            .memory = mem,
            .calibrator = cal,
            .llm_client = llm,
            .emotional_state = emotional,
            .topic_tracker = topics,
            .relationship = .{},
            .learner = learner,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        if (self.current_reasoning) |*chain| {
            chain.deinit();
        }
        self.topic_tracker.deinit();
        self.calibrator.deinit();
        self.memory.deinit();
        self.llm_client.deinit(self.allocator);
        if (self.learner) |*l| {
            l.deinit();
        }
        if (self.session_id) |*sid| {
            if (sid.user_id) |uid| {
                self.allocator.free(uid);
            }
        }
    }

    // ========================================================================
    // Conversation Management
    // ========================================================================

    /// Start a new conversation
    pub fn startConversation(self: *Self, user_id: ?[]const u8) !void {
        // End any existing conversation
        if (self.conversation_active) {
            try self.endConversation();
        }

        self.session_id = try core_types.SessionId.create(self.allocator, user_id);
        self.turn_count = 0;
        self.conversation_active = true;

        try self.memory.beginConversation();

        // Add system message if configured
        if (self.config.behavior.opinionated) {
            const system_msg = core_types.Message.system(
                \\You are Abbey, an opinionated, emotionally intelligent AI assistant.
                \\You have strong opinions based on evidence and reasoning.
                \\You adapt your communication style based on the user's emotional state.
                \\You acknowledge when you're uncertain and research when needed.
            );
            try self.memory.storeMessage(system_msg);
        }
    }

    /// End the current conversation
    pub fn endConversation(self: *Self) !void {
        if (!self.conversation_active) return;

        try self.memory.endConversation();

        // Consolidate memories
        if (self.config.memory.enable_consolidation) {
            try self.memory.consolidate();
        }

        self.conversation_active = false;
    }

    // ========================================================================
    // Main Processing Pipeline
    // ========================================================================

    /// Process a user message and generate a response
    pub fn process(self: *Self, user_input: []const u8) !Response {
        const start_time = core_types.getTimestampMs();

        // Ensure conversation is active
        if (!self.conversation_active) {
            try self.startConversation(null);
        }

        self.turn_count += 1;
        self.total_queries += 1;

        // 1. Emotional Analysis
        if (self.config.emotions.enable_detection) {
            self.emotional_state.detectFromText(user_input);
        }

        // 2. Topic Tracking
        if (self.config.behavior.opinionated) {
            try self.topic_tracker.updateFromMessage(user_input);
        }

        // 3. Store user message
        var user_msg = core_types.Message.user(user_input);
        user_msg.metadata = .{
            .emotion_detected = self.emotional_state.detected,
        };
        try self.memory.storeMessage(user_msg);

        // 4. Begin reasoning chain
        if (self.current_reasoning) |*old| {
            old.deinit();
        }
        self.current_reasoning = reasoning.ReasoningChain.init(self.allocator, user_input);

        // 5. Query Analysis & Confidence Calibration
        const query_analysis = calibration.QueryAnalyzer.analyzeQuery(user_input);
        const initial_confidence = self.assessInitialConfidence(user_input, query_analysis);

        try self.current_reasoning.?.addStep(
            .assessment,
            "Analyzing query complexity and confidence",
            initial_confidence,
        );

        // 6. Determine if research is needed
        const needs_research = self.config.behavior.research_first and
            initial_confidence.needsResearch();

        if (needs_research) {
            try self.current_reasoning.?.addStep(
                .research,
                "Query requires verification or research",
                .{
                    .level = .low,
                    .score = 0.4,
                    .reasoning = "Insufficient confidence for direct answer",
                },
            );
        }

        // 7. Build context
        const context_embedding: ?[]const f32 = null; // Would compute embedding here
        var hybrid_context = try self.memory.getHybridContext(
            context_embedding,
            self.config.memory.max_context_tokens,
            5,
        );
        defer hybrid_context.deinit(self.allocator);

        // 8. Generate response
        const response_content = try self.generateResponse(
            user_input,
            &hybrid_context,
            query_analysis,
        );

        // 9. Store assistant response
        var assistant_msg = core_types.Message.assistant(response_content);
        assistant_msg.metadata = .{
            .confidence = initial_confidence,
            .emotion_detected = self.emotional_state.detected,
        };
        try self.memory.storeMessage(assistant_msg);

        // 10. Finalize reasoning
        try self.current_reasoning.?.finalize();

        // 11. Update relationship
        self.relationship.recordInteraction(true);

        // 12. Calculate timing
        const end_time = core_types.getTimestampMs();
        const response_time = end_time - start_time;
        self.updateAverageResponseTime(response_time);

        // 13. Build response
        return Response{
            .content = response_content,
            .confidence = self.current_reasoning.?.getOverallConfidence(),
            .emotional_context = self.emotional_state,
            .reasoning_summary = try self.current_reasoning.?.getSummary(self.allocator),
            .topics = self.topic_tracker.getCurrentTopics(),
            .research_performed = needs_research,
            .generation_time_ms = response_time,
        };
    }

    /// Execute an iterative Ralph loop for a complex task
    /// Returns the final output from the agent. Caller owns the returned slice.
    pub fn runRalphLoop(self: *Self, goal: []const u8, max_iterations: usize) ![]u8 {
        // Ensure conversation is active
        if (!self.conversation_active) {
            try self.startConversation(null);
        }

        const ralph_persona = prompts.getPersona(.ralph);

        // Loop history
        var history = std.ArrayListUnmanaged(client.ChatMessage){};
        defer {
            for (history.items) |*msg| {
                if (msg.role.len > 0) {} // role is usually static string literal
                self.allocator.free(msg.content);
            }
            history.deinit(self.allocator);
        }

        // 1. System Prompt
        try history.append(self.allocator, .{
            .role = "system",
            .content = try self.allocator.dupe(u8, ralph_persona.system_prompt),
        });

        // 2. User Goal
        try history.append(self.allocator, .{
            .role = "user",
            .content = try self.allocator.dupe(u8, goal),
        });

        var iteration: usize = 0;
        var last_response: []u8 = try self.allocator.dupe(u8, ""); // Placeholder

        while (iteration < max_iterations) : (iteration += 1) {
            // Clean up previous response if it wasn't added to history (for the placeholder case)
            if (iteration == 0) self.allocator.free(last_response);

            // Make request
            const request = client.CompletionRequest{
                .messages = history.items,
                .model = self.config.llm.model,
                .temperature = ralph_persona.suggested_temperature,
                .max_tokens = self.config.behavior.max_tokens,
            };

            const response = try self.llm_client.complete(request);
            last_response = try self.allocator.dupe(u8, response.content);

            // Store agent response
            try history.append(self.allocator, .{
                .role = "assistant",
                .content = try self.allocator.dupe(u8, last_response),
            });

            // Update usage stats
            self.total_tokens_used += response.usage.total_tokens;
            self.turn_count += 1;

            // Check for explicit completion signal
            // If the agent says "TASK COMPLETED" or similar, we break.
            // But Ralph is designed to be verified.
            // For now, we rely on the loop injection to keep it going until max_iterations
            // OR if the agent outputs a specific stop token we define.
            // Let's assume for this MVP we run until max or if we detect a "final answer" pattern.

            // Inject Loop Prompt for next iteration
            if (iteration < max_iterations - 1) {
                const injection = try prompts.ralph.formatLoopInjection(self.allocator, iteration + 1, goal);
                try history.append(self.allocator, .{
                    .role = "system",
                    .content = injection,
                });
            }
        }

        return last_response;
    }

    /// Assess initial confidence for a query
    fn assessInitialConfidence(
        self: *Self,
        query: []const u8,
        analysis: calibration.QueryAnalyzer.QueryAnalysis,
    ) reasoning.Confidence {
        // Build evidence
        var evidence_buf: [8]calibration.Evidence = undefined;
        var evidence_count: usize = 0;

        // Training data evidence
        evidence_buf[evidence_count] = .{
            .source = .training_data,
            .strength = analysis.base_confidence,
            .reliability = 0.8,
            .recency = if (analysis.time_sensitive) @as(f32, 0.5) else @as(f32, 1.0),
        };
        evidence_count += 1;

        // Prior conversation evidence
        if (self.turn_count > 1) {
            evidence_buf[evidence_count] = .{
                .source = .prior_conversation,
                .strength = 0.6,
                .reliability = 0.7,
                .recency = 1.0,
            };
            evidence_count += 1;
        }

        // Calibrate
        const result = self.calibrator.calibrate(
            query,
            analysis.base_confidence,
            evidence_buf[0..evidence_count],
        );

        return .{
            .level = result.confidence_level,
            .score = result.posterior,
            .reasoning = result.explanation,
        };
    }

    /// Generate response using LLM
    fn generateResponse(
        self: *Self,
        query: []const u8,
        hybrid_context: *memory_mod.MemoryManager.HybridContext,
        analysis: calibration.QueryAnalyzer.QueryAnalysis,
    ) ![]u8 {
        _ = analysis;

        // Build messages
        var messages = std.ArrayListUnmanaged(client.ChatMessage){};
        defer messages.deinit(self.allocator);

        // Add context
        try messages.append(self.allocator, .{
            .role = "system",
            .content = hybrid_context.working,
        });

        if (hybrid_context.knowledge) |k| {
            try messages.append(self.allocator, .{
                .role = "system",
                .content = k,
            });
        }

        // Add user query
        try messages.append(self.allocator, .{
            .role = "user",
            .content = query,
        });

        // Adjust temperature based on emotional state
        var temperature = self.config.behavior.base_temperature;
        if (self.config.emotions.enable_adaptation) {
            temperature += self.emotional_state.detected.getTemperatureModifier();
            temperature = std.math.clamp(temperature, 0.1, 1.5);
        }

        // Make request
        const request = client.CompletionRequest{
            .messages = messages.items,
            .model = self.config.llm.model,
            .temperature = temperature,
            .max_tokens = self.config.behavior.max_tokens,
        };

        const response = try self.llm_client.complete(request);

        // Track usage
        self.total_tokens_used += response.usage.total_tokens;

        return self.allocator.dupe(u8, response.content);
    }

    fn updateAverageResponseTime(self: *Self, new_time: i64) void {
        const new_time_f = @as(f32, @floatFromInt(new_time));
        self.avg_response_time_ms = self.avg_response_time_ms * 0.9 + new_time_f * 0.1;
    }

    // ========================================================================
    // Learning & Feedback
    // ========================================================================

    /// Provide feedback on the last response
    pub fn provideFeedback(self: *Self, positive: bool) !void {
        self.relationship.recordInteraction(positive);

        // Record for calibration
        if (self.current_reasoning) |*chain| {
            const conf = chain.getOverallConfidence();
            try self.calibrator.recordPrediction(conf.score, positive);
        }

        // Update online learner if available
        if (self.learner) |*l| {
            if (positive) {
                l.running_loss *= 0.95; // Reduce loss on positive feedback
            }
        }
    }

    /// Add knowledge from user
    pub fn learnFromUser(self: *Self, content: []const u8, category: memory_mod.Knowledge.KnowledgeCategory) !u64 {
        return self.memory.storeKnowledge(content, category, .user_stated);
    }

    // ========================================================================
    // State Access
    // ========================================================================

    /// Get current emotional state
    pub fn getEmotionalState(self: *const Self) emotions.EmotionalState {
        return self.emotional_state;
    }

    /// Get relationship info
    pub fn getRelationship(self: *const Self) core_types.Relationship {
        return self.relationship;
    }

    /// Get engine statistics
    pub fn getStats(self: *const Self) EngineStats {
        return .{
            .turn_count = self.turn_count,
            .total_queries = self.total_queries,
            .total_tokens_used = self.total_tokens_used,
            .avg_response_time_ms = self.avg_response_time_ms,
            .relationship_score = self.relationship.rapport_score,
            .current_emotion = self.emotional_state.detected,
            .topics_discussed = self.topic_tracker.getTopicCount(),
            .memory_stats = self.memory.getStats(),
            .calibration_metrics = self.calibrator.getCalibrationMetrics(),
            .llm_backend = self.llm_client.getBackendName(),
            .conversation_active = self.conversation_active,
        };
    }

    /// Clear conversation but maintain learned state
    pub fn clearConversation(self: *Self) void {
        self.memory.clearWorking();
        if (self.current_reasoning) |*chain| {
            chain.deinit();
            self.current_reasoning = null;
        }
        self.turn_count = 0;
    }

    /// Full reset
    pub fn reset(self: *Self) void {
        self.clearConversation();
        self.memory.reset();
        self.topic_tracker.clear();
        self.relationship = .{};
        self.total_queries = 0;
        self.total_tokens_used = 0;
        self.avg_response_time_ms = 0;
        self.emotional_state = emotions.EmotionalState.init();
    }
};

// ============================================================================
// Response Type
// ============================================================================

pub const Response = struct {
    content: []const u8,
    confidence: reasoning.Confidence,
    emotional_context: emotions.EmotionalState,
    reasoning_summary: ?[]const u8,
    topics: []const []const u8,
    research_performed: bool,
    generation_time_ms: i64,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.reasoning_summary) |s| allocator.free(s);
    }
};

/// Engine statistics
pub const EngineStats = struct {
    turn_count: usize,
    total_queries: usize,
    total_tokens_used: usize,
    avg_response_time_ms: f32,
    relationship_score: f32,
    current_emotion: core_types.EmotionType,
    topics_discussed: usize,
    memory_stats: memory_mod.MemoryManager.MemoryStats,
    calibration_metrics: calibration.ConfidenceCalibrator.CalibrationMetrics,
    llm_backend: []const u8,
    conversation_active: bool,
};

// ============================================================================
// Tests
// ============================================================================

test "abbey engine initialization" {
    const allocator = std.testing.allocator;

    var engine = try AbbeyEngine.init(allocator, .{});
    defer engine.deinit();

    try std.testing.expect(!engine.conversation_active);
    try std.testing.expectEqual(@as(usize, 0), engine.turn_count);
}

test "abbey engine conversation lifecycle" {
    const allocator = std.testing.allocator;

    var engine = try AbbeyEngine.init(allocator, .{});
    defer engine.deinit();

    try engine.startConversation(null);
    try std.testing.expect(engine.conversation_active);

    try engine.endConversation();
    try std.testing.expect(!engine.conversation_active);
}

test "abbey engine stats" {
    const allocator = std.testing.allocator;

    var engine = try AbbeyEngine.init(allocator, .{});
    defer engine.deinit();

    const stats = engine.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_queries);
    try std.testing.expectEqualStrings("echo", stats.llm_backend);
}
