//! Abbey Theory of Mind (ToM) System
//!
//! Models user mental states, beliefs, and intentions:
//! - Belief tracking and updating
//! - Intention recognition
//! - Perspective taking
//! - Knowledge gap detection
//! - Emotional state inference
//! - Communication adaptation

const std = @import("std");
const types = @import("../../core/types.zig");
const emotions = @import("../emotions.zig");

// ============================================================================
// Mental Model Types
// ============================================================================

/// Complete mental model of a user
pub const MentalModel = struct {
    user_id: []const u8,

    // Belief system
    beliefs: BeliefSystem,

    // Knowledge state
    knowledge_state: KnowledgeState,

    // Intentions and goals
    intentions: IntentionTracker,

    // Emotional model
    emotional_model: EmotionalModel,

    // Communication preferences
    preferences: CommunicationPreferences,

    // Model confidence
    confidence: f32,
    last_updated: i64,
};

/// User's belief system
pub const BeliefSystem = struct {
    explicit_beliefs: std.StringHashMapUnmanaged(Belief),
    inferred_beliefs: std.StringHashMapUnmanaged(Belief),
    belief_conflicts: std.ArrayListUnmanaged(BeliefConflict),

    pub const Belief = struct {
        content: []const u8,
        confidence: f32,
        source: BeliefSource,
        timestamp: i64,
        supporting_evidence: []const []const u8,
    };

    pub const BeliefSource = enum {
        stated_directly,
        inferred_from_questions,
        inferred_from_behavior,
        assumed_common_knowledge,
        corrected_misconception,
    };

    pub const BeliefConflict = struct {
        belief_a: []const u8,
        belief_b: []const u8,
        conflict_type: ConflictType,
        resolution_status: ResolutionStatus,

        pub const ConflictType = enum {
            direct_contradiction,
            logical_inconsistency,
            temporal_inconsistency,
            scope_mismatch,
        };

        pub const ResolutionStatus = enum {
            unresolved,
            resolved_kept_a,
            resolved_kept_b,
            resolved_merged,
        };
    };
};

/// User's knowledge state model
pub const KnowledgeState = struct {
    known_topics: std.StringHashMapUnmanaged(TopicKnowledge),
    expertise_areas: std.ArrayListUnmanaged(ExpertiseArea),
    knowledge_gaps: std.ArrayListUnmanaged(KnowledgeGap),

    pub const TopicKnowledge = struct {
        topic: []const u8,
        estimated_level: KnowledgeLevel,
        confidence: f32,
        evidence_count: usize,
    };

    pub const KnowledgeLevel = enum(u8) {
        novice = 0,
        beginner = 1,
        intermediate = 2,
        advanced = 3,
        expert = 4,

        pub fn toFloat(self: KnowledgeLevel) f32 {
            return @as(f32, @floatFromInt(@intFromEnum(self))) / 4.0;
        }
    };

    pub const ExpertiseArea = struct {
        domain: []const u8,
        level: KnowledgeLevel,
        confidence: f32,
    };

    pub const KnowledgeGap = struct {
        topic: []const u8,
        detected_from: []const u8, // What question revealed the gap
        importance: f32,
        timestamp: i64,
    };
};

/// User intention tracking
pub const IntentionTracker = struct {
    current_goal: ?UserGoal,
    goal_stack: std.ArrayListUnmanaged(UserGoal),
    past_goals: std.ArrayListUnmanaged(CompletedGoal),

    pub const UserGoal = struct {
        description: []const u8,
        goal_type: GoalType,
        urgency: f32,
        clarity: f32,
        sub_goals: []const []const u8,
        blockers: []const []const u8,
    };

    pub const GoalType = enum {
        information_seeking,
        problem_solving,
        learning,
        task_completion,
        exploration,
        validation,
        social_connection,
        entertainment,
    };

    pub const CompletedGoal = struct {
        goal: UserGoal,
        outcome: GoalOutcome,
        completion_time: i64,
    };

    pub const GoalOutcome = enum {
        achieved,
        partially_achieved,
        abandoned,
        redirected,
    };
};

/// Emotional model of user
pub const EmotionalModel = struct {
    current_state: emotions.EmotionType,
    emotional_baseline: emotions.EmotionType,
    emotional_volatility: f32,
    trigger_patterns: std.ArrayListUnmanaged(EmotionalTrigger),
    emotional_history: std.ArrayListUnmanaged(EmotionalEvent),

    pub const EmotionalTrigger = struct {
        trigger: []const u8,
        response: emotions.EmotionType,
        intensity: f32,
        frequency: usize,
    };

    pub const EmotionalEvent = struct {
        emotion: emotions.EmotionType,
        intensity: f32,
        context: []const u8,
        timestamp: i64,
    };
};

/// Communication preferences
pub const CommunicationPreferences = struct {
    verbosity: VerbosityLevel,
    formality: FormalityLevel,
    technical_depth: f32,
    prefers_examples: bool,
    prefers_analogies: bool,
    humor_receptivity: f32,
    directness: f32,

    pub const VerbosityLevel = enum {
        minimal,
        concise,
        balanced,
        detailed,
        comprehensive,
    };

    pub const FormalityLevel = enum {
        casual,
        conversational,
        professional,
        formal,
        academic,
    };
};

// ============================================================================
// Theory of Mind Engine
// ============================================================================

/// Main ToM engine for modeling user mental states
pub const TheoryOfMind = struct {
    allocator: std.mem.Allocator,
    models: std.StringHashMapUnmanaged(MentalModel),
    default_model: DefaultModelConfig,

    const Self = @This();

    pub const DefaultModelConfig = struct {
        default_knowledge_level: KnowledgeState.KnowledgeLevel = .intermediate,
        default_formality: CommunicationPreferences.FormalityLevel = .conversational,
        default_verbosity: CommunicationPreferences.VerbosityLevel = .balanced,
        belief_decay_rate: f32 = 0.01,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .models = .{},
            .default_model = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.models.iterator();
        while (it.next()) |entry| {
            self.freeModel(entry.value_ptr);
        }
        self.models.deinit(self.allocator);
    }

    fn freeModel(self: *Self, model: *MentalModel) void {
        model.beliefs.explicit_beliefs.deinit(self.allocator);
        model.beliefs.inferred_beliefs.deinit(self.allocator);
        model.beliefs.belief_conflicts.deinit(self.allocator);
        model.knowledge_state.known_topics.deinit(self.allocator);
        model.knowledge_state.expertise_areas.deinit(self.allocator);
        model.knowledge_state.knowledge_gaps.deinit(self.allocator);
        model.intentions.goal_stack.deinit(self.allocator);
        model.intentions.past_goals.deinit(self.allocator);
        model.emotional_model.trigger_patterns.deinit(self.allocator);
        model.emotional_model.emotional_history.deinit(self.allocator);
    }

    /// Get or create mental model for user
    pub fn getModel(self: *Self, user_id: []const u8) !*MentalModel {
        const gop = try self.models.getOrPut(self.allocator, user_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = self.createDefaultModel(user_id);
        }
        return gop.value_ptr;
    }

    fn createDefaultModel(self: *Self, user_id: []const u8) MentalModel {
        return MentalModel{
            .user_id = user_id,
            .beliefs = .{
                .explicit_beliefs = .{},
                .inferred_beliefs = .{},
                .belief_conflicts = .{},
            },
            .knowledge_state = .{
                .known_topics = .{},
                .expertise_areas = .{},
                .knowledge_gaps = .{},
            },
            .intentions = .{
                .current_goal = null,
                .goal_stack = .{},
                .past_goals = .{},
            },
            .emotional_model = .{
                .current_state = .neutral,
                .emotional_baseline = .neutral,
                .emotional_volatility = 0.5,
                .trigger_patterns = .{},
                .emotional_history = .{},
            },
            .preferences = .{
                .verbosity = self.default_model.default_verbosity,
                .formality = self.default_model.default_formality,
                .technical_depth = 0.5,
                .prefers_examples = true,
                .prefers_analogies = true,
                .humor_receptivity = 0.5,
                .directness = 0.5,
            },
            .confidence = 0.3, // Low initial confidence
            .last_updated = types.getTimestampSec(),
        };
    }

    /// Infer user intention from message
    pub fn inferIntention(self: *Self, message: []const u8) IntentionInference {
        var lower_buf: [2048]u8 = undefined;
        const len = @min(message.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(message[i]);
        }
        const lower = lower_buf[0..len];

        // Pattern matching for intention types
        const info_patterns = [_][]const u8{ "what is", "how does", "explain", "tell me", "what are" };
        const problem_patterns = [_][]const u8{ "how do i", "help me", "fix", "solve", "issue", "problem", "error" };
        const learn_patterns = [_][]const u8{ "learn", "understand", "study", "teach me", "tutorial" };
        const validate_patterns = [_][]const u8{ "is this", "am i", "correct", "right", "should i" };

        var goal_type: IntentionTracker.GoalType = .exploration;
        var confidence: f32 = 0.3;

        for (info_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                goal_type = .information_seeking;
                confidence = 0.7;
                break;
            }
        }
        for (problem_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                goal_type = .problem_solving;
                confidence = 0.8;
                break;
            }
        }
        for (learn_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                goal_type = .learning;
                confidence = 0.7;
                break;
            }
        }
        for (validate_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                goal_type = .validation;
                confidence = 0.6;
                break;
            }
        }

        // Estimate urgency from punctuation and words
        var urgency: f32 = 0.5;
        if (std.mem.indexOf(u8, message, "!") != null) urgency += 0.2;
        if (std.mem.indexOf(u8, lower, "urgent") != null) urgency += 0.3;
        if (std.mem.indexOf(u8, lower, "asap") != null) urgency += 0.3;
        if (std.mem.indexOf(u8, lower, "please") != null) urgency += 0.1;
        urgency = @min(1.0, urgency);

        return IntentionInference{
            .goal_type = goal_type,
            .urgency = urgency,
            .confidence = confidence,
            .suggested_response_style = self.suggestResponseStyle(goal_type, urgency),
        };
    }

    fn suggestResponseStyle(self: *Self, goal: IntentionTracker.GoalType, urgency: f32) ResponseStyle {
        _ = self;
        return switch (goal) {
            .information_seeking => .{
                .include_explanation = true,
                .include_examples = true,
                .max_length = if (urgency > 0.7) .concise else .detailed,
            },
            .problem_solving => .{
                .include_explanation = urgency < 0.5,
                .include_examples = true,
                .max_length = .balanced,
            },
            .learning => .{
                .include_explanation = true,
                .include_examples = true,
                .max_length = .detailed,
            },
            .validation => .{
                .include_explanation = true,
                .include_examples = false,
                .max_length = .concise,
            },
            else => .{
                .include_explanation = true,
                .include_examples = true,
                .max_length = .balanced,
            },
        };
    }

    pub const IntentionInference = struct {
        goal_type: IntentionTracker.GoalType,
        urgency: f32,
        confidence: f32,
        suggested_response_style: ResponseStyle,
    };

    pub const ResponseStyle = struct {
        include_explanation: bool,
        include_examples: bool,
        max_length: LengthPreference,

        pub const LengthPreference = enum {
            minimal,
            concise,
            balanced,
            detailed,
        };
    };

    /// Detect knowledge gaps from user message
    pub fn detectKnowledgeGaps(self: *Self, message: []const u8, context: []const u8) []const KnowledgeGapIndicator {
        _ = self;
        _ = context;

        // Static buffer for results (simplified)
        var indicators: [8]KnowledgeGapIndicator = undefined;
        var count: usize = 0;

        var lower_buf: [2048]u8 = undefined;
        const len = @min(message.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(message[i]);
        }
        const lower = lower_buf[0..len];

        // Patterns indicating knowledge gaps
        const gap_patterns = [_]struct { pattern: []const u8, gap_type: GapType }{
            .{ .pattern = "what does", .gap_type = .definition },
            .{ .pattern = "i don't understand", .gap_type = .comprehension },
            .{ .pattern = "confused about", .gap_type = .comprehension },
            .{ .pattern = "how do you", .gap_type = .procedural },
            .{ .pattern = "why does", .gap_type = .causal },
            .{ .pattern = "difference between", .gap_type = .comparison },
            .{ .pattern = "what's the purpose", .gap_type = .purpose },
        };

        for (gap_patterns) |gp| {
            if (std.mem.indexOf(u8, lower, gp.pattern) != null and count < indicators.len) {
                indicators[count] = .{
                    .gap_type = gp.gap_type,
                    .topic = message, // Would extract actual topic in production
                    .confidence = 0.7,
                };
                count += 1;
            }
        }

        return indicators[0..count];
    }

    pub const KnowledgeGapIndicator = struct {
        gap_type: GapType,
        topic: []const u8,
        confidence: f32,
    };

    pub const GapType = enum {
        definition,
        comprehension,
        procedural,
        causal,
        comparison,
        purpose,
        context,
    };

    /// Take user's perspective to predict their understanding
    pub fn takePerspective(
        self: *Self,
        model: *const MentalModel,
        content: []const u8,
    ) PerspectiveAnalysis {
        // Estimate how well user would understand content
        var understanding_score: f32 = 0.5;
        var needs_simplification = false;
        var suggested_additions: [4][]const u8 = undefined;
        var addition_count: usize = 0;

        // Check against knowledge state
        const expertise = model.knowledge_state.expertise_areas;
        if (expertise.items.len > 0) {
            // Adjust based on expertise level
            for (expertise.items) |area| {
                understanding_score += area.level.toFloat() * 0.1;
            }
        }

        // Check if content might need clarification
        if (std.mem.indexOf(u8, content, "therefore") != null or
            std.mem.indexOf(u8, content, "consequently") != null)
        {
            if (understanding_score < 0.6) {
                needs_simplification = true;
            }
        }

        // Check preferences
        if (model.preferences.prefers_examples and addition_count < suggested_additions.len) {
            suggested_additions[addition_count] = "Consider adding an example";
            addition_count += 1;
        }
        if (model.preferences.prefers_analogies and addition_count < suggested_additions.len) {
            suggested_additions[addition_count] = "An analogy might help";
            addition_count += 1;
        }

        return PerspectiveAnalysis{
            .predicted_understanding = @min(1.0, understanding_score),
            .needs_simplification = needs_simplification,
            .suggested_additions = suggested_additions[0..addition_count],
            .emotional_impact = self.predictEmotionalImpact(model, content),
        };
    }

    fn predictEmotionalImpact(self: *Self, model: *const MentalModel, content: []const u8) EmotionalImpact {
        _ = self;
        _ = model;

        // Simple sentiment analysis
        var valence: f32 = 0;

        const positive = [_][]const u8{ "great", "good", "excellent", "wonderful", "helpful", "thank" };
        const negative = [_][]const u8{ "bad", "wrong", "error", "fail", "unfortunately", "sorry" };

        var lower_buf: [2048]u8 = undefined;
        const len = @min(content.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(content[i]);
        }
        const lower = lower_buf[0..len];

        for (positive) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) valence += 0.1;
        }
        for (negative) |n| {
            if (std.mem.indexOf(u8, lower, n) != null) valence -= 0.1;
        }

        valence = std.math.clamp(valence, -1.0, 1.0);

        return EmotionalImpact{
            .expected_valence = valence,
            .expected_emotion = if (valence > 0.2)
                .grateful
            else if (valence < -0.2)
                .disappointed
            else
                .neutral,
            .confidence = 0.5,
        };
    }

    pub const PerspectiveAnalysis = struct {
        predicted_understanding: f32,
        needs_simplification: bool,
        suggested_additions: []const []const u8,
        emotional_impact: EmotionalImpact,
    };

    pub const EmotionalImpact = struct {
        expected_valence: f32,
        expected_emotion: emotions.EmotionType,
        confidence: f32,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "theory of mind initialization" {
    const allocator = std.testing.allocator;

    var tom = TheoryOfMind.init(allocator);
    defer tom.deinit();

    const model = try tom.getModel("user123");
    try std.testing.expectEqualStrings("user123", model.user_id);
}

test "intention inference" {
    const allocator = std.testing.allocator;

    var tom = TheoryOfMind.init(allocator);
    defer tom.deinit();

    const inference = tom.inferIntention("How do I fix this error?");
    try std.testing.expectEqual(IntentionTracker.GoalType.problem_solving, inference.goal_type);
    try std.testing.expect(inference.confidence > 0.5);
}

test "knowledge gap detection" {
    const allocator = std.testing.allocator;

    var tom = TheoryOfMind.init(allocator);
    defer tom.deinit();

    const gaps = tom.detectKnowledgeGaps("What does polymorphism mean?", "");
    try std.testing.expect(gaps.len > 0);
    try std.testing.expectEqual(TheoryOfMind.GapType.definition, gaps[0].gap_type);
}
