//! Abbey Advanced AI Modules
//!
//! This module provides cutting-edge cognitive capabilities:
//! - Meta-learning: Learning to learn with MAML-inspired adaptation
//! - Theory of Mind: Modeling user mental states, beliefs, and intentions
//! - Compositional Reasoning: Breaking down complex problems into sub-problems
//! - Self-Reflection: Meta-cognitive evaluation and improvement

const std = @import("std");

// ============================================================================
// Sub-modules
// ============================================================================

pub const meta_learning = @import("meta_learning.zig");
pub const theory_of_mind = @import("theory_of_mind.zig");
pub const compositional_reasoning = @import("compositional_reasoning.zig");
pub const self_reflection = @import("self_reflection.zig");

// ============================================================================
// Type Re-exports (Meta-Learning)
// ============================================================================

pub const TaskProfile = meta_learning.TaskProfile;
pub const TaskDomain = meta_learning.TaskDomain;
pub const LearningStrategy = meta_learning.LearningStrategy;
pub const MetaLearner = meta_learning.MetaLearner;
pub const FewShotLearner = meta_learning.FewShotLearner;
pub const CurriculumScheduler = meta_learning.CurriculumScheduler;

// ============================================================================
// Type Re-exports (Theory of Mind)
// ============================================================================

pub const MentalModel = theory_of_mind.MentalModel;
pub const BeliefSystem = theory_of_mind.BeliefSystem;
pub const KnowledgeState = theory_of_mind.KnowledgeState;
pub const IntentionTracker = theory_of_mind.IntentionTracker;
pub const EmotionalModel = theory_of_mind.EmotionalModel;
pub const TheoryOfMind = theory_of_mind.TheoryOfMind;
pub const IntentionInference = theory_of_mind.IntentionInference;
pub const PerspectiveAnalysis = theory_of_mind.PerspectiveAnalysis;

// ============================================================================
// Type Re-exports (Compositional Reasoning)
// ============================================================================

pub const ProblemDecomposition = compositional_reasoning.ProblemDecomposition;
pub const SubProblem = compositional_reasoning.SubProblem;
pub const ExecutionPlan = compositional_reasoning.ExecutionPlan;
pub const ProblemDecomposer = compositional_reasoning.ProblemDecomposer;
pub const CounterfactualReasoner = compositional_reasoning.CounterfactualReasoner;
pub const Scenario = compositional_reasoning.Scenario;

// ============================================================================
// Type Re-exports (Self-Reflection)
// ============================================================================

pub const SelfEvaluation = self_reflection.SelfEvaluation;
pub const UncertaintyArea = self_reflection.UncertaintyArea;
pub const DetectedBias = self_reflection.DetectedBias;
pub const ReasoningQuality = self_reflection.ReasoningQuality;
pub const SelfReflectionEngine = self_reflection.SelfReflectionEngine;
pub const ImprovementTrend = self_reflection.ImprovementTrend;

// ============================================================================
// Integrated Advanced Cognitive System
// ============================================================================

/// Unified advanced cognitive system integrating all capabilities
pub const AdvancedCognition = struct {
    allocator: std.mem.Allocator,

    // Core components
    meta_learner: MetaLearner,
    theory_of_mind_engine: TheoryOfMind,
    problem_decomposer: ProblemDecomposer,
    counterfactual_reasoner: CounterfactualReasoner,
    self_reflection_engine: SelfReflectionEngine,

    // Integration state
    cognitive_load: f32,
    active_mental_models: usize,
    reasoning_depth: usize,

    const Self = @This();

    pub const Config = struct {
        max_sub_problems: usize = 10,
        max_reasoning_depth: usize = 5,
        meta_learning_rate: f32 = 0.01,
        enable_counterfactuals: bool = true,
        self_reflection_threshold: f32 = 0.7,
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
        return .{
            .allocator = allocator,
            .meta_learner = try MetaLearner.init(allocator, .{
                .meta_learning_rate = config.meta_learning_rate,
            }),
            .theory_of_mind_engine = TheoryOfMind.init(allocator),
            .problem_decomposer = ProblemDecomposer.init(allocator, .{
                .max_sub_problems = config.max_sub_problems,
                .max_depth = config.max_reasoning_depth,
            }),
            .counterfactual_reasoner = CounterfactualReasoner.init(allocator, .{}),
            .self_reflection_engine = SelfReflectionEngine.init(allocator, .{}),
            .cognitive_load = 0.0,
            .active_mental_models = 0,
            .reasoning_depth = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.meta_learner.deinit();
        self.theory_of_mind_engine.deinit();
        self.problem_decomposer.deinit();
        self.counterfactual_reasoner.deinit();
        self.self_reflection_engine.deinit();
    }

    /// Process a query with full cognitive integration
    pub fn process(self: *Self, user_id: []const u8, query: []const u8) !CognitiveResult {
        // 1. Get or create mental model for user
        const mental_model = try self.theory_of_mind_engine.getModel(user_id);
        self.active_mental_models = self.theory_of_mind_engine.models.count();

        // 2. Analyze task profile for meta-learning
        const task_profile = self.analyzeTaskProfile(query);
        const strategy = self.meta_learner.selectStrategy(task_profile);

        // 3. Decompose problem if complex
        var decomposition: ?ProblemDecomposition = null;
        if (task_profile.complexity > 0.6) {
            decomposition = try self.problem_decomposer.decompose(query);
            self.reasoning_depth = if (decomposition) |d| d.sub_problems.len else 0;
        }

        // 4. Infer user intention
        const intention = self.theory_of_mind_engine.inferIntention(query);

        // 5. Take perspective on potential response
        const perspective = self.theory_of_mind_engine.takePerspective(mental_model, query);

        // Update cognitive load
        self.cognitive_load = self.calculateCognitiveLoad(task_profile, decomposition);

        return .{
            .task_profile = task_profile,
            .learning_strategy = strategy,
            .decomposition = decomposition,
            .intention = intention,
            .perspective = perspective,
            .cognitive_load = self.cognitive_load,
            .reasoning_depth = self.reasoning_depth,
        };
    }

    /// Evaluate and learn from a response
    pub fn evaluateAndLearn(
        self: *Self,
        query: []const u8,
        response: []const u8,
        task_profile: TaskProfile,
        strategy: LearningStrategy,
        user_feedback_score: ?f32,
    ) !SelfEvaluation {
        // Self-evaluate the response
        const evaluation = try self.self_reflection_engine.evaluate(response, query, null);

        // Use feedback if available, otherwise use self-evaluation
        const success_score = user_feedback_score orelse evaluation.overall_quality;

        // Record outcome for meta-learning
        try self.meta_learner.recordOutcome(
            task_profile,
            strategy,
            success_score,
            100.0, // adaptation time placeholder
        );

        return evaluation;
    }

    fn analyzeTaskProfile(self: *Self, query: []const u8) TaskProfile {
        _ = self;

        // Simple heuristic analysis
        var complexity: f32 = 0.3;
        var novelty: f32 = 0.5;
        var ambiguity: f32 = 0.3;

        // Length-based complexity
        if (query.len > 200) complexity += 0.2;
        if (query.len > 500) complexity += 0.2;

        // Question indicators
        const has_how = std.mem.indexOf(u8, query, "how") != null;
        const has_why = std.mem.indexOf(u8, query, "why") != null;
        const has_what_if = std.mem.indexOf(u8, query, "what if") != null;

        if (has_how) complexity += 0.1;
        if (has_why) complexity += 0.15;
        if (has_what_if) {
            complexity += 0.2;
            novelty += 0.2;
        }

        // Ambiguity indicators
        const has_maybe = std.mem.indexOf(u8, query, "maybe") != null;
        const has_might = std.mem.indexOf(u8, query, "might") != null;
        const has_or = std.mem.indexOf(u8, query, " or ") != null;

        if (has_maybe or has_might) ambiguity += 0.2;
        if (has_or) ambiguity += 0.15;

        return .{
            .complexity = @min(1.0, complexity),
            .novelty = @min(1.0, novelty),
            .ambiguity = @min(1.0, ambiguity),
            .time_sensitivity = 0.5,
            .domain = .general,
            .context_dependency = 0.5,
        };
    }

    fn calculateCognitiveLoad(
        self: *Self,
        profile: TaskProfile,
        decomposition: ?ProblemDecomposition,
    ) f32 {
        _ = self;

        var load: f32 = profile.complexity * 0.4 + profile.ambiguity * 0.3;

        if (decomposition) |d| {
            load += @as(f32, @floatFromInt(d.sub_problems.len)) * 0.05;
        }

        return @min(1.0, load);
    }

    /// Get current cognitive state summary
    pub fn getCognitiveState(self: *const Self) CognitiveState {
        return .{
            .cognitive_load = self.cognitive_load,
            .active_mental_models = self.active_mental_models,
            .reasoning_depth = self.reasoning_depth,
            .improvement_trend = self.self_reflection_engine.getImprovementTrend(),
            .total_evaluations = self.self_reflection_engine.evaluations.items.len,
        };
    }
};

/// Result of cognitive processing
pub const CognitiveResult = struct {
    task_profile: TaskProfile,
    learning_strategy: LearningStrategy,
    decomposition: ?ProblemDecomposition,
    intention: IntentionInference,
    perspective: PerspectiveAnalysis,
    cognitive_load: f32,
    reasoning_depth: usize,
};

/// Current cognitive state
pub const CognitiveState = struct {
    cognitive_load: f32,
    active_mental_models: usize,
    reasoning_depth: usize,
    improvement_trend: ImprovementTrend,
    total_evaluations: usize,
};

// ============================================================================
// Tests
// ============================================================================

test "advanced cognition initialization" {
    const allocator = std.testing.allocator;

    var cognition = try AdvancedCognition.init(allocator, .{});
    defer cognition.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), cognition.cognitive_load);
    try std.testing.expectEqual(@as(usize, 0), cognition.active_mental_models);
}

test "cognitive processing" {
    const allocator = std.testing.allocator;

    var cognition = try AdvancedCognition.init(allocator, .{});
    defer cognition.deinit();

    const result = try cognition.process("user123", "How do I implement a neural network?");

    try std.testing.expect(result.task_profile.complexity > 0.3);
    try std.testing.expect(result.cognitive_load >= 0.0);
}

test "meta-learning integration" {
    const allocator = std.testing.allocator;

    var cognition = try AdvancedCognition.init(allocator, .{
        .meta_learning_rate = 0.02,
    });
    defer cognition.deinit();

    // Process and learn
    const result = try cognition.process("user1", "Explain quantum computing");
    const evaluation = try cognition.evaluateAndLearn(
        "Explain quantum computing",
        "Quantum computing uses qubits...",
        result.task_profile,
        result.learning_strategy,
        0.85,
    );

    try std.testing.expect(evaluation.overall_quality > 0.0);
}

test "cognitive state tracking" {
    const allocator = std.testing.allocator;

    var cognition = try AdvancedCognition.init(allocator, .{});
    defer cognition.deinit();

    _ = try cognition.process("user1", "Simple question");
    _ = try cognition.process("user2", "Another complex question with multiple parts");

    const state = cognition.getCognitiveState();
    try std.testing.expectEqual(@as(usize, 2), state.active_mental_models);
}

test {
    _ = meta_learning;
    _ = theory_of_mind;
    _ = compositional_reasoning;
    _ = self_reflection;
}
