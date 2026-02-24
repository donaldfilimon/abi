//! Abbey Compositional Reasoning System
//!
//! Breaks down complex problems into sub-problems:
//! - Query decomposition
//! - Dependency graph construction
//! - Parallel sub-problem solving
//! - Result composition and synthesis
//! - Multi-step planning
//! - Counterfactual reasoning

const std = @import("std");
const types = @import("../../core/types.zig");
const reasoning = @import("../reasoning.zig");

// ============================================================================
// Compositional Types
// ============================================================================

/// A decomposed problem with sub-problems
pub const ProblemDecomposition = struct {
    original_query: []const u8,
    problem_type: ProblemType,
    sub_problems: std.ArrayListUnmanaged(SubProblem),
    dependencies: std.ArrayListUnmanaged(Dependency),
    execution_plan: ?ExecutionPlan,
    synthesis_strategy: SynthesisStrategy,

    pub const ProblemType = enum {
        sequential, // Sub-problems must be solved in order
        parallel, // Sub-problems can be solved independently
        hierarchical, // Sub-problems have parent-child relationships
        iterative, // Sub-problems feed back into each other
        conditional, // Different paths based on conditions
    };

    pub const Dependency = struct {
        from: usize, // Index of sub-problem that produces output
        to: usize, // Index of sub-problem that needs input
        dependency_type: DependencyType,

        pub const DependencyType = enum {
            data_flow, // Output of 'from' is input to 'to'
            precondition, // 'from' must complete before 'to'
            soft, // 'from' provides helpful context for 'to'
        };
    };

    pub const SynthesisStrategy = enum {
        concatenate, // Simply join results
        merge, // Intelligently merge results
        summarize, // Summarize combined results
        select_best, // Choose best result
        weighted_combine, // Weight by confidence
        chain_of_thought, // Present as reasoning chain
    };
};

/// A sub-problem in the decomposition
pub const SubProblem = struct {
    id: usize,
    query: []const u8,
    sub_type: SubProblemType,
    priority: f32,
    estimated_difficulty: f32,
    requires_external: bool,
    status: Status,
    result: ?SubProblemResult,

    pub const SubProblemType = enum {
        fact_retrieval,
        computation,
        reasoning,
        comparison,
        synthesis,
        verification,
        generation,
        classification,
    };

    pub const Status = enum {
        pending,
        in_progress,
        completed,
        failed,
        skipped,
    };
};

/// Result of solving a sub-problem
pub const SubProblemResult = struct {
    content: []const u8,
    confidence: f32,
    reasoning_steps: []const []const u8,
    sources_used: []const []const u8,
    time_taken_ms: i64,
};

/// Execution plan for sub-problems
pub const ExecutionPlan = struct {
    stages: []const ExecutionStage,
    estimated_total_time_ms: i64,
    parallelism_factor: f32,

    pub const ExecutionStage = struct {
        sub_problem_ids: []const usize,
        can_parallelize: bool,
        timeout_ms: i64,
    };
};

// ============================================================================
// Problem Decomposer
// ============================================================================

/// Decomposes complex queries into sub-problems
pub const ProblemDecomposer = struct {
    allocator: std.mem.Allocator,
    decomposition_patterns: std.ArrayListUnmanaged(DecompositionPattern),
    max_sub_problems: usize,
    max_depth: usize,

    const Self = @This();

    pub const DecompositionPattern = struct {
        trigger: []const u8,
        decomposition_type: ProblemDecomposition.ProblemType,
        template: []const []const u8,
    };

    pub const InitConfig = struct {
        max_sub_problems: usize = 10,
        max_depth: usize = 3,
    };

    pub fn init(allocator: std.mem.Allocator, config: InitConfig) Self {
        return Self{
            .allocator = allocator,
            .decomposition_patterns = .{},
            .max_sub_problems = config.max_sub_problems,
            .max_depth = config.max_depth,
        };
    }

    pub fn deinit(self: *Self) void {
        self.decomposition_patterns.deinit(self.allocator);
    }

    /// Decompose a complex query into sub-problems
    pub fn decompose(self: *Self, query: []const u8) !ProblemDecomposition {
        var decomposition = ProblemDecomposition{
            .original_query = query,
            .problem_type = self.detectProblemType(query),
            .sub_problems = .{},
            .dependencies = .{},
            .execution_plan = null,
            .synthesis_strategy = .chain_of_thought,
        };
        errdefer {
            decomposition.sub_problems.deinit(self.allocator);
            decomposition.dependencies.deinit(self.allocator);
        }

        // Analyze query structure
        const analysis = self.analyzeQuery(query);

        // Generate sub-problems based on analysis
        try self.generateSubProblems(&decomposition, analysis);

        // Build dependency graph
        try self.buildDependencies(&decomposition);

        // Create execution plan
        decomposition.execution_plan = try self.createExecutionPlan(&decomposition);

        return decomposition;
    }

    fn detectProblemType(self: *Self, query: []const u8) ProblemDecomposition.ProblemType {
        _ = self;

        var lower_buf: [2048]u8 = undefined;
        const len = @min(query.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(query[i]);
        }
        const lower = lower_buf[0..len];

        // Sequential indicators
        if (std.mem.indexOf(u8, lower, "step by step") != null or
            std.mem.indexOf(u8, lower, "first") != null or
            std.mem.indexOf(u8, lower, "then") != null)
        {
            return .sequential;
        }

        // Parallel indicators
        if (std.mem.indexOf(u8, lower, " and ") != null or
            std.mem.indexOf(u8, lower, "compare") != null or
            std.mem.indexOf(u8, lower, "both") != null)
        {
            return .parallel;
        }

        // Conditional indicators
        if (std.mem.indexOf(u8, lower, "if ") != null or
            std.mem.indexOf(u8, lower, "depending") != null or
            std.mem.indexOf(u8, lower, "whether") != null)
        {
            return .conditional;
        }

        // Hierarchical indicators
        if (std.mem.indexOf(u8, lower, "explain") != null or
            std.mem.indexOf(u8, lower, "break down") != null)
        {
            return .hierarchical;
        }

        return .sequential;
    }

    fn analyzeQuery(self: *Self, query: []const u8) QueryAnalysis {
        _ = self;

        var analysis = QueryAnalysis{
            .components = .{},
            .connectors = .{},
            .modifiers = .{},
        };

        // Extract question words
        const question_words = [_][]const u8{ "what", "how", "why", "when", "where", "who", "which" };
        for (question_words) |qw| {
            if (std.mem.indexOf(u8, query, qw) != null) {
                analysis.question_type = qw;
                break;
            }
        }

        // Count logical connectors
        if (std.mem.indexOf(u8, query, " and ") != null) analysis.has_conjunction = true;
        if (std.mem.indexOf(u8, query, " or ") != null) analysis.has_disjunction = true;
        if (std.mem.indexOf(u8, query, " but ") != null) analysis.has_contrast = true;

        // Estimate complexity
        analysis.estimated_complexity = @as(f32, @floatFromInt(query.len)) / 200.0;
        if (analysis.has_conjunction) analysis.estimated_complexity += 0.2;
        if (analysis.has_disjunction) analysis.estimated_complexity += 0.3;

        return analysis;
    }

    const QueryAnalysis = struct {
        components: std.ArrayListUnmanaged([]const u8) = .{},
        connectors: std.ArrayListUnmanaged([]const u8) = .{},
        modifiers: std.ArrayListUnmanaged([]const u8) = .{},
        question_type: ?[]const u8 = null,
        has_conjunction: bool = false,
        has_disjunction: bool = false,
        has_contrast: bool = false,
        estimated_complexity: f32 = 0.5,
    };

    fn generateSubProblems(
        self: *Self,
        decomposition: *ProblemDecomposition,
        analysis: QueryAnalysis,
    ) !void {
        // Generate sub-problems based on problem type and analysis
        switch (decomposition.problem_type) {
            .sequential => {
                // Create step-by-step sub-problems
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 0,
                    .query = "Understand the context and requirements",
                    .sub_type = .fact_retrieval,
                    .priority = 1.0,
                    .estimated_difficulty = 0.3,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 1,
                    .query = "Analyze and reason about the problem",
                    .sub_type = .reasoning,
                    .priority = 0.9,
                    .estimated_difficulty = analysis.estimated_complexity,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 2,
                    .query = "Synthesize the final answer",
                    .sub_type = .synthesis,
                    .priority = 0.8,
                    .estimated_difficulty = 0.4,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
            },
            .parallel => {
                // Create independent sub-problems
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 0,
                    .query = "Analyze first component",
                    .sub_type = .reasoning,
                    .priority = 1.0,
                    .estimated_difficulty = analysis.estimated_complexity * 0.6,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 1,
                    .query = "Analyze second component",
                    .sub_type = .reasoning,
                    .priority = 1.0,
                    .estimated_difficulty = analysis.estimated_complexity * 0.6,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 2,
                    .query = "Compare and synthesize",
                    .sub_type = .comparison,
                    .priority = 0.8,
                    .estimated_difficulty = 0.5,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
            },
            .conditional => {
                // Create branching sub-problems
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 0,
                    .query = "Evaluate the condition",
                    .sub_type = .reasoning,
                    .priority = 1.0,
                    .estimated_difficulty = 0.4,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 1,
                    .query = "Handle positive case",
                    .sub_type = .reasoning,
                    .priority = 0.9,
                    .estimated_difficulty = analysis.estimated_complexity * 0.5,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 2,
                    .query = "Handle negative case",
                    .sub_type = .reasoning,
                    .priority = 0.9,
                    .estimated_difficulty = analysis.estimated_complexity * 0.5,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
            },
            .hierarchical => {
                // Create parent-child sub-problems
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 0,
                    .query = "High-level overview",
                    .sub_type = .synthesis,
                    .priority = 1.0,
                    .estimated_difficulty = 0.5,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 1,
                    .query = "Detailed breakdown",
                    .sub_type = .reasoning,
                    .priority = 0.9,
                    .estimated_difficulty = analysis.estimated_complexity,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
            },
            .iterative => {
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 0,
                    .query = "Initial analysis",
                    .sub_type = .reasoning,
                    .priority = 1.0,
                    .estimated_difficulty = 0.4,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
                try decomposition.sub_problems.append(self.allocator, .{
                    .id = 1,
                    .query = "Refinement iteration",
                    .sub_type = .verification,
                    .priority = 0.9,
                    .estimated_difficulty = 0.5,
                    .requires_external = false,
                    .status = .pending,
                    .result = null,
                });
            },
        }
    }

    fn buildDependencies(
        self: *Self,
        decomposition: *ProblemDecomposition,
    ) !void {
        // Build dependency graph based on problem type
        switch (decomposition.problem_type) {
            .sequential => {
                // Linear dependencies
                for (1..decomposition.sub_problems.items.len) |i| {
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = i - 1,
                        .to = i,
                        .dependency_type = .precondition,
                    });
                }
            },
            .parallel => {
                // All parallel sub-problems depend on synthesis
                const synthesis_idx = decomposition.sub_problems.items.len - 1;
                for (0..synthesis_idx) |i| {
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = i,
                        .to = synthesis_idx,
                        .dependency_type = .data_flow,
                    });
                }
            },
            .conditional => {
                // Branches depend on condition
                if (decomposition.sub_problems.items.len >= 3) {
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = 0,
                        .to = 1,
                        .dependency_type = .precondition,
                    });
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = 0,
                        .to = 2,
                        .dependency_type = .precondition,
                    });
                }
            },
            .hierarchical => {
                // Children depend on parent
                if (decomposition.sub_problems.items.len >= 2) {
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = 0,
                        .to = 1,
                        .dependency_type = .soft,
                    });
                }
            },
            .iterative => {
                // Circular soft dependency
                if (decomposition.sub_problems.items.len >= 2) {
                    try decomposition.dependencies.append(self.allocator, .{
                        .from = 0,
                        .to = 1,
                        .dependency_type = .data_flow,
                    });
                }
            },
        }
    }

    fn createExecutionPlan(
        self: *Self,
        decomposition: *const ProblemDecomposition,
    ) !ExecutionPlan {
        // Simplified execution planning
        var stages = std.ArrayListUnmanaged(ExecutionPlan.ExecutionStage).empty;

        switch (decomposition.problem_type) {
            .parallel => {
                // First stage: parallel sub-problems
                const parallel_ids = [_]usize{ 0, 1 };
                try stages.append(self.allocator, .{
                    .sub_problem_ids = &parallel_ids,
                    .can_parallelize = true,
                    .timeout_ms = 5000,
                });
                // Second stage: synthesis
                const synthesis_ids = [_]usize{2};
                try stages.append(self.allocator, .{
                    .sub_problem_ids = &synthesis_ids,
                    .can_parallelize = false,
                    .timeout_ms = 3000,
                });
            },
            else => {
                // Sequential execution
                for (0..decomposition.sub_problems.items.len) |i| {
                    const ids = [_]usize{i};
                    try stages.append(self.allocator, .{
                        .sub_problem_ids = &ids,
                        .can_parallelize = false,
                        .timeout_ms = 3000,
                    });
                }
            },
        }

        const owned_stages = try stages.toOwnedSlice(self.allocator);
        return ExecutionPlan{
            .stages = owned_stages,
            .estimated_total_time_ms = @as(i64, @intCast(owned_stages.len)) * 3000,
            .parallelism_factor = if (decomposition.problem_type == .parallel) 0.5 else 1.0,
        };
    }

    /// Clean up a decomposition
    pub fn freeDecomposition(self: *Self, decomposition: *ProblemDecomposition) void {
        if (decomposition.execution_plan) |plan| {
            self.allocator.free(plan.stages);
        }
        decomposition.sub_problems.deinit(self.allocator);
        decomposition.dependencies.deinit(self.allocator);
    }
};

// ============================================================================
// Counterfactual Reasoner
// ============================================================================

/// Reasons about "what if" scenarios
pub const CounterfactualReasoner = struct {
    allocator: std.mem.Allocator,
    causal_models: std.StringHashMapUnmanaged(CausalModel),

    const Self = @This();

    pub const CausalModel = struct {
        variables: []const []const u8,
        relationships: []const CausalRelation,
    };

    pub const CausalRelation = struct {
        cause: []const u8,
        effect: []const u8,
        strength: f32,
        mechanism: []const u8,
    };

    pub const CounterfactualQuery = struct {
        original_state: []const u8,
        intervention: []const u8,
        target_variable: []const u8,
    };

    pub const CounterfactualResult = struct {
        predicted_outcome: []const u8,
        confidence: f32,
        reasoning_chain: []const []const u8,
        assumptions: []const []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .causal_models = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.causal_models.deinit(self.allocator);
    }

    /// Reason about a counterfactual scenario
    pub fn reason(self: *Self, _: CounterfactualQuery) CounterfactualResult {
        _ = self;

        // Simplified counterfactual reasoning
        return CounterfactualResult{
            .predicted_outcome = "Hypothetical outcome based on intervention",
            .confidence = 0.6, // Lower confidence for counterfactuals
            .reasoning_chain = &[_][]const u8{
                "Identified causal mechanism",
                "Applied intervention to model",
                "Propagated effects through causal graph",
                "Derived predicted outcome",
            },
            .assumptions = &[_][]const u8{
                "Causal structure remains stable",
                "No unmodeled confounders",
                "Intervention is valid",
            },
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "problem decomposer initialization" {
    const allocator = std.testing.allocator;

    var decomposer = ProblemDecomposer.init(allocator, .{});
    defer decomposer.deinit();

    try std.testing.expectEqual(@as(usize, 10), decomposer.max_sub_problems);
}

test "problem decomposition" {
    const allocator = std.testing.allocator;

    var decomposer = ProblemDecomposer.init(allocator, .{});
    defer decomposer.deinit();

    var decomposition = try decomposer.decompose("What is X and how does it compare to Y?");
    defer decomposer.freeDecomposition(&decomposition);

    try std.testing.expect(decomposition.sub_problems.items.len > 0);
}

test "problem type detection" {
    const allocator = std.testing.allocator;

    var decomposer = ProblemDecomposer.init(allocator, .{});
    defer decomposer.deinit();

    const sequential = decomposer.detectProblemType("First do X, then do Y");
    try std.testing.expectEqual(ProblemDecomposition.ProblemType.sequential, sequential);

    const parallel = decomposer.detectProblemType("Compare A and B");
    try std.testing.expectEqual(ProblemDecomposition.ProblemType.parallel, parallel);
}

test "counterfactual reasoner" {
    const allocator = std.testing.allocator;

    var reasoner = CounterfactualReasoner.init(allocator);
    defer reasoner.deinit();

    const result = reasoner.reason(.{
        .original_state = "Current situation",
        .intervention = "Change X",
        .target_variable = "Y",
    });

    try std.testing.expect(result.confidence > 0);
    try std.testing.expect(result.reasoning_chain.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
