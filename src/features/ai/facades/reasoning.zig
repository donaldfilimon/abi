//! AI Reasoning Module — Abbey, RAG, Eval, Templates, Explore
//!
//! This module provides advanced AI reasoning capabilities: the Abbey engine
//! (meta-learning, self-reflection, theory of mind), retrieval-augmented
//! generation, evaluation frameworks, prompt templates, codebase exploration,
//! multi-model orchestration, and document understanding.
//!
//! Gated by `-Dfeat-reasoning`.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../../core/config");

// ============================================================================
// Sub-module re-exports (from features/ai/)
// ============================================================================

pub const abbey = @import("../abbey");

pub const rag = if (build_options.feat_reasoning)
    @import("../rag")
else
    @import("../rag/stub");

pub const eval = if (build_options.feat_reasoning)
    @import("../eval")
else
    @import("../eval/stub");

pub const templates = if (build_options.feat_reasoning)
    @import("../templates")
else
    @import("../templates/stub");

pub const explore = if (build_options.feat_explore)
    @import("../explore")
else
    @import("../explore/stub");

pub const orchestration = if (build_options.feat_reasoning)
    @import("../orchestration")
else
    @import("../orchestration/stub");

pub const documents = if (build_options.feat_reasoning)
    @import("../documents")
else
    @import("../documents/stub");

// ============================================================================
// Convenience type re-exports
// ============================================================================

// Abbey
pub const AbbeyEngine = abbey.AbbeyEngine;
pub const Abbey = abbey.Abbey;
pub const AbbeyStats = abbey.Stats;
pub const ReasoningChain = abbey.ReasoningChain;
pub const ReasoningStep = abbey.ReasoningStep;
pub const ConversationContext = abbey.ConversationContext;

// Explore
pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;

// Orchestration
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

// Documents
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

// ============================================================================
// Error
// ============================================================================

pub const Error = error{
    ReasoningDisabled,
    InvalidConfig,
};

// ============================================================================
// Context
// ============================================================================

/// Reasoning module Context. The reasoning submodules (Abbey, RAG, eval, etc.)
/// are largely stateless collections of types and functions, so the Context is
/// lightweight — it primarily serves as the framework integration handle.
pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        _: config_module.AiConfig,
    ) !*Context {
        if (!isEnabled()) return error.ReasoningDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Module-level functions
// ============================================================================

pub fn isEnabled() bool {
    return build_options.feat_reasoning;
}

// ============================================================================
// Tests
// ============================================================================

test "ai_reasoning module loads" {
    try std.testing.expect(@TypeOf(ExploreAgent) != void);
    try std.testing.expect(@TypeOf(Orchestrator) != void);
    try std.testing.expect(@TypeOf(DocumentPipeline) != void);
}

test "ai_reasoning isEnabled reflects build flag" {
    try std.testing.expectEqual(build_options.feat_reasoning, isEnabled());
}

test "ai_reasoning type re-exports distinct types" {
    try std.testing.expect(@TypeOf(ExploreConfig) != void);
    try std.testing.expect(@TypeOf(OrchestrationConfig) != void);
    try std.testing.expect(@TypeOf(PipelineConfig) != void);
    try std.testing.expect(@TypeOf(RoutingStrategy) != void);
    try std.testing.expect(@TypeOf(DocumentFormat) != void);
    try std.testing.expect(@TypeOf(EntityType) != void);
}

test {
    std.testing.refAllDecls(@This());
}
