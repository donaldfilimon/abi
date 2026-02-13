//! AI Reasoning Stub Module
//!
//! Provides API-compatible no-op implementations when AI reasoning is disabled.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

pub const Error = error{
    ReasoningDisabled,
    InvalidConfig,
};

// Sub-module stubs
pub const abbey = @import("../ai/abbey/stub.zig");
pub const rag = @import("../ai/rag/stub.zig");
pub const eval = @import("../ai/eval/stub.zig");
pub const templates = @import("../ai/templates/stub.zig");
pub const explore = @import("../ai/explore/stub.zig");
pub const orchestration = @import("../ai/orchestration/stub.zig");
pub const documents = @import("../ai/documents/stub.zig");

// Re-exports
pub const AbbeyEngine = abbey.AbbeyEngine;
pub const Abbey = abbey.Abbey;
pub const AbbeyStats = abbey.Stats;
pub const ReasoningChain = abbey.ReasoningChain;
pub const ReasoningStep = abbey.ReasoningStep;
pub const ConversationContext = abbey.ConversationContext;
pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;
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

pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        _: config_module.AiConfig,
    ) !*Context {
        _ = allocator;
        return error.ReasoningDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}
