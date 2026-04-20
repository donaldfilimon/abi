//! Multi-model orchestration facade.

const std = @import("std");
const build_options = @import("build_options");
const orchestrator = @import("orchestrator/mod.zig");

pub const types = @import("types.zig");
pub const router = @import("router.zig");
pub const ensemble = @import("ensemble.zig");
pub const fallback = @import("fallback.zig");

pub const Router = router.Router;
pub const RoutingStrategy = types.RoutingStrategy;
pub const TaskType = types.TaskType;
pub const RouteResult = types.RouteResult;

pub const Ensemble = ensemble.Ensemble;
pub const EnsembleMethod = types.EnsembleMethod;
pub const ModelResponse = types.ModelResponse;
pub const AggregationMetadata = types.AggregationMetadata;
pub const EnsembleResult = types.EnsembleResult;

pub const FallbackManager = fallback.FallbackManager;
pub const FallbackPolicy = types.FallbackPolicy;
pub const HealthStatus = types.HealthStatus;

pub const OrchestrationError = types.OrchestrationError;
pub const OrchestrationConfig = types.OrchestrationConfig;
pub const ModelBackend = types.ModelBackend;
pub const Capability = types.Capability;
pub const ModelConfig = types.ModelConfig;
pub const ModelEntry = types.ModelEntry;
pub const OrchestratorStats = types.OrchestratorStats;

pub const Orchestrator = orchestrator.Orchestrator;

pub fn isEnabled() bool {
    return build_options.feat_ai;
}

test {
    _ = router;
    _ = ensemble;
    _ = fallback;
    std.testing.refAllDecls(@This());
}
