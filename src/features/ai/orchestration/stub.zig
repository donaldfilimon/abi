//! Orchestration stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const OrchestrationError = types.OrchestrationError;
pub const OrchestrationConfig = types.OrchestrationConfig;
pub const ModelBackend = types.ModelBackend;
pub const Capability = types.Capability;
pub const ModelConfig = types.ModelConfig;
pub const RoutingStrategy = types.RoutingStrategy;
pub const TaskType = types.TaskType;
pub const RouteResult = types.RouteResult;
pub const EnsembleMethod = types.EnsembleMethod;
pub const EnsembleResult = types.EnsembleResult;
pub const FallbackPolicy = types.FallbackPolicy;
pub const HealthStatus = types.HealthStatus;
pub const ModelEntry = types.ModelEntry;
pub const OrchestratorStats = types.OrchestratorStats;

// --- Router ---
pub const router = struct {
    pub const RoutingStrategy_ = RoutingStrategy;
    pub const TaskType_ = TaskType;
    pub const RouteResult_ = RouteResult;
    pub const Router_ = Router;
};

pub const Router = struct {
    pub fn init(_: std.mem.Allocator, _: RoutingStrategy) Router {
        return .{};
    }
    pub fn deinit(_: *Router) void {}
};

// --- Ensemble ---
pub const ensemble = struct {
    pub const EnsembleMethod_ = EnsembleMethod;
    pub const EnsembleResult_ = EnsembleResult;
    pub const Ensemble_ = Ensemble;
};

pub const Ensemble = struct {
    pub fn init(_: std.mem.Allocator, _: EnsembleMethod) Ensemble {
        return .{};
    }
    pub fn deinit(_: *Ensemble) void {}
};

// --- Fallback ---
pub const fallback = struct {
    pub const FallbackManager_ = FallbackManager;
    pub const FallbackPolicy_ = FallbackPolicy;
    pub const HealthStatus_ = HealthStatus;
};

pub const FallbackManager = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) FallbackManager {
        return .{};
    }
    pub fn deinit(_: *FallbackManager) void {}
};

// --- Orchestrator ---
pub const Orchestrator = struct {
    pub fn init(_: std.mem.Allocator, _: OrchestrationConfig) OrchestrationError!Orchestrator {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Orchestrator) void {}
    pub fn registerModel(_: *Orchestrator, _: ModelConfig) OrchestrationError!void {
        return error.FeatureDisabled;
    }
    pub fn unregisterModel(_: *Orchestrator, _: []const u8) OrchestrationError!void {
        return error.FeatureDisabled;
    }
    pub fn getModel(_: *Orchestrator, _: []const u8) ?*ModelEntry {
        return null;
    }
    pub fn setModelEnabled(_: *Orchestrator, _: []const u8, _: bool) OrchestrationError!void {
        return error.FeatureDisabled;
    }
    pub fn setModelHealth(_: *Orchestrator, _: []const u8, _: HealthStatus) OrchestrationError!void {
        return error.FeatureDisabled;
    }
    pub fn route(_: *Orchestrator, _: []const u8, _: ?TaskType) OrchestrationError!RouteResult {
        return error.FeatureDisabled;
    }
    pub fn execute(_: *Orchestrator, _: []const u8, _: ?TaskType, _: std.mem.Allocator) OrchestrationError![]u8 {
        return error.FeatureDisabled;
    }
    pub fn executeEnsemble(_: *Orchestrator, _: []const u8, _: ?TaskType, _: std.mem.Allocator) OrchestrationError!EnsembleResult {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *Orchestrator) OrchestratorStats {
        return .{};
    }
    pub fn listModels(_: *Orchestrator, _: std.mem.Allocator) OrchestrationError![][]const u8 {
        return error.FeatureDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
