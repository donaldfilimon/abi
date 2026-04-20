const std = @import("std");
const build_options = @import("build_options");
const sync = @import("../../../../foundation/mod.zig").sync;
const ensemble = @import("../ensemble.zig");
const fallback = @import("../fallback.zig");
const router = @import("../router.zig");
const types = @import("../types.zig");
const registry = @import("registry.zig");
const selection = @import("selection.zig");
const execution = @import("execution.zig");

const Mutex = sync.Mutex;

pub const Orchestrator = struct {
    allocator: std.mem.Allocator,
    config: types.OrchestrationConfig,
    models: std.StringHashMapUnmanaged(types.ModelEntry),
    router_instance: router.Router,
    ensemble_instance: ?ensemble.Ensemble,
    fallback_manager: fallback.FallbackManager,
    round_robin_index: usize = 0,
    mutex: Mutex = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        config: types.OrchestrationConfig,
    ) types.OrchestrationError!Orchestrator {
        if (!build_options.feat_ai) return types.OrchestrationError.OrchestrationDisabled;

        return .{
            .allocator = allocator,
            .config = config,
            .models = .empty,
            .router_instance = router.Router.init(allocator, config.strategy),
            .ensemble_instance = if (config.enable_ensemble)
                ensemble.Ensemble.init(allocator, config.ensemble_method)
            else
                null,
            .fallback_manager = fallback.FallbackManager.init(allocator, .{
                .max_retries = config.max_retries,
                .health_check_interval_ms = config.health_check_interval_ms,
            }),
        };
    }

    pub fn deinit(self: *Orchestrator) void {
        registry.deinit(self);
    }

    pub fn registerModel(
        self: *Orchestrator,
        config: types.ModelConfig,
    ) types.OrchestrationError!void {
        return registry.registerModel(self, config);
    }

    pub fn unregisterModel(
        self: *Orchestrator,
        model_id: []const u8,
    ) types.OrchestrationError!void {
        return registry.unregisterModel(self, model_id);
    }

    pub fn getModel(self: *Orchestrator, model_id: []const u8) ?*types.ModelEntry {
        return registry.getModel(self, model_id);
    }

    pub fn setModelEnabled(
        self: *Orchestrator,
        model_id: []const u8,
        enabled: bool,
    ) types.OrchestrationError!void {
        return registry.setModelEnabled(self, model_id, enabled);
    }

    pub fn setModelHealth(
        self: *Orchestrator,
        model_id: []const u8,
        status: types.HealthStatus,
    ) types.OrchestrationError!void {
        return registry.setModelHealth(self, model_id, status);
    }

    pub fn route(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?types.TaskType,
    ) types.OrchestrationError!types.RouteResult {
        return selection.route(self, prompt, task_type);
    }

    pub fn execute(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?types.TaskType,
        response_allocator: std.mem.Allocator,
    ) types.OrchestrationError![]u8 {
        return execution.execute(self, prompt, task_type, response_allocator);
    }

    pub fn executeEnsemble(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?types.TaskType,
        response_allocator: std.mem.Allocator,
    ) types.OrchestrationError!types.EnsembleResult {
        return execution.executeEnsemble(self, prompt, task_type, response_allocator);
    }

    pub fn getStats(self: *Orchestrator) types.OrchestratorStats {
        return registry.getStats(self);
    }

    pub fn listModels(self: *Orchestrator, allocator: std.mem.Allocator) ![][]const u8 {
        return registry.listModels(self, allocator);
    }
};

test {
    std.testing.refAllDecls(@This());
}
