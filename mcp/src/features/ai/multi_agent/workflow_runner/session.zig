const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const messaging = @import("../messaging.zig");
const types = @import("../types.zig");
const time = @import("../../../../foundation/mod.zig").time;

pub const RunSession = struct {
    allocator: std.mem.Allocator,
    layers: []const []const []const u8,
    tracker: workflow_mod.ExecutionTracker,
    task_id: u64,
    overall_timer: ?time.Timer,
    step_results: std.StringHashMapUnmanaged(types.StepResult) = .empty,
    stats: types.WorkflowStats,
    result_taken: bool = false,

    pub fn bootstrap(runner: anytype, def: *const workflow_mod.WorkflowDef) !RunSession {
        const validation = try def.validate(runner.allocator);
        if (!validation.valid) {
            return types.RunError.InvalidWorkflow;
        }

        if (runner.agent_map.count() == 0) {
            return types.RunError.NoAgents;
        }

        const layers = def.computeLayers(runner.allocator) catch return types.RunError.OutOfMemory;
        errdefer {
            for (layers) |layer| runner.allocator.free(layer);
            runner.allocator.free(layers);
        }

        var tracker = workflow_mod.ExecutionTracker.init(runner.allocator, def.*) catch
            return types.RunError.OutOfMemory;
        errdefer tracker.deinit();

        runner.profile_registry.loadPresets() catch |err| {
            std.log.warn("Failed to load profile presets: {t}", .{err});
        };

        const task_id = messaging.taskId(def.id);
        runner.event_bus.taskStarted(task_id);

        return .{
            .allocator = runner.allocator,
            .layers = layers,
            .tracker = tracker,
            .task_id = task_id,
            .overall_timer = time.Timer.start() catch null,
            .stats = .{
                .total_steps = @intCast(def.steps.len),
            },
        };
    }

    pub fn deinit(self: *RunSession) void {
        if (!self.result_taken) {
            var iter = self.step_results.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.output) |output| {
                    self.allocator.free(output);
                }
            }
            self.step_results.deinit(self.allocator);
        }

        self.tracker.deinit();
        for (self.layers) |layer| self.allocator.free(layer);
        self.allocator.free(self.layers);
    }

    pub fn fail(self: *RunSession, runner: anytype, detail: []const u8) types.WorkflowResult {
        self.stats.total_duration_ms = elapsedMs(self.overall_timer);
        runner.event_bus.taskFailed(self.task_id, detail);
        return self.intoResult(false, null);
    }

    pub fn intoResult(
        self: *RunSession,
        success: bool,
        final_output: ?[]const u8,
    ) types.WorkflowResult {
        const step_results = self.step_results;
        self.step_results = .empty;
        self.result_taken = true;

        return .{
            .success = success,
            .step_results = step_results,
            .final_output = final_output,
            .stats = self.stats,
            .allocator = self.allocator,
        };
    }
};

fn elapsedMs(timer: ?time.Timer) u64 {
    if (timer) |resolved| {
        var t = resolved;
        return t.read() / std.time.ns_per_ms;
    }
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
