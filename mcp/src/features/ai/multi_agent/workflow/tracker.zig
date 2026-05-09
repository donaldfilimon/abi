//! Workflow execution tracker.

const std = @import("std");
const types = @import("types.zig");
const definition = @import("definition.zig");
const StepStatus = types.StepStatus;
const StepResult = types.StepResult;
const WorkflowStatus = types.WorkflowStatus;
const WorkflowDef = definition.WorkflowDef;

pub const ExecutionTracker = struct {
    allocator: std.mem.Allocator,
    workflow: WorkflowDef,
    status: WorkflowStatus,
    step_statuses: std.StringHashMapUnmanaged(StepStatus),
    step_results: std.StringHashMapUnmanaged(StepResult),

    pub fn init(allocator: std.mem.Allocator, workflow: WorkflowDef) !ExecutionTracker {
        var step_statuses = std.StringHashMapUnmanaged(StepStatus).empty;
        errdefer step_statuses.deinit(allocator);

        for (workflow.steps) |step| {
            const initial: StepStatus = if (step.depends_on.len == 0) .ready else .pending;
            try step_statuses.put(allocator, step.id, initial);
        }

        return .{
            .allocator = allocator,
            .workflow = workflow,
            .status = .created,
            .step_statuses = step_statuses,
            .step_results = .{},
        };
    }

    pub fn deinit(self: *ExecutionTracker) void {
        self.step_statuses.deinit(self.allocator);
        self.step_results.deinit(self.allocator);
    }

    pub fn getStepStatus(self: *const ExecutionTracker, step_id: []const u8) ?StepStatus {
        return self.step_statuses.get(step_id);
    }

    pub fn markRunning(self: *ExecutionTracker, step_id: []const u8) void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .running;
        }
        self.status = .running;
    }

    pub fn markCompleted(self: *ExecutionTracker, step_id: []const u8, result: StepResult) !void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .completed;
        }
        try self.step_results.put(self.allocator, step_id, result);
        self.updateReadiness();
        self.checkOverallStatus();
    }

    pub fn markFailed(self: *ExecutionTracker, step_id: []const u8, result: StepResult) !void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .failed;
        }
        try self.step_results.put(self.allocator, step_id, result);

        if (self.workflow.getStep(step_id)) |step| {
            if (step.is_critical) {
                self.skipDependents(step_id);
            }
        }

        self.updateReadiness();
        self.checkOverallStatus();
    }

    pub fn readySteps(self: *const ExecutionTracker, allocator: std.mem.Allocator) ![]const []const u8 {
        var result: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .ready) {
                try result.append(allocator, entry.key_ptr.*);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn isComplete(self: *const ExecutionTracker) bool {
        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.isTerminal()) return false;
        }
        return true;
    }

    pub fn progress(self: *const ExecutionTracker) Progress {
        var completed: usize = 0;
        var failed: usize = 0;
        var running: usize = 0;
        var total: usize = 0;

        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            total += 1;
            switch (entry.value_ptr.*) {
                .completed => completed += 1,
                .failed => failed += 1,
                .running => running += 1,
                else => {},
            }
        }

        return .{
            .completed = completed,
            .failed = failed,
            .running = running,
            .total = total,
        };
    }

    pub const Progress = struct {
        completed: usize,
        failed: usize,
        running: usize,
        total: usize,
    };

    fn updateReadiness(self: *ExecutionTracker) void {
        for (self.workflow.steps) |step| {
            const current = self.step_statuses.get(step.id) orelse continue;
            if (current != .pending) continue;

            var all_met = true;
            for (step.depends_on) |dep_id| {
                const dep_status = self.step_statuses.get(dep_id) orelse .pending;
                if (dep_status != .completed) {
                    all_met = false;
                    break;
                }
            }

            if (all_met) {
                if (self.step_statuses.getPtr(step.id)) |ptr| {
                    ptr.* = .ready;
                }
            }
        }
    }

    fn skipDependents(self: *ExecutionTracker, failed_step_id: []const u8) void {
        for (self.workflow.steps) |step| {
            for (step.depends_on) |dep_id| {
                if (std.mem.eql(u8, dep_id, failed_step_id)) {
                    if (self.step_statuses.getPtr(step.id)) |ptr| {
                        if (!ptr.*.isTerminal()) {
                            ptr.* = .skipped;
                        }
                    }
                    self.skipDependents(step.id);
                    break;
                }
            }
        }
    }

    fn checkOverallStatus(self: *ExecutionTracker) void {
        if (!self.isComplete()) return;

        var any_failed = false;
        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .failed) {
                any_failed = true;
                break;
            }
        }

        self.status = if (any_failed) .failed else .completed;
    }
};

test {
    std.testing.refAllDecls(@This());
}
