//! Workflow DAG definition and validation.

const std = @import("std");
const types = @import("types.zig");
const Step = types.Step;
const StepStatus = types.StepStatus;
const ValidationResult = types.ValidationResult;

pub const WorkflowDef = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    steps: []const Step,

    pub fn validate(self: WorkflowDef, allocator: std.mem.Allocator) !ValidationResult {
        if (self.steps.len == 0) {
            return .{ .valid = false, .error_message = "workflow has no steps" };
        }

        for (self.steps) |step| {
            for (step.depends_on) |dep_id| {
                if (!self.hasStep(dep_id)) {
                    return .{ .valid = false, .error_message = "missing dependency reference" };
                }
            }
        }

        if (try self.hasCycle(allocator)) {
            return .{ .valid = false, .error_message = "workflow contains a cycle" };
        }

        return .{ .valid = true, .error_message = "" };
    }

    pub fn hasStep(self: WorkflowDef, id: []const u8) bool {
        for (self.steps) |step| {
            if (std.mem.eql(u8, step.id, id)) return true;
        }
        return false;
    }

    pub fn getStep(self: WorkflowDef, id: []const u8) ?Step {
        for (self.steps) |step| {
            if (std.mem.eql(u8, step.id, id)) return step;
        }
        return null;
    }

    pub fn computeLayers(self: WorkflowDef, allocator: std.mem.Allocator) ![]const []const []const u8 {
        var step_to_layer = std.StringHashMapUnmanaged(usize).empty;
        defer step_to_layer.deinit(allocator);

        for (self.steps) |step| {
            try step_to_layer.put(allocator, step.id, 0);
        }

        var changed = true;
        while (changed) {
            changed = false;
            for (self.steps) |step| {
                var max_dep_layer: usize = 0;
                for (step.depends_on) |dep_id| {
                    if (step_to_layer.get(dep_id)) |dep_layer| {
                        max_dep_layer = @max(max_dep_layer, dep_layer + 1);
                    }
                }
                const current = step_to_layer.get(step.id) orelse 0;
                if (max_dep_layer > current) {
                    try step_to_layer.put(allocator, step.id, max_dep_layer);
                    changed = true;
                }
            }
        }

        var max_layer: usize = 0;
        var iter = step_to_layer.iterator();
        while (iter.next()) |entry| {
            max_layer = @max(max_layer, entry.value_ptr.*);
        }

        var layers: std.ArrayListUnmanaged([]const []const u8) = .empty;
        errdefer {
            for (layers.items) |layer| allocator.free(layer);
            layers.deinit(allocator);
        }

        for (0..max_layer + 1) |layer_idx| {
            var layer_steps: std.ArrayListUnmanaged([]const u8) = .empty;
            errdefer layer_steps.deinit(allocator);

            for (self.steps) |step| {
                if ((step_to_layer.get(step.id) orelse 0) == layer_idx) {
                    try layer_steps.append(allocator, step.id);
                }
            }

            try layers.append(allocator, try layer_steps.toOwnedSlice(allocator));
        }

        return layers.toOwnedSlice(allocator);
    }

    fn hasCycle(self: WorkflowDef, allocator: std.mem.Allocator) !bool {
        const colors = try allocator.alloc(u8, self.steps.len);
        defer allocator.free(colors);
        @memset(colors, 0);

        for (0..self.steps.len) |i| {
            if (colors[i] == 0) {
                if (self.dfsHasCycle(i, colors)) return true;
            }
        }
        return false;
    }

    fn dfsHasCycle(self: WorkflowDef, idx: usize, colors: []u8) bool {
        colors[idx] = 1;
        const step = self.steps[idx];
        for (step.depends_on) |dep_id| {
            for (self.steps, 0..) |s, j| {
                if (std.mem.eql(u8, s.id, dep_id)) {
                    if (colors[j] == 1) return true;
                    if (colors[j] == 0 and self.dfsHasCycle(j, colors)) return true;
                    break;
                }
            }
        }
        colors[idx] = 2;
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
