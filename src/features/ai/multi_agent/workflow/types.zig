//! Workflow DAG types.

const std = @import("std");
const roles = @import("../roles.zig");

pub const StepStatus = enum {
    pending,
    ready,
    running,
    completed,
    failed,
    skipped,

    pub fn isTerminal(self: StepStatus) bool {
        return self == .completed or self == .failed or self == .skipped;
    }
};

pub const Step = struct {
    id: []const u8,
    description: []const u8,
    depends_on: []const []const u8,
    required_capabilities: []const roles.Capability,
    input_keys: []const []const u8,
    output_key: []const u8,
    prompt_template: []const u8,
    is_critical: bool = true,
    timeout_ms: u64 = 0,
    assigned_profile: []const u8 = "",
};

pub const StepResult = struct {
    step_id: []const u8,
    status: StepStatus,
    output: []const u8,
    error_message: []const u8,
    duration_ns: u64,
    assigned_profile: []const u8,
};

pub const WorkflowStatus = enum {
    created,
    running,
    completed,
    failed,
    cancelled,
};

pub const ValidationResult = struct {
    valid: bool,
    error_message: []const u8,
};

test {
    std.testing.refAllDecls(@This());
}
