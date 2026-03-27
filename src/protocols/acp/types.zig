//! Shared ACP types used by both mod.zig and stub.zig.
//!
//! This file is intentionally free of runtime dependencies (no foundation,
//! no json_utils) so that it can be imported safely regardless of feature
//! flags.

const std = @import("std");

// =============================================================================
// Error sets
// =============================================================================

/// Error returned when a task state transition is not allowed.
pub const TransitionError = error{InvalidTransition};

// =============================================================================
// TaskStatus
// =============================================================================

/// Task status in the ACP lifecycle.
pub const TaskStatus = enum {
    submitted,
    working,
    input_required,
    completed,
    failed,
    canceled,

    /// Returns whether transitioning from `self` to `target` is allowed.
    pub fn canTransitionTo(self: TaskStatus, target: TaskStatus) bool {
        // Any state can transition to canceled.
        if (target == .canceled) return true;

        return switch (self) {
            .submitted => target == .working,
            .working => target == .completed or target == .failed or target == .input_required,
            .input_required => target == .working,
            // Terminal states cannot transition (except to canceled, handled above).
            .completed, .failed, .canceled => false,
        };
    }

    /// Attempt a state transition. Returns the new status or error.InvalidTransition.
    pub fn transition(self: TaskStatus, target: TaskStatus) TransitionError!TaskStatus {
        if (self.canTransitionTo(target)) return target;
        return error.InvalidTransition;
    }

    pub fn toString(self: TaskStatus) []const u8 {
        return switch (self) {
            .submitted => "submitted",
            .working => "working",
            .input_required => "input-required",
            .completed => "completed",
            .failed => "failed",
            .canceled => "canceled",
        };
    }

    /// Parse a status from its wire-format string. Returns null for unknown values.
    pub fn fromString(s: []const u8) ?TaskStatus {
        inline for (std.meta.fields(TaskStatus)) |f| {
            const variant: TaskStatus = @enumFromInt(f.value);
            if (std.mem.eql(u8, s, variant.toString())) return variant;
        }
        return null;
    }
};

// =============================================================================
// Message
// =============================================================================

/// A single message in an ACP task conversation.
pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

// =============================================================================
// Tests
// =============================================================================

test "TaskStatus fromString round-trips toString" {
    const all = std.meta.fields(TaskStatus);
    inline for (all) |f| {
        const variant: TaskStatus = @enumFromInt(f.value);
        try std.testing.expectEqual(variant, TaskStatus.fromString(variant.toString()).?);
    }
    try std.testing.expectEqual(null, TaskStatus.fromString("invalid"));
}

test "TaskStatus toString" {
    try std.testing.expectEqualStrings("submitted", TaskStatus.submitted.toString());
    try std.testing.expectEqualStrings("completed", TaskStatus.completed.toString());
    try std.testing.expectEqualStrings("input-required", TaskStatus.input_required.toString());
}

test "valid transition: submitted -> working" {
    try std.testing.expectEqual(TaskStatus.working, try TaskStatus.submitted.transition(.working));
}

test "valid transition: any state -> canceled" {
    const all_statuses = [_]TaskStatus{ .submitted, .working, .input_required, .completed, .failed, .canceled };
    for (all_statuses) |status| {
        try std.testing.expectEqual(TaskStatus.canceled, try status.transition(.canceled));
    }
}

test "invalid transition: submitted -> completed" {
    try std.testing.expectError(error.InvalidTransition, TaskStatus.submitted.transition(.completed));
}

test {
    std.testing.refAllDecls(@This());
}
