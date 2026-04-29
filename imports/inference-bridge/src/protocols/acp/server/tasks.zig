//! ACP Task types and lifecycle management.
//!
//! Implements a task state machine enforcing valid transitions:
//!   submitted -> working
//!   working -> completed | failed | input_required
//!   input_required -> working
//!   any -> canceled

const std = @import("std");
const json_utils = @import("json_utils.zig");
const appendEscaped = json_utils.appendEscaped;
const foundation_time = @import("../../../foundation/time.zig");
const types = @import("../types.zig");

// Re-export shared types so existing importers are unaffected.
pub const TransitionError = types.TransitionError;
pub const TaskStatus = types.TaskStatus;

/// Record of a single state transition with wall-clock timestamp.
pub const StateChange = struct {
    from: TaskStatus,
    to: TaskStatus,
    timestamp_ms: i64,
};

/// ACP Task
pub const Task = struct {
    id: []const u8,
    status: TaskStatus,
    messages: std.ArrayListUnmanaged(Message),
    /// Wall-clock timestamp (ms) when the task was created.
    created_at_ms: i64,
    /// Wall-clock timestamp (ms) of the last status change.
    updated_at_ms: i64,
    /// History of state transitions.
    history: std.ArrayListUnmanaged(StateChange),

    /// Re-export shared Message type so existing `Task.Message` access works.
    pub const Message = types.Message;

    /// Transition task to a new status, enforcing the state machine.
    /// Records the transition in history and updates the timestamp.
    pub fn transitionTo(self: *Task, allocator: std.mem.Allocator, target: TaskStatus) TransitionError!void {
        const new_status = try self.status.transition(target);
        const now = foundation_time.unixMs();
        self.history.append(allocator, .{
            .from = self.status,
            .to = new_status,
            .timestamp_ms = now,
        }) catch {
            // OOM during history append — still apply the transition
            // so the task isn't stuck in a stale state.
        };
        self.status = new_status;
        self.updated_at_ms = now;
    }

    pub fn deinit(self: *Task, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        for (self.messages.items) |msg| {
            allocator.free(msg.role);
            allocator.free(msg.content);
        }
        self.messages.deinit(allocator);
        self.history.deinit(allocator);
    }

    /// Serialize task to JSON
    pub fn toJson(self: *const Task, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"id\":\"");
        try appendEscaped(allocator, &buf, self.id);
        try buf.appendSlice(allocator, "\",\"status\":\"");
        try buf.appendSlice(allocator, self.status.toString());
        try buf.appendSlice(allocator, "\",\"messages\":[");

        for (self.messages.items, 0..) |msg, i| {
            if (i > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"role\":\"");
            try appendEscaped(allocator, &buf, msg.role);
            try buf.appendSlice(allocator, "\",\"parts\":[{\"type\":\"text\",\"text\":\"");
            try appendEscaped(allocator, &buf, msg.content);
            try buf.appendSlice(allocator, "\"}]}");
        }

        try buf.appendSlice(allocator, "]}");
        return buf.toOwnedSlice(allocator);
    }
};

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "Task toJson" {
    const allocator = std.testing.allocator;
    var task = Task{
        .id = try allocator.dupe(u8, "task-1"),
        .status = .working,
        .messages = .empty,
        .created_at_ms = 0,
        .updated_at_ms = 0,
        .history = .empty,
    };
    defer task.deinit(allocator);

    const role = try allocator.dupe(u8, "user");
    const content = try allocator.dupe(u8, "test message");
    try task.messages.append(allocator, .{ .role = role, .content = content });

    const json = try task.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "task-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "working") != null);
}

test "TaskStatus toString" {
    try std.testing.expectEqualStrings("submitted", TaskStatus.submitted.toString());
    try std.testing.expectEqualStrings("completed", TaskStatus.completed.toString());
    try std.testing.expectEqualStrings("input-required", TaskStatus.input_required.toString());
}

test "valid transition: submitted -> working" {
    try std.testing.expectEqual(TaskStatus.working, try TaskStatus.submitted.transition(.working));
}

test "valid transition: working -> completed" {
    try std.testing.expectEqual(TaskStatus.completed, try TaskStatus.working.transition(.completed));
}

test "valid transition: working -> failed" {
    try std.testing.expectEqual(TaskStatus.failed, try TaskStatus.working.transition(.failed));
}

test "valid transition: working -> input_required" {
    try std.testing.expectEqual(TaskStatus.input_required, try TaskStatus.working.transition(.input_required));
}

test "valid transition: input_required -> working" {
    try std.testing.expectEqual(TaskStatus.working, try TaskStatus.input_required.transition(.working));
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

test "invalid transition: completed -> working" {
    try std.testing.expectError(error.InvalidTransition, TaskStatus.completed.transition(.working));
}

test "invalid transition: failed -> working" {
    try std.testing.expectError(error.InvalidTransition, TaskStatus.failed.transition(.working));
}

test "invalid transition: submitted -> failed" {
    try std.testing.expectError(error.InvalidTransition, TaskStatus.submitted.transition(.failed));
}

test "Task.transitionTo records history" {
    const allocator = std.testing.allocator;
    var task = Task{
        .id = try allocator.dupe(u8, "task-hist"),
        .status = .submitted,
        .messages = .empty,
        .created_at_ms = 0,
        .updated_at_ms = 0,
        .history = .empty,
    };
    defer task.deinit(allocator);

    try task.transitionTo(allocator, .working);
    try std.testing.expectEqual(TaskStatus.working, task.status);
    try std.testing.expectEqual(@as(usize, 1), task.history.items.len);
    try std.testing.expectEqual(TaskStatus.submitted, task.history.items[0].from);
    try std.testing.expectEqual(TaskStatus.working, task.history.items[0].to);
    try std.testing.expect(task.updated_at_ms != 0 or task.history.items[0].timestamp_ms != 0);

    try task.transitionTo(allocator, .completed);
    try std.testing.expectEqual(TaskStatus.completed, task.status);
    try std.testing.expectEqual(@as(usize, 2), task.history.items.len);
}

test "Task.transitionTo rejects invalid transition" {
    const allocator = std.testing.allocator;
    var task = Task{
        .id = try allocator.dupe(u8, "task-inv"),
        .status = .submitted,
        .messages = .empty,
        .created_at_ms = 0,
        .updated_at_ms = 0,
        .history = .empty,
    };
    defer task.deinit(allocator);

    try std.testing.expectError(error.InvalidTransition, task.transitionTo(allocator, .completed));
    // Status should remain unchanged after rejected transition
    try std.testing.expectEqual(TaskStatus.submitted, task.status);
    try std.testing.expectEqual(@as(usize, 0), task.history.items.len);
}

test {
    std.testing.refAllDecls(@This());
}
