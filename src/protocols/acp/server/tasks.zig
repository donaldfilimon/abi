//! ACP Task types and lifecycle management.

const std = @import("std");
const json_utils = @import("json_utils.zig");
const appendEscaped = json_utils.appendEscaped;

/// Task status in the ACP lifecycle
pub const TaskStatus = enum {
    submitted,
    working,
    input_required,
    completed,
    failed,
    canceled,

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
};

/// ACP Task
pub const Task = struct {
    id: []const u8,
    status: TaskStatus,
    messages: std.ArrayListUnmanaged(Message),

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn deinit(self: *Task, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        for (self.messages.items) |msg| {
            allocator.free(msg.role);
            allocator.free(msg.content);
        }
        self.messages.deinit(allocator);
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

test "Task toJson" {
    const allocator = std.testing.allocator;
    var task = Task{
        .id = try allocator.dupe(u8, "task-1"),
        .status = .working,
        .messages = .empty,
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
