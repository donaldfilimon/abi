//! Task Management Types
//!
//! Core types for the unified task system supporting personal tasks,
//! distributed compute tasks, and project roadmap items.

const std = @import("std");

/// Task priority levels
pub const Priority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    critical = 3,

    pub fn toString(self: Priority) []const u8 {
        return switch (self) {
            .low => "low",
            .normal => "normal",
            .high => "high",
            .critical => "critical",
        };
    }

    pub fn fromString(s: []const u8) ?Priority {
        if (std.mem.eql(u8, s, "low")) return .low;
        if (std.mem.eql(u8, s, "normal")) return .normal;
        if (std.mem.eql(u8, s, "high")) return .high;
        if (std.mem.eql(u8, s, "critical")) return .critical;
        return null;
    }
};

/// Task status
pub const Status = enum(u8) {
    pending = 0,
    in_progress = 1,
    completed = 2,
    cancelled = 3,
    blocked = 4,

    pub fn toString(self: Status) []const u8 {
        return switch (self) {
            .pending => "pending",
            .in_progress => "in_progress",
            .completed => "completed",
            .cancelled => "cancelled",
            .blocked => "blocked",
        };
    }

    pub fn fromString(s: []const u8) ?Status {
        if (std.mem.eql(u8, s, "pending")) return .pending;
        if (std.mem.eql(u8, s, "in_progress")) return .in_progress;
        if (std.mem.eql(u8, s, "completed")) return .completed;
        if (std.mem.eql(u8, s, "cancelled")) return .cancelled;
        if (std.mem.eql(u8, s, "blocked")) return .blocked;
        return null;
    }
};

/// Task category for organization
pub const Category = enum(u8) {
    personal = 0,
    roadmap = 1,
    compute = 2,
    bug = 3,
    feature = 4,

    pub fn toString(self: Category) []const u8 {
        return switch (self) {
            .personal => "personal",
            .roadmap => "roadmap",
            .compute => "compute",
            .bug => "bug",
            .feature => "feature",
        };
    }

    pub fn fromString(s: []const u8) ?Category {
        if (std.mem.eql(u8, s, "personal")) return .personal;
        if (std.mem.eql(u8, s, "roadmap")) return .roadmap;
        if (std.mem.eql(u8, s, "compute")) return .compute;
        if (std.mem.eql(u8, s, "bug")) return .bug;
        if (std.mem.eql(u8, s, "feature")) return .feature;
        return null;
    }
};

/// Core task structure
pub const Task = struct {
    id: u64,
    title: []const u8,
    description: ?[]const u8 = null,
    status: Status = .pending,
    priority: Priority = .normal,
    category: Category = .personal,
    tags: []const []const u8 = &.{},
    created_at: i64,
    updated_at: i64,
    due_date: ?i64 = null,
    completed_at: ?i64 = null,
    blocked_by: ?u64 = null,
    parent_id: ?u64 = null,

    /// Check if task is actionable (not blocked or completed)
    pub fn isActionable(self: *const Task) bool {
        return self.status == .pending or self.status == .in_progress;
    }

    /// Check if task is overdue
    pub fn isOverdue(self: *const Task) bool {
        if (self.due_date) |due| {
            if (self.status == .completed or self.status == .cancelled) return false;
            return std.time.timestamp() > due;
        }
        return false;
    }
};

/// Filter criteria for querying tasks
pub const Filter = struct {
    status: ?Status = null,
    priority: ?Priority = null,
    category: ?Category = null,
    tag: ?[]const u8 = null,
    overdue_only: bool = false,
    parent_id: ?u64 = null,
};

/// Sort options for task lists
pub const SortBy = enum {
    created,
    updated,
    priority,
    due_date,
    status,
};

/// Task statistics
pub const Stats = struct {
    total: usize = 0,
    pending: usize = 0,
    in_progress: usize = 0,
    completed: usize = 0,
    cancelled: usize = 0,
    blocked: usize = 0,
    overdue: usize = 0,
};

test "Priority string conversion" {
    try std.testing.expectEqualStrings("high", Priority.high.toString());
    try std.testing.expectEqual(Priority.high, Priority.fromString("high").?);
    try std.testing.expectEqual(@as(?Priority, null), Priority.fromString("invalid"));
}

test "Status string conversion" {
    try std.testing.expectEqualStrings("pending", Status.pending.toString());
    try std.testing.expectEqual(Status.completed, Status.fromString("completed").?);
}

test "Task actionable check" {
    const task = Task{
        .id = 1,
        .title = "Test",
        .created_at = 0,
        .updated_at = 0,
        .status = .pending,
    };
    try std.testing.expect(task.isActionable());

    const completed = Task{
        .id = 2,
        .title = "Done",
        .created_at = 0,
        .updated_at = 0,
        .status = .completed,
    };
    try std.testing.expect(!completed.isActionable());
}
