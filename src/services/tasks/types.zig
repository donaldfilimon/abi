//! Task Types and Enums
//!
//! Core type definitions for the task management system.

const std = @import("std");
const time_utils = @import("../shared/utils.zig");

// ============================================================================
// Enums
// ============================================================================

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

/// Sort options for task lists
pub const SortBy = enum {
    created,
    updated,
    priority,
    due_date,
    status,

    pub fn toString(self: SortBy) []const u8 {
        return switch (self) {
            .created => "created",
            .updated => "updated",
            .priority => "priority",
            .due_date => "due_date",
            .status => "status",
        };
    }

    pub fn fromString(s: []const u8) ?SortBy {
        if (std.mem.eql(u8, s, "created")) return .created;
        if (std.mem.eql(u8, s, "updated")) return .updated;
        if (std.mem.eql(u8, s, "priority")) return .priority;
        if (std.mem.eql(u8, s, "due_date") or std.mem.eql(u8, s, "due")) return .due_date;
        if (std.mem.eql(u8, s, "status")) return .status;
        return null;
    }
};

// ============================================================================
// Core Structs
// ============================================================================

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
            return time_utils.unixSeconds() > due;
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
    sort_by: SortBy = .created,
    sort_descending: bool = true,
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

// ============================================================================
// Manager Types
// ============================================================================

pub const ManagerError = error{
    TaskNotFound,
    InvalidOperation,
    PersistenceFailed,
    ParseError,
} || std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.Dir.ReadFileAllocError || std.Io.File.Writer.Error;

pub const ManagerConfig = struct {
    /// Optional storage path override. Empty uses the platform primary path.
    storage_path: []const u8 = "",
    auto_save: bool = true,
};

pub const AddOptions = struct {
    description: ?[]const u8 = null,
    priority: Priority = .normal,
    category: Category = .personal,
    tags: []const []const u8 = &.{},
    due_date: ?i64 = null,
    parent_id: ?u64 = null,
};

// ============================================================================
// Tests
// ============================================================================

test "Priority string conversion" {
    try std.testing.expectEqualStrings("high", Priority.high.toString());
    try std.testing.expectEqual(Priority.high, Priority.fromString("high").?);
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
}

test "SortBy string conversion" {
    try std.testing.expectEqualStrings("priority", SortBy.priority.toString());
    try std.testing.expectEqual(SortBy.due_date, SortBy.fromString("due_date").?);
    try std.testing.expectEqual(SortBy.due_date, SortBy.fromString("due").?);
}

test {
    std.testing.refAllDecls(@This());
}
