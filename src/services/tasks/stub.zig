//! Task Management Stub Module
//!
//! Provides API compatibility with mod.zig while returning TasksDisabled for all operations.
//! Types are kept minimal - only essential ones needed for compile-time checking.

const std = @import("std");

// ============================================================================
// Error Types
// ============================================================================

pub const ManagerError = error{
    TasksDisabled,
    TaskNotFound,
    InvalidOperation,
    PersistenceFailed,
    ParseError,
    OutOfMemory,
};

// ============================================================================
// Enums (minimal for type compatibility)
// ============================================================================

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
// Core Types (minimal definitions for type compatibility)
// ============================================================================

pub const Task = struct {
    id: u64 = 0,
    title: []const u8 = "",
    description: ?[]const u8 = null,
    status: Status = .pending,
    priority: Priority = .normal,
    category: Category = .personal,
    tags: []const []const u8 = &.{},
    created_at: i64 = 0,
    updated_at: i64 = 0,
    due_date: ?i64 = null,
    completed_at: ?i64 = null,
    blocked_by: ?u64 = null,
    parent_id: ?u64 = null,

    pub fn isActionable(self: *const Task) bool {
        return self.status == .pending or self.status == .in_progress;
    }

    pub fn isOverdue(_: *const Task) bool {
        return false;
    }
};

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

pub const Stats = struct {
    total: usize = 0,
    pending: usize = 0,
    in_progress: usize = 0,
    completed: usize = 0,
    cancelled: usize = 0,
    blocked: usize = 0,
    overdue: usize = 0,
};

pub const ManagerConfig = struct {
    storage_path: []const u8 = ".abi/tasks.json",
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

pub const RoadmapItem = struct {
    id: []const u8 = "",
    title: []const u8 = "",
    description: []const u8 = "",
    priority: Priority = .normal,
    category: Category = .roadmap,
};

// ============================================================================
// Sub-module Stubs (namespace compatibility)
// ============================================================================

pub const types = struct {
    pub const Task = @import("stub.zig").Task;
    pub const Priority = @import("stub.zig").Priority;
    pub const Status = @import("stub.zig").Status;
    pub const Category = @import("stub.zig").Category;
    pub const Filter = @import("stub.zig").Filter;
    pub const SortBy = @import("stub.zig").SortBy;
    pub const Stats = @import("stub.zig").Stats;
    pub const ManagerConfig = @import("stub.zig").ManagerConfig;
    pub const AddOptions = @import("stub.zig").AddOptions;
    pub const ManagerError = @import("stub.zig").ManagerError;
};

pub const persistence = struct {
    pub fn save(_: std.mem.Allocator, _: []const u8, _: anytype, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn load(_: std.mem.Allocator, _: []const u8, _: anytype, _: *u64, _: anytype) ManagerError!void {
        return error.TasksDisabled;
    }
};

pub const querying = struct {
    pub fn list(_: anytype, _: std.mem.Allocator, _: Filter) ManagerError![]Task {
        return error.TasksDisabled;
    }
    pub fn getStats(_: anytype) Stats {
        return .{};
    }
};

pub const lifecycle = struct {
    pub fn add(_: std.mem.Allocator, _: anytype, _: anytype, _: *u64, _: []const u8, _: AddOptions) ManagerError!u64 {
        return error.TasksDisabled;
    }
    pub fn get(_: anytype, _: u64) ?Task {
        return null;
    }
    pub fn setStatus(_: anytype, _: u64, _: Status) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn delete(_: anytype, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setDueDate(_: anytype, _: u64, _: ?i64) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setBlockedBy(_: anytype, _: u64, _: ?u64) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setPriority(_: anytype, _: u64, _: Priority) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setCategory(_: anytype, _: u64, _: Category) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setTitle(_: std.mem.Allocator, _: anytype, _: anytype, _: u64, _: []const u8) ManagerError!void {
        return error.TasksDisabled;
    }
    pub fn setDescription(_: std.mem.Allocator, _: anytype, _: anytype, _: u64, _: ?[]const u8) ManagerError!void {
        return error.TasksDisabled;
    }
};

pub const roadmap = struct {
    pub const RoadmapItem = @import("stub.zig").RoadmapItem;

    pub fn importAll(_: std.mem.Allocator, _: anytype, _: *u64, _: anytype) !usize {
        return error.TasksDisabled;
    }
};

// ============================================================================
// Manager (main interface stub)
// ============================================================================

pub const Manager = struct {
    allocator: std.mem.Allocator,
    config: ManagerConfig,

    pub fn init(allocator: std.mem.Allocator, config: ManagerConfig) ManagerError!Manager {
        _ = allocator;
        _ = config;
        return error.TasksDisabled;
    }

    pub fn deinit(_: *Manager) void {}

    // Lifecycle Operations
    pub fn add(_: *Manager, _: []const u8, _: AddOptions) ManagerError!u64 {
        return error.TasksDisabled;
    }

    pub fn get(_: *const Manager, _: u64) ?Task {
        return null;
    }

    pub fn setStatus(_: *Manager, _: u64, _: Status) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn complete(_: *Manager, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn start(_: *Manager, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn cancel(_: *Manager, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn delete(_: *Manager, _: u64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setDueDate(_: *Manager, _: u64, _: ?i64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setBlockedBy(_: *Manager, _: u64, _: ?u64) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setPriority(_: *Manager, _: u64, _: Priority) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setCategory(_: *Manager, _: u64, _: Category) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setTitle(_: *Manager, _: u64, _: []const u8) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn setDescription(_: *Manager, _: u64, _: ?[]const u8) ManagerError!void {
        return error.TasksDisabled;
    }

    // Querying Operations
    pub fn list(_: *const Manager, _: std.mem.Allocator, _: Filter) ManagerError![]Task {
        return error.TasksDisabled;
    }

    pub fn getStats(_: *const Manager) Stats {
        return .{};
    }

    // Persistence Operations
    pub fn save(_: *Manager) ManagerError!void {
        return error.TasksDisabled;
    }

    pub fn load(_: *Manager) ManagerError!void {
        return error.TasksDisabled;
    }

    // Roadmap Integration
    pub fn importRoadmap(_: *Manager) !usize {
        return error.TasksDisabled;
    }
};

// ============================================================================
// Module Lifecycle
// ============================================================================

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) ManagerError!void {
    return error.TasksDisabled;
}

pub fn deinit() void {}
