//! Task Management Module
//!
//! Provides unified task tracking for personal tasks, project roadmap
//! items, and distributed compute jobs.
//!
//! ## Usage
//!
//! ```zig
//! const tasks = @import("tasks/mod.zig");
//!
//! var manager = try tasks.Manager.init(allocator, .{});
//! defer manager.deinit();
//!
//! const id = try manager.add("Fix bug", .{ .priority = .high });
//! try manager.complete(id);
//! ```

const std = @import("std");
const app_paths = @import("../shared/app_paths.zig");

// Re-export types
pub const types = @import("types.zig");
pub const persistence = @import("persistence.zig");
pub const querying = @import("querying.zig");
pub const lifecycle = @import("lifecycle.zig");
pub const roadmap = @import("roadmap.zig");

// Type re-exports for convenience
pub const Task = types.Task;
pub const Priority = types.Priority;
pub const Status = types.Status;
pub const Category = types.Category;
pub const Filter = types.Filter;
pub const SortBy = types.SortBy;
pub const Stats = types.Stats;
pub const ManagerConfig = types.ManagerConfig;
pub const AddOptions = types.AddOptions;
pub const ManagerError = types.ManagerError;
pub const RoadmapItem = roadmap.RoadmapItem;

/// Task Manager - main interface for task operations
pub const Manager = struct {
    allocator: std.mem.Allocator,
    config: ManagerConfig,
    owns_resolved_storage_paths: bool,
    tasks: std.AutoHashMapUnmanaged(u64, Task),
    next_id: u64,
    dirty: bool,

    // Owned string storage
    strings: std.ArrayListUnmanaged([]u8),

    pub fn init(allocator: std.mem.Allocator, config: ManagerConfig) ManagerError!Manager {
        var resolved_config = config;
        var owns_resolved_storage_paths = false;
        const default_config: ManagerConfig = .{};

        if (config.storage_path.len == 0 or std.mem.eql(u8, config.storage_path, default_config.storage_path)) {
            const resolved_path = app_paths.resolvePath(allocator, "tasks.json") catch |err| switch (err) {
                error.NoHomeDirectory => return error.PersistenceFailed,
                error.OutOfMemory => return error.OutOfMemory,
            };
            resolved_config.storage_path = resolved_path;
            owns_resolved_storage_paths = true;
        }

        errdefer if (owns_resolved_storage_paths) {
            allocator.free(resolved_config.storage_path);
        };

        var self = Manager{
            .allocator = allocator,
            .config = resolved_config,
            .owns_resolved_storage_paths = owns_resolved_storage_paths,
            .tasks = .{},
            .next_id = 1,
            .dirty = false,
            .strings = .{},
        };

        // Try to load existing tasks
        self.load() catch |err| switch (err) {
            error.FileNotFound => {}, // No existing file is OK
            else => return err,
        };

        return self;
    }

    pub fn deinit(self: *Manager) void {
        if (self.dirty and self.config.auto_save) {
            self.save() catch |err| {
                std.log.debug("Failed to auto-save tasks during deinit: {t}", .{err});
            };
        }

        // Free all owned strings
        for (self.strings.items) |s| {
            self.allocator.free(s);
        }
        self.strings.deinit(self.allocator);

        if (self.owns_resolved_storage_paths) {
            self.allocator.free(self.config.storage_path);
        }

        self.tasks.deinit(self.allocator);
    }

    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    /// Add a new task
    pub fn add(self: *Manager, title: []const u8, options: AddOptions) ManagerError!u64 {
        const id = try lifecycle.add(
            self.allocator,
            &self.tasks,
            &self.strings,
            &self.next_id,
            title,
            options,
        );
        self.dirty = true;
        if (self.config.auto_save) try self.save();
        return id;
    }

    /// Get a task by ID
    pub fn get(self: *const Manager, id: u64) ?Task {
        return lifecycle.get(&self.tasks, id);
    }

    /// Update task status
    pub fn setStatus(self: *Manager, id: u64, status: Status) ManagerError!void {
        try lifecycle.setStatus(&self.tasks, id, status);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Mark task as completed
    pub fn complete(self: *Manager, id: u64) ManagerError!void {
        return self.setStatus(id, .completed);
    }

    /// Mark task as in progress
    pub fn start(self: *Manager, id: u64) ManagerError!void {
        return self.setStatus(id, .in_progress);
    }

    /// Cancel a task
    pub fn cancel(self: *Manager, id: u64) ManagerError!void {
        return self.setStatus(id, .cancelled);
    }

    /// Delete a task
    pub fn delete(self: *Manager, id: u64) ManagerError!void {
        try lifecycle.delete(&self.tasks, id);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Set due date for a task
    pub fn setDueDate(self: *Manager, id: u64, due_date: ?i64) ManagerError!void {
        try lifecycle.setDueDate(&self.tasks, id, due_date);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Set task as blocked by another task
    pub fn setBlockedBy(self: *Manager, id: u64, blocker_id: ?u64) ManagerError!void {
        try lifecycle.setBlockedBy(&self.tasks, id, blocker_id);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task priority
    pub fn setPriority(self: *Manager, id: u64, priority: Priority) ManagerError!void {
        try lifecycle.setPriority(&self.tasks, id, priority);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task category
    pub fn setCategory(self: *Manager, id: u64, category: Category) ManagerError!void {
        try lifecycle.setCategory(&self.tasks, id, category);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task title
    pub fn setTitle(self: *Manager, id: u64, title: []const u8) ManagerError!void {
        try lifecycle.setTitle(self.allocator, &self.tasks, &self.strings, id, title);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task description
    pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void {
        try lifecycle.setDescription(self.allocator, &self.tasks, &self.strings, id, description);
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    // ========================================================================
    // Querying Operations
    // ========================================================================

    /// List tasks with optional filter and sorting
    pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task {
        return querying.list(&self.tasks, allocator, filter);
    }

    /// Get statistics
    pub fn getStats(self: *const Manager) Stats {
        return querying.getStats(&self.tasks);
    }

    // ========================================================================
    // Persistence Operations
    // ========================================================================

    /// Save tasks to file
    pub fn save(self: *Manager) ManagerError!void {
        try persistence.save(self.allocator, self.config.storage_path, &self.tasks, self.next_id);
        self.dirty = false;
    }

    /// Load tasks from file
    pub fn load(self: *Manager) ManagerError!void {
        try persistence.load(
            self.allocator,
            self.config.storage_path,
            &self.tasks,
            &self.next_id,
            &self.strings,
        );
        self.dirty = false;
    }

    // ========================================================================
    // Roadmap Integration
    // ========================================================================

    /// Import all roadmap items as tasks
    pub fn importRoadmap(self: *Manager) !usize {
        const count = try roadmap.importAll(
            self.allocator,
            &self.tasks,
            &self.next_id,
            &self.strings,
        );
        self.dirty = true;
        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Manager basic operations" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_tasks.json",
        .auto_save = false,
    });
    defer manager.deinit();

    const id = try manager.add("Test task", .{ .priority = .high });
    try std.testing.expect(id == 1);

    const task = manager.get(id).?;
    try std.testing.expectEqualStrings("Test task", task.title);
    try std.testing.expectEqual(Priority.high, task.priority);

    try manager.complete(id);
    const updated = manager.get(id).?;
    try std.testing.expectEqual(Status.completed, updated.status);
}

test "Manager list sorting by priority" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_tasks_sort.json",
        .auto_save = false,
    });
    defer manager.deinit();

    // Add tasks with different priorities
    _ = try manager.add("Low priority task", .{ .priority = .low });
    _ = try manager.add("High priority task", .{ .priority = .high });
    _ = try manager.add("Normal priority task", .{ .priority = .normal });

    // List sorted by priority descending (high first)
    const sorted_desc = try manager.list(std.testing.allocator, .{
        .sort_by = .priority,
        .sort_descending = true,
    });
    defer std.testing.allocator.free(sorted_desc);

    try std.testing.expectEqual(@as(usize, 3), sorted_desc.len);
    try std.testing.expectEqual(Priority.high, sorted_desc[0].priority);
    try std.testing.expectEqual(Priority.normal, sorted_desc[1].priority);
    try std.testing.expectEqual(Priority.low, sorted_desc[2].priority);

    // List sorted by priority ascending (low first)
    const sorted_asc = try manager.list(std.testing.allocator, .{
        .sort_by = .priority,
        .sort_descending = false,
    });
    defer std.testing.allocator.free(sorted_asc);

    try std.testing.expectEqual(Priority.low, sorted_asc[0].priority);
    try std.testing.expectEqual(Priority.normal, sorted_asc[1].priority);
    try std.testing.expectEqual(Priority.high, sorted_asc[2].priority);
}

test {
    std.testing.refAllDecls(@This());
}
