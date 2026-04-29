//! Task Management Module
//!
//! Provides unified task tracking for profilel tasks, project roadmap
//! items, and distributed compute jobs.
//!
//! ## Usage
//!
//! ```zig
//! const tasks = @import("tasks");
//!
//! var manager = try tasks.Manager.init(allocator, .{});
//! defer manager.deinit();
//!
//! const id = try manager.add("Fix bug", .{ .priority = .high });
//! try manager.complete(id);
//! ```

const std = @import("std");
const app_paths = @import("../foundation/app_paths.zig");
const time_utils = @import("../foundation/mod.zig").utils;

// Re-export types
pub const types = @import("types.zig");
pub const roadmap = @import("roadmap.zig");
pub const roadmap_catalog = roadmap.catalog;

/// A task representing a unit of work.
pub const Task = types.Task;
/// Task priority levels.
pub const Priority = types.Priority;
/// Task lifecycle statuses.
pub const Status = types.Status;
/// Task categories.
pub const Category = types.Category;
/// Filter criteria for listing tasks.
pub const Filter = types.Filter;
/// Sorting criteria for task lists.
pub const SortBy = types.SortBy;
/// Task statistics and metrics.
pub const Stats = types.Stats;
/// Configuration for the Task Manager.
pub const ManagerConfig = types.ManagerConfig;
/// Options for adding a new task.
pub const AddOptions = types.AddOptions;
/// Error set for task operations.
pub const ManagerError = types.ManagerError;
/// An item from the project roadmap.
pub const RoadmapItem = roadmap.RoadmapItem;

/// Helper structs for ZON persistence
const PersistedTask = struct {
    id: u64,
    title: []const u8,
    status: []const u8,
    priority: []const u8,
    category: []const u8,
    created_at: i64,
    updated_at: i64,
    description: ?[]const u8 = null,
    due_date: ?i64 = null,
    completed_at: ?i64 = null,
    parent_id: ?u64 = null,
};

const PersistedData = struct {
    next_id: u64,
    tasks: []const PersistedTask,
};

/// Duplicate a string and store it in the strings list.
fn dupeString(allocator: std.mem.Allocator, strings: *std.ArrayListUnmanaged([]u8), s: []const u8) ManagerError![]const u8 {
    const duped = allocator.dupe(u8, s) catch return error.OutOfMemory;
    strings.append(allocator, duped) catch return error.OutOfMemory;
    return duped;
}

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
            const resolved_path = app_paths.resolvePath(allocator, "tasks.zon") catch |err| switch (err) {
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
            .tasks = .empty,
            .next_id = 1,
            .dirty = false,
            .strings = .empty,
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
        const now = time_utils.unixSeconds();
        const id = self.next_id;
        self.next_id += 1;

        const owned_title = try dupeString(self.allocator, &self.strings, title);
        const owned_desc = if (options.description) |d| try dupeString(self.allocator, &self.strings, d) else null;

        const task = Task{
            .id = id,
            .title = owned_title,
            .description = owned_desc,
            .priority = options.priority,
            .category = options.category,
            .created_at = now,
            .updated_at = now,
            .due_date = options.due_date,
            .parent_id = options.parent_id,
        };

        self.tasks.put(self.allocator, id, task) catch return error.OutOfMemory;

        self.dirty = true;
        if (self.config.auto_save) try self.save();
        return id;
    }

    /// Get a task by ID
    pub fn get(self: *const Manager, id: u64) ?Task {
        return self.tasks.get(id);
    }

    /// Update task status
    pub fn setStatus(self: *Manager, id: u64, status: Status) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        ptr.status = status;
        ptr.updated_at = time_utils.unixSeconds();
        if (status == .completed) {
            ptr.completed_at = ptr.updated_at;
        }
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
        if (!self.tasks.remove(id)) return error.TaskNotFound;
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Set due date for a task
    pub fn setDueDate(self: *Manager, id: u64, due_date: ?i64) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        ptr.due_date = due_date;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Set task as blocked by another task
    pub fn setBlockedBy(self: *Manager, id: u64, blocker_id: ?u64) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        if (blocker_id) |bid| {
            if (!self.tasks.contains(bid)) return error.TaskNotFound;
        }
        ptr.blocked_by = blocker_id;
        ptr.updated_at = time_utils.unixSeconds();
        if (blocker_id != null) {
            ptr.status = .blocked;
        } else if (ptr.status == .blocked) {
            ptr.status = .pending;
        }
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task priority
    pub fn setPriority(self: *Manager, id: u64, priority: Priority) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        ptr.priority = priority;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task category
    pub fn setCategory(self: *Manager, id: u64, category: Category) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        ptr.category = category;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task title
    pub fn setTitle(self: *Manager, id: u64, title: []const u8) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        const owned_title = try dupeString(self.allocator, &self.strings, title);
        ptr.title = owned_title;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task description
    pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        const owned_desc = if (description) |d| try dupeString(self.allocator, &self.strings, d) else null;
        ptr.description = owned_desc;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    // ========================================================================
    // Querying Operations
    // ========================================================================

    /// List tasks with optional filter and sorting
    pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task {
        var result = std.ArrayListUnmanaged(Task).empty;
        errdefer result.deinit(allocator);

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            if (matchesFilter(&task, filter)) {
                result.append(allocator, task) catch return error.OutOfMemory;
            }
        }

        const Context = struct {
            sort_by: SortBy,
            descending: bool,

            pub fn lessThan(ctx: @This(), a: Task, b: Task) bool {
                const cmp = switch (ctx.sort_by) {
                    .created => compare(a.created_at, b.created_at),
                    .updated => compare(a.updated_at, b.updated_at),
                    .priority => compare(@intFromEnum(a.priority), @intFromEnum(b.priority)),
                    .due_date => compareDueDate(a.due_date, b.due_date),
                    .status => compare(@intFromEnum(a.status), @intFromEnum(b.status)),
                };
                return if (ctx.descending) cmp == .gt else cmp == .lt;
            }

            fn compare(a: anytype, b: @TypeOf(a)) std.math.Order {
                return std.math.order(a, b);
            }

            fn compareDueDate(a: ?i64, b: ?i64) std.math.Order {
                if (a == null and b == null) return .eq;
                if (a == null) return .gt;
                if (b == null) return .lt;
                return std.math.order(a.?, b.?);
            }
        };

        std.mem.sortUnstable(Task, result.items, Context{
            .sort_by = filter.sort_by,
            .descending = filter.sort_descending,
        }, Context.lessThan);

        return result.toOwnedSlice(allocator) catch error.OutOfMemory;
    }

    /// Check if a task matches the given filter
    fn matchesFilter(task: *const Task, filter: Filter) bool {
        if (filter.status) |s| if (task.status != s) return false;
        if (filter.priority) |p| if (task.priority != p) return false;
        if (filter.category) |c| if (task.category != c) return false;
        if (filter.overdue_only and !task.isOverdue()) return false;
        if (filter.parent_id) |pid| if (task.parent_id != pid) return false;
        return true;
    }

    /// Get statistics
    pub fn getStats(self: *const Manager) Stats {
        var stats = Stats{};
        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            stats.total += 1;
            switch (task.status) {
                .pending => stats.pending += 1,
                .in_progress => stats.in_progress += 1,
                .completed => stats.completed += 1,
                .cancelled => stats.cancelled += 1,
                .blocked => stats.blocked += 1,
            }
            if (task.isOverdue()) stats.overdue += 1;
        }
        return stats;
    }

    // ========================================================================
    // Persistence Operations
    // ========================================================================

    /// Save tasks to file
    pub fn save(self: *Manager) ManagerError!void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = .empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const dir_path = std.fs.path.dirname(self.config.storage_path) orelse ".";
        std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| {
            std.log.warn("persistence: failed to create directory '{s}': {t}", .{ dir_path, err });
        };

        var file = std.Io.Dir.cwd().createFile(io, self.config.storage_path, .{ .truncate = true }) catch return error.PersistenceFailed;
        defer file.close(io);

        var persisted_tasks = std.ArrayListUnmanaged(PersistedTask).empty;
        defer persisted_tasks.deinit(self.allocator);

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const t = entry.value_ptr.*;
            persisted_tasks.append(self.allocator, .{
                .id = t.id,
                .title = t.title,
                .status = t.status.toString(),
                .priority = t.priority.toString(),
                .category = t.category.toString(),
                .created_at = t.created_at,
                .updated_at = t.updated_at,
                .description = t.description,
                .due_date = t.due_date,
                .completed_at = t.completed_at,
                .parent_id = t.parent_id,
            }) catch return error.OutOfMemory;
        }

        const data = PersistedData{
            .next_id = self.next_id,
            .tasks = persisted_tasks.items,
        };

        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        var writer = out.writer;
        std.zon.stringify.serialize(data, .{}, &writer) catch return error.PersistenceFailed;

        const slice = out.toOwnedSlice() catch return error.PersistenceFailed;
        file.writeStreamingAll(io, slice) catch return error.PersistenceFailed;
        self.dirty = false;
    }

    /// Load tasks from file
    pub fn load(self: *Manager) ManagerError!void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = .empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const contents = std.Io.Dir.cwd().readFileAlloc(
            io,
            self.config.storage_path,
            self.allocator,
            .limited(1024 * 1024),
        ) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return error.PersistenceFailed,
        };
        defer self.allocator.free(contents);

        const contents_z = self.allocator.dupeZ(u8, contents) catch return error.OutOfMemory;
        defer self.allocator.free(contents_z);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        const data = std.zon.parse.fromSliceAlloc(PersistedData, arena_allocator, contents_z, null, .{}) catch {
            return error.ParseError;
        };
        self.next_id = data.next_id;

        for (data.tasks) |pt| {
            const task = Task{
                .id = pt.id,
                .title = try dupeString(self.allocator, &self.strings, pt.title),
                .status = Status.fromString(pt.status) orelse .pending,
                .priority = Priority.fromString(pt.priority) orelse .normal,
                .category = Category.fromString(pt.category) orelse .profilel,
                .created_at = pt.created_at,
                .updated_at = pt.updated_at,
                .description = if (pt.description) |d| try dupeString(self.allocator, &self.strings, d) else null,
                .due_date = pt.due_date,
                .completed_at = pt.completed_at,
                .parent_id = pt.parent_id,
            };
            self.tasks.put(self.allocator, task.id, task) catch return error.OutOfMemory;
        }
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
        .storage_path = ".zig-cache/test_tasks.zon",
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
