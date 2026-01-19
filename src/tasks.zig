//! Task Management Module
//!
//! Provides unified task tracking for personal tasks, project roadmap
//! items, and distributed compute jobs.
//!
//! ## Usage
//!
//! ```zig
//! const tasks = @import("tasks.zig");
//!
//! var manager = try tasks.Manager.init(allocator, .{});
//! defer manager.deinit();
//!
//! const id = try manager.add("Fix bug", .{ .priority = .high });
//! try manager.complete(id);
//! ```

const std = @import("std");
const time_utils = @import("shared/utils.zig");

// ============================================================================
// Types
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
// Manager
// ============================================================================

pub const ManagerError = error{
    TaskNotFound,
    InvalidOperation,
    PersistenceFailed,
    ParseError,
} || std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.Dir.ReadFileAllocError || std.Io.File.Writer.Error;

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

pub const Manager = struct {
    allocator: std.mem.Allocator,
    config: ManagerConfig,
    tasks: std.AutoHashMapUnmanaged(u64, Task),
    next_id: u64,
    dirty: bool,

    // Owned string storage
    strings: std.ArrayListUnmanaged([]u8),

    pub fn init(allocator: std.mem.Allocator, config: ManagerConfig) ManagerError!Manager {
        var self = Manager{
            .allocator = allocator,
            .config = config,
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
            self.save() catch {};
        }

        // Free all owned strings
        for (self.strings.items) |s| {
            self.allocator.free(s);
        }
        self.strings.deinit(self.allocator);

        self.tasks.deinit(self.allocator);
    }

    /// Add a new task
    pub fn add(self: *Manager, title: []const u8, options: AddOptions) ManagerError!u64 {
        const now = time_utils.unixSeconds();
        const id = self.next_id;
        self.next_id += 1;

        // Duplicate strings we need to own
        const owned_title = try self.dupeString(title);
        const owned_desc = if (options.description) |d| try self.dupeString(d) else null;

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

        try self.tasks.put(self.allocator, id, task);
        self.dirty = true;

        if (self.config.auto_save) {
            try self.save();
        }

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
        // Verify blocker exists if set
        if (blocker_id) |bid| {
            if (!self.tasks.contains(bid)) return error.TaskNotFound;
        }
        ptr.blocked_by = blocker_id;
        ptr.updated_at = time_utils.unixSeconds();
        // Also set status to blocked if a blocker is specified
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
        const owned_title = try self.dupeString(title);
        ptr.title = owned_title;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// Update task description
    pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void {
        const ptr = self.tasks.getPtr(id) orelse return error.TaskNotFound;
        const owned_desc = if (description) |d| try self.dupeString(d) else null;
        ptr.description = owned_desc;
        ptr.updated_at = time_utils.unixSeconds();
        self.dirty = true;
        if (self.config.auto_save) try self.save();
    }

    /// List tasks with optional filter and sorting
    pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task {
        var result = std.ArrayListUnmanaged(Task){};
        errdefer result.deinit(allocator);

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            if (self.matchesFilter(&task, filter)) {
                try result.append(allocator, task);
            }
        }

        // Sort the results
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
                // Tasks without due dates sort to the end
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

        return result.toOwnedSlice(allocator);
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

    fn matchesFilter(_: *const Manager, task: *const Task, filter: Filter) bool {
        if (filter.status) |s| if (task.status != s) return false;
        if (filter.priority) |p| if (task.priority != p) return false;
        if (filter.category) |c| if (task.category != c) return false;
        if (filter.overdue_only and !task.isOverdue()) return false;
        if (filter.parent_id) |pid| if (task.parent_id != pid) return false;
        return true;
    }

    fn dupeString(self: *Manager, s: []const u8) ManagerError![]const u8 {
        const owned = try self.allocator.dupe(u8, s);
        try self.strings.append(self.allocator, owned);
        return owned;
    }

    /// Save tasks to file
    pub fn save(self: *Manager) ManagerError!void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const dir_path = std.fs.path.dirname(self.config.storage_path) orelse ".";
        std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};

        var file = std.Io.Dir.cwd().createFile(io, self.config.storage_path, .{ .truncate = true }) catch |err| {
            return err;
        };
        defer file.close(io);

        var json_buffer = std.ArrayListUnmanaged(u8).empty;
        defer json_buffer.deinit(self.allocator);

        try json_buffer.appendSlice(self.allocator, "{\n  \"next_id\": ");
        try json_buffer.print(self.allocator, "{d}", .{self.next_id});
        try json_buffer.appendSlice(self.allocator, ",\n  \"tasks\": [\n");

        var first = true;
        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            if (!first) try json_buffer.appendSlice(self.allocator, ",\n");
            first = false;
            try self.writeTask(&json_buffer, entry.value_ptr.*);
        }

        try json_buffer.appendSlice(self.allocator, "\n  ]\n}\n");

        try file.writeStreamingAll(io, json_buffer.items);
        self.dirty = false;
    }

    fn writeTask(self: *const Manager, buf: *std.ArrayListUnmanaged(u8), task: Task) !void {
        try buf.appendSlice(self.allocator, "    {");
        try buf.print(self.allocator, "\"id\":{d},", .{task.id});
        try buf.appendSlice(self.allocator, "\"title\":\"");
        try writeJsonString(self.allocator, buf, task.title);
        try buf.appendSlice(self.allocator, "\",");
        try buf.print(self.allocator, "\"status\":\"{s}\",", .{task.status.toString()});
        try buf.print(self.allocator, "\"priority\":\"{s}\",", .{task.priority.toString()});
        try buf.print(self.allocator, "\"category\":\"{s}\",", .{task.category.toString()});
        try buf.print(self.allocator, "\"created_at\":{d},", .{task.created_at});
        try buf.print(self.allocator, "\"updated_at\":{d}", .{task.updated_at});
        if (task.description) |d| {
            try buf.appendSlice(self.allocator, ",\"description\":\"");
            try writeJsonString(self.allocator, buf, d);
            try buf.append(self.allocator, '"');
        }
        if (task.due_date) |d| {
            try buf.print(self.allocator, ",\"due_date\":{d}", .{d});
        }
        if (task.completed_at) |c| {
            try buf.print(self.allocator, ",\"completed_at\":{d}", .{c});
        }
        if (task.parent_id) |p| {
            try buf.print(self.allocator, ",\"parent_id\":{d}", .{p});
        }
        try buf.append(self.allocator, '}');
    }

    /// Load tasks from file
    pub fn load(self: *Manager) ManagerError!void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const content = std.Io.Dir.cwd().readFileAlloc(
            io,
            self.config.storage_path,
            self.allocator,
            .limited(1024 * 1024),
        ) catch |err| {
            return err;
        };
        defer self.allocator.free(content);

        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch {
            return error.ParseError;
        };
        defer parsed.deinit();

        const root = parsed.value.object;

        if (root.get("next_id")) |nid| {
            self.next_id = @intCast(nid.integer);
        }

        if (root.get("tasks")) |tasks_val| {
            for (tasks_val.array.items) |task_val| {
                const obj = task_val.object;
                const task = try self.parseTask(obj);
                try self.tasks.put(self.allocator, task.id, task);
            }
        }

        self.dirty = false;
    }

    fn parseTask(self: *Manager, obj: std.json.ObjectMap) ManagerError!Task {
        const id: u64 = @intCast(obj.get("id").?.integer);
        const title = try self.dupeString(obj.get("title").?.string);
        const status = Status.fromString(obj.get("status").?.string) orelse .pending;
        const priority = Priority.fromString(obj.get("priority").?.string) orelse .normal;
        const category = Category.fromString(obj.get("category").?.string) orelse .personal;
        const created_at: i64 = obj.get("created_at").?.integer;
        const updated_at: i64 = obj.get("updated_at").?.integer;

        var task = Task{
            .id = id,
            .title = title,
            .status = status,
            .priority = priority,
            .category = category,
            .created_at = created_at,
            .updated_at = updated_at,
        };

        if (obj.get("description")) |d| {
            task.description = try self.dupeString(d.string);
        }
        if (obj.get("due_date")) |d| {
            task.due_date = d.integer;
        }
        if (obj.get("completed_at")) |c| {
            task.completed_at = c.integer;
        }
        if (obj.get("parent_id")) |p| {
            task.parent_id = @intCast(p.integer);
        }

        return task;
    }
};

fn writeJsonString(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => try buf.append(allocator, c),
        }
    }
}

// ============================================================================
// Roadmap Integration
// ============================================================================

pub const roadmap = struct {
    pub const RoadmapItem = struct {
        title: []const u8,
        category: []const u8,
        timeline: []const u8,
        description: ?[]const u8 = null,
    };

    pub const incomplete_items = [_]RoadmapItem{
        .{
            .title = "Record video tutorials",
            .category = "Documentation",
            .timeline = "Short-term",
            .description = "Record and produce video tutorials from existing scripts in docs/tutorials/videos/",
        },
        .{
            .title = "FPGA/ASIC hardware acceleration research",
            .category = "Research & Innovation",
            .timeline = "Long-term (2027+)",
            .description = "Experimental hardware acceleration using FPGA and ASIC for vector operations",
        },
        .{
            .title = "Novel index structures research",
            .category = "Research & Innovation",
            .timeline = "Long-term (2027+)",
            .description = "Research and implement novel index structures for improved search performance",
        },
        .{
            .title = "AI-optimized workloads",
            .category = "Research & Innovation",
            .timeline = "Long-term (2027+)",
            .description = "Optimize workloads specifically for AI/ML inference patterns",
        },
        .{
            .title = "Academic collaborations",
            .category = "Research & Innovation",
            .timeline = "Long-term (2027+)",
            .description = "Research partnerships, paper publications, conference presentations",
        },
        .{
            .title = "Community governance RFC process",
            .category = "Community & Growth",
            .timeline = "Long-term (2027+)",
            .description = "Establish RFC process, voting mechanism, contribution recognition",
        },
        .{
            .title = "Education and certification program",
            .category = "Community & Growth",
            .timeline = "Long-term (2027+)",
            .description = "Training courses, certification program, university partnerships",
        },
        .{
            .title = "Commercial support services",
            .category = "Enterprise Features",
            .timeline = "Long-term (2028+)",
            .description = "SLA offerings, priority support, custom development services",
        },
        .{
            .title = "AWS Lambda integration",
            .category = "Cloud Integration",
            .timeline = "Long-term (2028+)",
            .description = "Deploy ABI functions to AWS Lambda",
        },
        .{
            .title = "Google Cloud Functions integration",
            .category = "Cloud Integration",
            .timeline = "Long-term (2028+)",
            .description = "Deploy ABI functions to Google Cloud Functions",
        },
        .{
            .title = "Azure Functions integration",
            .category = "Cloud Integration",
            .timeline = "Long-term (2028+)",
            .description = "Deploy ABI functions to Azure Functions",
        },
    };

    pub fn importAll(manager: *Manager) !usize {
        var count: usize = 0;
        for (incomplete_items) |item| {
            var exists = false;
            var iter = manager.tasks.iterator();
            while (iter.next()) |entry| {
                if (std.mem.eql(u8, entry.value_ptr.title, item.title)) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                _ = try manager.add(item.title, .{
                    .description = item.description,
                    .category = .roadmap,
                    .priority = if (std.mem.eql(u8, item.timeline, "Short-term")) .high else .low,
                });
                count += 1;
            }
        }
        return count;
    }
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

test "Roadmap incomplete_items count" {
    try std.testing.expectEqual(@as(usize, 11), roadmap.incomplete_items.len);
}

test "SortBy string conversion" {
    try std.testing.expectEqualStrings("priority", SortBy.priority.toString());
    try std.testing.expectEqual(SortBy.due_date, SortBy.fromString("due_date").?);
    try std.testing.expectEqual(SortBy.due_date, SortBy.fromString("due").?);
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
