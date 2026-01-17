//! Task Manager
//!
//! Handles task CRUD operations, persistence, and queries.

const std = @import("std");
const types = @import("types.zig");
const time_utils = @import("../shared/utils/time.zig");

const Task = types.Task;
const Priority = types.Priority;
const Status = types.Status;
const Category = types.Category;
const Filter = types.Filter;
const Stats = types.Stats;

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

    /// List tasks with optional filter
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
        // Initialize I/O backend for Zig 0.16
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        // Ensure directory exists
        const dir_path = std.fs.path.dirname(self.config.storage_path) orelse ".";
        std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};

        var file = std.Io.Dir.cwd().createFile(io, self.config.storage_path, .{ .truncate = true }) catch |err| {
            return err;
        };
        defer file.close(io);

        // Build JSON content in memory first
        var json_buffer = std.ArrayListUnmanaged(u8).empty;
        defer json_buffer.deinit(self.allocator);

        // Write simple JSON manually
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

        // Write to file
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
        // Initialize I/O backend for Zig 0.16
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

        // Parse JSON using std.json
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
    try std.testing.expectEqual(Status.pending, task.status);

    try manager.complete(id);
    const updated = manager.get(id).?;
    try std.testing.expectEqual(Status.completed, updated.status);
}

test "Manager stats" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_tasks2.json",
        .auto_save = false,
    });
    defer manager.deinit();

    _ = try manager.add("Task 1", .{});
    _ = try manager.add("Task 2", .{});
    const id3 = try manager.add("Task 3", .{});
    try manager.complete(id3);

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.total);
    try std.testing.expectEqual(@as(usize, 2), stats.pending);
    try std.testing.expectEqual(@as(usize, 1), stats.completed);
}

test "Manager list with filter" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_tasks3.json",
        .auto_save = false,
    });
    defer manager.deinit();

    _ = try manager.add("High priority", .{ .priority = .high });
    _ = try manager.add("Low priority", .{ .priority = .low });
    _ = try manager.add("Another high", .{ .priority = .high });

    const high_tasks = try manager.list(std.testing.allocator, .{ .priority = .high });
    defer std.testing.allocator.free(high_tasks);

    try std.testing.expectEqual(@as(usize, 2), high_tasks.len);
}
