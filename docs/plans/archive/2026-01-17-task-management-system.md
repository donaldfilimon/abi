---
title: "2026-01-17-task-management-system"
tags: []
---
# Task Management System Implementation Plan
> **Codebase Status:** Synced with repository as of 2026-01-18.

> **Status:** In Progress 🔄

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified task management system that provides CLI-based personal task tracking, integrates with the distributed compute scheduler, and organizes project roadmap items.

**Architecture:** Three-layer design: (1) Core `Task` abstraction in `src/tasks/` with persistence, (2) CLI command `tools/cli/commands/task.zig` for user interaction, (3) Integration hooks into existing `TaskScheduler` for distributed execution. Uses JSON file storage for simplicity.

**Tech Stack:** Zig 0.16, JSON serialization via `std.json`, file-based persistence, existing CLI patterns from `tools/cli/`

---

## Phase 1: Core Task Module

### Task 1.1: Create Task Types

**Files:**
- Create: `src/tasks/types.zig`
- Create: `src/tasks/mod.zig`

**Step 1: Create the types file**

Create `src/tasks/types.zig`:

```zig
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
```

**Step 2: Create the module entry point**

Create `src/tasks/mod.zig`:

```zig
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

pub const types = @import("types.zig");

pub const Task = types.Task;
pub const Priority = types.Priority;
pub const Status = types.Status;
pub const Category = types.Category;
pub const Filter = types.Filter;
pub const SortBy = types.SortBy;
pub const Stats = types.Stats;

// Manager will be added in Task 1.2
pub const Manager = @import("manager.zig").Manager;
pub const ManagerError = @import("manager.zig").ManagerError;

test {
    _ = types;
    _ = @import("manager.zig");
}
```

**Step 3: Verify compilation**

Run: `zig build-lib src/tasks/types.zig -femit-bin=nul`
Expected: Clean compilation

**Step 4: Commit**

```bash
git add src/tasks/types.zig src/tasks/mod.zig
git commit -m "feat(tasks): add core task types

- Task struct with id, title, status, priority, category
- Priority, Status, Category enums with string conversion
- Filter and SortBy for queries
- Stats for aggregates"
```

---

### Task 1.2: Create Task Manager

**Files:**
- Create: `src/tasks/manager.zig`
- Modify: `src/tasks/mod.zig` (already imports manager)

**Step 1: Create the manager file**

Create `src/tasks/manager.zig`:

```zig
//! Task Manager
//!
//! Handles task CRUD operations, persistence, and queries.

const std = @import("std");
const types = @import("types.zig");

const Task = types.Task;
const Priority = types.Priority;
const Status = types.Status;
const Category = types.Category;
const Filter = types.Filter;
const SortBy = types.SortBy;
const Stats = types.Stats;

pub const ManagerError = error{
    TaskNotFound,
    InvalidOperation,
    PersistenceFailed,
    ParseError,
} || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError || std.fs.File.WriteError;

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
        const now = std.time.timestamp();
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
        ptr.updated_at = std.time.timestamp();
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

    fn matchesFilter(self: *const Manager, task: *const Task, filter: Filter) bool {
        _ = self;
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
        // Ensure directory exists
        const dir_path = std.fs.path.dirname(self.config.storage_path) orelse ".";
        std.fs.cwd().makePath(dir_path) catch {};

        var file = std.fs.cwd().createFile(self.config.storage_path, .{}) catch |err| {
            return err;
        };
        defer file.close();

        var writer = file.writer();

        // Write simple JSON manually
        try writer.writeAll("{\n  \"next_id\": ");
        try std.fmt.format(writer, "{d}", .{self.next_id});
        try writer.writeAll(",\n  \"tasks\": [\n");

        var first = true;
        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            if (!first) try writer.writeAll(",\n");
            first = false;
            try self.writeTask(writer, entry.value_ptr.*);
        }

        try writer.writeAll("\n  ]\n}\n");
        self.dirty = false;
    }

    fn writeTask(self: *const Manager, writer: anytype, task: Task) !void {
        _ = self;
        try writer.writeAll("    {");
        try std.fmt.format(writer, "\"id\":{d},", .{task.id});
        try writer.writeAll("\"title\":\"");
        try writeJsonString(writer, task.title);
        try writer.writeAll("\",");
        try std.fmt.format(writer, "\"status\":\"{s}\",", .{task.status.toString()});
        try std.fmt.format(writer, "\"priority\":\"{s}\",", .{task.priority.toString()});
        try std.fmt.format(writer, "\"category\":\"{s}\",", .{task.category.toString()});
        try std.fmt.format(writer, "\"created_at\":{d},", .{task.created_at});
        try std.fmt.format(writer, "\"updated_at\":{d}", .{task.updated_at});
        if (task.description) |d| {
            try writer.writeAll(",\"description\":\"");
            try writeJsonString(writer, d);
            try writer.writeAll("\"");
        }
        if (task.due_date) |d| {
            try std.fmt.format(writer, ",\"due_date\":{d}", .{d});
        }
        if (task.completed_at) |c| {
            try std.fmt.format(writer, ",\"completed_at\":{d}", .{c});
        }
        if (task.parent_id) |p| {
            try std.fmt.format(writer, ",\"parent_id\":{d}", .{p});
        }
        try writer.writeAll("}");
    }

    /// Load tasks from file
    pub fn load(self: *Manager) ManagerError!void {
        const file = try std.fs.cwd().openFile(self.config.storage_path, .{});
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 1024 * 1024) catch |err| {
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

fn writeJsonString(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(c),
        }
    }
}

test "Manager basic operations" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = "/tmp/abi_test_tasks.json",
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
        .storage_path = "/tmp/abi_test_tasks2.json",
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
```

**Step 2: Verify compilation**

Run: `zig build-lib src/tasks/manager.zig -femit-bin=nul`
Expected: Clean compilation

**Step 3: Run tests**

Run: `zig test src/tasks/manager.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/tasks/manager.zig
git commit -m "feat(tasks): add task manager with persistence

- CRUD operations (add, get, setStatus, delete)
- Convenience methods (complete, start, cancel)
- JSON file persistence with auto-save
- Filtering and statistics
- Tests for basic operations"
```

---

## Phase 2: CLI Integration

### Task 2.1: Create Task CLI Command

**Files:**
- Create: `tools/cli/commands/task.zig`
- Modify: `tools/cli/mod.zig` (add task command)
- Modify: `tools/cli/commands/mod.zig` (export task)

**Step 1: Create the task command**

Create `tools/cli/commands/task.zig`:

```zig
//! Task Management CLI
//!
//! Commands:
//!   task add <title> [--priority=<p>] [--category=<c>]
//!   task list [--status=<s>] [--priority=<p>]
//!   task show <id>
//!   task done <id>
//!   task start <id>
//!   task cancel <id>
//!   task delete <id>
//!   task stats

const std = @import("std");
const tasks = @import("../../../src/tasks/mod.zig");
const utils = @import("../utils/mod.zig");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (!parser.hasMore() or parser.wantsHelp()) {
        printHelp();
        return;
    }

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {}", .{err});
        return;
    };
    defer manager.deinit();

    const command = parser.next().?;

    if (std.mem.eql(u8, command, "add")) {
        try runAdd(allocator, &parser, &manager);
    } else if (std.mem.eql(u8, command, "list") or std.mem.eql(u8, command, "ls")) {
        try runList(allocator, &parser, &manager);
    } else if (std.mem.eql(u8, command, "show")) {
        try runShow(&parser, &manager);
    } else if (std.mem.eql(u8, command, "done")) {
        try runDone(&parser, &manager);
    } else if (std.mem.eql(u8, command, "start")) {
        try runStart(&parser, &manager);
    } else if (std.mem.eql(u8, command, "cancel")) {
        try runCancel(&parser, &manager);
    } else if (std.mem.eql(u8, command, "delete") or std.mem.eql(u8, command, "rm")) {
        try runDelete(&parser, &manager);
    } else if (std.mem.eql(u8, command, "stats")) {
        runStats(&manager);
    } else {
        utils.output.printError("Unknown command: {s}", .{command});
        printHelp();
    }
}

fn runAdd(allocator: std.mem.Allocator, parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const title = parser.next() orelse {
        utils.output.printError("Usage: task add <title> [options]", .{});
        return;
    };

    const priority_str = parser.consumeOption(&.{ "--priority", "-p" });
    const priority = if (priority_str) |p| tasks.Priority.fromString(p) orelse .normal else .normal;

    const category_str = parser.consumeOption(&.{ "--category", "-c" });
    const category = if (category_str) |c| tasks.Category.fromString(c) orelse .personal else .personal;

    const desc = parser.consumeOption(&.{ "--desc", "-d" });

    const id = manager.add(title, .{
        .priority = priority,
        .category = category,
        .description = desc,
    }) catch |err| {
        utils.output.printError("Failed to add task: {}", .{err});
        return;
    };

    utils.output.printSuccess("Created task #{d}: {s}", .{ id, title });
    _ = allocator;
}

fn runList(allocator: std.mem.Allocator, parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    var filter = tasks.Filter{};

    if (parser.consumeOption(&.{ "--status", "-s" })) |s| {
        filter.status = tasks.Status.fromString(s);
    }
    if (parser.consumeOption(&.{ "--priority", "-p" })) |p| {
        filter.priority = tasks.Priority.fromString(p);
    }
    if (parser.consumeOption(&.{ "--category", "-c" })) |c| {
        filter.category = tasks.Category.fromString(c);
    }
    if (parser.consumeFlag(&.{"--overdue"})) {
        filter.overdue_only = true;
    }

    const task_list = manager.list(allocator, filter) catch |err| {
        utils.output.printError("Failed to list tasks: {}", .{err});
        return;
    };
    defer allocator.free(task_list);

    if (task_list.len == 0) {
        std.debug.print("No tasks found.\n", .{});
        return;
    }

    std.debug.print("\n", .{});
    std.debug.print("{s:<5} {s:<10} {s:<10} {s:<12} {s}\n", .{ "ID", "Status", "Priority", "Category", "Title" });
    std.debug.print("{s}\n", .{"-" ** 70});

    for (task_list) |task| {
        const status_marker = switch (task.status) {
            .pending => "[ ]",
            .in_progress => "[~]",
            .completed => "[x]",
            .cancelled => "[-]",
            .blocked => "[!]",
        };
        std.debug.print("#{d:<4} {s:<10} {s:<10} {s:<12} {s}\n", .{
            task.id,
            status_marker,
            task.priority.toString(),
            task.category.toString(),
            task.title,
        });
    }
    std.debug.print("\n", .{});
}

fn runShow(parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const id_str = parser.next() orelse {
        utils.output.printError("Usage: task show <id>", .{});
        return;
    };

    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    const task = manager.get(id) orelse {
        utils.output.printError("Task #{d} not found", .{id});
        return;
    };

    std.debug.print("\nTask #{d}\n", .{task.id});
    std.debug.print("  Title:    {s}\n", .{task.title});
    std.debug.print("  Status:   {s}\n", .{task.status.toString()});
    std.debug.print("  Priority: {s}\n", .{task.priority.toString()});
    std.debug.print("  Category: {s}\n", .{task.category.toString()});
    if (task.description) |d| {
        std.debug.print("  Description: {s}\n", .{d});
    }
    std.debug.print("\n", .{});
}

fn runDone(parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const id = try parseTaskId(parser);
    manager.complete(id) catch |err| {
        utils.output.printError("Failed to complete task: {}", .{err});
        return;
    };
    utils.output.printSuccess("Completed task #{d}", .{id});
}

fn runStart(parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const id = try parseTaskId(parser);
    manager.start(id) catch |err| {
        utils.output.printError("Failed to start task: {}", .{err});
        return;
    };
    utils.output.printSuccess("Started task #{d}", .{id});
}

fn runCancel(parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const id = try parseTaskId(parser);
    manager.cancel(id) catch |err| {
        utils.output.printError("Failed to cancel task: {}", .{err});
        return;
    };
    utils.output.printSuccess("Cancelled task #{d}", .{id});
}

fn runDelete(parser: *utils.args.ArgParser, manager: *tasks.Manager) !void {
    const id = try parseTaskId(parser);
    manager.delete(id) catch |err| {
        utils.output.printError("Failed to delete task: {}", .{err});
        return;
    };
    utils.output.printSuccess("Deleted task #{d}", .{id});
}

fn runStats(manager: *tasks.Manager) void {
    const stats = manager.getStats();

    std.debug.print("\nTask Statistics\n", .{});
    std.debug.print("{s}\n", .{"-" ** 30});
    std.debug.print("  Total:       {d}\n", .{stats.total});
    std.debug.print("  Pending:     {d}\n", .{stats.pending});
    std.debug.print("  In Progress: {d}\n", .{stats.in_progress});
    std.debug.print("  Completed:   {d}\n", .{stats.completed});
    std.debug.print("  Cancelled:   {d}\n", .{stats.cancelled});
    std.debug.print("  Blocked:     {d}\n", .{stats.blocked});
    if (stats.overdue > 0) {
        std.debug.print("  Overdue:     {d} (!)\n", .{stats.overdue});
    }
    std.debug.print("\n", .{});
}

fn parseTaskId(parser: *utils.args.ArgParser) !u64 {
    const id_str = parser.next() orelse {
        utils.output.printError("Usage: task <command> <id>", .{});
        return error.MissingArgument;
    };

    return std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return error.InvalidArgument;
    };
}

const CommandError = error{
    MissingArgument,
    InvalidArgument,
};

fn printHelp() void {
    std.debug.print(
        \\
        \\Task Management
        \\
        \\USAGE:
        \\  abi task <command> [args] [options]
        \\
        \\COMMANDS:
        \\  add <title>     Add a new task
        \\  list, ls        List tasks (with optional filters)
        \\  show <id>       Show task details
        \\  done <id>       Mark task as completed
        \\  start <id>      Mark task as in-progress
        \\  cancel <id>     Cancel a task
        \\  delete, rm <id> Delete a task
        \\  stats           Show task statistics
        \\
        \\OPTIONS:
        \\  --priority, -p <low|normal|high|critical>
        \\  --category, -c <personal|roadmap|compute|bug|feature>
        \\  --status, -s   <pending|in_progress|completed|cancelled|blocked>
        \\  --desc, -d     Description text
        \\  --overdue      Show only overdue tasks
        \\
        \\EXAMPLES:
        \\  abi task add "Fix bug" --priority=high --category=bug
        \\  abi task list --status=pending
        \\  abi task done 1
        \\  abi task stats
        \\
    , .{});
}
```

**Step 2: Add task to commands/mod.zig**

Modify `tools/cli/commands/mod.zig` to export task:

```zig
pub const task = @import("task.zig");
```

**Step 3: Add task command to dispatcher**

Modify `tools/cli/mod.zig`, add after the existing command checks:

```zig
if (std.mem.eql(u8, command, "task")) {
    try commands.task.run(allocator, args[2..]);
    return;
}
```

**Step 4: Build and test**

Run: `zig build`
Expected: Clean build

Run: `zig build run -- task --help`
Expected: Shows task command help

Run: `zig build run -- task add "Test task" --priority=high`
Expected: Shows "Created task #1: Test task"

Run: `zig build run -- task list`
Expected: Shows the task in a table

**Step 5: Commit**

```bash
git add tools/cli/commands/task.zig tools/cli/commands/mod.zig tools/cli/mod.zig
git commit -m "feat(cli): add task management command

Commands: add, list, show, done, start, cancel, delete, stats
Options: --priority, --category, --status, --desc, --overdue
Persistence: JSON file at .abi/tasks.json"
```

---

## Phase 3: Roadmap Integration

### Task 3.1: Import Roadmap Items as Tasks

**Files:**
- Create: `src/tasks/roadmap.zig`
- Modify: `src/tasks/mod.zig` (export roadmap)

**Step 1: Create roadmap importer**

Create `src/tasks/roadmap.zig`:

```zig
//! Roadmap Item Importer
//!
//! Imports incomplete roadmap items from ROADMAP.md as tasks.

const std = @import("std");
const Manager = @import("manager.zig").Manager;
const types = @import("types.zig");

pub const RoadmapItem = struct {
    title: []const u8,
    category: []const u8,
    timeline: []const u8,
    description: ?[]const u8 = null,
};

/// Predefined roadmap items from ROADMAP.md analysis
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

/// Import all incomplete roadmap items as tasks
pub fn importAll(manager: *Manager) !usize {
    var count: usize = 0;

    for (incomplete_items) |item| {
        // Check if already exists (by title)
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
```

**Step 2: Add to mod.zig**

Add to `src/tasks/mod.zig`:

```zig
pub const roadmap = @import("roadmap.zig");
```

**Step 3: Add CLI command for import**

Add to `tools/cli/commands/task.zig` in the command dispatch:

```zig
} else if (std.mem.eql(u8, command, "import-roadmap")) {
    try runImportRoadmap(&manager);
}
```

And add the handler:

```zig
fn runImportRoadmap(manager: *tasks.Manager) !void {
    const count = tasks.roadmap.importAll(manager) catch |err| {
        utils.output.printError("Failed to import roadmap: {}", .{err});
        return;
    };
    utils.output.printSuccess("Imported {d} roadmap items as tasks", .{count});
}
```

**Step 4: Test import**

Run: `zig build run -- task import-roadmap`
Expected: Shows "Imported 11 roadmap items as tasks"

Run: `zig build run -- task list --category=roadmap`
Expected: Shows all roadmap items

**Step 5: Commit**

```bash
git add src/tasks/roadmap.zig src/tasks/mod.zig tools/cli/commands/task.zig
git commit -m "feat(tasks): add roadmap item import

- Predefined list of incomplete ROADMAP.md items
- import-roadmap command to create tasks from roadmap
- Automatically sets category=roadmap and appropriate priority"
```

---

## Phase 4: ABI Integration

### Task 4.1: Add Tasks to abi.zig

**Files:**
- Modify: `src/abi.zig` (add tasks export)

**Step 1: Add tasks import to abi.zig**

Add to `src/abi.zig` with other imports:

```zig
pub const tasks = @import("tasks/mod.zig");
```

**Step 2: Verify build**

Run: `zig build`
Expected: Clean build

Run: `zig build test --summary all`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/abi.zig
git commit -m "feat(abi): export tasks module

Tasks module now accessible via abi.tasks for programmatic use"
```

---

## Phase 5: Documentation

### Task 5.1: Add Task Module Documentation

**Files:**
- Create: `src/tasks/README.md`
- Modify: `CLAUDE.md` (add tasks to CLI commands table)

**Step 1: Create module README**

Create `src/tasks/README.md`:

```markdown
# Task Management Module

**Status:** ✅ Complete

## Overview

Unified task management for personal tasks, project roadmap items, and future distributed compute job tracking.

## CLI Usage

```bash
# Add tasks
abi task add "Fix bug" --priority=high --category=bug
abi task add "Write docs" --desc="API documentation"

# List and filter
abi task list
abi task list --status=pending
abi task list --category=roadmap --priority=high

# Manage status
abi task done 1
abi task start 2
abi task cancel 3

# View details
abi task show 1
abi task stats

# Import roadmap items
abi task import-roadmap
```

## Programmatic Usage

```zig
const abi = @import("abi");

var manager = try abi.tasks.Manager.init(allocator, .{});
defer manager.deinit();

// Add task
const id = try manager.add("My task", .{
    .priority = .high,
    .category = .feature,
});

// Query
const pending = try manager.list(allocator, .{ .status = .pending });
defer allocator.free(pending);

// Update
try manager.complete(id);
```

## Storage

Tasks are stored in `.abi/tasks.json` with auto-save enabled by default.

## Task Properties

- **id:** Unique identifier
- **title:** Task name
- **description:** Optional detailed description
- **status:** pending, in_progress, completed, cancelled, blocked
- **priority:** low, normal, high, critical
- **category:** personal, roadmap, compute, bug, feature
- **due_date:** Optional deadline (Unix timestamp)
- **parent_id:** Optional parent task for subtasks
```

**Step 2: Update CLAUDE.md CLI table**

Add to the CLI Commands table in CLAUDE.md:

```markdown
| `task` | Task management (add, list, done, stats, import-roadmap) |
```

**Step 3: Commit**

```bash
git add src/tasks/README.md CLAUDE.md
git commit -m "docs: add task module documentation

- Module README with CLI and programmatic usage
- Updated CLAUDE.md CLI commands table"
```

---

## Verification

### Final Verification Steps

**Step 1: Clean build**

Run: `rm -rf .zig-cache zig-out && zig build`
Expected: Clean build

**Step 2: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 3: Test CLI workflow**

```bash
# Full workflow test
zig build run -- task add "Test task 1" --priority=high
zig build run -- task add "Test task 2" --category=bug
zig build run -- task list
zig build run -- task done 1
zig build run -- task stats
zig build run -- task import-roadmap
zig build run -- task list --category=roadmap
```

Expected: All commands work correctly

**Step 4: Verify persistence**

Run: `cat .abi/tasks.json`
Expected: JSON file with all tasks

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: task management system complete

- Core task module with types, manager, persistence
- CLI with full CRUD operations
- Roadmap import integration
- Documentation complete
- All tests passing"
```

---

## Success Criteria

1. ✅ `zig build run -- task add "Title"` creates a task
2. ✅ `zig build run -- task list` shows tasks in table format
3. ✅ `zig build run -- task done <id>` marks task complete
4. ✅ `zig build run -- task stats` shows aggregate statistics
5. ✅ `zig build run -- task import-roadmap` imports ROADMAP.md items
6. ✅ Tasks persist to `.abi/tasks.json`
7. ✅ All tests pass
8. ✅ Programmatic access via `abi.tasks`

