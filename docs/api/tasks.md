# tasks API Reference

> Centralized task management system

**Source:** [`src/tasks.zig`](../../src/tasks.zig)

---

Task Management Module

Provides unified task tracking for personal tasks, project roadmap
items, and distributed compute jobs.

## Usage

```zig
const tasks = @import("tasks.zig");

var manager = try tasks.Manager.init(allocator, .{});
defer manager.deinit();

const id = try manager.add("Fix bug", .{ .priority = .high });
try manager.complete(id);
```

---

## API

### `pub const Priority`

<sup>**type**</sup>

Task priority levels

### `pub const Status`

<sup>**type**</sup>

Task status

### `pub const Category`

<sup>**type**</sup>

Task category for organization

### `pub const Task`

<sup>**type**</sup>

Core task structure

### `pub fn isActionable(self: *const Task) bool`

<sup>**fn**</sup>

Check if task is actionable (not blocked or completed)

### `pub fn isOverdue(self: *const Task) bool`

<sup>**fn**</sup>

Check if task is overdue

### `pub const Filter`

<sup>**type**</sup>

Filter criteria for querying tasks

### `pub const SortBy`

<sup>**type**</sup>

Sort options for task lists

### `pub const Stats`

<sup>**type**</sup>

Task statistics

### `pub fn add(self: *Manager, title: []const u8, options: AddOptions) ManagerError!u64`

<sup>**fn**</sup>

Add a new task

### `pub fn get(self: *const Manager, id: u64) ?Task`

<sup>**fn**</sup>

Get a task by ID

### `pub fn setStatus(self: *Manager, id: u64, status: Status) ManagerError!void`

<sup>**fn**</sup>

Update task status

### `pub fn complete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup>

Mark task as completed

### `pub fn start(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup>

Mark task as in progress

### `pub fn cancel(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup>

Cancel a task

### `pub fn delete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup>

Delete a task

### `pub fn setDueDate(self: *Manager, id: u64, due_date: ?i64) ManagerError!void`

<sup>**fn**</sup>

Set due date for a task

### `pub fn setBlockedBy(self: *Manager, id: u64, blocker_id: ?u64) ManagerError!void`

<sup>**fn**</sup>

Set task as blocked by another task

### `pub fn setPriority(self: *Manager, id: u64, priority: Priority) ManagerError!void`

<sup>**fn**</sup>

Update task priority

### `pub fn setCategory(self: *Manager, id: u64, category: Category) ManagerError!void`

<sup>**fn**</sup>

Update task category

### `pub fn setTitle(self: *Manager, id: u64, title: []const u8) ManagerError!void`

<sup>**fn**</sup>

Update task title

### `pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void`

<sup>**fn**</sup>

Update task description

### `pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task`

<sup>**fn**</sup>

List tasks with optional filter and sorting

### `pub fn getStats(self: *const Manager) Stats`

<sup>**fn**</sup>

Get statistics

### `pub fn save(self: *Manager) ManagerError!void`

<sup>**fn**</sup>

Save tasks to file

### `pub fn load(self: *Manager) ManagerError!void`

<sup>**fn**</sup>

Load tasks from file

---

*Generated automatically by `zig build gendocs`*
