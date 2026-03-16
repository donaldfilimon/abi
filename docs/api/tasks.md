---
title: tasks API
purpose: Generated API reference for tasks
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# tasks

> Task Management Module

Provides unified task tracking for personal tasks, project roadmap
items, and distributed compute jobs.

## Usage

```zig
const tasks = @import("tasks");

var manager = try tasks.Manager.init(allocator, .{});
defer manager.deinit();

const id = try manager.add("Fix bug", .{ .priority = .high });
try manager.complete(id);
```

**Source:** [`src/services/tasks/mod.zig`](../../src/services/tasks/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-task"></a>`pub const Task`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L30)

A task representing a unit of work.

### <a id="pub-const-priority"></a>`pub const Priority`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L32)

Task priority levels.

### <a id="pub-const-status"></a>`pub const Status`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L34)

Task lifecycle statuses.

### <a id="pub-const-category"></a>`pub const Category`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L36)

Task categories.

### <a id="pub-const-filter"></a>`pub const Filter`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L38)

Filter criteria for listing tasks.

### <a id="pub-const-sortby"></a>`pub const SortBy`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L40)

Sorting criteria for task lists.

### <a id="pub-const-stats"></a>`pub const Stats`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L42)

Task statistics and metrics.

### <a id="pub-const-managerconfig"></a>`pub const ManagerConfig`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L44)

Configuration for the Task Manager.

### <a id="pub-const-addoptions"></a>`pub const AddOptions`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L46)

Options for adding a new task.

### <a id="pub-const-managererror"></a>`pub const ManagerError`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L48)

Error set for task operations.

### <a id="pub-const-roadmapitem"></a>`pub const RoadmapItem`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L50)

An item from the project roadmap.

### <a id="pub-const-manager"></a>`pub const Manager`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L53)

Task Manager - main interface for task operations

### <a id="pub-fn-add-self-manager-title-const-u8-options-addoptions-managererror-u64"></a>`pub fn add(self: *Manager, title: []const u8, options: AddOptions) ManagerError!u64`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L126)

Add a new task

### <a id="pub-fn-get-self-const-manager-id-u64-task"></a>`pub fn get(self: *const Manager, id: u64) ?Task`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L141)

Get a task by ID

### <a id="pub-fn-setstatus-self-manager-id-u64-status-status-managererror-void"></a>`pub fn setStatus(self: *Manager, id: u64, status: Status) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L146)

Update task status

### <a id="pub-fn-complete-self-manager-id-u64-managererror-void"></a>`pub fn complete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L153)

Mark task as completed

### <a id="pub-fn-start-self-manager-id-u64-managererror-void"></a>`pub fn start(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L158)

Mark task as in progress

### <a id="pub-fn-cancel-self-manager-id-u64-managererror-void"></a>`pub fn cancel(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L163)

Cancel a task

### <a id="pub-fn-delete-self-manager-id-u64-managererror-void"></a>`pub fn delete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L168)

Delete a task

### <a id="pub-fn-setduedate-self-manager-id-u64-due-date-i64-managererror-void"></a>`pub fn setDueDate(self: *Manager, id: u64, due_date: ?i64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L175)

Set due date for a task

### <a id="pub-fn-setblockedby-self-manager-id-u64-blocker-id-u64-managererror-void"></a>`pub fn setBlockedBy(self: *Manager, id: u64, blocker_id: ?u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L182)

Set task as blocked by another task

### <a id="pub-fn-setpriority-self-manager-id-u64-priority-priority-managererror-void"></a>`pub fn setPriority(self: *Manager, id: u64, priority: Priority) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L189)

Update task priority

### <a id="pub-fn-setcategory-self-manager-id-u64-category-category-managererror-void"></a>`pub fn setCategory(self: *Manager, id: u64, category: Category) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L196)

Update task category

### <a id="pub-fn-settitle-self-manager-id-u64-title-const-u8-managererror-void"></a>`pub fn setTitle(self: *Manager, id: u64, title: []const u8) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L203)

Update task title

### <a id="pub-fn-setdescription-self-manager-id-u64-description-const-u8-managererror-void"></a>`pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L210)

Update task description

### <a id="pub-fn-list-self-const-manager-allocator-std-mem-allocator-filter-filter-managererror-task"></a>`pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L221)

List tasks with optional filter and sorting

### <a id="pub-fn-getstats-self-const-manager-stats"></a>`pub fn getStats(self: *const Manager) Stats`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L226)

Get statistics

### <a id="pub-fn-save-self-manager-managererror-void"></a>`pub fn save(self: *Manager) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L235)

Save tasks to file

### <a id="pub-fn-load-self-manager-managererror-void"></a>`pub fn load(self: *Manager) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L241)

Load tasks from file

### <a id="pub-fn-importroadmap-self-manager-usize"></a>`pub fn importRoadmap(self: *Manager) !usize`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L257)

Import all roadmap items as tasks



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
