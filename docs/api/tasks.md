# tasks

> Task Management Module

Provides unified task tracking for personal tasks, project roadmap
items, and distributed compute jobs.

## Usage

```zig
const tasks = @import("tasks/mod.zig");

var manager = try tasks.Manager.init(allocator, .{});
defer manager.deinit();

const id = try manager.add("Fix bug", .{ .priority = .high });
try manager.complete(id);
```

**Source:** [`src/services/tasks/mod.zig`](../../src/services/tasks/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-manager"></a>`pub const Manager`

<sup>**const**</sup> | [source](../../src/services/tasks/mod.zig#L43)

Task Manager - main interface for task operations

### <a id="pub-fn-add-self-manager-title-const-u8-options-addoptions-managererror-u64"></a>`pub fn add(self: *Manager, title: []const u8, options: AddOptions) ManagerError!u64`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L116)

Add a new task

### <a id="pub-fn-get-self-const-manager-id-u64-task"></a>`pub fn get(self: *const Manager, id: u64) ?Task`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L131)

Get a task by ID

### <a id="pub-fn-setstatus-self-manager-id-u64-status-status-managererror-void"></a>`pub fn setStatus(self: *Manager, id: u64, status: Status) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L136)

Update task status

### <a id="pub-fn-complete-self-manager-id-u64-managererror-void"></a>`pub fn complete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L143)

Mark task as completed

### <a id="pub-fn-start-self-manager-id-u64-managererror-void"></a>`pub fn start(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L148)

Mark task as in progress

### <a id="pub-fn-cancel-self-manager-id-u64-managererror-void"></a>`pub fn cancel(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L153)

Cancel a task

### <a id="pub-fn-delete-self-manager-id-u64-managererror-void"></a>`pub fn delete(self: *Manager, id: u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L158)

Delete a task

### <a id="pub-fn-setduedate-self-manager-id-u64-due-date-i64-managererror-void"></a>`pub fn setDueDate(self: *Manager, id: u64, due_date: ?i64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L165)

Set due date for a task

### <a id="pub-fn-setblockedby-self-manager-id-u64-blocker-id-u64-managererror-void"></a>`pub fn setBlockedBy(self: *Manager, id: u64, blocker_id: ?u64) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L172)

Set task as blocked by another task

### <a id="pub-fn-setpriority-self-manager-id-u64-priority-priority-managererror-void"></a>`pub fn setPriority(self: *Manager, id: u64, priority: Priority) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L179)

Update task priority

### <a id="pub-fn-setcategory-self-manager-id-u64-category-category-managererror-void"></a>`pub fn setCategory(self: *Manager, id: u64, category: Category) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L186)

Update task category

### <a id="pub-fn-settitle-self-manager-id-u64-title-const-u8-managererror-void"></a>`pub fn setTitle(self: *Manager, id: u64, title: []const u8) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L193)

Update task title

### <a id="pub-fn-setdescription-self-manager-id-u64-description-const-u8-managererror-void"></a>`pub fn setDescription(self: *Manager, id: u64, description: ?[]const u8) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L200)

Update task description

### <a id="pub-fn-list-self-const-manager-allocator-std-mem-allocator-filter-filter-managererror-task"></a>`pub fn list(self: *const Manager, allocator: std.mem.Allocator, filter: Filter) ManagerError![]Task`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L211)

List tasks with optional filter and sorting

### <a id="pub-fn-getstats-self-const-manager-stats"></a>`pub fn getStats(self: *const Manager) Stats`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L216)

Get statistics

### <a id="pub-fn-save-self-manager-managererror-void"></a>`pub fn save(self: *Manager) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L225)

Save tasks to file

### <a id="pub-fn-load-self-manager-managererror-void"></a>`pub fn load(self: *Manager) ManagerError!void`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L231)

Load tasks from file

### <a id="pub-fn-importroadmap-self-manager-usize"></a>`pub fn importRoadmap(self: *Manager) !usize`

<sup>**fn**</sup> | [source](../../src/services/tasks/mod.zig#L247)

Import all roadmap items as tasks



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
