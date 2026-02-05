# Tasks Module
> **Last reviewed:** 2026-01-31

A unified task management system for personal tasks, project roadmap items, and distributed compute jobs.

## Overview

The tasks module provides a complete task tracking system with support for:
- **Task Creation & Management** - Add, update, delete, and complete tasks
- **Priority & Status Tracking** - 4 priority levels (low, normal, high, critical) and 5 statuses (pending, in_progress, completed, cancelled, blocked)
- **Organization** - Categorize tasks (personal, roadmap, compute, bug, feature)
- **Querying & Filtering** - List tasks with flexible filtering and sorting
- **Persistence** - Automatic saving/loading to JSON storage
- **Roadmap Integration** - Import roadmap items as tasks
- **Task Dependencies** - Block tasks that depend on other tasks

## Key Types

### Task
Represents a single task with metadata:
- `id: u64` - Unique identifier
- `title: []const u8` - Task title
- `description: ?[]const u8` - Optional description
- `status: Status` - Current status (pending, in_progress, completed, cancelled, blocked)
- `priority: Priority` - Priority level (low, normal, high, critical)
- `category: Category` - Task category
- `created_at: i64` - Unix timestamp of creation
- `updated_at: i64` - Unix timestamp of last update
- `due_date: ?i64` - Optional due date (Unix timestamp)
- `completed_at: ?i64` - Completion timestamp if completed
- `blocked_by: ?u64` - ID of blocking task if any
- `parent_id: ?u64` - ID of parent task if any

Methods:
- `isActionable()` - Check if task is pending or in progress
- `isOverdue()` - Check if task is past its due date

### Priority
```zig
pub const Priority = enum { low, normal, high, critical };
```

### Status
```zig
pub const Status = enum { pending, in_progress, completed, cancelled, blocked };
```

### Category
```zig
pub const Category = enum { personal, roadmap, compute, bug, feature };
```

### Manager
Main interface for task operations. Handles all task lifecycle, querying, and persistence.

## Basic Usage

```zig
const abi = @import("abi");
const std = @import("std");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Initialize manager
var manager = try abi.tasks.Manager.init(allocator, .{
    .storage_path = ".abi/tasks.json",
    .auto_save = true,
});
defer manager.deinit();

// Add a task
const task_id = try manager.add("Fix critical bug", .{
    .priority = .high,
    .category = .bug,
    .due_date = std.time.timestamp() + (24 * 3600), // Due tomorrow
});

// Start working on the task
try manager.start(task_id);

// Complete the task
try manager.complete(task_id);

// Query tasks
const tasks = try manager.list(allocator, .{
    .status = .pending,
    .priority = .high,
    .sort_by = .priority,
    .sort_descending = true,
});
defer allocator.free(tasks);

// Get statistics
const stats = manager.getStats();
std.debug.print("Total: {}, Pending: {}, Overdue: {}\n", .{
    stats.total, stats.pending, stats.overdue
});

// Cleanup
try manager.save();
```

## CLI Integration

The tasks module is integrated with the CLI via the `task` command:

```bash
# Add a task
zig build run -- task add "Fix authentication" --priority high --category bug

# List tasks
zig build run -- task list                    # All tasks
zig build run -- task list --status pending   # Filter by status
zig build run -- task list --priority high    # Filter by priority
zig build run -- task ls                      # Alias for list

# Show task details
zig build run -- task show <id>

# Update task status
zig build run -- task done <id>               # Mark completed
zig build run -- task start <id>              # Mark in progress
zig build run -- task cancel <id>             # Mark cancelled

# Set task properties
zig build run -- task priority <id> <level>   # Change priority
zig build run -- task category <id> <name>    # Change category
zig build run -- task due <id> <date>         # Set due date

# Get statistics
zig build run -- task stats
```

## Lifecycle Operations

### Adding Tasks
```zig
const id = try manager.add("Task title", .{
    .description = "Task description",
    .priority = .high,
    .category = .feature,
    .due_date = optional_unix_timestamp,
    .parent_id = optional_parent_task_id,
});
```

### Status Updates
```zig
try manager.setStatus(task_id, .in_progress);
try manager.complete(task_id);        // Shorthand for .completed
try manager.start(task_id);           // Shorthand for .in_progress
try manager.cancel(task_id);          // Shorthand for .cancelled
```

### Property Updates
```zig
try manager.setPriority(task_id, .high);
try manager.setCategory(task_id, .bug);
try manager.setDueDate(task_id, new_unix_timestamp);
try manager.setTitle(task_id, "New title");
try manager.setDescription(task_id, "New description");
try manager.setBlockedBy(task_id, blocker_task_id);
```

### Deletion
```zig
try manager.delete(task_id);
```

## Querying & Filtering

### Filter Options
```zig
pub const Filter = struct {
    status: ?Status = null,              // Filter by status
    priority: ?Priority = null,          // Filter by priority
    category: ?Category = null,          // Filter by category
    tag: ?[]const u8 = null,            // Filter by tag
    overdue_only: bool = false,         // Only overdue tasks
    parent_id: ?u64 = null,             // Only subtasks of parent
    sort_by: SortBy = .created,         // Sort key
    sort_descending: bool = true,       // Sort direction
};
```

### Query Examples
```zig
// High-priority pending tasks, sorted by due date
const urgent = try manager.list(allocator, .{
    .status = .pending,
    .priority = .high,
    .sort_by = .due_date,
});
defer allocator.free(urgent);

// All overdue tasks
const overdue = try manager.list(allocator, .{
    .overdue_only = true,
});
defer allocator.free(overdue);

// Get statistics
const stats = manager.getStats();
// stats contains: total, pending, in_progress, completed, cancelled, blocked, overdue
```

## Persistence

Tasks are automatically persisted to JSON when `auto_save` is enabled:

```zig
// Auto-save enabled (default)
var manager = try Manager.init(allocator, .{
    .auto_save = true,
});

// Manual save/load
try manager.save();
try manager.load();
```

The storage file location can be customized via `storage_path` in `ManagerConfig`.

## Roadmap Integration

Import roadmap items as tasks:

```zig
const count = try manager.importRoadmap();
std.debug.print("Imported {} roadmap items\n", .{count});
```

## Module Structure

| File | Purpose |
|------|---------|
| `mod.zig` | Main Manager API and lifecycle/query operations |
| `types.zig` | Core types (Task, Priority, Status, Category, Filter, etc.) |
| `lifecycle.zig` | Task creation, updates, and deletion |
| `querying.zig` | List, filter, and statistics operations |
| `persistence.zig` | JSON serialization and storage |
| `roadmap.zig` | Roadmap item integration |

## Error Handling

The module uses `ManagerError` for error handling:

```zig
pub const ManagerError = error{
    TaskNotFound,           // Task ID not found
    InvalidOperation,       // Invalid operation on task
    PersistenceFailed,      // Save/load failed
    ParseError,            // JSON parse error
} || std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.Dir.ReadFileAllocError || std.Io.File.Writer.Error;
```

## Testing

Run task module tests:

```bash
zig test src/services/tasks/mod.zig
zig test src/services/tasks/types.zig
zig test src/tasks.zig
```

Or run full test suite:

```bash
zig build test --summary all
```
