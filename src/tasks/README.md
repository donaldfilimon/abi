# Task Management Module

**Status:** Complete

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
