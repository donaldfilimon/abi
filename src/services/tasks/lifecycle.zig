//! Task Lifecycle
//!
//! CRUD operations and status management for tasks.

const std = @import("std");
const types = @import("types.zig");
const persistence = @import("persistence.zig");

const Task = types.Task;
const Status = types.Status;
const Priority = types.Priority;
const Category = types.Category;
const AddOptions = types.AddOptions;
const ManagerError = types.ManagerError;

const time_utils = @import("../shared/utils.zig");

/// Add a new task
pub fn add(
    allocator: std.mem.Allocator,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    strings: *std.ArrayListUnmanaged([]u8),
    next_id: *u64,
    title: []const u8,
    options: AddOptions,
) ManagerError!u64 {
    const now = time_utils.unixSeconds();
    const id = next_id.*;
    next_id.* += 1;

    // Duplicate strings we need to own
    const owned_title = try persistence.dupeString(allocator, strings, title);
    const owned_desc = if (options.description) |d| try persistence.dupeString(allocator, strings, d) else null;

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

    try tasks.put(allocator, id, task);
    return id;
}

/// Get a task by ID
pub fn get(tasks: *const std.AutoHashMapUnmanaged(u64, Task), id: u64) ?Task {
    return tasks.get(id);
}

/// Delete a task
pub fn delete(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
) ManagerError!void {
    if (!tasks.remove(id)) return error.TaskNotFound;
}

/// Update task status
pub fn setStatus(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
    status: Status,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    ptr.status = status;
    ptr.updated_at = time_utils.unixSeconds();
    if (status == .completed) {
        ptr.completed_at = ptr.updated_at;
    }
}

/// Mark task as completed
pub fn complete(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
) ManagerError!void {
    return setStatus(tasks, id, .completed);
}

/// Mark task as in progress
pub fn start(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
) ManagerError!void {
    return setStatus(tasks, id, .in_progress);
}

/// Cancel a task
pub fn cancel(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
) ManagerError!void {
    return setStatus(tasks, id, .cancelled);
}

/// Set due date for a task
pub fn setDueDate(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
    due_date: ?i64,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    ptr.due_date = due_date;
    ptr.updated_at = time_utils.unixSeconds();
}

/// Set task as blocked by another task
pub fn setBlockedBy(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
    blocker_id: ?u64,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    // Verify blocker exists if set
    if (blocker_id) |bid| {
        if (!tasks.contains(bid)) return error.TaskNotFound;
    }
    ptr.blocked_by = blocker_id;
    ptr.updated_at = time_utils.unixSeconds();
    // Also set status to blocked if a blocker is specified
    if (blocker_id != null) {
        ptr.status = .blocked;
    } else if (ptr.status == .blocked) {
        ptr.status = .pending;
    }
}

/// Update task priority
pub fn setPriority(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
    priority: Priority,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    ptr.priority = priority;
    ptr.updated_at = time_utils.unixSeconds();
}

/// Update task category
pub fn setCategory(
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    id: u64,
    category: Category,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    ptr.category = category;
    ptr.updated_at = time_utils.unixSeconds();
}

/// Update task title
pub fn setTitle(
    allocator: std.mem.Allocator,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    strings: *std.ArrayListUnmanaged([]u8),
    id: u64,
    title: []const u8,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    const owned_title = try persistence.dupeString(allocator, strings, title);
    ptr.title = owned_title;
    ptr.updated_at = time_utils.unixSeconds();
}

/// Update task description
pub fn setDescription(
    allocator: std.mem.Allocator,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    strings: *std.ArrayListUnmanaged([]u8),
    id: u64,
    description: ?[]const u8,
) ManagerError!void {
    const ptr = tasks.getPtr(id) orelse return error.TaskNotFound;
    const owned_desc = if (description) |d| try persistence.dupeString(allocator, strings, d) else null;
    ptr.description = owned_desc;
    ptr.updated_at = time_utils.unixSeconds();
}

test {
    std.testing.refAllDecls(@This());
}
