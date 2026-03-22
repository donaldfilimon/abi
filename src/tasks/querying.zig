//! Task Querying
//!
//! Filtering, sorting, and statistics for task lists.

const std = @import("std");
const types = @import("types.zig");

const Task = types.Task;
const Filter = types.Filter;
const SortBy = types.SortBy;
const Stats = types.Stats;
const ManagerError = types.ManagerError;

/// List tasks with optional filter and sorting
pub fn list(
    tasks: *const std.AutoHashMapUnmanaged(u64, Task),
    allocator: std.mem.Allocator,
    filter: Filter,
) ManagerError![]Task {
    var result = std.ArrayListUnmanaged(Task).empty;
    errdefer result.deinit(allocator);

    var iter = tasks.iterator();
    while (iter.next()) |entry| {
        const task = entry.value_ptr.*;
        if (matchesFilter(&task, filter)) {
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

/// Check if a task matches the given filter
pub fn matchesFilter(task: *const Task, filter: Filter) bool {
    if (filter.status) |s| if (task.status != s) return false;
    if (filter.priority) |p| if (task.priority != p) return false;
    if (filter.category) |c| if (task.category != c) return false;
    if (filter.overdue_only and !task.isOverdue()) return false;
    if (filter.parent_id) |pid| if (task.parent_id != pid) return false;
    return true;
}

/// Calculate statistics from tasks
pub fn getStats(tasks: *const std.AutoHashMapUnmanaged(u64, Task)) Stats {
    var stats = Stats{};
    var iter = tasks.iterator();
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

test {
    std.testing.refAllDecls(@This());
}
