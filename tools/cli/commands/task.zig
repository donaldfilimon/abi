//! Task CLI command.
//!
//! Provides task management operations including add, list, show, complete,
//! start, cancel, delete, stats, and roadmap import.
//!
//! Commands:
//! - task add <title> - Add a new task
//! - task list / task ls - List tasks with optional filters
//! - task show <id> - Show task details
//! - task done <id> - Mark task completed
//! - task start <id> - Mark task in-progress
//! - task cancel <id> - Cancel task
//! - task delete <id> / task rm <id> - Delete task
//! - task stats - Show statistics
//! - task import-roadmap - Import roadmap items

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const tasks = abi.tasks;
const time_utils = abi.internal.utils.time;

/// Run the task command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "add")) {
        try runAdd(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "list") or std.mem.eql(u8, command, "ls")) {
        try runList(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "show")) {
        try runShow(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "done")) {
        try runDone(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "start")) {
        try runStart(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "cancel")) {
        try runCancel(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "delete") or std.mem.eql(u8, command, "rm")) {
        try runDelete(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "stats")) {
        try runStats(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "import-roadmap")) {
        try runImportRoadmap(allocator);
        return;
    }

    utils.output.printError("Unknown task command: {s}", .{command});
    printHelp();
}

fn runAdd(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var title: ?[]const u8 = null;
    var description: ?[]const u8 = null;
    var priority: tasks.Priority = .normal;
    var category: tasks.Category = .personal;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &.{ "--priority", "-p" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                priority = tasks.Priority.fromString(val) orelse {
                    utils.output.printError("Invalid priority: {s}. Valid: low, normal, high, critical", .{val});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--category", "-c" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                category = tasks.Category.fromString(val) orelse {
                    utils.output.printError("Invalid category: {s}. Valid: personal, roadmap, compute, bug, feature", .{val});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--desc", "--description", "-d" })) {
            if (i < args.len) {
                description = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        // Positional: title
        if (title == null) {
            title = arg;
        }
    }

    if (title == null) {
        utils.output.printError("Task title is required", .{});
        std.debug.print("Usage: abi task add <title> [--priority <p>] [--category <c>] [--desc <text>]\n", .{});
        return;
    }

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    const id = manager.add(title.?, .{
        .description = description,
        .priority = priority,
        .category = category,
    }) catch |err| {
        utils.output.printError("Failed to add task: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Created task #{d}: {s}", .{ id, title.? });
}

fn runList(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var filter = tasks.Filter{};

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &.{ "--status", "-s" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                filter.status = tasks.Status.fromString(val) orelse {
                    utils.output.printError("Invalid status: {s}. Valid: pending, in_progress, completed, cancelled, blocked", .{val});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--priority", "-p" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                filter.priority = tasks.Priority.fromString(val) orelse {
                    utils.output.printError("Invalid priority: {s}. Valid: low, normal, high, critical", .{val});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--category", "-c" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                filter.category = tasks.Category.fromString(val) orelse {
                    utils.output.printError("Invalid category: {s}. Valid: personal, roadmap, compute, bug, feature", .{val});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--overdue", "-o" })) {
            filter.overdue_only = true;
            continue;
        }
    }

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    const task_list = manager.list(allocator, filter) catch |err| {
        utils.output.printError("Failed to list tasks: {t}", .{err});
        return;
    };
    defer allocator.free(task_list);

    if (task_list.len == 0) {
        utils.output.printInfo("No tasks found", .{});
        return;
    }

    std.debug.print("\n", .{});
    utils.output.printHeader("Tasks");
    std.debug.print("\n", .{});

    // Print table header
    std.debug.print("  {s:<6} {s:<12} {s:<10} {s:<10} {s}\n", .{ "ID", "STATUS", "PRIORITY", "CATEGORY", "TITLE" });
    std.debug.print("  {s:-<6} {s:-<12} {s:-<10} {s:-<10} {s:-<40}\n", .{ "", "", "", "", "" });

    for (task_list) |task| {
        const status_str = task.status.toString();
        const priority_str = task.priority.toString();
        const category_str = task.category.toString();

        // Truncate title if too long
        const max_title_len: usize = 50;
        const title_display = if (task.title.len > max_title_len)
            task.title[0..max_title_len]
        else
            task.title;

        // Color based on status
        const status_color = switch (task.status) {
            .pending => utils.output.Color.yellow,
            .in_progress => utils.output.Color.cyan,
            .completed => utils.output.Color.green,
            .cancelled => utils.output.Color.dim,
            .blocked => utils.output.Color.red,
        };

        std.debug.print("  {d:<6} {s}{s:<12}{s} {s:<10} {s:<10} {s}", .{
            task.id,
            status_color,
            status_str,
            utils.output.Color.reset,
            priority_str,
            category_str,
            title_display,
        });

        if (task.title.len > max_title_len) {
            std.debug.print("...", .{});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n  Total: {d} task(s)\n\n", .{task_list.len});
}

fn runShow(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Task ID required", .{});
        std.debug.print("Usage: abi task show <id>\n", .{});
        return;
    }

    const id_str = std.mem.sliceTo(args[0], 0);
    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    const task = manager.get(id) orelse {
        utils.output.printError("Task #{d} not found", .{id});
        return;
    };

    std.debug.print("\n", .{});
    utils.output.printHeader("Task Details");
    std.debug.print("\n", .{});

    utils.output.printKeyValue("ID", std.fmt.allocPrint(allocator, "{d}", .{task.id}) catch "?");
    utils.output.printKeyValue("Title", task.title);
    utils.output.printKeyValue("Status", task.status.toString());
    utils.output.printKeyValue("Priority", task.priority.toString());
    utils.output.printKeyValue("Category", task.category.toString());

    if (task.description) |desc| {
        utils.output.printKeyValue("Description", desc);
    }

    // Format timestamps
    const created_str = formatTimestamp(allocator, task.created_at) catch "unknown";
    defer if (created_str.len > 0 and !std.mem.eql(u8, created_str, "unknown")) allocator.free(created_str);
    utils.output.printKeyValue("Created", created_str);

    const updated_str = formatTimestamp(allocator, task.updated_at) catch "unknown";
    defer if (updated_str.len > 0 and !std.mem.eql(u8, updated_str, "unknown")) allocator.free(updated_str);
    utils.output.printKeyValue("Updated", updated_str);

    if (task.due_date) |due| {
        const due_str = formatTimestamp(allocator, due) catch "unknown";
        defer if (due_str.len > 0 and !std.mem.eql(u8, due_str, "unknown")) allocator.free(due_str);
        utils.output.printKeyValue("Due", due_str);
        if (task.isOverdue()) {
            std.debug.print("  {s}OVERDUE{s}\n", .{ utils.output.Color.red, utils.output.Color.reset });
        }
    }

    if (task.completed_at) |comp| {
        const comp_str = formatTimestamp(allocator, comp) catch "unknown";
        defer if (comp_str.len > 0 and !std.mem.eql(u8, comp_str, "unknown")) allocator.free(comp_str);
        utils.output.printKeyValue("Completed", comp_str);
    }

    if (task.parent_id) |parent| {
        const parent_str = std.fmt.allocPrint(allocator, "{d}", .{parent}) catch "?";
        defer allocator.free(parent_str);
        utils.output.printKeyValue("Parent Task", parent_str);
    }

    std.debug.print("\n", .{});
}

fn runDone(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Task ID required", .{});
        std.debug.print("Usage: abi task done <id>\n", .{});
        return;
    }

    const id_str = std.mem.sliceTo(args[0], 0);
    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    manager.complete(id) catch |err| {
        if (err == error.TaskNotFound) {
            utils.output.printError("Task #{d} not found", .{id});
        } else {
            utils.output.printError("Failed to complete task: {t}", .{err});
        }
        return;
    };

    utils.output.printSuccess("Task #{d} marked as completed", .{id});
}

fn runStart(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Task ID required", .{});
        std.debug.print("Usage: abi task start <id>\n", .{});
        return;
    }

    const id_str = std.mem.sliceTo(args[0], 0);
    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    manager.start(id) catch |err| {
        if (err == error.TaskNotFound) {
            utils.output.printError("Task #{d} not found", .{id});
        } else {
            utils.output.printError("Failed to start task: {t}", .{err});
        }
        return;
    };

    utils.output.printSuccess("Task #{d} marked as in-progress", .{id});
}

fn runCancel(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Task ID required", .{});
        std.debug.print("Usage: abi task cancel <id>\n", .{});
        return;
    }

    const id_str = std.mem.sliceTo(args[0], 0);
    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    manager.cancel(id) catch |err| {
        if (err == error.TaskNotFound) {
            utils.output.printError("Task #{d} not found", .{id});
        } else {
            utils.output.printError("Failed to cancel task: {t}", .{err});
        }
        return;
    };

    utils.output.printSuccess("Task #{d} cancelled", .{id});
}

fn runDelete(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Task ID required", .{});
        std.debug.print("Usage: abi task delete <id>\n", .{});
        return;
    }

    const id_str = std.mem.sliceTo(args[0], 0);
    const id = std.fmt.parseInt(u64, id_str, 10) catch {
        utils.output.printError("Invalid task ID: {s}", .{id_str});
        return;
    };

    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    manager.delete(id) catch |err| {
        if (err == error.TaskNotFound) {
            utils.output.printError("Task #{d} not found", .{id});
        } else {
            utils.output.printError("Failed to delete task: {t}", .{err});
        }
        return;
    };

    utils.output.printSuccess("Task #{d} deleted", .{id});
}

fn runStats(allocator: std.mem.Allocator) !void {
    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    const stats = manager.getStats();

    std.debug.print("\n", .{});
    utils.output.printHeader("Task Statistics");
    std.debug.print("\n", .{});

    std.debug.print("  Total:       {d}\n", .{stats.total});
    std.debug.print("  {s}Pending:{s}     {d}\n", .{ utils.output.Color.yellow, utils.output.Color.reset, stats.pending });
    std.debug.print("  {s}In Progress:{s} {d}\n", .{ utils.output.Color.cyan, utils.output.Color.reset, stats.in_progress });
    std.debug.print("  {s}Completed:{s}   {d}\n", .{ utils.output.Color.green, utils.output.Color.reset, stats.completed });
    std.debug.print("  {s}Cancelled:{s}   {d}\n", .{ utils.output.Color.dim, utils.output.Color.reset, stats.cancelled });
    std.debug.print("  {s}Blocked:{s}     {d}\n", .{ utils.output.Color.red, utils.output.Color.reset, stats.blocked });
    std.debug.print("  {s}Overdue:{s}     {d}\n", .{ utils.output.Color.red, utils.output.Color.reset, stats.overdue });

    if (stats.total > 0) {
        const completion_rate = @as(f64, @floatFromInt(stats.completed)) / @as(f64, @floatFromInt(stats.total)) * 100.0;
        std.debug.print("\n  Completion rate: {d:.1}%\n", .{completion_rate});
    }

    std.debug.print("\n", .{});
}

fn runImportRoadmap(allocator: std.mem.Allocator) !void {
    var manager = tasks.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("Failed to initialize task manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    const count = tasks.roadmap.importAll(&manager) catch |err| {
        utils.output.printError("Failed to import roadmap items: {t}", .{err});
        return;
    };

    if (count == 0) {
        utils.output.printInfo("No new roadmap items to import (all items already exist)", .{});
    } else {
        utils.output.printSuccess("Imported {d} roadmap item(s)", .{count});
    }
}

fn formatTimestamp(allocator: std.mem.Allocator, timestamp: i64) ![]const u8 {
    // Simple timestamp formatting - just show as relative time or ISO-like format
    const now = time_utils.unixSeconds();
    const diff = now - timestamp;

    if (diff < 0) {
        // Future date
        return std.fmt.allocPrint(allocator, "in {d}s", .{-diff});
    } else if (diff < 60) {
        return std.fmt.allocPrint(allocator, "{d}s ago", .{diff});
    } else if (diff < 3600) {
        return std.fmt.allocPrint(allocator, "{d}m ago", .{@divFloor(diff, 60)});
    } else if (diff < 86400) {
        return std.fmt.allocPrint(allocator, "{d}h ago", .{@divFloor(diff, 3600)});
    } else {
        return std.fmt.allocPrint(allocator, "{d}d ago", .{@divFloor(diff, 86400)});
    }
}

fn printHelp() void {
    const help_text =
        \\Usage: abi task <command> [options]
        \\
        \\Task management for personal tasks, project items, and compute jobs.
        \\
        \\Commands:
        \\  add <title>          Add a new task
        \\  list, ls             List tasks with optional filters
        \\  show <id>            Show task details
        \\  done <id>            Mark task as completed
        \\  start <id>           Mark task as in-progress
        \\  cancel <id>          Cancel a task
        \\  delete <id>, rm <id> Delete a task
        \\  stats                Show task statistics
        \\  import-roadmap       Import roadmap items as tasks
        \\  help                 Show this help message
        \\
        \\Add Options:
        \\  -p, --priority <p>   Priority: low, normal, high, critical (default: normal)
        \\  -c, --category <c>   Category: personal, roadmap, compute, bug, feature
        \\  -d, --desc <text>    Task description
        \\
        \\List Filters:
        \\  -s, --status <s>     Filter by status: pending, in_progress, completed,
        \\                       cancelled, blocked
        \\  -p, --priority <p>   Filter by priority
        \\  -c, --category <c>   Filter by category
        \\  -o, --overdue        Show only overdue tasks
        \\
        \\Examples:
        \\  abi task add "Fix login bug" --priority high --category bug
        \\  abi task add "Write tests" -p normal -c feature -d "Unit tests for auth"
        \\  abi task list
        \\  abi task ls --status pending --priority high
        \\  abi task ls -s in_progress
        \\  abi task show 1
        \\  abi task start 1
        \\  abi task done 1
        \\  abi task rm 1
        \\  abi task stats
        \\  abi task import-roadmap
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
