//! Process Management Tools for Agent Actions
//!
//! Provides tools for listing, inspecting, and managing OS processes:
//! - List running processes
//! - Kill a process by PID
//! - Spawn background processes

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const os = @import("../../../services/shared/os.zig");

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const ToolRegistry = tool.ToolRegistry;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ToolExecutionError = tool.ToolExecutionError;

// ============================================================================
// List Processes Tool
// ============================================================================

fn executeListProcesses(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    _ = args;
    var result = os.exec(ctx.allocator, "ps aux --sort=-%mem 2>/dev/null | head -50 || ps aux | head -50") catch {
        return ToolResult.fromError(ctx.allocator, "Failed to list processes");
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    }
    ctx.allocator.free(result.stdout);
    return ToolResult.fromError(ctx.allocator, "Process listing failed");
}

pub const list_processes_tool = Tool{
    .name = "list_processes",
    .description = "List running processes sorted by memory usage (top 50)",
    .parameters = &[_]Parameter{},
    .execute = &executeListProcesses,
};

// ============================================================================
// Kill Process Tool
// ============================================================================

fn executeKillProcess(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const pid_val = obj.get("pid") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: pid");
    };
    const pid_int: i64 = switch (pid_val) {
        .integer => |i| i,
        else => return ToolResult.fromError(ctx.allocator, "pid must be an integer"),
    };

    const signal_name = if (obj.get("signal")) |sig_val| switch (sig_val) {
        .string => |s| s,
        else => "terminate",
    } else "terminate";

    const signal: os.Signal = if (std.mem.eql(u8, signal_name, "kill"))
        .kill
    else if (std.mem.eql(u8, signal_name, "interrupt"))
        .interrupt
    else
        .terminate;

    const pid: os.Pid = @intCast(pid_int);
    os.kill(pid, signal) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to kill process");
    };

    const output = std.fmt.allocPrint(ctx.allocator, "Sent {s} signal to process {d}", .{
        signal_name,
        pid_int,
    }) catch return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const kill_process_tool = Tool{
    .name = "kill_process",
    .description = "Send a signal to a process by PID (default: terminate)",
    .parameters = &[_]Parameter{
        .{ .name = "pid", .type = .integer, .required = true, .description = "Process ID to signal" },
        .{ .name = "signal", .type = .string, .required = false, .description = "Signal: terminate (default), kill, interrupt" },
    },
    .execute = &executeKillProcess,
};

// ============================================================================
// Spawn Background Tool
// ============================================================================

fn executeSpawnBackground(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const cmd_val = obj.get("command") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: command");
    };
    const command = switch (cmd_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "command must be a string"),
    };

    // Safety: wrap with nohup and redirect output
    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "nohup {s} > /dev/null 2>&1 &", .{command}) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn background process");
    };
    defer result.deinit();

    const output = std.fmt.allocPrint(ctx.allocator, "Background process started: {s}", .{command}) catch
        return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const spawn_background_tool = Tool{
    .name = "spawn_background",
    .description = "Spawn a command as a background process (nohup)",
    .parameters = &[_]Parameter{
        .{ .name = "command", .type = .string, .required = true, .description = "Command to run in background" },
    },
    .execute = &executeSpawnBackground,
};

// ============================================================================
// Registration
// ============================================================================

pub const all_tools = [_]*const Tool{
    &list_processes_tool,
    &kill_process_tool,
    &spawn_background_tool,
};

pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "list_processes_tool creation" {
    try std.testing.expectEqualStrings("list_processes", list_processes_tool.name);
}

test "kill_process_tool creation" {
    try std.testing.expectEqualStrings("kill_process", kill_process_tool.name);
    try std.testing.expectEqual(@as(usize, 2), kill_process_tool.parameters.len);
}

test "spawn_background_tool creation" {
    try std.testing.expectEqualStrings("spawn_background", spawn_background_tool.name);
    try std.testing.expectEqual(@as(usize, 1), spawn_background_tool.parameters.len);
}

test "all_tools count" {
    try std.testing.expectEqual(@as(usize, 3), all_tools.len);
}
