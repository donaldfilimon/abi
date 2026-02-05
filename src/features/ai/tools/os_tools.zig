//! OS Tools for Agent Actions
//!
//! Provides cross-platform OS operations that agents can use:
//! - System information queries
//! - Environment variable access
//! - Shell command execution
//! - Clipboard operations
//! - System notifications
//! - Process management

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const os = @import("../../../services/shared/os.zig");

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const ToolRegistry = tool.ToolRegistry;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ParameterType = tool.ParameterType;
const ToolExecutionError = tool.ToolExecutionError;

// ============================================================================
// System Info Tool
// ============================================================================

fn executeSystemInfo(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    var info = os.getSystemInfo(ctx.allocator) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to get system information");
    };
    defer info.deinit();

    const output = std.fmt.allocPrint(ctx.allocator,
        \\System Information:
        \\  OS: {s} {s}
        \\  Hostname: {s}
        \\  Username: {s}
        \\  Home: {s}
        \\  Temp: {s}
        \\  CWD: {s}
        \\  CPUs: {d}
        \\  Page Size: {d}
        \\  Total Memory: {d} bytes
    , .{
        info.os_name,
        info.os_version,
        info.hostname,
        info.username,
        info.home_dir,
        info.temp_dir,
        info.current_dir,
        info.cpu_count,
        info.page_size,
        info.total_memory,
    }) catch return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const system_info_tool = Tool{
    .name = "system_info",
    .description = "Get comprehensive system information including OS, hostname, username, directories, CPU count, and memory",
    .parameters = &[_]Parameter{},
    .execute = &executeSystemInfo,
};

// ============================================================================
// Get Environment Variable Tool
// ============================================================================

fn executeGetEnv(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const name_val = obj.get("name") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: name");
    };

    const name = switch (name_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'name' must be a string"),
    };

    if (os.Env.get(name)) |value| {
        const output = std.fmt.allocPrint(ctx.allocator, "{s}={s}", .{ name, value }) catch return error.OutOfMemory;
        return ToolResult.init(ctx.allocator, true, output);
    } else {
        const output = std.fmt.allocPrint(ctx.allocator, "Environment variable '{s}' not found", .{name}) catch return error.OutOfMemory;
        return ToolResult.init(ctx.allocator, false, output);
    }
}

pub const get_env_tool = Tool{
    .name = "get_env",
    .description = "Get the value of an environment variable",
    .parameters = &[_]Parameter{
        .{
            .name = "name",
            .type = .string,
            .required = true,
            .description = "Name of the environment variable to retrieve",
        },
    },
    .execute = &executeGetEnv,
};

// ============================================================================
// Expand Environment Variables Tool
// ============================================================================

fn executeExpandEnv(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const input_val = obj.get("input") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: input");
    };

    const input = switch (input_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'input' must be a string"),
    };

    const expanded = os.Env.expand(ctx.allocator, input) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to expand: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    return ToolResult.init(ctx.allocator, true, expanded);
}

pub const expand_env_tool = Tool{
    .name = "expand_env",
    .description = "Expand environment variables in a string ($VAR, ${VAR}, or %VAR% on Windows)",
    .parameters = &[_]Parameter{
        .{
            .name = "input",
            .type = .string,
            .required = true,
            .description = "String containing environment variables to expand",
        },
    },
    .execute = &executeExpandEnv,
};

// ============================================================================
// Execute Shell Command Tool
// ============================================================================

fn executeShellCommand(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const command_val = obj.get("command") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: command");
    };

    const command = switch (command_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'command' must be a string"),
    };

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Command execution failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        const output = std.fmt.allocPrint(ctx.allocator, "Exit code: {d}\nStdout: {s}\nStderr: {s}", .{
            result.exit_code,
            result.stdout,
            result.stderr,
        }) catch return error.OutOfMemory;
        ctx.allocator.free(result.stdout);
        return ToolResult.init(ctx.allocator, false, output);
    }
}

pub const shell_command_tool = Tool{
    .name = "shell",
    .description = "Execute a shell command and capture its output",
    .parameters = &[_]Parameter{
        .{
            .name = "command",
            .type = .string,
            .required = true,
            .description = "Shell command to execute",
        },
    },
    .execute = &executeShellCommand,
};

// ============================================================================
// Clipboard Copy Tool
// ============================================================================

fn executeClipboardCopy(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const text_val = obj.get("text") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: text");
    };

    const text = switch (text_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'text' must be a string"),
    };

    os.Clipboard.copy(ctx.allocator, text) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Clipboard copy failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    const output = ctx.allocator.dupe(u8, "Text copied to clipboard") catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const clipboard_copy_tool = Tool{
    .name = "clipboard_copy",
    .description = "Copy text to the system clipboard",
    .parameters = &[_]Parameter{
        .{
            .name = "text",
            .type = .string,
            .required = true,
            .description = "Text to copy to clipboard",
        },
    },
    .execute = &executeClipboardCopy,
};

// ============================================================================
// Clipboard Paste Tool
// ============================================================================

fn executeClipboardPaste(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const text = os.Clipboard.paste(ctx.allocator) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Clipboard paste failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    return ToolResult.init(ctx.allocator, true, text);
}

pub const clipboard_paste_tool = Tool{
    .name = "clipboard_paste",
    .description = "Get text from the system clipboard",
    .parameters = &[_]Parameter{},
    .execute = &executeClipboardPaste,
};

// ============================================================================
// System Notification Tool
// ============================================================================

fn executeNotify(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const title_val = obj.get("title") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: title");
    };
    const message_val = obj.get("message") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: message");
    };

    const title = switch (title_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'title' must be a string"),
    };
    const message = switch (message_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'message' must be a string"),
    };

    os.notify(ctx.allocator, title, message) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Notification failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    const output = ctx.allocator.dupe(u8, "Notification sent") catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const notify_tool = Tool{
    .name = "notify",
    .description = "Send a system notification (desktop notification)",
    .parameters = &[_]Parameter{
        .{
            .name = "title",
            .type = .string,
            .required = true,
            .description = "Notification title",
        },
        .{
            .name = "message",
            .type = .string,
            .required = true,
            .description = "Notification message body",
        },
    },
    .execute = &executeNotify,
};

// ============================================================================
// Path Operations Tool
// ============================================================================

fn executePathOp(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const op_val = obj.get("operation") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: operation");
    };
    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };

    const operation = switch (op_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'operation' must be a string"),
    };
    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };

    var output: []u8 = undefined;

    if (std.mem.eql(u8, operation, "basename")) {
        output = ctx.allocator.dupe(u8, os.Path.basename(path)) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, operation, "dirname")) {
        output = ctx.allocator.dupe(u8, os.Path.dirname(path)) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, operation, "extension")) {
        output = ctx.allocator.dupe(u8, os.Path.extension(path)) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, operation, "is_absolute")) {
        const is_abs = os.Path.isAbsolute(path);
        output = ctx.allocator.dupe(u8, if (is_abs) "true" else "false") catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, operation, "normalize")) {
        output = os.Path.normalize(ctx.allocator, path) catch return error.OutOfMemory;
    } else {
        const msg = std.fmt.allocPrint(ctx.allocator, "Unknown operation: {s}. Valid: basename, dirname, extension, is_absolute, normalize", .{operation}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }

    return ToolResult.init(ctx.allocator, true, output);
}

pub const path_op_tool = Tool{
    .name = "path_op",
    .description = "Perform cross-platform path operations (basename, dirname, extension, is_absolute, normalize)",
    .parameters = &[_]Parameter{
        .{
            .name = "operation",
            .type = .string,
            .required = true,
            .description = "Path operation: basename, dirname, extension, is_absolute, normalize",
            .enum_values = &[_][]const u8{ "basename", "dirname", "extension", "is_absolute", "normalize" },
        },
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to operate on",
        },
    },
    .execute = &executePathOp,
};

// ============================================================================
// Get Process Info Tool
// ============================================================================

fn executeGetProcess(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const pid = os.getpid();
    const ppid = os.getppid();
    const is_tty = os.isatty();
    const is_ci = os.isCI();

    const output = std.fmt.allocPrint(ctx.allocator,
        \\Process Information:
        \\  PID: {d}
        \\  Parent PID: {d}
        \\  Is TTY: {s}
        \\  Is CI: {s}
        \\  Platform: {s}
        \\  Is Desktop: {s}
    , .{
        pid,
        ppid,
        if (is_tty) "yes" else "no",
        if (is_ci) "yes" else "no",
        os.getOsName(),
        if (os.is_desktop) "yes" else "no",
    }) catch return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const process_info_tool = Tool{
    .name = "process_info",
    .description = "Get current process information (PID, parent PID, TTY status, CI detection)",
    .parameters = &[_]Parameter{},
    .execute = &executeGetProcess,
};

// ============================================================================
// Get Home Directory Tool
// ============================================================================

fn executeGetHome(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const home = os.getHomeDir(ctx.allocator) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to get home directory: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    return ToolResult.init(ctx.allocator, true, home);
}

pub const get_home_tool = Tool{
    .name = "get_home",
    .description = "Get the user's home directory path",
    .parameters = &[_]Parameter{},
    .execute = &executeGetHome,
};

// ============================================================================
// Get Temp Directory Tool
// ============================================================================

fn executeGetTemp(ctx: *Context, _: json.Value) ToolExecutionError!ToolResult {
    const temp = os.getTempDir(ctx.allocator) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to get temp directory: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    return ToolResult.init(ctx.allocator, true, temp);
}

pub const get_temp_tool = Tool{
    .name = "get_temp",
    .description = "Get the system temporary directory path",
    .parameters = &[_]Parameter{},
    .execute = &executeGetTemp,
};

// ============================================================================
// Registration
// ============================================================================

/// All OS tools for easy registration
pub const all_tools = [_]*const Tool{
    &system_info_tool,
    &get_env_tool,
    &expand_env_tool,
    &shell_command_tool,
    &clipboard_copy_tool,
    &clipboard_paste_tool,
    &notify_tool,
    &path_op_tool,
    &process_info_tool,
    &get_home_tool,
    &get_temp_tool,
};

/// Register all OS tools with a registry
pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "system_info_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("system_info", system_info_tool.name);
}

test "all_tools count" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 11), all_tools.len);
}
