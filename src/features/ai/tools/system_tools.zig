//! System and Service Tools for Agent Actions
//!
//! Provides tools for system administration tasks:
//! - Service status queries
//! - Package/dependency listing
//! - System package inspection
//! - File watching

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
// Service Status Tool
// ============================================================================

fn executeServiceStatus(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const svc_val = obj.get("service");
    if (svc_val) |sv| {
        const service_name = switch (sv) {
            .string => |s| s,
            else => return ToolResult.fromError(ctx.allocator, "service must be a string"),
        };

        // Validate service name to prevent injection
        for (service_name) |c| {
            if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_' and c != '.') {
                return ToolResult.fromError(ctx.allocator, "Invalid service name characters");
            }
        }

        const cmd = std.fmt.allocPrint(ctx.allocator, "systemctl status {s} 2>/dev/null || launchctl list | grep {s} 2>/dev/null || echo 'Service manager not available'", .{ service_name, service_name }) catch
            return error.OutOfMemory;
        defer ctx.allocator.free(cmd);

        var result = os.exec(ctx.allocator, cmd) catch {
            return ToolResult.fromError(ctx.allocator, "Failed to query service status");
        };
        ctx.allocator.free(result.stderr);

        return ToolResult.init(ctx.allocator, true, result.stdout);
    }

    // No service specified — list all
    var result = os.exec(ctx.allocator, "systemctl list-units --type=service --no-pager 2>/dev/null | head -30 || launchctl list 2>/dev/null | head -30 || echo 'Service manager not available'") catch {
        return ToolResult.fromError(ctx.allocator, "Failed to list services");
    };
    ctx.allocator.free(result.stderr);

    return ToolResult.init(ctx.allocator, true, result.stdout);
}

pub const service_status_tool = Tool{
    .name = "service_status",
    .description = "Query systemd/launchd service status (or list all services)",
    .parameters = &[_]Parameter{
        .{ .name = "service", .type = .string, .required = false, .description = "Service name (omit to list all)" },
    },
    .execute = &executeServiceStatus,
};

// ============================================================================
// List Dependencies Tool
// ============================================================================

fn executeListDeps(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const path = if (obj.get("path")) |pv| switch (pv) {
        .string => |s| s,
        else => ".",
    } else ".";

    // Try multiple project types
    const cmd = std.fmt.allocPrint(ctx.allocator,
        \\test -f {s}/build.zig.zon && cat {s}/build.zig.zon || \
        \\test -f {s}/package.json && cat {s}/package.json | head -50 || \
        \\test -f {s}/Cargo.toml && cat {s}/Cargo.toml | head -50 || \
        \\test -f {s}/requirements.txt && cat {s}/requirements.txt || \
        \\test -f {s}/go.mod && cat {s}/go.mod || \
        \\echo 'No recognized dependency file found'
    , .{ path, path, path, path, path, path, path, path, path, path }) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(cmd);

    var result = os.exec(ctx.allocator, cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to read dependency files");
    };
    ctx.allocator.free(result.stderr);

    return ToolResult.init(ctx.allocator, true, result.stdout);
}

pub const list_deps_tool = Tool{
    .name = "list_deps",
    .description = "List project dependencies (build.zig.zon, package.json, Cargo.toml, etc.)",
    .parameters = &[_]Parameter{
        .{ .name = "path", .type = .string, .required = false, .description = "Project root path (default: current directory)" },
    },
    .execute = &executeListDeps,
};

// ============================================================================
// System Packages Tool
// ============================================================================

fn executeSystemPackages(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    _ = args;
    var result = os.exec(ctx.allocator, "dpkg -l 2>/dev/null | tail -40 || brew list 2>/dev/null | head -40 || pacman -Q 2>/dev/null | head -40 || rpm -qa 2>/dev/null | head -40 || echo 'No package manager detected'") catch {
        return ToolResult.fromError(ctx.allocator, "Failed to list system packages");
    };
    ctx.allocator.free(result.stderr);

    return ToolResult.init(ctx.allocator, true, result.stdout);
}

pub const system_packages_tool = Tool{
    .name = "system_packages",
    .description = "List installed system packages (dpkg/brew/pacman/rpm)",
    .parameters = &[_]Parameter{},
    .execute = &executeSystemPackages,
};

// ============================================================================
// Watch File Tool
// ============================================================================

fn executeWatchFile(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };
    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "path must be a string"),
    };

    if (tool.hasPathTraversal(path)) {
        return ToolResult.fromError(ctx.allocator, "Path traversal detected");
    }

    // Short timeout to avoid blocking forever
    const cmd = std.fmt.allocPrint(ctx.allocator, "timeout 5 inotifywait -e modify,create,delete {s} 2>/dev/null || timeout 5 fswatch --one-event {s} 2>/dev/null || echo 'File watching not available (install inotify-tools or fswatch)'", .{ path, path }) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(cmd);

    var result = os.exec(ctx.allocator, cmd) catch {
        return ToolResult.fromError(ctx.allocator, "File watch failed");
    };
    ctx.allocator.free(result.stderr);

    return ToolResult.init(ctx.allocator, true, result.stdout);
}

pub const watch_file_tool = Tool{
    .name = "watch_file",
    .description = "Watch a file or directory for changes (5 second timeout)",
    .parameters = &[_]Parameter{
        .{ .name = "path", .type = .string, .required = true, .description = "File or directory path to watch" },
    },
    .execute = &executeWatchFile,
};

// ============================================================================
// Registration
// ============================================================================

pub const all_tools = [_]*const Tool{
    &service_status_tool,
    &list_deps_tool,
    &system_packages_tool,
    &watch_file_tool,
};

pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "service_status_tool creation" {
    try std.testing.expectEqualStrings("service_status", service_status_tool.name);
    try std.testing.expectEqual(@as(usize, 1), service_status_tool.parameters.len);
}

test "list_deps_tool creation" {
    try std.testing.expectEqualStrings("list_deps", list_deps_tool.name);
}

test "system_packages_tool creation" {
    try std.testing.expectEqualStrings("system_packages", system_packages_tool.name);
}

test "watch_file_tool creation" {
    try std.testing.expectEqualStrings("watch_file", watch_file_tool.name);
    try std.testing.expectEqual(@as(usize, 1), watch_file_tool.parameters.len);
}

test "all_tools count" {
    try std.testing.expectEqual(@as(usize, 4), all_tools.len);
}
