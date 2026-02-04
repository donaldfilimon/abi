//! Search Tools for Agent Actions
//!
//! Provides search operations that agents can use:
//! - Grep (pattern matching in files)
//! - Find (locate files by name)
//!
//! Note: These tools use shell commands internally for Zig 0.16 compatibility.

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const os = @import("../../shared/os.zig");

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const ToolRegistry = tool.ToolRegistry;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ParameterType = tool.ParameterType;
const ToolExecutionError = tool.ToolExecutionError;

// ============================================================================
// Grep Tool
// ============================================================================

fn executeGrep(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const pattern_val = obj.get("pattern") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: pattern");
    };

    const pattern = switch (pattern_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'pattern' must be a string"),
    };

    // Get optional path (default to current directory)
    var search_path: []const u8 = ctx.working_directory;
    var path_allocated = false;
    if (obj.get("path")) |path_val| {
        switch (path_val) {
            .string => |s| {
                if (std.fs.path.isAbsolute(s)) {
                    search_path = s;
                } else {
                    search_path = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, s }) catch return error.OutOfMemory;
                    path_allocated = true;
                }
            },
            else => {},
        }
    }
    defer if (path_allocated) ctx.allocator.free(search_path);

    // Get optional file glob
    var file_glob: []const u8 = "*";
    if (obj.get("glob")) |glob_val| {
        switch (glob_val) {
            .string => |s| file_glob = s,
            else => {},
        }
    }

    // Get context lines
    var context_flag: []const u8 = "";
    var context_allocated = false;
    if (obj.get("context_before")) |cb| {
        switch (cb) {
            .integer => |i| {
                if (i > 0) {
                    context_flag = std.fmt.allocPrint(ctx.allocator, "-B {d} ", .{i}) catch return error.OutOfMemory;
                    context_allocated = true;
                }
            },
            else => {},
        }
    }
    if (obj.get("context_after")) |ca| {
        switch (ca) {
            .integer => |i| {
                if (i > 0) {
                    const after_flag = std.fmt.allocPrint(ctx.allocator, "-A {d} ", .{i}) catch return error.OutOfMemory;
                    if (context_allocated) {
                        const combined = std.fmt.allocPrint(ctx.allocator, "{s}{s}", .{ context_flag, after_flag }) catch return error.OutOfMemory;
                        ctx.allocator.free(context_flag);
                        ctx.allocator.free(after_flag);
                        context_flag = combined;
                    } else {
                        context_flag = after_flag;
                        context_allocated = true;
                    }
                }
            },
            else => {},
        }
    }
    defer if (context_allocated) ctx.allocator.free(context_flag);

    // Case insensitive flag
    var case_flag: []const u8 = "";
    if (obj.get("case_insensitive")) |ci| {
        switch (ci) {
            .bool => |b| {
                if (b) case_flag = "-i ";
            },
            else => {},
        }
    }

    // Build grep command - use grep -rn for recursive search with line numbers
    const command = std.fmt.allocPrint(ctx.allocator, "grep -rn {s}{s}--include=\"{s}\" \"{s}\" \"{s}\" 2>/dev/null | head -500", .{
        case_flag,
        context_flag,
        file_glob,
        pattern,
        search_path,
    }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    const result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Grep failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    // grep returns exit code 1 when no matches found (not an error)
    if (result.exit_code == 0 or result.exit_code == 1) {
        if (result.stdout.len == 0) {
            ctx.allocator.free(result.stdout);
            const output = ctx.allocator.dupe(u8, "No matches found") catch return error.OutOfMemory;
            return ToolResult.init(ctx.allocator, true, output);
        }
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Grep failed: {s}", .{result.stderr}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const grep_tool = Tool{
    .name = "grep",
    .description = "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
    .parameters = &[_]Parameter{
        .{
            .name = "pattern",
            .type = .string,
            .required = true,
            .description = "Pattern to search for (supports regex)",
        },
        .{
            .name = "path",
            .type = .string,
            .required = false,
            .description = "File or directory to search in (default: working directory)",
        },
        .{
            .name = "glob",
            .type = .string,
            .required = false,
            .description = "File glob pattern to filter files (e.g., '*.zig')",
        },
        .{
            .name = "context_before",
            .type = .integer,
            .required = false,
            .description = "Number of lines to show before each match",
        },
        .{
            .name = "context_after",
            .type = .integer,
            .required = false,
            .description = "Number of lines to show after each match",
        },
        .{
            .name = "case_insensitive",
            .type = .boolean,
            .required = false,
            .description = "If true, search case-insensitively (default: false)",
        },
    },
    .execute = &executeGrep,
};

// ============================================================================
// Find Tool
// ============================================================================

fn executeFind(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const name_val = obj.get("name") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: name");
    };

    const name = switch (name_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'name' must be a string"),
    };

    // Get optional path
    var search_path: []const u8 = ctx.working_directory;
    var path_allocated = false;
    if (obj.get("path")) |path_val| {
        switch (path_val) {
            .string => |s| {
                if (std.fs.path.isAbsolute(s)) {
                    search_path = s;
                } else {
                    search_path = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, s }) catch return error.OutOfMemory;
                    path_allocated = true;
                }
            },
            else => {},
        }
    }
    defer if (path_allocated) ctx.allocator.free(search_path);

    // Get type filter
    var type_flag: []const u8 = "";
    if (obj.get("type")) |type_val| {
        switch (type_val) {
            .string => |s| {
                if (std.mem.eql(u8, s, "file") or std.mem.eql(u8, s, "f")) {
                    type_flag = "-type f ";
                } else if (std.mem.eql(u8, s, "directory") or std.mem.eql(u8, s, "d")) {
                    type_flag = "-type d ";
                }
            },
            else => {},
        }
    }

    // Build find command
    const command = std.fmt.allocPrint(ctx.allocator, "find \"{s}\" {s}-name \"{s}\" 2>/dev/null | head -1000", .{
        search_path,
        type_flag,
        name,
    }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Find failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        if (result.stdout.len == 0) {
            ctx.allocator.free(result.stdout);
            const output = ctx.allocator.dupe(u8, "No files found") catch return error.OutOfMemory;
            return ToolResult.init(ctx.allocator, true, output);
        }
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Find failed: {s}", .{result.stderr}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const find_tool = Tool{
    .name = "find",
    .description = "Find files and directories by name. Supports glob patterns.",
    .parameters = &[_]Parameter{
        .{
            .name = "name",
            .type = .string,
            .required = true,
            .description = "File name pattern to search for (supports * and ? wildcards)",
        },
        .{
            .name = "path",
            .type = .string,
            .required = false,
            .description = "Directory to search in (default: working directory)",
        },
        .{
            .name = "type",
            .type = .string,
            .required = false,
            .description = "Filter by type: 'file' (or 'f'), 'directory' (or 'd')",
            .enum_values = &[_][]const u8{ "file", "f", "directory", "d" },
        },
    },
    .execute = &executeFind,
};

// ============================================================================
// Registration
// ============================================================================

/// All search tools for easy registration
pub const all_tools = [_]*const Tool{
    &grep_tool,
    &find_tool,
};

/// Register all search tools with a registry
pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "grep_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("grep", grep_tool.name);
}

test "find_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("find", find_tool.name);
}

test "all_tools count" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 2), all_tools.len);
}
