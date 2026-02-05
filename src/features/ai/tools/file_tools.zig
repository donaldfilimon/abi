//! File Tools for Agent Actions
//!
//! Provides file system operations that agents can use:
//! - Read file contents
//! - Write file contents
//! - List directory contents
//! - Glob pattern matching
//! - File existence checks
//!
//! Note: These tools use shell commands internally for Zig 0.16 compatibility.

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
// Read File Tool
// ============================================================================

fn executeReadFile(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };

    // Get optional offset and limit
    var offset_str: []const u8 = "";
    var limit_str: []const u8 = "";

    if (obj.get("offset")) |off_val| {
        switch (off_val) {
            .integer => |i| {
                if (i > 0) {
                    offset_str = std.fmt.allocPrint(ctx.allocator, " | tail -n +{d}", .{i + 1}) catch return error.OutOfMemory;
                }
            },
            else => {},
        }
    }

    if (obj.get("limit")) |lim_val| {
        switch (lim_val) {
            .integer => |i| {
                if (i > 0) {
                    limit_str = std.fmt.allocPrint(ctx.allocator, " | head -n {d}", .{i}) catch return error.OutOfMemory;
                }
            },
            else => {},
        }
    }
    defer if (offset_str.len > 0) ctx.allocator.free(offset_str);
    defer if (limit_str.len > 0) ctx.allocator.free(limit_str);

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        path
    else blk: {
        const joined = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
        break :blk joined;
    };
    defer if (!std.fs.path.isAbsolute(path)) ctx.allocator.free(full_path);

    // Use cat -n to get line numbers
    const command = std.fmt.allocPrint(ctx.allocator, "cat -n \"{s}\"{s}{s}", .{ full_path, offset_str, limit_str }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to read file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to read file '{s}': {s}", .{ path, result.stderr }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const read_file_tool = Tool{
    .name = "read_file",
    .description = "Read the contents of a file. Returns lines with line numbers. Use offset and limit for large files.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the file to read (absolute or relative to working directory)",
        },
        .{
            .name = "offset",
            .type = .integer,
            .required = false,
            .description = "Line number to start reading from (1-indexed, default: 0)",
        },
        .{
            .name = "limit",
            .type = .integer,
            .required = false,
            .description = "Maximum number of lines to read (default: unlimited)",
        },
    },
    .execute = &executeReadFile,
};

// ============================================================================
// Write File Tool
// ============================================================================

fn executeWriteFile(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };
    const content_val = obj.get("content") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: content");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };
    const content = switch (content_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'content' must be a string"),
    };

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        path
    else blk: {
        const joined = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
        break :blk joined;
    };
    defer if (!std.fs.path.isAbsolute(path)) ctx.allocator.free(full_path);

    // Ensure parent directory exists
    if (std.fs.path.dirname(full_path)) |dir| {
        const mkdir_cmd = std.fmt.allocPrint(ctx.allocator, "mkdir -p \"{s}\"", .{dir}) catch return error.OutOfMemory;
        defer ctx.allocator.free(mkdir_cmd);
        _ = os.exec(ctx.allocator, mkdir_cmd) catch |err| {
            std.debug.print("Warning: Failed to create directory '{s}': {t}\n", .{ dir, err });
        };
    }

    // Escape content for shell and write using heredoc
    // Use base64 encoding to safely pass binary/special content
    const encoded = std.base64.standard.Allocator.encode(ctx.allocator, content) catch return error.OutOfMemory;
    defer ctx.allocator.free(encoded);

    const command = std.fmt.allocPrint(ctx.allocator, "echo \"{s}\" | base64 -d > \"{s}\"", .{ encoded, full_path }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to write file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);
    defer ctx.allocator.free(result.stdout);

    if (result.success()) {
        const output = std.fmt.allocPrint(ctx.allocator, "Successfully wrote {d} bytes to '{s}'", .{ content.len, path }) catch return error.OutOfMemory;
        return ToolResult.init(ctx.allocator, true, output);
    } else {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to write file '{s}': {s}", .{ path, result.stderr }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const write_file_tool = Tool{
    .name = "write_file",
    .description = "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing files.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the file to write (absolute or relative to working directory)",
        },
        .{
            .name = "content",
            .type = .string,
            .required = true,
            .description = "Content to write to the file",
        },
    },
    .execute = &executeWriteFile,
};

// ============================================================================
// List Directory Tool
// ============================================================================

fn executeListDir(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };

    // Check for recursive flag
    var recursive = false;
    if (obj.get("recursive")) |rec_val| {
        switch (rec_val) {
            .bool => |b| recursive = b,
            else => {},
        }
    }

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        path
    else blk: {
        const joined = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
        break :blk joined;
    };
    defer if (!std.fs.path.isAbsolute(path)) ctx.allocator.free(full_path);

    // Use ls or find command
    const command = if (recursive)
        std.fmt.allocPrint(ctx.allocator, "find \"{s}\" -type f -o -type d | head -1000", .{full_path}) catch return error.OutOfMemory
    else
        std.fmt.allocPrint(ctx.allocator, "ls -la \"{s}\"", .{full_path}) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to list directory '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to list directory '{s}': {s}", .{ path, result.stderr }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const list_dir_tool = Tool{
    .name = "list_dir",
    .description = "List contents of a directory. Shows file details with ls -la, or recursive listing with find.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the directory to list",
        },
        .{
            .name = "recursive",
            .type = .boolean,
            .required = false,
            .description = "If true, list recursively (default: false)",
        },
    },
    .execute = &executeListDir,
};

// ============================================================================
// File Exists Tool
// ============================================================================

fn executeFileExists(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        path
    else blk: {
        const joined = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
        break :blk joined;
    };
    defer if (!std.fs.path.isAbsolute(path)) ctx.allocator.free(full_path);

    // Use test command to check existence and type
    const command = std.fmt.allocPrint(ctx.allocator, "if [ -e \"{s}\" ]; then if [ -d \"{s}\" ]; then echo \"directory\"; elif [ -f \"{s}\" ]; then stat -c%%s \"{s}\" 2>/dev/null || stat -f%%z \"{s}\" 2>/dev/null; else echo \"other\"; fi; else echo \"not_found\"; fi", .{ full_path, full_path, full_path, full_path, full_path }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Error checking file: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        const trimmed = std.mem.trim(u8, result.stdout, " \n\r\t");
        if (std.mem.eql(u8, trimmed, "not_found")) {
            ctx.allocator.free(result.stdout);
            const output = ctx.allocator.dupe(u8, "false") catch return error.OutOfMemory;
            return ToolResult.init(ctx.allocator, true, output);
        } else if (std.mem.eql(u8, trimmed, "directory")) {
            ctx.allocator.free(result.stdout);
            const output = ctx.allocator.dupe(u8, "true (directory)") catch return error.OutOfMemory;
            return ToolResult.init(ctx.allocator, true, output);
        } else if (std.mem.eql(u8, trimmed, "other")) {
            ctx.allocator.free(result.stdout);
            const output = ctx.allocator.dupe(u8, "true (other)") catch return error.OutOfMemory;
            return ToolResult.init(ctx.allocator, true, output);
        } else {
            // It's a file with size
            const output = std.fmt.allocPrint(ctx.allocator, "true (file, {s} bytes)", .{trimmed}) catch return error.OutOfMemory;
            ctx.allocator.free(result.stdout);
            return ToolResult.init(ctx.allocator, true, output);
        }
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Error checking file: {s}", .{result.stderr}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const file_exists_tool = Tool{
    .name = "file_exists",
    .description = "Check if a file or directory exists. Returns 'true (type, size)' or 'false'.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to check",
        },
    },
    .execute = &executeFileExists,
};

// ============================================================================
// Glob Tool
// ============================================================================

fn executeGlob(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;
    const pattern_val = obj.get("pattern") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: pattern");
    };

    const pattern = switch (pattern_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'pattern' must be a string"),
    };

    // Get optional path
    var base_path: []const u8 = ctx.working_directory;
    if (obj.get("path")) |path_val| {
        switch (path_val) {
            .string => |s| {
                if (std.fs.path.isAbsolute(s)) {
                    base_path = s;
                } else {
                    base_path = std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, s }) catch return error.OutOfMemory;
                }
            },
            else => {},
        }
    }

    // Use find with -name pattern
    const command = std.fmt.allocPrint(ctx.allocator, "find \"{s}\" -name \"{s}\" 2>/dev/null | head -1000", .{ base_path, pattern }) catch return error.OutOfMemory;
    defer ctx.allocator.free(command);

    var result = os.exec(ctx.allocator, command) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Glob search failed: {t}", .{err}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    } else {
        ctx.allocator.free(result.stdout);
        const msg = std.fmt.allocPrint(ctx.allocator, "Glob search failed: {s}", .{result.stderr}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }
}

pub const glob_tool = Tool{
    .name = "glob",
    .description = "Find files matching a glob pattern. Uses find -name for pattern matching.",
    .parameters = &[_]Parameter{
        .{
            .name = "pattern",
            .type = .string,
            .required = true,
            .description = "Glob pattern to match (e.g., '*.zig', '*.ts')",
        },
        .{
            .name = "path",
            .type = .string,
            .required = false,
            .description = "Base directory to search in (default: working directory)",
        },
    },
    .execute = &executeGlob,
};

// ============================================================================
// Registration
// ============================================================================

/// All file tools for easy registration
pub const all_tools = [_]*const Tool{
    &read_file_tool,
    &write_file_tool,
    &list_dir_tool,
    &file_exists_tool,
    &glob_tool,
};

/// Register all file tools with a registry
pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "read_file_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("read_file", read_file_tool.name);
}

test "write_file_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("write_file", write_file_tool.name);
}

test "all_tools count" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 5), all_tools.len);
}
