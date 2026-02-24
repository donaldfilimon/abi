//! Edit Tools for Agent Actions
//!
//! Provides code editing operations that agents can use:
//! - Edit (precise string replacement)
//! - Insert (insert content at line)
//! - Delete (delete lines)

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const ToolRegistry = tool.ToolRegistry;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ParameterType = tool.ParameterType;
const ToolExecutionError = tool.ToolExecutionError;

/// File I/O helpers using Zig 0.16 std.Io backend.
fn readFile(allocator: std.mem.Allocator, path: []const u8, max_size: usize) ![]u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_size));
}

fn writeFile(allocator: std.mem.Allocator, path: []const u8, data: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, data);
}

// ============================================================================
// Edit Tool (String Replacement)
// ============================================================================

fn executeEdit(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;

    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };
    const old_string_val = obj.get("old_string") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: old_string");
    };
    const new_string_val = obj.get("new_string") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: new_string");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };
    if (tool.hasPathTraversal(path)) return ToolResult.fromError(ctx.allocator, "Path contains directory traversal");
    const old_string = switch (old_string_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'old_string' must be a string"),
    };
    const new_string = switch (new_string_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'new_string' must be a string"),
    };

    // Check replace_all flag
    var replace_all = false;
    if (obj.get("replace_all")) |ra| {
        switch (ra) {
            .bool => |b| replace_all = b,
            else => {},
        }
    }

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        ctx.allocator.dupe(u8, path) catch return error.OutOfMemory
    else
        std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
    defer ctx.allocator.free(full_path);

    // Read file
    const max_size = 10 * 1024 * 1024; // 10MB max
    const content = readFile(ctx.allocator, full_path, max_size) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to read file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(content);

    // Count occurrences
    var occurrence_count: usize = 0;
    var search_pos: usize = 0;
    while (std.mem.indexOf(u8, content[search_pos..], old_string)) |pos| {
        occurrence_count += 1;
        search_pos += pos + old_string.len;
    }

    if (occurrence_count == 0) {
        const msg = std.fmt.allocPrint(ctx.allocator, "String not found in file: '{s}'", .{old_string[0..@min(50, old_string.len)]}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }

    if (!replace_all and occurrence_count > 1) {
        const msg = std.fmt.allocPrint(ctx.allocator, "String found {d} times. Use replace_all=true or provide more context to make it unique.", .{occurrence_count}) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }

    // Perform replacement
    var new_content: std.ArrayListUnmanaged(u8) = .{};
    defer new_content.deinit(ctx.allocator);

    var replacements: usize = 0;
    var last_end: usize = 0;
    var search_start: usize = 0;

    while (std.mem.indexOf(u8, content[search_start..], old_string)) |rel_pos| {
        const pos = search_start + rel_pos;

        // Append content before match
        new_content.appendSlice(ctx.allocator, content[last_end..pos]) catch return error.OutOfMemory;

        // Append replacement
        new_content.appendSlice(ctx.allocator, new_string) catch return error.OutOfMemory;

        last_end = pos + old_string.len;
        search_start = last_end;
        replacements += 1;

        if (!replace_all) break;
    }

    // Append remaining content
    new_content.appendSlice(ctx.allocator, content[last_end..]) catch return error.OutOfMemory;

    // Write file
    writeFile(ctx.allocator, full_path, new_content.items) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to write file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    const output = std.fmt.allocPrint(ctx.allocator, "Successfully replaced {d} occurrence(s) in '{s}'", .{ replacements, path }) catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const edit_tool = Tool{
    .name = "edit",
    .description = "Perform exact string replacement in a file. old_string must be unique unless replace_all is true.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the file to edit",
        },
        .{
            .name = "old_string",
            .type = .string,
            .required = true,
            .description = "The exact string to replace (must be unique in the file)",
        },
        .{
            .name = "new_string",
            .type = .string,
            .required = true,
            .description = "The replacement string",
        },
        .{
            .name = "replace_all",
            .type = .boolean,
            .required = false,
            .description = "If true, replace all occurrences (default: false)",
        },
    },
    .execute = &executeEdit,
};

// ============================================================================
// Insert Lines Tool
// ============================================================================

fn executeInsertLines(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;

    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };
    const line_val = obj.get("line") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: line");
    };
    const content_val = obj.get("content") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: content");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };
    if (tool.hasPathTraversal(path)) return ToolResult.fromError(ctx.allocator, "Path contains directory traversal");
    const line_num: usize = switch (line_val) {
        .integer => |i| @intCast(@max(1, i)),
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'line' must be an integer"),
    };
    const content = switch (content_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'content' must be a string"),
    };

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        ctx.allocator.dupe(u8, path) catch return error.OutOfMemory
    else
        std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
    defer ctx.allocator.free(full_path);

    // Read file
    const max_size = 10 * 1024 * 1024; // 10MB max
    const file_content = readFile(ctx.allocator, full_path, max_size) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to read file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(file_content);

    // Split into lines
    var lines: std.ArrayListUnmanaged([]const u8) = .{};
    defer lines.deinit(ctx.allocator);

    var iter = std.mem.splitScalar(u8, file_content, '\n');
    while (iter.next()) |l| {
        lines.append(ctx.allocator, l) catch return error.OutOfMemory;
    }

    // Insert at position (1-indexed)
    const insert_pos = @min(line_num - 1, lines.items.len);

    // Build new content
    var new_content: std.ArrayListUnmanaged(u8) = .{};
    defer new_content.deinit(ctx.allocator);

    for (lines.items[0..insert_pos]) |l| {
        new_content.appendSlice(ctx.allocator, l) catch return error.OutOfMemory;
        new_content.append(ctx.allocator, '\n') catch return error.OutOfMemory;
    }

    new_content.appendSlice(ctx.allocator, content) catch return error.OutOfMemory;
    if (content.len == 0 or content[content.len - 1] != '\n') {
        new_content.append(ctx.allocator, '\n') catch return error.OutOfMemory;
    }

    for (lines.items[insert_pos..]) |l| {
        new_content.appendSlice(ctx.allocator, l) catch return error.OutOfMemory;
        new_content.append(ctx.allocator, '\n') catch return error.OutOfMemory;
    }

    // Remove trailing newline if original file didn't have one
    if (new_content.items.len > 0 and file_content.len > 0 and file_content[file_content.len - 1] != '\n') {
        _ = new_content.pop();
    }

    // Write file
    writeFile(ctx.allocator, full_path, new_content.items) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to write file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    // Count inserted lines
    var inserted_lines: usize = 1;
    for (content) |c| {
        if (c == '\n') inserted_lines += 1;
    }

    const output = std.fmt.allocPrint(ctx.allocator, "Inserted {d} line(s) at line {d} in '{s}'", .{ inserted_lines, line_num, path }) catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const insert_lines_tool = Tool{
    .name = "insert_lines",
    .description = "Insert content at a specific line number. Lines are 1-indexed.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the file to edit",
        },
        .{
            .name = "line",
            .type = .integer,
            .required = true,
            .description = "Line number to insert at (1-indexed)",
        },
        .{
            .name = "content",
            .type = .string,
            .required = true,
            .description = "Content to insert",
        },
    },
    .execute = &executeInsertLines,
};

// ============================================================================
// Delete Lines Tool
// ============================================================================

fn executeDeleteLines(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = args.object;

    const path_val = obj.get("path") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: path");
    };
    const start_val = obj.get("start") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: start");
    };
    const end_val = obj.get("end") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: end");
    };

    const path = switch (path_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'path' must be a string"),
    };
    if (tool.hasPathTraversal(path)) return ToolResult.fromError(ctx.allocator, "Path contains directory traversal");
    const start_line: usize = switch (start_val) {
        .integer => |i| @intCast(@max(1, i)),
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'start' must be an integer"),
    };
    const end_line: usize = switch (end_val) {
        .integer => |i| @intCast(@max(1, i)),
        else => return ToolResult.fromError(ctx.allocator, "Parameter 'end' must be an integer"),
    };

    if (start_line > end_line) {
        return ToolResult.fromError(ctx.allocator, "start must be less than or equal to end");
    }

    // Resolve path relative to working directory
    const full_path = if (std.fs.path.isAbsolute(path))
        ctx.allocator.dupe(u8, path) catch return error.OutOfMemory
    else
        std.fs.path.join(ctx.allocator, &[_][]const u8{ ctx.working_directory, path }) catch return error.OutOfMemory;
    defer ctx.allocator.free(full_path);

    // Read file
    const max_size = 10 * 1024 * 1024; // 10MB max
    const file_content = readFile(ctx.allocator, full_path, max_size) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to read file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };
    defer ctx.allocator.free(file_content);

    // Split into lines
    var lines: std.ArrayListUnmanaged([]const u8) = .{};
    defer lines.deinit(ctx.allocator);

    var iter = std.mem.splitScalar(u8, file_content, '\n');
    while (iter.next()) |l| {
        lines.append(ctx.allocator, l) catch return error.OutOfMemory;
    }

    // Validate line numbers
    if (start_line > lines.items.len) {
        const msg = std.fmt.allocPrint(ctx.allocator, "start line {d} exceeds file length ({d} lines)", .{ start_line, lines.items.len }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    }

    const actual_end = @min(end_line, lines.items.len);

    // Build new content (skip deleted lines)
    var new_content: std.ArrayListUnmanaged(u8) = .{};
    defer new_content.deinit(ctx.allocator);

    for (lines.items, 1..) |l, cur_line_num| {
        if (cur_line_num >= start_line and cur_line_num <= actual_end) continue;

        new_content.appendSlice(ctx.allocator, l) catch return error.OutOfMemory;
        new_content.append(ctx.allocator, '\n') catch return error.OutOfMemory;
    }

    // Remove trailing newline if original file didn't have one
    if (new_content.items.len > 0 and file_content.len > 0 and file_content[file_content.len - 1] != '\n') {
        _ = new_content.pop();
    }

    // Write file
    writeFile(ctx.allocator, full_path, new_content.items) catch |err| {
        const msg = std.fmt.allocPrint(ctx.allocator, "Failed to write file '{s}': {t}", .{ path, err }) catch return error.OutOfMemory;
        return ToolResult.fromError(ctx.allocator, msg);
    };

    const deleted_count = actual_end - start_line + 1;
    const output = std.fmt.allocPrint(ctx.allocator, "Deleted {d} line(s) ({d}-{d}) from '{s}'", .{ deleted_count, start_line, actual_end, path }) catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const delete_lines_tool = Tool{
    .name = "delete_lines",
    .description = "Delete lines from start to end (inclusive). Lines are 1-indexed.",
    .parameters = &[_]Parameter{
        .{
            .name = "path",
            .type = .string,
            .required = true,
            .description = "Path to the file to edit",
        },
        .{
            .name = "start",
            .type = .integer,
            .required = true,
            .description = "First line to delete (1-indexed)",
        },
        .{
            .name = "end",
            .type = .integer,
            .required = true,
            .description = "Last line to delete (1-indexed, inclusive)",
        },
    },
    .execute = &executeDeleteLines,
};

// ============================================================================
// Registration
// ============================================================================

/// All edit tools for easy registration
pub const all_tools = [_]*const Tool{
    &edit_tool,
    &insert_lines_tool,
    &delete_lines_tool,
};

/// Register all edit tools with a registry
pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "edit_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("edit", edit_tool.name);
}

test "insert_lines_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("insert_lines", insert_lines_tool.name);
}

test "delete_lines_tool creation" {
    const testing = std.testing;
    try testing.expectEqualStrings("delete_lines", delete_lines_tool.name);
}

test "all_tools count" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 3), all_tools.len);
}

test {
    std.testing.refAllDecls(@This());
}
