//! Zig 0.16 Migration Scanner - Modern Version
//!
//! Scans the codebase for deprecated Zig 0.15 patterns and suggests fixes
//! Uses Zig 0.16 I/O patterns to scan itself.

const std = @import("std");

pub const Pattern = struct {
    name: []const u8,
    description: []const u8,
    search_term: []const u8,
    fix: []const u8,
};

pub const patterns = [_]Pattern{
    .{
        .name = "std.fs.cwd()",
        .description = "Replace std.fs.cwd() with std.Io.Dir.cwd() with Io context",
        .search_term = "std.fs.cwd()",
        .fix =
        \\ var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        \\ defer io_backend.deinit();
        \\ const io = io_backend.io();
        \\ var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        ,
    },
    .{
        .name = "std.time.sleep()",
        .description = "Replace std.time.sleep() with utils.sleepMs()",
        .search_term = "std.time.sleep(",
        .fix =
        \\ // Use shared utility:
        \\ const time = @import("../shared/utils.zig");
        \\ time.sleepMs(ms);
        ,
    },
    .{
        .name = "std.time.nanoTimestamp()",
        .description = "Replace std.time.nanoTimestamp() with std.time.Timer or utils",
        .search_term = "std.time.nanoTimestamp()",
        .fix =
        \\ // For high-precision timing:
        \\ var timer = std.time.Timer.start() catch return error.TimerFailed;
        \\ const elapsed_ns = timer.read();
        \\ 
        \\ // For timestamp:
        \\ const time = @import("../shared/utils.zig");
        \\ const timestamp_ns = time.nowNanoseconds();
        ,
    },
    .{
        .name = "@errorName() in format strings",
        .description = "Replace @errorName(err) with {t} format specifier",
        .search_term = "@errorName(",
        .fix =
        \\ // OLD: std.log.err("Error: {s}", .{@errorName(err)});
        \\ // NEW: std.log.err("Error: {t}", .{err});
        ,
    },
    .{
        .name = "std.io.AnyReader",
        .description = "Replace std.io.AnyReader with std.Io.Reader",
        .search_term = "std.io.AnyReader",
        .fix = "Replace with std.Io.Reader",
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.debug.print("Scanning for Zig 0.16 migration issues...\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Scan from current directory (project root)
    try scanDirectory(allocator, io, ".");
}

fn scanDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    var dir = cwd.openDir(io, dir_path, .{}) catch {
        std.debug.print("Skipping directory '{s}' (not found)\n", .{dir_path});
        return;
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        const full_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(full_path);

        if (entry.kind == .directory) {
            try scanDirectory(allocator, io, full_path);
        } else if (std.mem.endsWith(u8, entry.name, ".zig")) {
            try scanFile(allocator, io, full_path);
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, io: std.Io, file_path: []const u8) !void {
    // Read entire file content
    const content = std.Io.Dir.cwd().readFileAlloc(io, file_path, allocator, .unlimited) catch {
        std.debug.print("Could not read file: {s}\n", .{file_path});
        return;
    };
    defer allocator.free(content);

    var found_issues = false;

    for (patterns) |pattern| {
        var occurrences: usize = 0;
        var pos: usize = 0;

        // Simple linear search for pattern
        while (true) {
            if (std.mem.indexOfPos(u8, content, pos, pattern.search_term)) |idx| {
                occurrences += 1;
                pos = idx + pattern.search_term.len;

                if (!found_issues) {
                    std.debug.print("\n{s}:\n", .{file_path});
                    std.debug.print("{s}\n", .{"-" ** 80});
                    found_issues = true;
                }

                // Show context for first occurrence
                if (occurrences == 1) {
                    const line_num = lineNumber(content, idx);
                    const line_start = lineStart(content, idx);
                    const line_end = lineEnd(content, idx);
                    const line = content[line_start..line_end];
                    std.debug.print("   Line {d}: {s}\n", .{ line_num, std.mem.trim(u8, line, " \t\r\n") });
                }
            } else {
                break;
            }
        }

        if (occurrences > 0) {
            if (!found_issues) {
                std.debug.print("\n{s}:\n", .{file_path});
                std.debug.print("{s}\n", .{"-" ** 80});
                found_issues = true;
            }

            std.debug.print("⚠️  {s}: {s}\n", .{ pattern.name, pattern.description });
            std.debug.print("   Occurrences: {d}\n", .{occurrences});

            if (pattern.fix.len > 0) {
                std.debug.print("   Suggested fix:\n{s}\n", .{pattern.fix});
            }
        }
    }

    if (found_issues) {
        std.debug.print("\n", .{});
    }
}

fn lineNumber(content: []const u8, pos: usize) usize {
    var line: usize = 1;
    for (content[0..pos]) |c| {
        if (c == '\n') line += 1;
    }
    return line;
}

fn lineStart(content: []const u8, pos: usize) usize {
    var start = pos;
    while (start > 0 and content[start - 1] != '\n') {
        start -= 1;
    }
    return start;
}

fn lineEnd(content: []const u8, pos: usize) usize {
    var end = pos;
    while (end < content.len and content[end] != '\n') {
        end += 1;
    }
    return end;
}
