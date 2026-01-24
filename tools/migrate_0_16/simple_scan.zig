//! Zig 0.16 Migration Scanner - Simple Version
//!
//! Scans the codebase for deprecated Zig 0.15 patterns and suggests fixes
//! for Zig 0.16 compatibility.

const std = @import("std");

pub const Pattern = struct {
    name: []const u8,
    description: []const u8,
    old_pattern: []const u8,
    fix: []const u8,
};

pub const patterns = [_]Pattern{
    .{
        .name = "std.fs.cwd()",
        .description = "Replace std.fs.cwd() with std.Io.Dir.cwd() with Io context",
        .old_pattern = "std\\.fs\\.cwd\\(\\)",
        .fix =
        \\ var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        \\ defer io_backend.deinit();
        \\ const io = io_backend.io();
        \\ // Replace std.fs.cwd() with std.Io.Dir.cwd(io)
        ,
    },
    .{
        .name = "std.time.sleep()",
        .description = "Replace std.time.sleep() with std.Io.Clock.Duration.sleep() or use shared utils",
        .old_pattern = "std\\.time\\.sleep\\(",
        .fix =
        \\ // Use shared utility:
        \\ const time = @import("../shared/utils.zig");
        \\ time.sleepMs(ms);
        ,
    },
    .{
        .name = "std.time.nanoTimestamp()",
        .description = "Replace std.time.nanoTimestamp() with std.time.Timer for high-precision timing",
        .old_pattern = "std\\.time\\.nanoTimestamp\\(\\)",
        .fix =
        \\ // For high-precision timing:
        \\ var timer = std.time.Timer.start() catch return error.TimerFailed;
        \\ // ... work ...
        \\ const elapsed_ns = timer.read();
        \\ 
        \\ // For simple timestamp:
        \\ const time = @import("../shared/utils.zig");
        \\ const timestamp_ms = time.unixMs();
        ,
    },
    .{
        .name = "@errorName() in format strings",
        .description = "Replace @errorName(err) with {t} format specifier",
        .old_pattern = "@errorName\\(",
        .fix =
        \\ // OLD: std.log.err(\"Error: {s}\", .{@errorName(err)});
        \\ // NEW: std.log.err(\"Error: {t}\", .{err});
        ,
    },
    .{
        .name = "std.io.AnyReader",
        .description = "Replace std.io.AnyReader with std.Io.Reader",
        .old_pattern = "std\\.io\\.AnyReader",
        .fix = "Replace with std.Io.Reader",
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Scanning for Zig 0.16 migration issues...\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Scan src directory
    try scanDirectory(allocator, "src");

    // Scan benchmarks directory
    try scanDirectory(allocator, "benchmarks");

    // Scan examples directory
    try scanDirectory(allocator, "examples");
}

fn scanDirectory(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch {
        std.debug.print("Skipping directory '{s}' (not found)\n", .{dir_path});
        return;
    };
    defer dir.close();

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        const full_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(full_path);

        if (entry.kind == .directory) {
            // Recursively scan subdirectories
            try scanDirectory(allocator, full_path);
        } else if (std.mem.endsWith(u8, entry.name, ".zig")) {
            try scanFile(allocator, full_path);
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const file = std.fs.cwd().openFile(file_path, .{}) catch {
        std.debug.print("Could not open file: {s}\n", .{file_path});
        return;
    };
    defer file.close();

    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var found_issues = false;

    for (patterns) |pattern| {
        // Count occurrences of pattern
        var occurrences: usize = 0;
        var pos: usize = 0;

        while (std.mem.indexOf(u8, content[pos..], pattern.old_pattern)) |idx| {
            occurrences += 1;
            pos += idx + pattern.old_pattern.len;
        }

        if (occurrences > 0) {
            if (!found_issues) {
                std.debug.print("\n{s}:\n", .{file_path});
                std.debug.print("{s}\n", .{std.fmt.comptimePrint("-", .{}) ** 80});
                found_issues = true;
            }

            std.debug.print("⚠️  {s}: {s}\n", .{ pattern.name, pattern.description });
            std.debug.print("   Occurrences: {d}\n", .{occurrences});

            // Try to show context for first occurrence
            if (std.mem.indexOf(u8, content, pattern.old_pattern)) |first_pos| {
                const start_line = lineNumber(content, first_pos);
                const line_start = lineStart(content, first_pos);
                const line_end = lineEnd(content, first_pos);
                const line = content[line_start..line_end];
                std.debug.print("   Line {d}: {s}\n", .{ start_line, std.mem.trim(u8, line, " \t\r\n") });
            }

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
