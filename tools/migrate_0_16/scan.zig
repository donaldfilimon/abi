//! Zig 0.16 Migration Scanner
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

    // Initialize I/O backend for Zig 0.16
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const cwd = std.Io.Dir.cwd();

    std.debug.print("Scanning for Zig 0.16 migration issues...\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Scan src directory
    try scanDirectory(allocator, cwd, io, "src");

    // Scan benchmarks directory
    try scanDirectory(allocator, cwd, io, "benchmarks");

    // Scan examples directory
    try scanDirectory(allocator, cwd, io, "examples");
}

fn scanDirectory(allocator: std.mem.Allocator, cwd: std.Io.Dir, io: std.Io, dir_path: []const u8) !void {
    var dir = cwd.openDir(io, dir_path, .{}) catch {
        std.debug.print("Skipping directory '{s}' (not found)\n", .{dir_path});
        return;
    };
    defer dir.close(io);

    var iterator = dir.iterate(io);
    while (try iterator.next(io, null)) |entry| {
        const full_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(full_path);

        if (entry.kind == .directory) {
            // Recursively scan subdirectories
            try scanDirectory(allocator, cwd, io, full_path);
        } else if (std.mem.endsWith(u8, entry.name, ".zig")) {
            try scanFile(allocator, cwd, io, full_path);
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, cwd: std.Io.Dir, io: std.Io, file_path: []const u8) !void {
    const file = cwd.openFile(io, file_path, .{}) catch {
        std.debug.print("Could not open file: {s}\n", .{file_path});
        return;
    };
    defer file.close(io);

    const content = try file.readFileAlloc(io, allocator, .unlimited) catch {
        std.debug.print("Could not read file: {s}\n", .{file_path});
        return;
    };
    defer allocator.free(content);

    var found_issues = false;

    for (patterns) |pattern| {
        // Simple regex-like search
        if (std.mem.indexOf(u8, content, pattern.name) != null) {
            if (!found_issues) {
                std.debug.print("\n{s}:\n", .{file_path});
                std.debug.print("{s}\n", .{std.fmt.comptimePrint("-", .{}) ** 80});
                found_issues = true;
            }

            // Actually search for the old_pattern, not just the name
            var pattern_occurrences: usize = 0;
            var line_number: usize = 1;
            var line_start: usize = 0;

            for (content, 0..) |char, i| {
                if (char == '\n') {
                    line_number += 1;
                    line_start = i + 1;
                }
                // Check for pattern at position i
                var found = true;
                for (pattern.old_pattern, 0..) |pattern_char, j| {
                    if (i + j >= content.len or content[i + j] != pattern_char) {
                        found = false;
                        break;
                    }
                }
                if (found) {
                    pattern_occurrences += 1;
                    const line_end = std.mem.indexOfScalar(u8, content[i..], '\n') orelse content.len - i;
                    const line = content[line_start..@min(line_start + line_end, content.len)];
                    std.debug.print("   Line {d}: {s}\n", .{ line_number, std.mem.trim(u8, line, " \t\r\n") });
                }
            }

            if (pattern_occurrences > 0) {
                std.debug.print("⚠️  {s}: {s}\n", .{ pattern.name, pattern.description });
                std.debug.print("   Occurrences: {d}\n", .{pattern_occurrences});

                if (pattern.fix.len > 0) {
                    std.debug.print("   Suggested fix:\n{s}\n", .{pattern.fix});
                }
            }
        }
    }

    if (found_issues) {
        std.debug.print("\n", .{});
    }
}
