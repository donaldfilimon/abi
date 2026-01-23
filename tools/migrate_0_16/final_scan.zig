//! Zig 0.16 Migration Scanner - Final Working Version
//!
//! Uses simple file I/O without complex I/O backend to scan for patterns.

const std = @import("std");

pub const Pattern = struct {
    name: []const u8,
    description: []const u8,
    search_text: []const u8,
    fix: []const u8,
};

pub const patterns = [_]Pattern{
    .{
        .name = "@errorName() in format strings",
        .description = "Replace @errorName(err) with {t} format specifier in format strings",
        .search_text = "@errorName(",
        .fix =
        \\ // OLD: std.debug.print("Error: {s}", .{@errorName(err)});
        \\ // NEW: std.debug.print("Error: {t}", .{err});
        ,
    },
    .{
        .name = "std.time.sleep()",
        .description = "Replace std.time.sleep() with utils.sleepMs()",
        .search_text = "std.time.sleep(",
        .fix =
        \\ // Use shared utility:
        \\ const time = @import("../shared/utils.zig");
        \\ time.sleepMs(ms);
        ,
    },
    .{
        .name = "std.io.AnyReader",
        .description = "Replace std.io.AnyReader with std.Io.Reader",
        .search_text = "std.io.AnyReader",
        .fix = "Replace with std.Io.Reader",
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend (single instance for whole program)
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.debug.print("Scanning for Zig 0.16 migration issues...\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Scan directories
    std.debug.print("Compiled with Zig 0.16\n", .{});
    try scanDirectory(allocator, io, "src");
    try scanDirectory(allocator, io, "benchmarks");
    try scanDirectory(allocator, io, "examples");

    std.debug.print("\n========================================\n", .{});
    std.debug.print("Scan complete.\n", .{});
}

fn scanDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    var dir = cwd.openDir(io, dir_path, .{}) catch {
        // Skip if directory doesn't exist
        return;
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        const full_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(full_path);

        if (entry.kind == .directory) {
            // Skip .git and other special directories
            if (!std.mem.startsWith(u8, entry.name, ".") and !std.mem.eql(u8, entry.name, ".zig-cache")) {
                try scanDirectory(allocator, io, full_path);
            }
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
        var count: usize = 0;
        var pos: usize = 0;

        // Count occurrences of pattern
        while (true) {
            if (std.mem.indexOfPos(u8, content, pos, pattern.search_text)) |idx| {
                count += 1;
                pos = idx + pattern.search_text.len;
            } else {
                break;
            }
        }

        if (count > 0) {
            if (!found_issues) {
                std.debug.print("\n{s}:\n", .{file_path});
                std.debug.print("{s}\n", .{"-" ** 80});
                found_issues = true;
            }

            std.debug.print("⚠️  {s}: {s}\n", .{ pattern.name, pattern.description });
            std.debug.print("   Occurrences: {d}\n", .{count});

            if (pattern.fix.len > 0) {
                std.debug.print("   Suggested fix:\n{s}\n", .{pattern.fix});
            }
        }
    }

    if (found_issues) {
        std.debug.print("{s}\n", .{"-" ** 80});
    }
}
