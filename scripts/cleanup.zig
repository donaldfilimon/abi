// Helper script to analyze the `src` tree for unused or duplicate modules.
// Run with: zig run scripts/cleanup.zig
// The script prints a report and optionally deletes unreferenced files.

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const err = gpa.deinit();
        if (err != .leak) {
            std.log.err("Memory leak\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var io = std.Io.Threaded.init(allocator);
    defer io.deinit();

    // Map from file path to index
    var all_files = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    defer all_files.deinit();
    try walkSrcFiles(allocator, &io, &all_files);

    // Build import graph
    var graph = std.AutoHashMap([]const u8, []const []const u8).init(allocator);
    defer graph.deinit();
    for (all_files.items) |file| {
        const imports = try parseImports(file, allocator, &io);
        try graph.put(file, imports);
    }

    // Detect unused files: those that are never imported by any other file
    var used = std.AutoHashSet([]const u8).init(allocator);
    defer used.deinit();
    for (graph.values()) |deps| {
        for (deps) |dep| {
            try used.put(dep, {});
        }
    }

    std.log.info("Total Zig files: {d}", .{all_files.items.len});
    std.log.info("Unused modules:", .{});
    for (all_files.items) |file| {
        if (!used.contains(file)) {
            std.log.info("  {s}", .{file});
        }
    }

    // (Optional) detection of duplicate helpers is much more involved and
    // requires semantic comparison. We leave that as a TODO.
}

fn walkSrcFiles(allocator: std.mem.Allocator, io: *std.Io.Threaded, files: *std.ArrayList([]const u8)) !void {
    const cwd = try std.Io.Dir.cwd(io.*);
    var walker = try cwd.walk(allocator);
    defer walker.deinit();
    while (try walker.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.basename, ".zig")) {
            const full = try std.fs.path.join(allocator, &[_][]const u8{ "src", entry.path });
            try files.append(full);
        }
    }
}

fn parseImports(path: []const u8, allocator: std.mem.Allocator, io: *std.Io.Threaded) ![]const []const u8 {
    const cwd = try std.Io.Dir.cwd(io.*);
    const content = try cwd.readFileAlloc(allocator, path, std.math.maxInt(usize));
    defer allocator.free(content);
    var imports = std.ArrayList([]const u8).init(allocator);
    defer imports.deinit();
    var reader = std.io.fixedBufferStream(content);
    var line_buf: [256]u8 = undefined;
    while (true) {
        const line = reader.reader().readUntilDelimiterOrEof(&line_buf, '\n') catch break;
        if (std.mem.indexOf(u8, line, "@import(")) |idx| {
            const start = idx + "@import(".len;
            const end = std.mem.indexOf(u8, line[start..], ")") orelse continue;
            const imp = line[start .. start + end];
            try imports.append(imp);
        }
    }
    return imports.toOwnedSlice();
}
