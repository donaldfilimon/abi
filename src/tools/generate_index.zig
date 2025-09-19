const std = @import("std");

// Generates a tree-like index of the repository.
// Usage: zig run src/tools/generate_index.zig [root_dir] [output_file]
// If root_dir is omitted, defaults to "src".
// If output_file is omitted, defaults to "repo_index.txt" in the current working directory.

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(!gpa.deinit());

    const alloc = gpa.allocator();

    // Parse command‑line arguments
    const args = try std.process.argsAlloc(alloc);
    defer alloc.free(args);

    const root_path: []const u8 = if (args.len > 1) args[1] else "src";
    const out_path: []const u8 = if (args.len > 2) args[2] else "repo_index.txt";

    // Open output file
    const out_file = try std.fs.cwd().createFile(out_path, .{ .truncate = true });
    defer out_file.close();

    // Start walking the tree
    try walk(root_path, 0, out_file.writer(), alloc);
}

fn walk(path: []const u8, depth: u32, out: anytype, alloc: std.mem.Allocator) !void {
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        // Build indentation string
        var indent: [64]u8 = undefined;
        const indent_len = @min(depth * 4, @sizeOf(indent));
        for (0..indent_len) |i| indent[i] = ' ';

        const marker = if (entry.kind == .directory) "└── " else "    ";

        try out.print("{s}{s}{s}\n", .{ indent[0..indent_len], marker, entry.name });

        if (entry.kind == .directory) {
            const child_path = try std.fs.path.join(alloc, &.{ path, entry.name });
            defer alloc.free(child_path);

            try walk(child_path, depth + 1, out, alloc);
        }
    }
}
