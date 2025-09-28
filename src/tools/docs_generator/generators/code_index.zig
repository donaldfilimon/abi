const std = @import("std");

pub fn generateCodeApiIndex(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("# Code API Index\n\n");
    try writer.writeAll("This index lists each Zig source file and the public functions it exposes. The generator is intentionally lightweight so we can spot missing docs during CI runs.\n\n");

    var dir = std.fs.cwd();
    var src_dir = dir.openDir("src", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => {
            try writer.writeAll("No `src` directory present.\n");
            try flush("docs/generated/CODE_API_INDEX.md", buffer.items);
            return;
        },
        else => return err,
    };
    defer src_dir.close();

    try walkDirectory(allocator, &src_dir, "src", writer);

    try flush("docs/generated/CODE_API_INDEX.md", buffer.items);
}

fn walkDirectory(allocator: std.mem.Allocator, dir: *std.fs.Dir, prefix: []const u8, writer: anytype) !void {
    var it = dir.iterate();
    var entries = std.ArrayList(std.fs.Dir.Entry).init(allocator);
    defer entries.deinit();

    while (try it.next()) |entry| {
        try entries.append(entry);
    }

    std.mem.sort(std.fs.Dir.Entry, entries.items, {}, struct {
        fn lessThan(_: void, a: std.fs.Dir.Entry, b: std.fs.Dir.Entry) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);

    for (entries.items) |entry| {
        const full_path = try std.fs.path.join(allocator, &.{ prefix, entry.name });
        defer allocator.free(full_path);

        switch (entry.kind) {
            .directory => {
                var child = try dir.openDir(entry.name, .{ .iterate = true });
                defer child.close();
                try walkDirectory(allocator, &child, full_path, writer);
            },
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;
                try emitFile(allocator, dir, entry.name, full_path, writer);
            },
            else => {},
        }
    }
}

fn emitFile(allocator: std.mem.Allocator, dir: *std.fs.Dir, name: []const u8, full_path: []const u8, writer: anytype) !void {
    try writer.print("## `{s}`\n\n", .{full_path});

    const contents = try dir.readFileAlloc(allocator, name, 1 << 19);
    defer allocator.free(contents);

    var line_it = std.mem.splitScalar(u8, contents, '\n');
    var found: bool = false;
    while (line_it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "pub fn ")) {
            try writer.print("- {s}\n", .{trimmed});
            found = true;
        }
    }

    if (!found) {
        try writer.writeAll("- (no public functions)\n");
    }

    try writer.writeAll("\n");
}

fn flush(path: []const u8, data: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(data);
}
