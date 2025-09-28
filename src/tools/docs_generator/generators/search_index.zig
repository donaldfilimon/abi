const std = @import("std");

pub fn generateSearchIndex(allocator: std.mem.Allocator) !void {
    var dir = std.fs.cwd().openDir("docs/generated", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => {
            try std.fs.cwd().makePath("docs/generated");
            return flushEmpty();
        },
        else => return err,
    };
    defer dir.close();

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("[\n");
    var first = true;

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".md")) continue;

        const file_path = try std.fmt.allocPrint(allocator, "generated/{s}", .{entry.name});
        defer allocator.free(file_path);

        const contents = try dir.readFileAlloc(allocator, entry.name, 1 << 19);
        defer allocator.free(contents);

        const title = try extractTitle(allocator, contents);
        defer allocator.free(title);

        if (!first) {
            try writer.writeAll(",\n");
        } else {
            first = false;
        }

        try writer.writeAll("  { \"file\": ");
        try writeJsonString(writer, file_path);
        try writer.writeAll(", \"title\": ");
        try writeJsonString(writer, title);
        try writer.writeAll(" }");
    }

    if (!first) {
        try writer.writeAll("\n");
    }
    try writer.writeAll("]\n");

    try flush("docs/generated/search_index.json", buffer.items);
}

fn flushEmpty() !void {
    try flush("docs/generated/search_index.json", "[]\n");
}

fn extractTitle(allocator: std.mem.Allocator, contents: []const u8) ![]const u8 {
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (trimmed[0] != '#') continue;

        var idx: usize = 0;
        while (idx < trimmed.len and (trimmed[idx] == '#' or trimmed[idx] == ' ')) : (idx += 1) {}
        if (idx >= trimmed.len) continue;
        return try allocator.dupe(u8, trimmed[idx..]);
    }
    return allocator.dupe(u8, "Untitled");
}

fn writeJsonString(writer: anytype, text: []const u8) !void {
    try writer.writeByte('"');
    for (text) |ch| {
        switch (ch) {
            '"', '\\' => {
                try writer.writeByte('\\');
                try writer.writeByte(ch);
            },
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(ch),
        }
    }
    try writer.writeByte('"');
}

fn flush(path: []const u8, data: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(data);
}
