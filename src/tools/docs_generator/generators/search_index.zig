const std = @import("std");

fn docPathLessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

pub fn generateSearchIndex(allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Ensure output dir exists
    try std.fs.cwd().makePath("docs/generated");

    // Collect Markdown files in docs/generated
    var dir = std.fs.cwd().openDir("docs/generated", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return, // nothing to index yet
        else => return err,
    };
    defer dir.close();

    var files = std.ArrayListUnmanaged([]const u8){};
    defer files.deinit(a);

    var it = dir.iterate();
    while (it.next() catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".md")) {
            const rel = try std.fs.path.join(a, &[_][]const u8{ "generated", entry.name });
            try files.append(a, rel);
        }
    }

    std.sort.block([]const u8, files.items, {}, docPathLessThan);

    var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
    defer out.close();

    try out.writeAll("[\n");
    var first = true;

    for (files.items) |rel| {
        const full = try std.fs.path.join(a, &[_][]const u8{ "docs", rel });
        // Normalize relative path for web (forward slashes)
        const rel_web = try a.dupe(u8, rel);
        for (rel_web) |*ch| {
            if (ch.* == std.fs.path.sep) ch.* = '/';
        }
        var title_buf: []const u8 = "";
        var excerpt_buf: []const u8 = "";
        getTitleAndExcerpt(a, full, &title_buf, &excerpt_buf) catch {
            // Fallbacks
            title_buf = std.fs.path.basename(rel);
            excerpt_buf = "";
        };

        if (!first) {
            try out.writeAll(",\n");
        } else {
            first = false;
        }

        try out.writeAll("  {\"file\": ");
        try writeJsonString(out, rel_web);
        try out.writeAll(", \"title\": ");
        try writeJsonString(out, title_buf);
        try out.writeAll(", \"excerpt\": ");
        try writeJsonString(out, excerpt_buf);
        try out.writeAll("}");
    }

    try out.writeAll("\n]\n");
}

fn getTitleAndExcerpt(allocator: std.mem.Allocator, path: []const u8, title_out: *[]const u8, excerpt_out: *[]const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const max_bytes: usize = 16 * 1024 * 1024;
    const data = try std.fs.cwd().readFileAlloc(path, allocator, std.Io.Limit.limited(max_bytes));
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var first_heading: ?[]const u8 = null;
    var in_code = false;

    var excerpt: std.ArrayListUnmanaged(u8) = .empty;
    defer excerpt.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) continue;

        if (first_heading == null and std.mem.startsWith(u8, trimmed, "#")) {
            var j: usize = 0;
            while (j < trimmed.len and trimmed[j] == '#') j += 1;
            const after = std.mem.trim(u8, trimmed[j..], " \t");
            if (after.len > 0) first_heading = after;
            continue;
        }

        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue; // skip headings in excerpt
        if (trimmed[0] == '|') continue; // skip tables

        // Append to excerpt up to ~300 chars
        if (excerpt.items.len > 0) try excerpt.append(allocator, ' ');
        var k: usize = 0;
        while (k < trimmed.len and excerpt.items.len < 300) : (k += 1) {
            const c = trimmed[k];
            if (c == '`') continue;
            try excerpt.append(allocator, c);
        }
        if (excerpt.items.len >= 300) break;
    }

    if (first_heading) |h| {
        title_out.* = try allocator.dupe(u8, h);
    } else {
        const base = std.fs.path.basename(path);
        title_out.* = try allocator.dupe(u8, base);
    }
    excerpt_out.* = try allocator.dupe(u8, excerpt.items);
}

fn writeJsonString(out: std.fs.File, s: []const u8) !void {
    try out.writeAll("\"");
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        switch (c) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => {
                var buf: [1]u8 = .{c};
                try out.writeAll(buf[0..1]);
            },
        }
    }
    try out.writeAll("\"");

}
