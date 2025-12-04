const std = @import("std");

const DocEntry = struct {
    os_path: []const u8,
    web_path: []const u8,
};

fn docPathLessThan(_: void, lhs: DocEntry, rhs: DocEntry) bool {
    return std.mem.lessThan(u8, lhs.web_path, rhs.web_path);
}

fn shouldSkipPath(rel_web: []const u8) bool {
    if (std.mem.startsWith(u8, rel_web, "generated/")) return false;

    const allowlist = [_][]const u8{
        "AGENTS_EXECUTIVE_SUMMARY.md",
    };

    for (allowlist) |allowed| {
        if (std.mem.eql(u8, rel_web, allowed)) return false;
    }

    // Skip everything else to avoid surfacing legacy Markdown duplicates.
    return true;
}

pub fn generateSearchIndex(allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Ensure output dir exists even if no Markdown files were produced yet.
    try std.fs.cwd().makePath("docs/generated");

    var files = std.ArrayListUnmanaged(DocEntry){};
    defer files.deinit(a);

    var docs_dir = std.fs.cwd().openDir("docs", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer docs_dir.close();

    var walker = try docs_dir.walk(a);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".md")) continue;

        const os_rel = try a.dupe(u8, entry.path);
        var web_rel = try a.dupe(u8, os_rel);
        for (web_rel) |*ch| {
            if (ch.* == std.fs.path.sep) ch.* = '/';
        }

        if (shouldSkipPath(web_rel)) continue;

        try files.append(a, .{ .os_path = os_rel, .web_path = web_rel });
    }

    std.sort.block(DocEntry, files.items, {}, docPathLessThan);

    var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
    defer out.close();

    try out.writeAll("[\n");
    var first = true;

    for (files.items) |entry| {
        const full = try std.fs.path.join(a, &[_][]const u8{ "docs", entry.os_path });
        var title_buf: []const u8 = "";
        var excerpt_buf: []const u8 = "";
        getTitleAndExcerpt(a, full, &title_buf, &excerpt_buf) catch {
            title_buf = std.fs.path.basename(entry.os_path);
            excerpt_buf = "";
        };

        if (!first) {
            try out.writeAll(",\n");
        } else {
            first = false;
        }

        try out.writeAll("  {\"file\": ");
        try writeJsonString(out, entry.web_path);
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
    const data = try std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var first_heading: ?[]const u8 = null;
    var in_code = false;
    var in_front_matter = false;
    var front_matter_processed = false;

    var excerpt: std.ArrayListUnmanaged(u8) = .empty;
    defer excerpt.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (!front_matter_processed) {
            if (trimmed.len == 0 and !in_front_matter) {
                continue;
            }

            if (std.mem.eql(u8, trimmed, "---")) {
                in_front_matter = !in_front_matter;
                if (!in_front_matter) {
                    front_matter_processed = true;
                }
                continue;
            }

            if (!in_front_matter) {
                front_matter_processed = true;
            }
        }

        if (in_front_matter) continue;

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
        if (trimmed[0] == '<') continue; // skip HTML scaffolding
        if (std.mem.eql(u8, trimmed, "---")) continue; // skip horizontal rules

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
