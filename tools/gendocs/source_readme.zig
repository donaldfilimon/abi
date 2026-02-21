const std = @import("std");
const model = @import("model.zig");

pub fn collectReadmeSummaries(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.ReadmeSummary {
    var root = try cwd.openDir(io, "src", .{ .iterate = true });
    defer root.close(io);

    var summaries = std.ArrayListUnmanaged(model.ReadmeSummary).empty;
    errdefer {
        for (summaries.items) |summary| summary.deinit(allocator);
        summaries.deinit(allocator);
    }

    try walkReadmes(allocator, io, root, "src", &summaries);
    insertionSortReadmes(summaries.items);
    return summaries.toOwnedSlice(allocator);
}

fn walkReadmes(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    prefix: []const u8,
    out: *std.ArrayListUnmanaged(model.ReadmeSummary),
) !void {
    var it = dir.iterate();
    while (true) {
        const maybe_entry = try it.next(io);
        const entry = maybe_entry orelse break;

        const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, entry.name });
        defer allocator.free(path);

        switch (entry.kind) {
            .file => {
                if (!std.mem.eql(u8, entry.name, "README.md")) continue;
                const summary = try parseReadmeFile(allocator, io, path);
                try out.append(allocator, summary);
            },
            .directory => {
                var child = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
                defer child.close(io);
                try walkReadmes(allocator, io, child, path, out);
            },
            else => {},
        }
    }
}

fn parseReadmeFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !model.ReadmeSummary {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(content);

    var title: []const u8 = "";
    var paragraph = std.ArrayListUnmanaged(u8).empty;
    defer paragraph.deinit(allocator);

    var lines = std.mem.splitScalar(u8, content, '\n');
    var in_paragraph = false;

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");

        if (title.len == 0 and std.mem.startsWith(u8, line, "# ")) {
            title = std.mem.trim(u8, line[2..], " \t\r");
            continue;
        }

        if (line.len == 0) {
            if (in_paragraph and paragraph.items.len > 0) break;
            continue;
        }

        if (std.mem.startsWith(u8, line, "#")) continue;
        if (std.mem.startsWith(u8, line, "|")) continue;
        if (std.mem.startsWith(u8, line, "```")) continue;

        in_paragraph = true;
        if (paragraph.items.len > 0) try paragraph.append(allocator, ' ');
        try paragraph.appendSlice(allocator, line);
    }

    return .{
        .path = try allocator.dupe(u8, path),
        .title = try allocator.dupe(u8, if (title.len > 0) title else path),
        .summary = try allocator.dupe(u8, std.mem.trim(u8, paragraph.items, " \t\r\n")),
    };
}

fn insertionSortReadmes(items: []model.ReadmeSummary) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.compareReadmes({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

test "readme summary parser extracts heading and paragraph" {
    const sample =
        \\# Module Name
        \\
        \\This module does useful work.
        \\It keeps data deterministic.
        \\
        \\## Next
    ;
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const path = "tools/gendocs/.tmp_readme_test.md";
    {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, sample);
    }
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    const summary = try parseReadmeFile(std.testing.allocator, io, path);
    defer summary.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("Module Name", summary.title);
    try std.testing.expect(std.mem.indexOf(u8, summary.summary, "deterministic") != null);
}
