const std = @import("std");
const model = @import("model.zig");

const SubcommandDef = struct {
    key: []const u8,
    values: []const []const u8,

    fn deinit(self: SubcommandDef, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        for (self.values) |item| allocator.free(item);
        allocator.free(self.values);
    }
};

pub fn discoverCommands(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.CliCommand {
    const source = try cwd.readFileAlloc(io, "tools/cli/commands/mod.zig", allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(source);

    var subcommands = try parseSubcommandArrays(allocator, source);
    defer {
        for (subcommands.items) |entry| entry.deinit(allocator);
        subcommands.deinit(allocator);
    }

    var commands = std.ArrayListUnmanaged(model.CliCommand).empty;
    errdefer {
        for (commands.items) |cmd| cmd.deinit(allocator);
        commands.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const name_pos_rel = std.mem.indexOfPos(u8, source, cursor, ".name = \"") orelse break;
        const name_pos = name_pos_rel;

        const block_start = std.mem.lastIndexOfScalar(u8, source[0..name_pos], '{') orelse {
            cursor = name_pos + 1;
            continue;
        };

        const block_end = findMatchingBrace(source, block_start) orelse {
            cursor = name_pos + 1;
            continue;
        };

        const block = source[block_start .. block_end + 1];

        const name = extractQuotedAfter(allocator, block, ".name = \"") orelse {
            cursor = block_end + 1;
            continue;
        };
        errdefer allocator.free(name);

        const description = extractQuotedAfter(allocator, block, ".description = \"") orelse try allocator.dupe(u8, "");
        errdefer allocator.free(description);

        const aliases = try extractQuotedArrayAfter(allocator, block, ".aliases = &.{");
        errdefer {
            for (aliases) |alias| allocator.free(alias);
            allocator.free(aliases);
        }

        const subcommands_ref = extractIdentifierAfter(block, ".subcommands = &");
        var command_subcommands = std.ArrayListUnmanaged([]const u8).empty;
        defer command_subcommands.deinit(allocator);

        if (subcommands_ref) |ref| {
            if (lookupSubcommands(subcommands.items, ref)) |items| {
                for (items) |sub| {
                    try command_subcommands.append(allocator, try allocator.dupe(u8, sub));
                }
            }
        }

        try commands.append(allocator, .{
            .name = name,
            .description = description,
            .aliases = aliases,
            .subcommands = try command_subcommands.toOwnedSlice(allocator),
        });

        cursor = block_end + 1;
    }

    insertionSortCommands(commands.items);
    return commands.toOwnedSlice(allocator);
}

fn parseSubcommandArrays(allocator: std.mem.Allocator, source: []const u8) !std.ArrayListUnmanaged(SubcommandDef) {
    var out = std.ArrayListUnmanaged(SubcommandDef).empty;
    errdefer {
        for (out.items) |entry| entry.deinit(allocator);
        out.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const const_pos = std.mem.indexOfPos(u8, source, cursor, "const ") orelse break;
        const after_const = source[const_pos + "const ".len ..];
        const eq_pos = std.mem.indexOf(u8, after_const, " = [_][]const u8{") orelse {
            cursor = const_pos + "const ".len;
            continue;
        };

        const key = std.mem.trim(u8, after_const[0..eq_pos], " \t\r\n");
        if (!std.mem.endsWith(u8, key, "_subcommands")) {
            cursor = const_pos + "const ".len;
            continue;
        }

        const list_start = const_pos + "const ".len + eq_pos + " = [_][]const u8{".len;
        const list_end = std.mem.indexOfPos(u8, source, list_start, "};") orelse {
            cursor = list_start;
            continue;
        };

        const list_text = source[list_start..list_end];
        const values = try parseQuotedList(allocator, list_text);

        try out.append(allocator, .{
            .key = try allocator.dupe(u8, key),
            .values = values,
        });

        cursor = list_end + 2;
    }

    return out;
}

fn parseQuotedList(allocator: std.mem.Allocator, input: []const u8) ![]const []const u8 {
    var items = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (items.items) |item| allocator.free(item);
        items.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const q1 = std.mem.indexOfPos(u8, input, cursor, "\"") orelse break;
        const q2 = std.mem.indexOfPos(u8, input, q1 + 1, "\"") orelse break;
        const value = input[q1 + 1 .. q2];
        try items.append(allocator, try allocator.dupe(u8, value));
        cursor = q2 + 1;
    }

    return items.toOwnedSlice(allocator);
}

fn extractQuotedAfter(allocator: std.mem.Allocator, haystack: []const u8, needle: []const u8) ?[]u8 {
    const start = std.mem.indexOf(u8, haystack, needle) orelse return null;
    const tail = haystack[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, tail, '"') orelse return null;
    return allocator.dupe(u8, tail[0..end]) catch null;
}

fn extractQuotedArrayAfter(allocator: std.mem.Allocator, haystack: []const u8, needle: []const u8) ![]const []const u8 {
    const start = std.mem.indexOf(u8, haystack, needle) orelse return allocator.dupe([]const u8, &.{});
    const tail = haystack[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, tail, '}') orelse return allocator.dupe([]const u8, &.{});
    const list_text = tail[0..end];
    return parseQuotedList(allocator, list_text);
}

fn extractIdentifierAfter(haystack: []const u8, needle: []const u8) ?[]const u8 {
    const start = std.mem.indexOf(u8, haystack, needle) orelse return null;
    const tail = haystack[start + needle.len ..];
    var end: usize = 0;
    while (end < tail.len and (std.ascii.isAlphabetic(tail[end]) or std.ascii.isDigit(tail[end]) or tail[end] == '_')) : (end += 1) {}
    if (end == 0) return null;
    return tail[0..end];
}

fn lookupSubcommands(defs: []const SubcommandDef, key: []const u8) ?[]const []const u8 {
    for (defs) |def| {
        if (std.mem.eql(u8, def.key, key)) return def.values;
    }
    return null;
}

fn findMatchingBrace(source: []const u8, open_idx: usize) ?usize {
    var depth: usize = 0;
    var in_string = false;
    var i = open_idx;
    while (i < source.len) : (i += 1) {
        const ch = source[i];
        if (ch == '"' and (i == 0 or source[i - 1] != '\\')) {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        switch (ch) {
            '{' => depth += 1,
            '}' => {
                depth -|= 1;
                if (depth == 0) return i;
            },
            else => {},
        }
    }
    return null;
}

fn insertionSortCommands(items: []model.CliCommand) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const value = items[i];
        var j = i;
        while (j > 0 and model.compareCommands({}, value, items[j - 1])) : (j -= 1) {
            items[j] = items[j - 1];
        }
        items[j] = value;
    }
}

test "parseQuotedList parses quoted tokens" {
    const parsed = try parseQuotedList(std.testing.allocator, "\"a\", \"b\", \"c\"");
    defer {
        for (parsed) |item| std.testing.allocator.free(item);
        std.testing.allocator.free(parsed);
    }
    try std.testing.expectEqual(@as(usize, 3), parsed.len);
    try std.testing.expectEqualStrings("b", parsed[1]);
}
