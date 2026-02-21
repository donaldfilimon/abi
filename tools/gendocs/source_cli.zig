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
    const source = try cwd.readFileAlloc(io, "tools/cli/tests/catalog.zig", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(source);

    var subcommands = try parseSubcommandArrays(allocator, source);
    defer {
        for (subcommands.items) |entry| entry.deinit(allocator);
        subcommands.deinit(allocator);
    }

    const commands_block_start = std.mem.indexOf(u8, source, "pub const commands = [_]CommandSpec{") orelse {
        return allocator.dupe(model.CliCommand, &.{});
    };

    const array_open = std.mem.indexOfPos(u8, source, commands_block_start, "{") orelse {
        return allocator.dupe(model.CliCommand, &.{});
    };
    const array_close = findMatchingBrace(source, array_open) orelse {
        return allocator.dupe(model.CliCommand, &.{});
    };

    const block = source[array_open + 1 .. array_close];

    var commands = std.ArrayListUnmanaged(model.CliCommand).empty;
    errdefer {
        for (commands.items) |cmd| cmd.deinit(allocator);
        commands.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const entry_rel = std.mem.indexOfPos(u8, block, cursor, ".{") orelse break;
        const entry_start = entry_rel;
        const entry_end = findMatchingBrace(block, entry_start + 1) orelse break;
        const entry = block[entry_start .. entry_end + 1];

        const name = extractQuotedAfter(allocator, entry, ".name = \"") orelse {
            cursor = entry_end + 1;
            continue;
        };
        errdefer allocator.free(name);

        const description = extractQuotedAfter(allocator, entry, ".description = \"") orelse try allocator.dupe(u8, "");
        errdefer allocator.free(description);

        const aliases = try extractQuotedArrayAfter(allocator, entry, ".aliases = &.{");
        errdefer {
            for (aliases) |alias| allocator.free(alias);
            allocator.free(aliases);
        }

        const subcommands_ref = extractIdentifierAfter(entry, ".subcommands = &");
        var command_subcommands = std.ArrayListUnmanaged([]const u8).empty;
        defer command_subcommands.deinit(allocator);

        if (subcommands_ref) |ref| {
            if (lookupSubcommands(subcommands.items, ref)) |items| {
                for (items) |sub| {
                    try command_subcommands.append(allocator, try allocator.dupe(u8, sub));
                }
            }
        }

        if (command_subcommands.items.len == 0) {
            if (lookupSubcommandsByCommandName(subcommands.items, name)) |items| {
                if (std.mem.eql(u8, name, "bench")) {
                    std.debug.print("DBG bench fallback hit {d}\n", .{items.len});
                }
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

        cursor = entry_end + 1;
    }

    std.mem.sort(model.CliCommand, commands.items, {}, model.compareCommands);
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
        const const_pos = std.mem.indexOfPos(u8, source, cursor, "pub const ") orelse break;
        const after_const = source[const_pos + "pub const ".len ..];

        const line_end_rel = std.mem.indexOfScalar(u8, after_const, '\n') orelse after_const.len;
        const header_line = after_const[0..line_end_rel];
        const eq_pos = std.mem.indexOf(u8, header_line, " = [_][]const u8{") orelse {
            cursor = const_pos + "pub const ".len;
            continue;
        };

        const key = std.mem.trim(u8, header_line[0..eq_pos], " \t\r\n");
        if (!std.mem.endsWith(u8, key, "_subcommands")) {
            cursor = const_pos + "pub const ".len;
            continue;
        }

        const list_start = const_pos + "pub const ".len + eq_pos + " = [_][]const u8{".len;
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

fn lookupSubcommandsByCommandName(defs: []const SubcommandDef, command_name: []const u8) ?[]const []const u8 {
    var key_buf: [128]u8 = undefined;
    if (command_name.len + "_subcommands".len > key_buf.len) return null;

    var index: usize = 0;
    for (command_name) |ch| {
        key_buf[index] = if (ch == '-') '_' else ch;
        index += 1;
    }
    @memcpy(key_buf[index .. index + "_subcommands".len], "_subcommands");
    index += "_subcommands".len;

    return lookupSubcommands(defs, key_buf[0..index]);
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

test "parseQuotedList parses quoted tokens" {
    const parsed = try parseQuotedList(std.testing.allocator, "\"a\", \"b\", \"c\"");
    defer {
        for (parsed) |item| std.testing.allocator.free(item);
        std.testing.allocator.free(parsed);
    }
    try std.testing.expectEqual(@as(usize, 3), parsed.len);
    try std.testing.expectEqualStrings("b", parsed[1]);
}

test "discover parser finds command names in catalog source" {
    const source =
        \\pub const llm_subcommands = [_][]const u8{ "run", "session" };
        \\pub const commands = [_]CommandSpec{
        \\    .{ .name = "llm", .description = "desc", .aliases = &.{"chat"}, .subcommands = &llm_subcommands },
        \\};
    ;

    var defs = try parseSubcommandArrays(std.testing.allocator, source);
    defer {
        for (defs.items) |entry| entry.deinit(std.testing.allocator);
        defs.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), defs.items.len);
    try std.testing.expectEqualStrings("llm_subcommands", defs.items[0].key);
}
