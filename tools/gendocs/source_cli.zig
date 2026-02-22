const std = @import("std");
const model = @import("model.zig");

/// Discover CLI commands by scanning actual command module files.
///
/// Reads `tools/cli/commands/mod.zig` to find all `pub const X = @import("path");`
/// lines, then reads each imported file to extract `pub const meta: command_mod.Meta`
/// blocks containing name, description, aliases, subcommands, and children.
pub fn discoverCommands(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.CliCommand {
    // Step 1: Read the command registry (mod.zig) to discover import paths.
    const mod_source = cwd.readFileAlloc(io, "tools/cli/commands/mod.zig", allocator, .limited(256 * 1024)) catch {
        return allocator.dupe(model.CliCommand, &.{});
    };
    defer allocator.free(mod_source);

    var import_paths = std.ArrayListUnmanaged(ImportEntry).empty;
    defer {
        for (import_paths.items) |entry| {
            allocator.free(entry.path);
        }
        import_paths.deinit(allocator);
    }

    try parseImportPaths(allocator, mod_source, &import_paths);

    // Step 2: For each import, read the file and parse its meta block.
    var commands = std.ArrayListUnmanaged(model.CliCommand).empty;
    errdefer {
        for (commands.items) |cmd| cmd.deinit(allocator);
        commands.deinit(allocator);
    }

    for (import_paths.items) |entry| {
        const file_path = try std.fmt.allocPrint(allocator, "tools/cli/commands/{s}", .{entry.path});
        defer allocator.free(file_path);

        const source = cwd.readFileAlloc(io, file_path, allocator, .limited(1 * 1024 * 1024)) catch continue;
        defer allocator.free(source);

        if (try parseMetaBlock(allocator, source)) |cmd| {
            try commands.append(allocator, cmd);
        }
    }

    std.mem.sort(model.CliCommand, commands.items, {}, model.compareCommands);
    return commands.toOwnedSlice(allocator);
}

// ─── Import path parsing ─────────────────────────────────────────────────────

const ImportEntry = struct {
    path: []const u8,
};

/// Parse `pub const X = @import("path");` lines from mod.zig source.
/// Skips lines that are comments or appear after the `command_modules` tuple
/// (those are infrastructure, not command imports).
fn parseImportPaths(
    allocator: std.mem.Allocator,
    source: []const u8,
    out: *std.ArrayListUnmanaged(ImportEntry),
) !void {
    const marker = "pub const ";
    const import_prefix = "= @import(\"";

    var cursor: usize = 0;
    while (cursor < source.len) {
        // Find start of next line.
        const line_start = cursor;
        const line_end = std.mem.indexOfScalarPos(u8, source, cursor, '\n') orelse source.len;
        const line = std.mem.trim(u8, source[line_start..line_end], " \t\r");
        cursor = if (line_end < source.len) line_end + 1 else source.len;

        // Stop when we hit the comptime-derived section (non-import declarations).
        if (std.mem.startsWith(u8, line, "const command_modules")) break;
        if (std.mem.startsWith(u8, line, "pub const descriptors")) break;

        // Must start with `pub const`.
        if (!std.mem.startsWith(u8, line, marker)) continue;

        // Must contain `= @import("`.
        const import_pos = std.mem.indexOf(u8, line, import_prefix) orelse continue;
        const path_start = import_pos + import_prefix.len;
        const path_end = std.mem.indexOfPos(u8, line, path_start, "\"") orelse continue;
        const import_path = line[path_start..path_end];

        // Skip non-command imports (e.g., std, command_mod, CommandDescriptor).
        if (!std.mem.endsWith(u8, import_path, ".zig")) continue;

        try out.append(allocator, .{
            .path = try allocator.dupe(u8, import_path),
        });
    }
}

// ─── Meta block parsing ──────────────────────────────────────────────────────

/// Parse the `pub const meta: command_mod.Meta = .{ ... };` block from a command
/// source file and return a `model.CliCommand`, or null if no meta is found.
fn parseMetaBlock(allocator: std.mem.Allocator, source: []const u8) !?model.CliCommand {
    const meta_needle = "pub const meta:";
    const meta_start = std.mem.indexOf(u8, source, meta_needle) orelse return null;

    // Find the opening `.{` of the meta struct literal.
    const dot_brace = std.mem.indexOfPos(u8, source, meta_start, ".{") orelse return null;
    // The brace we track is the `{` after the dot.
    const open_brace = dot_brace + 1;
    const close_brace = findMatchingBrace(source, open_brace) orelse return null;
    const meta_body = source[open_brace + 1 .. close_brace];

    // Extract .name
    const name = extractQuotedField(allocator, meta_body, ".name") orelse return null;
    errdefer allocator.free(name);

    // Extract .description
    const description = extractQuotedField(allocator, meta_body, ".description") orelse try allocator.dupe(u8, "");
    errdefer allocator.free(description);

    // Extract .aliases
    const aliases = try extractStringArray(allocator, meta_body, ".aliases");
    errdefer {
        for (aliases) |a| allocator.free(a);
        allocator.free(aliases);
    }

    // Extract .subcommands
    const subcommands_raw = try extractStringArray(allocator, meta_body, ".subcommands");
    defer {
        for (subcommands_raw) |s| allocator.free(s);
        allocator.free(subcommands_raw);
    }

    // Extract child names from .children block and merge (deduplicated).
    const child_names = try extractChildNames(allocator, meta_body);
    defer {
        for (child_names) |c| allocator.free(c);
        allocator.free(child_names);
    }

    // Merge subcommands_raw + child_names, deduplicating.
    var merged = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (merged.items) |s| allocator.free(s);
        merged.deinit(allocator);
    }

    for (subcommands_raw) |sub| {
        try merged.append(allocator, try allocator.dupe(u8, sub));
    }

    for (child_names) |child| {
        if (!containsString(merged.items, child)) {
            try merged.append(allocator, try allocator.dupe(u8, child));
        }
    }

    return .{
        .name = name,
        .description = description,
        .aliases = aliases,
        .subcommands = try merged.toOwnedSlice(allocator),
    };
}

// ─── Field extraction helpers ────────────────────────────────────────────────

/// Extract a quoted string value from a field like `.name = "value"`.
fn extractQuotedField(allocator: std.mem.Allocator, body: []const u8, field: []const u8) ?[]u8 {
    // Find the field name in the body.
    var search_pos: usize = 0;
    while (true) {
        const field_pos = std.mem.indexOfPos(u8, body, search_pos, field) orelse return null;
        // Ensure this is the start of a field assignment (preceded by whitespace/comma/brace).
        // Look for `= "` after the field name.
        const after_field = body[field_pos + field.len ..];
        const trimmed = std.mem.trimStart(u8, after_field, " \t\r\n");
        if (std.mem.startsWith(u8, trimmed, "= \"")) {
            const quote_start = field_pos + field.len + (@intFromPtr(trimmed.ptr) - @intFromPtr(after_field.ptr)) + "= \"".len;
            const quote_end = std.mem.indexOfPos(u8, body, quote_start, "\"") orelse return null;
            return allocator.dupe(u8, body[quote_start..quote_end]) catch return null;
        }
        // Not a direct string assignment (could be `= &.{` etc.), skip.
        search_pos = field_pos + field.len;
    }
}

/// Extract an array of quoted strings from a field like `.aliases = &.{ "a", "b" }`.
fn extractStringArray(allocator: std.mem.Allocator, body: []const u8, field: []const u8) ![]const []const u8 {
    const field_pos = std.mem.indexOf(u8, body, field) orelse return allocator.dupe([]const u8, &.{});
    const after_field = body[field_pos + field.len ..];
    const trimmed = std.mem.trimStart(u8, after_field, " \t\r\n");

    // Expect `= &.{`
    if (!std.mem.startsWith(u8, trimmed, "= &.{")) return allocator.dupe([]const u8, &.{});

    const brace_offset = field_pos + field.len + (@intFromPtr(trimmed.ptr) - @intFromPtr(after_field.ptr)) + "= &.{".len;
    const close = std.mem.indexOfPos(u8, body, brace_offset, "}") orelse return allocator.dupe([]const u8, &.{});

    return parseQuotedList(allocator, body[brace_offset..close]);
}

/// Extract child names from a `.children = &.{ .{ .name = "x", ... }, ... }` block.
fn extractChildNames(allocator: std.mem.Allocator, body: []const u8) ![]const []const u8 {
    const field = ".children";
    const field_pos = std.mem.indexOf(u8, body, field) orelse return allocator.dupe([]const u8, &.{});
    const after_field = body[field_pos + field.len ..];
    const trimmed = std.mem.trimStart(u8, after_field, " \t\r\n");

    // Expect `= &.{`
    if (!std.mem.startsWith(u8, trimmed, "= &.{")) return allocator.dupe([]const u8, &.{});

    // Find the matching `}` for the outer array brace.
    const ampersand_dot_brace = field_pos + field.len + (@intFromPtr(trimmed.ptr) - @intFromPtr(after_field.ptr)) + "= &".len;
    // The brace is right after `&.`
    const outer_open = ampersand_dot_brace + 1; // position of `{` in `&.{`
    const outer_close = findMatchingBrace(body, outer_open) orelse return allocator.dupe([]const u8, &.{});
    const children_body = body[outer_open + 1 .. outer_close];

    // Scan for `.name = "..."` within each child struct literal.
    var names = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const name_field = std.mem.indexOfPos(u8, children_body, cursor, ".name") orelse break;
        const after_name = children_body[name_field + ".name".len ..];
        const after_name_trimmed = std.mem.trimStart(u8, after_name, " \t\r\n");
        if (std.mem.startsWith(u8, after_name_trimmed, "= \"")) {
            const q_start = name_field + ".name".len + (@intFromPtr(after_name_trimmed.ptr) - @intFromPtr(after_name.ptr)) + "= \"".len;
            const q_end = std.mem.indexOfPos(u8, children_body, q_start, "\"") orelse break;
            try names.append(allocator, try allocator.dupe(u8, children_body[q_start..q_end]));
            cursor = q_end + 1;
        } else {
            cursor = name_field + ".name".len;
        }
    }

    return names.toOwnedSlice(allocator);
}

// ─── Utility functions ───────────────────────────────────────────────────────

/// Parse quoted strings from a comma-separated list like `"a", "b", "c"`.
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

/// Find the index of the closing `}` that matches the opening `{` at `open_idx`.
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

/// Check if a string slice is already present in a list.
fn containsString(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "parseQuotedList parses quoted tokens" {
    const parsed = try parseQuotedList(std.testing.allocator, "\"a\", \"b\", \"c\"");
    defer {
        for (parsed) |item| std.testing.allocator.free(item);
        std.testing.allocator.free(parsed);
    }
    try std.testing.expectEqual(@as(usize, 3), parsed.len);
    try std.testing.expectEqualStrings("a", parsed[0]);
    try std.testing.expectEqualStrings("b", parsed[1]);
    try std.testing.expectEqualStrings("c", parsed[2]);
}

test "parseMetaBlock extracts simple command" {
    const source =
        \\const std = @import("std");
        \\const command_mod = @import("../command.zig");
        \\
        \\pub const meta: command_mod.Meta = .{
        \\    .name = "db",
        \\    .description = "Database operations (add, query, stats)",
        \\    .aliases = &.{"ls"},
        \\    .subcommands = &.{ "add", "query", "stats", "help" },
        \\};
    ;

    const cmd = (try parseMetaBlock(std.testing.allocator, source)).?;
    defer cmd.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("db", cmd.name);
    try std.testing.expectEqualStrings("Database operations (add, query, stats)", cmd.description);
    try std.testing.expectEqual(@as(usize, 1), cmd.aliases.len);
    try std.testing.expectEqualStrings("ls", cmd.aliases[0]);
    try std.testing.expectEqual(@as(usize, 4), cmd.subcommands.len);
    try std.testing.expectEqualStrings("add", cmd.subcommands[0]);
    try std.testing.expectEqualStrings("help", cmd.subcommands[3]);
}

test "parseMetaBlock extracts command with children and deduplicates" {
    const source =
        \\pub const meta: command_mod.Meta = .{
        \\    .name = "config",
        \\    .description = "Configuration management",
        \\    .subcommands = &.{ "init", "show", "help" },
        \\    .children = &.{
        \\        .{ .name = "init", .description = "Generate config", .handler = .{ .basic = wrapInit } },
        \\        .{ .name = "show", .description = "Display config", .handler = .{ .basic = wrapShow } },
        \\        .{ .name = "extra", .description = "Extra child", .handler = .{ .basic = wrapExtra } },
        \\    },
        \\};
    ;

    const cmd = (try parseMetaBlock(std.testing.allocator, source)).?;
    defer cmd.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("config", cmd.name);
    try std.testing.expectEqualStrings("Configuration management", cmd.description);
    // "init" and "show" come from subcommands; "extra" is added from children (not duplicated).
    try std.testing.expectEqual(@as(usize, 4), cmd.subcommands.len);
    try std.testing.expectEqualStrings("init", cmd.subcommands[0]);
    try std.testing.expectEqualStrings("show", cmd.subcommands[1]);
    try std.testing.expectEqualStrings("help", cmd.subcommands[2]);
    try std.testing.expectEqualStrings("extra", cmd.subcommands[3]);
}

test "parseMetaBlock handles forward command without aliases or subcommands" {
    const source =
        \\pub const meta: command_mod.Meta = .{
        \\    .name = "launch-ui",
        \\    .description = "Launch interactive UI command menu",
        \\    .io_mode = .io,
        \\    .forward = .{
        \\        .target = "ui",
        \\        .prepend_args = &[_][:0]const u8{"launch"},
        \\        .warning = "'abi launch-ui' is deprecated; use 'abi ui launch'.",
        \\    },
        \\};
    ;

    const cmd = (try parseMetaBlock(std.testing.allocator, source)).?;
    defer cmd.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("launch-ui", cmd.name);
    try std.testing.expectEqualStrings("Launch interactive UI command menu", cmd.description);
    try std.testing.expectEqual(@as(usize, 0), cmd.aliases.len);
    try std.testing.expectEqual(@as(usize, 0), cmd.subcommands.len);
}

test "parseMetaBlock returns null for file without meta" {
    const source =
        \\const std = @import("std");
        \\pub fn run() void {}
    ;
    const result = try parseMetaBlock(std.testing.allocator, source);
    try std.testing.expect(result == null);
}

test "parseImportPaths extracts import entries from mod.zig" {
    const source =
        \\const std = @import("std");
        \\const command_mod = @import("../command.zig");
        \\
        \\pub const db = @import("db.zig");
        \\pub const agent = @import("agent.zig");
        \\pub const bench = @import("bench/mod.zig");
        \\pub const llm = @import("llm/mod.zig");
        \\
        \\const command_modules = .{
        \\    db, agent, bench, llm,
        \\};
    ;

    var entries = std.ArrayListUnmanaged(ImportEntry).empty;
    defer {
        for (entries.items) |e| std.testing.allocator.free(e.path);
        entries.deinit(std.testing.allocator);
    }

    try parseImportPaths(std.testing.allocator, source, &entries);

    try std.testing.expectEqual(@as(usize, 4), entries.items.len);
    try std.testing.expectEqualStrings("db.zig", entries.items[0].path);
    try std.testing.expectEqualStrings("agent.zig", entries.items[1].path);
    try std.testing.expectEqualStrings("bench/mod.zig", entries.items[2].path);
    try std.testing.expectEqualStrings("llm/mod.zig", entries.items[3].path);
}

test "findMatchingBrace handles nested braces" {
    const source = "{ { inner } outer }";
    const result = findMatchingBrace(source, 0);
    try std.testing.expectEqual(@as(?usize, 18), result);
}

test "findMatchingBrace handles strings with braces" {
    const source =
        \\.{ .name = "test{}", .value = 42 }
    ;
    const result = findMatchingBrace(source, 1);
    try std.testing.expect(result != null);
    // The closing brace should be the last character.
    try std.testing.expectEqual(@as(u8, '}'), source[result.?]);
}

test "extractQuotedField extracts correct value" {
    const body =
        \\ .name = "hello", .description = "world desc",
    ;
    const name = extractQuotedField(std.testing.allocator, body, ".name");
    defer if (name) |n| std.testing.allocator.free(n);
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("hello", name.?);

    const desc = extractQuotedField(std.testing.allocator, body, ".description");
    defer if (desc) |d| std.testing.allocator.free(d);
    try std.testing.expect(desc != null);
    try std.testing.expectEqualStrings("world desc", desc.?);
}

test "extractStringArray extracts array values" {
    const body =
        \\ .aliases = &.{ "chat", "reasoning", "serve" },
    ;
    const result = try extractStringArray(std.testing.allocator, body, ".aliases");
    defer {
        for (result) |item| std.testing.allocator.free(item);
        std.testing.allocator.free(result);
    }
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqualStrings("chat", result[0]);
    try std.testing.expectEqualStrings("reasoning", result[1]);
    try std.testing.expectEqualStrings("serve", result[2]);
}

test "extractStringArray returns empty for missing field" {
    const body =
        \\ .name = "test",
    ;
    const result = try extractStringArray(std.testing.allocator, body, ".aliases");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}
