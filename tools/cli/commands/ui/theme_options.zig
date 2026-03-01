//! Shared theme option parsing for interactive TUI commands.
//!
//! Supports:
//! - --theme <name>
//! - --list-themes
//! - --help/-h/help passthrough detection

const std = @import("std");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");

pub const ParseResult = struct {
    initial_theme: ?*const tui.Theme,
    list_themes: bool,
    wants_help: bool,
    remaining_args: []const [:0]const u8,
    allocator: std.mem.Allocator,
    owns_remaining: bool,

    pub fn deinit(self: *ParseResult) void {
        if (self.owns_remaining) {
            self.allocator.free(self.remaining_args);
            self.owns_remaining = false;
            self.remaining_args = &.{};
        }
    }
};

pub fn parseThemeArgs(
    allocator: std.mem.Allocator,
    args: []const [:0]const u8,
) !ParseResult {
    var remaining: std.ArrayListUnmanaged([:0]const u8) = .empty;
    errdefer remaining.deinit(allocator);

    const wants_help = utils.args.containsHelpArgs(args);
    var list_themes = false;
    var initial_theme: ?*const tui.Theme = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (std.mem.eql(u8, arg, "--list-themes")) {
            list_themes = true;
            continue;
        }

        if (std.mem.eql(u8, arg, "--theme")) {
            const next_i = i + 1;
            if (next_i >= args.len) {
                if (!wants_help) {
                    utils.output.printError("Missing theme name after --theme.", .{});
                    printThemeHint();
                    return error.InvalidArgument;
                }
                continue;
            }

            const theme_name = std.mem.sliceTo(args[next_i], 0);
            i = next_i;

            const resolved = tui.themes.lookupTheme(theme_name);
            if (resolved == null) {
                if (!wants_help) {
                    utils.output.printError("Unknown theme: {s}", .{theme_name});
                    printThemeHint();
                    return error.InvalidArgument;
                }
                continue;
            }
            initial_theme = resolved.?;
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--help", "-h", "help" })) {
            continue;
        }

        try remaining.append(allocator, args[i]);
    }

    const remaining_args = try remaining.toOwnedSlice(allocator);

    return .{
        .initial_theme = initial_theme,
        .list_themes = list_themes,
        .wants_help = wants_help,
        .remaining_args = remaining_args,
        .allocator = allocator,
        .owns_remaining = true,
    };
}

pub fn printAvailableThemes() void {
    const names = tui.themes.themeNames();
    utils.output.println("Available themes:", .{});
    for (names) |name| {
        utils.output.println("  {s}", .{name});
    }
}

pub fn printThemeHint() void {
    const names = tui.themes.themeNames();
    utils.output.print("Valid themes: ", .{});
    for (names, 0..) |name, idx| {
        if (idx > 0) utils.output.print(", ", .{});
        utils.output.print("{s}", .{name});
    }
    utils.output.println("", .{});
    utils.output.println("Use --list-themes to see available names.", .{});
}

pub fn themeNotificationMessage(theme_name: []const u8) []const u8 {
    return if (std.mem.eql(u8, theme_name, "default"))
        "Theme: default"
    else if (std.mem.eql(u8, theme_name, "monokai"))
        "Theme: monokai"
    else if (std.mem.eql(u8, theme_name, "solarized"))
        "Theme: solarized"
    else if (std.mem.eql(u8, theme_name, "nord"))
        "Theme: nord"
    else if (std.mem.eql(u8, theme_name, "gruvbox"))
        "Theme: gruvbox"
    else if (std.mem.eql(u8, theme_name, "high_contrast"))
        "Theme: high_contrast"
    else if (std.mem.eql(u8, theme_name, "minimal"))
        "Theme: minimal"
    else
        "Theme changed";
}

test "parseThemeArgs parses valid theme and list flag" {
    const allocator = std.testing.allocator;
    const args = [_][:0]const u8{ "--theme", "nord", "--list-themes" };

    var parsed = try parseThemeArgs(allocator, &args);
    defer parsed.deinit();

    try std.testing.expect(parsed.initial_theme != null);
    try std.testing.expectEqualStrings("nord", parsed.initial_theme.?.name);
    try std.testing.expect(parsed.list_themes);
    try std.testing.expect(!parsed.wants_help);
    try std.testing.expectEqual(@as(usize, 0), parsed.remaining_args.len);
}

test "parseThemeArgs keeps unknown args as remaining" {
    const allocator = std.testing.allocator;
    const args = [_][:0]const u8{ "--theme", "default", "--foo", "bar" };

    var parsed = try parseThemeArgs(allocator, &args);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 2), parsed.remaining_args.len);
    try std.testing.expectEqualStrings("--foo", parsed.remaining_args[0]);
    try std.testing.expectEqualStrings("bar", parsed.remaining_args[1]);
}

test "themeNotificationMessage returns named messages and fallback" {
    try std.testing.expectEqualStrings("Theme: nord", themeNotificationMessage("nord"));
    try std.testing.expectEqualStrings("Theme changed", themeNotificationMessage("unknown"));
}

test {
    std.testing.refAllDecls(@This());
}
