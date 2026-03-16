//! Subcommand dispatch for CLI commands.
//!
//! Reduces boilerplate when a command has multiple subcommands (e.g. gpu backends,
//! gpu devices, llm generate). Use runSubcommand after handling top-level help.

const std = @import("std");
const args = @import("args.zig");

/// A single subcommand: name/aliases and handler.
pub const Command = struct {
    /// Primary name and aliases (e.g. &.{ "devices", "list" }).
    names: []const []const u8,
    /// Handler; receives allocator and parser (parser has remaining args after subcommand).
    run: *const fn (std.mem.Allocator, *args.ArgParser) anyerror!void,
};

/// Run help → default (if no subcommand) → matching command → on_unknown + help.
/// Call this after creating the parser and handling top-level --help.
pub fn runSubcommand(
    allocator: std.mem.Allocator,
    parser: *args.ArgParser,
    commands: []const Command,
    default_action: ?*const fn (std.mem.Allocator, *args.ArgParser) anyerror!void,
    print_help: *const fn (std.mem.Allocator) void,
    on_unknown: *const fn ([]const u8) void,
) !void {
    if (parser.wantsHelp()) {
        print_help(allocator);
        return;
    }
    if (!parser.hasMore()) {
        if (default_action) |action| {
            try action(allocator, parser);
        } else {
            print_help(allocator);
        }
        return;
    }
    const cmd = parser.next().?;
    for (commands) |c| {
        if (args.matchesAny(cmd, c.names)) {
            try c.run(allocator, parser);
            return;
        }
    }
    on_unknown(cmd);
    print_help(allocator);
}

const TestState = struct {
    var help_calls: usize = 0;
    var default_calls: usize = 0;
    var alpha_calls: usize = 0;
    var beta_calls: usize = 0;
    var unknown_calls: usize = 0;
    var last_unknown: [32]u8 = [_]u8{0} ** 32;
    var last_unknown_len: usize = 0;

    fn reset() void {
        help_calls = 0;
        default_calls = 0;
        alpha_calls = 0;
        beta_calls = 0;
        unknown_calls = 0;
        last_unknown = [_]u8{0} ** 32;
        last_unknown_len = 0;
    }

    fn help(_: std.mem.Allocator) void {
        help_calls += 1;
    }

    fn runDefault(_: std.mem.Allocator, parser: *args.ArgParser) !void {
        try std.testing.expect(!parser.hasMore());
        default_calls += 1;
    }

    fn runAlpha(_: std.mem.Allocator, parser: *args.ArgParser) !void {
        try std.testing.expect(!parser.hasMore());
        alpha_calls += 1;
    }

    fn runBeta(_: std.mem.Allocator, parser: *args.ArgParser) !void {
        try std.testing.expect(!parser.hasMore());
        beta_calls += 1;
    }

    fn onUnknown(command: []const u8) void {
        unknown_calls += 1;
        last_unknown_len = @min(command.len, last_unknown.len);
        @memcpy(last_unknown[0..last_unknown_len], command[0..last_unknown_len]);
    }

    fn unknownSlice() []const u8 {
        return last_unknown[0..last_unknown_len];
    }
};

const test_commands = [_]Command{
    .{ .names = &.{ "alpha", "a" }, .run = TestState.runAlpha },
    .{ .names = &.{"beta"}, .run = TestState.runBeta },
};

test "runSubcommand dispatches default action when command is missing" {
    TestState.reset();
    const argv = [_][:0]const u8{};
    var parser = args.ArgParser.init(std.testing.allocator, &argv);

    try runSubcommand(
        std.testing.allocator,
        &parser,
        &test_commands,
        TestState.runDefault,
        TestState.help,
        TestState.onUnknown,
    );

    try std.testing.expectEqual(@as(usize, 1), TestState.default_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.help_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.unknown_calls);
}

test "runSubcommand dispatches explicit command and aliases" {
    TestState.reset();
    const argv = [_][:0]const u8{"a"};
    var parser = args.ArgParser.init(std.testing.allocator, &argv);

    try runSubcommand(
        std.testing.allocator,
        &parser,
        &test_commands,
        TestState.runDefault,
        TestState.help,
        TestState.onUnknown,
    );

    try std.testing.expectEqual(@as(usize, 1), TestState.alpha_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.help_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.unknown_calls);
}

test "runSubcommand prints help when requested" {
    TestState.reset();
    const argv = [_][:0]const u8{"--help"};
    var parser = args.ArgParser.init(std.testing.allocator, &argv);

    try runSubcommand(
        std.testing.allocator,
        &parser,
        &test_commands,
        TestState.runDefault,
        TestState.help,
        TestState.onUnknown,
    );

    try std.testing.expectEqual(@as(usize, 1), TestState.help_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.default_calls);
    try std.testing.expectEqual(@as(usize, 0), TestState.unknown_calls);
}

test "runSubcommand reports unknown command and then prints help" {
    TestState.reset();
    const argv = [_][:0]const u8{"unknown"};
    var parser = args.ArgParser.init(std.testing.allocator, &argv);

    try runSubcommand(
        std.testing.allocator,
        &parser,
        &test_commands,
        null,
        TestState.help,
        TestState.onUnknown,
    );

    try std.testing.expectEqual(@as(usize, 1), TestState.help_calls);
    try std.testing.expectEqual(@as(usize, 1), TestState.unknown_calls);
    try std.testing.expectEqualStrings("unknown", TestState.unknownSlice());
}

test {
    std.testing.refAllDecls(@This());
}
