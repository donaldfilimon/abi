//! CLI and TUI Integration Tests
//!
//! Comprehensive test suite for CLI commands and TUI interactions.
//! Tests are designed to run cross-platform (Linux, Windows, macOS).

const std = @import("std");
const builtin = @import("builtin");

const utils = @import("utils/mod.zig");
const args = @import("utils/args.zig");
const tui = @import("tui/mod.zig");

// ============================================================================
// CLI Argument Parsing Tests
// ============================================================================

test "ArgParser: basic navigation" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "--name", "value", "--flag" };

    var parser = args.ArgParser.init(allocator, &test_args);

    try std.testing.expectEqualStrings("--name", parser.current().?);
    try std.testing.expectEqualStrings("value", parser.peek().?);
    try std.testing.expectEqualStrings("--name", parser.next().?);
    try std.testing.expectEqualStrings("value", parser.current().?);
    try std.testing.expect(parser.hasMore());
}

test "ArgParser: consumeOption with aliases" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "-n", "myname", "--other" };

    var parser = args.ArgParser.init(allocator, &test_args);
    const value = parser.consumeOption(&[_][]const u8{ "--name", "-n" });

    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("myname", value.?);
    try std.testing.expectEqualStrings("--other", parser.current().?);
}

test "ArgParser: consumeInt with default" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "--count", "42", "--missing" };

    var parser = args.ArgParser.init(allocator, &test_args);

    const count = parser.consumeInt(u32, &[_][]const u8{ "--count", "-c" }, 0);
    try std.testing.expectEqual(@as(u32, 42), count);

    // Missing option should return default
    const missing = parser.consumeInt(u32, &[_][]const u8{"--nonexistent"}, 999);
    try std.testing.expectEqual(@as(u32, 999), missing);
}

test "ArgParser: consumeFloat" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "--rate", "0.001" };

    var parser = args.ArgParser.init(allocator, &test_args);
    const rate = parser.consumeFloat(f32, &[_][]const u8{ "--rate", "-r" }, 0.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.001), rate, 0.0001);
}

test "ArgParser: consumeFlag" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "--verbose", "--dry-run", "--name", "test" };

    var parser = args.ArgParser.init(allocator, &test_args);

    try std.testing.expect(parser.consumeFlag(&[_][]const u8{ "--verbose", "-v" }));
    try std.testing.expect(parser.consumeFlag(&[_][]const u8{"--dry-run"}));
    try std.testing.expect(!parser.consumeFlag(&[_][]const u8{"--missing"}));
    try std.testing.expectEqualStrings("--name", parser.current().?);
}

test "ArgParser: wantsHelp" {
    const allocator = std.testing.allocator;

    const help_variants = [_][]const [:0]const u8{
        &[_][:0]const u8{"help"},
        &[_][:0]const u8{"--help"},
        &[_][:0]const u8{"-h"},
    };

    for (help_variants) |variant| {
        var parser = args.ArgParser.init(allocator, variant);
        try std.testing.expect(parser.wantsHelp());
    }

    const non_help = [_][:0]const u8{"--other"};
    var parser = args.ArgParser.init(allocator, &non_help);
    try std.testing.expect(!parser.wantsHelp());
}

test "ArgParser: remaining arguments" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "--opt", "val", "pos1", "pos2" };

    var parser = args.ArgParser.init(allocator, &test_args);
    _ = parser.consumeOption(&[_][]const u8{"--opt"});

    const remaining = parser.remaining();
    try std.testing.expectEqual(@as(usize, 2), remaining.len);
    try std.testing.expectEqualStrings("pos1", args.toSlice(remaining[0]));
    try std.testing.expectEqualStrings("pos2", args.toSlice(remaining[1]));
}

test "ArgParser: skip" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{ "a", "b", "c", "d" };

    var parser = args.ArgParser.init(allocator, &test_args);
    parser.skip(2);

    try std.testing.expectEqualStrings("c", parser.current().?);
}

// ============================================================================
// Enum Parsing Tests
// ============================================================================

test "parseEnum: case insensitive" {
    const TestEnum = enum { quick, medium, thorough, deep };

    try std.testing.expectEqual(TestEnum.quick, args.parseEnum(TestEnum, "quick").?);
    try std.testing.expectEqual(TestEnum.quick, args.parseEnum(TestEnum, "QUICK").?);
    try std.testing.expectEqual(TestEnum.quick, args.parseEnum(TestEnum, "Quick").?);
    try std.testing.expectEqual(TestEnum.medium, args.parseEnum(TestEnum, "medium").?);
    try std.testing.expectEqual(TestEnum.thorough, args.parseEnum(TestEnum, "THOROUGH").?);
    try std.testing.expect(args.parseEnum(TestEnum, "invalid") == null);
}

test "enumNames: generates comma-separated list" {
    const TestEnum = enum { a, bb, ccc };
    const names = args.enumNames(TestEnum);
    try std.testing.expectEqualStrings("a, bb, ccc", names);
}

// ============================================================================
// CLI Error Context Tests
// ============================================================================

test "CliError: basic formatting" {
    const err = args.cliError(error.FileNotFound, "Config file not found");
    try std.testing.expectEqual(error.FileNotFound, err.code);
    try std.testing.expectEqualStrings("Config file not found", err.context);
    try std.testing.expect(err.suggestion == null);
}

test "CliError: with suggestion" {
    const err = args.cliErrorWithSuggestion(
        error.InvalidArgument,
        "Invalid batch size",
        "Use a positive integer (e.g., --batch-size 32)",
    );
    try std.testing.expectEqual(error.InvalidArgument, err.code);
    try std.testing.expect(err.suggestion != null);
    try std.testing.expectEqualStrings("Use a positive integer (e.g., --batch-size 32)", err.suggestion.?);
}

// ============================================================================
// TUI Event Types Tests
// ============================================================================

test "TUI Key: character detection" {
    const key_j = tui.Key{ .code = .character, .char = 'j' };
    try std.testing.expect(tui.events.isChar(key_j, 'j'));
    try std.testing.expect(!tui.events.isChar(key_j, 'k'));

    const key_enter = tui.Key{ .code = .enter };
    try std.testing.expect(!tui.events.isChar(key_enter, '\n'));
}

test "TUI Modifiers: packed struct" {
    var mods = tui.Modifiers{};
    try std.testing.expect(!mods.ctrl);
    try std.testing.expect(!mods.alt);
    try std.testing.expect(!mods.shift);

    mods.ctrl = true;
    mods.shift = true;
    try std.testing.expect(mods.ctrl);
    try std.testing.expect(!mods.alt);
    try std.testing.expect(mods.shift);
}

test "TUI Event: union discrimination" {
    const key_event = tui.Event{ .key = .{ .code = .up } };
    const mouse_event = tui.Event{ .mouse = .{ .row = 10, .col = 20, .button = .left, .pressed = true } };

    switch (key_event) {
        .key => |key| try std.testing.expectEqual(tui.KeyCode.up, key.code),
        .mouse => unreachable,
    }

    switch (mouse_event) {
        .key => unreachable,
        .mouse => |mouse| {
            try std.testing.expectEqual(@as(u16, 10), mouse.row);
            try std.testing.expectEqual(@as(u16, 20), mouse.col);
            try std.testing.expectEqual(tui.MouseButton.left, mouse.button);
            try std.testing.expect(mouse.pressed);
        },
    }
}

// ============================================================================
// TUI Terminal Tests (Cross-Platform)
// ============================================================================

test "TUI Terminal: initialization" {
    const allocator = std.testing.allocator;
    var term = tui.Terminal.init(allocator);
    defer term.deinit();

    try std.testing.expect(!term.active);
}

test "TUI Terminal: size detection" {
    const allocator = std.testing.allocator;
    var term = tui.Terminal.init(allocator);
    defer term.deinit();

    const size = term.size();
    // Terminal should report some size (at least defaults)
    try std.testing.expect(size.rows > 0);
    try std.testing.expect(size.cols > 0);
}

// ============================================================================
// Cross-Platform Path Tests
// ============================================================================

test "cross-platform path separator" {
    const sep = std.fs.path.sep_str;

    if (builtin.os.tag == .windows) {
        try std.testing.expectEqualStrings("\\", sep);
    } else {
        try std.testing.expectEqualStrings("/", sep);
    }

    // Test path construction
    const path = ".abi" ++ sep ++ "sessions";
    try std.testing.expect(path.len > 0);
}

test "cross-platform temp directory" {
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    try std.testing.expect(tmp_dir.sub_path.len > 0);
}

// ============================================================================
// matchesAny Tests
// ============================================================================

test "matchesAny: standard usage" {
    try std.testing.expect(args.matchesAny("--help", &[_][]const u8{ "--help", "-h", "help" }));
    try std.testing.expect(args.matchesAny("-h", &[_][]const u8{ "--help", "-h", "help" }));
    try std.testing.expect(args.matchesAny("help", &[_][]const u8{ "--help", "-h", "help" }));
    try std.testing.expect(!args.matchesAny("--version", &[_][]const u8{ "--help", "-h", "help" }));
}

test "matchesAny: empty options" {
    try std.testing.expect(!args.matchesAny("anything", &[_][]const u8{}));
}

test "matchesAny: single option" {
    try std.testing.expect(args.matchesAny("exact", &[_][]const u8{"exact"}));
    try std.testing.expect(!args.matchesAny("different", &[_][]const u8{"exact"}));
}

// ============================================================================
// toSlice Tests
// ============================================================================

test "toSlice: null-terminated to slice" {
    const nt: [:0]const u8 = "hello";
    const slice = args.toSlice(nt);
    try std.testing.expectEqualStrings("hello", slice);
    try std.testing.expectEqual(@as(usize, 5), slice.len);
}

// ============================================================================
// Integration-Style Tests
// ============================================================================

test "integration: parse training config style args" {
    const allocator = std.testing.allocator;
    const test_args = [_][:0]const u8{
        "--epochs",     "10",
        "--batch-size", "32",
        "--lr",         "0.001",
        "--verbose",
    };

    var parser = args.ArgParser.init(allocator, &test_args);

    const epochs = parser.consumeInt(u32, &[_][]const u8{ "--epochs", "-e" }, 1);
    const batch_size = parser.consumeInt(u32, &[_][]const u8{ "--batch-size", "-b" }, 16);
    const lr = parser.consumeFloat(f32, &[_][]const u8{ "--lr", "--learning-rate" }, 0.01);
    const verbose = parser.consumeFlag(&[_][]const u8{ "--verbose", "-v" });

    try std.testing.expectEqual(@as(u32, 10), epochs);
    try std.testing.expectEqual(@as(u32, 32), batch_size);
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), lr, 0.0001);
    try std.testing.expect(verbose);
    try std.testing.expect(!parser.hasMore());
}

test "integration: parse explore config style args" {
    const allocator = std.testing.allocator;
    const ExploreLevel = enum { quick, medium, thorough, deep };

    const test_args = [_][:0]const u8{
        "--level",          "thorough",
        "--max-files",      "100",
        "--case-sensitive", "my search query",
    };

    var parser = args.ArgParser.init(allocator, &test_args);

    var level: ExploreLevel = .medium;
    var max_files: usize = 50;
    var case_sensitive = false;
    var query: ?[]const u8 = null;

    while (parser.hasMore()) {
        if (parser.consumeOption(&[_][]const u8{ "--level", "-l" })) |val| {
            level = args.parseEnum(ExploreLevel, val) orelse .medium;
            continue;
        }
        const max_files_override = parser.consumeInt(usize, &[_][]const u8{"--max-files"}, 0);
        if (max_files_override > 0) {
            max_files = max_files_override;
            continue;
        }
        if (parser.consumeFlag(&[_][]const u8{ "--case-sensitive", "-c" })) {
            case_sensitive = true;
            continue;
        }
        // Positional argument
        query = parser.next();
    }

    try std.testing.expectEqual(ExploreLevel.thorough, level);
    try std.testing.expectEqual(@as(usize, 100), max_files);
    try std.testing.expect(case_sensitive);
    try std.testing.expect(query != null);
    try std.testing.expectEqualStrings("my search query", query.?);
}

// ============================================================================
// Platform Detection Tests
// ============================================================================

test "platform detection" {
    const os = builtin.os.tag;

    // Just verify detection works
    const is_windows = os == .windows;
    const is_linux = os == .linux;
    const is_macos = os == .macos;

    // At least one should be true (or another supported platform)
    const known_platform = is_windows or is_linux or is_macos or
        os == .freebsd or os == .netbsd or os == .openbsd;
    try std.testing.expect(known_platform);
}

test "pointer size detection" {
    const ptr_size = @sizeOf(*u8);

    // Should be either 32-bit or 64-bit
    try std.testing.expect(ptr_size == 4 or ptr_size == 8);
}

// ============================================================================
// Run all tests
// ============================================================================

pub fn main() !void {
    std.debug.print("CLI/TUI tests should be run with: zig test tools/cli/tests.zig\n", .{});
}
