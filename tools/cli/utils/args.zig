//! Argument parsing utilities for CLI commands.
//!
//! Provides helpers for parsing command-line arguments with type safety,
//! reducing boilerplate in CLI command implementations.

const std = @import("std");

/// Check if text matches any of the provided options.
pub fn matchesAny(text: []const u8, options: []const []const u8) bool {
    for (options) |option| {
        if (std.mem.eql(u8, text, option)) return true;
    }
    return false;
}

fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (needle, 0..) |c, i| {
        if (std.ascii.toLower(c) != std.ascii.toLower(haystack[i])) return false;
    }
    return true;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (startsWithIgnoreCase(haystack[i..], needle)) return true;
    }
    return false;
}

/// Suggest a command based on a partial or mistyped input.
/// Tries exact (case-insensitive), prefix, substring, then Levenshtein distance.
pub fn suggestCommand(input: []const u8, commands: []const []const u8) ?[]const u8 {
    if (input.len == 0) return null;

    // 1. Exact match (case-insensitive)
    for (commands) |cmd| {
        if (std.ascii.eqlIgnoreCase(input, cmd)) return cmd;
    }

    // 2. Prefix match (either direction)
    for (commands) |cmd| {
        if (startsWithIgnoreCase(cmd, input) or startsWithIgnoreCase(input, cmd)) return cmd;
    }

    // 3. Substring match
    for (commands) |cmd| {
        if (containsIgnoreCase(cmd, input)) return cmd;
    }

    // 4. Fuzzy match via Levenshtein edit distance (max distance 3)
    const help_utils = @import("help.zig");
    var best: ?[]const u8 = null;
    var best_dist: usize = 4; // threshold: only suggest if distance <= 3
    for (commands) |cmd| {
        const dist = help_utils.editDistance(input, cmd);
        if (dist < best_dist) {
            best_dist = dist;
            best = cmd;
        }
    }
    return best;
}

/// Convert a null-terminated argument to a regular slice.
pub fn toSlice(arg: [:0]const u8) []const u8 {
    return arg[0..];
}

/// Check if any argument requests help.
pub fn containsHelpArgs(args: []const [:0]const u8) bool {
    for (args) |arg| {
        const slice = arg[0..];
        if (matchesAny(slice, &[_][]const u8{ "help", "--help", "-h" })) return true;
    }
    return false;
}

/// Parse a node status string to enum value.
pub fn parseNodeStatus(text: []const u8) ?@import("abi").network.NodeStatus {
    if (std.ascii.eqlIgnoreCase(text, "healthy")) return .healthy;
    if (std.ascii.eqlIgnoreCase(text, "degraded")) return .degraded;
    if (std.ascii.eqlIgnoreCase(text, "offline")) return .offline;
    return null;
}

/// Argument parser for structured CLI argument handling.
/// Reduces boilerplate for common parsing patterns.
pub const ArgParser = struct {
    args: []const [:0]const u8,
    index: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, args: []const [:0]const u8) ArgParser {
        return .{
            .args = args,
            .allocator = allocator,
        };
    }

    /// Get the current argument as a slice, or null if exhausted.
    pub fn current(self: *const ArgParser) ?[]const u8 {
        if (self.index >= self.args.len) return null;
        return self.args[self.index][0..];
    }

    /// Advance to next argument and return current.
    pub fn next(self: *ArgParser) ?[]const u8 {
        const curr = self.current();
        if (curr != null) self.index += 1;
        return curr;
    }

    /// Peek at next argument without advancing.
    pub fn peek(self: *const ArgParser) ?[]const u8 {
        if (self.index + 1 >= self.args.len) return null;
        return self.args[self.index + 1][0..];
    }

    /// Check if current argument matches any of the options.
    pub fn matches(self: *const ArgParser, options: []const []const u8) bool {
        const curr = self.current() orelse return false;
        return matchesAny(curr, options);
    }

    /// Try to consume an option with a value (e.g., "--name value").
    /// Returns the value if the current arg matches and has a next value.
    pub fn consumeOption(self: *ArgParser, options: []const []const u8) ?[]const u8 {
        if (!self.matches(options)) return null;
        _ = self.next(); // consume the flag
        return self.next(); // return the value
    }

    /// Try to consume an option and parse as integer.
    pub fn consumeInt(self: *ArgParser, comptime T: type, options: []const []const u8, default: T) T {
        const value = self.consumeOption(options) orelse return default;
        return std.fmt.parseInt(T, value, 10) catch default;
    }

    /// Try to consume an option and parse as float.
    pub fn consumeFloat(self: *ArgParser, comptime T: type, options: []const []const u8, default: T) T {
        const value = self.consumeOption(options) orelse return default;
        return std.fmt.parseFloat(T, value) catch default;
    }

    /// Try to consume a boolean flag (no value needed).
    pub fn consumeFlag(self: *ArgParser, options: []const []const u8) bool {
        if (!self.matches(options)) return false;
        _ = self.next();
        return true;
    }

    /// Check if help is requested.
    pub fn wantsHelp(self: *const ArgParser) bool {
        const curr = self.current() orelse return false;
        return matchesAny(curr, &[_][]const u8{ "help", "--help", "-h" });
    }

    /// Check if any argument requests help.
    pub fn containsHelp(self: *const ArgParser) bool {
        for (self.args) |arg| {
            const slice = arg[0..];
            if (matchesAny(slice, &[_][]const u8{ "help", "--help", "-h" })) return true;
        }
        return false;
    }

    /// Check if there are more arguments.
    pub fn hasMore(self: *const ArgParser) bool {
        return self.index < self.args.len;
    }

    /// Get remaining arguments as slice.
    pub fn remaining(self: *const ArgParser) []const [:0]const u8 {
        if (self.index >= self.args.len) return &.{};
        return self.args[self.index..];
    }

    /// Skip n arguments.
    pub fn skip(self: *ArgParser, n: usize) void {
        self.index = @min(self.index + n, self.args.len);
    }
};

/// Error context for CLI operations with improved error messages.
pub const CliError = struct {
    code: anyerror,
    context: []const u8,
    suggestion: ?[]const u8 = null,

    pub fn format(
        self: CliError,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("Error: {s} ({t})", .{ self.context, self.code });
        if (self.suggestion) |s| {
            try writer.print("\nSuggestion: {s}", .{s});
        }
    }

    pub fn print(self: CliError) void {
        std.debug.print("Error: {s} ({t})\n", .{ self.context, self.code });
        if (self.suggestion) |s| {
            std.debug.print("Suggestion: {s}\n", .{s});
        }
    }
};

/// Create a CLI error with context.
pub fn cliError(err: anyerror, context: []const u8) CliError {
    return .{ .code = err, .context = context };
}

/// Create a CLI error with context and suggestion.
pub fn cliErrorWithSuggestion(err: anyerror, context: []const u8, suggestion: []const u8) CliError {
    return .{ .code = err, .context = context, .suggestion = suggestion };
}

/// Parse a required positional ID argument (common pattern for task/model commands).
/// Prints an error and returns null if missing or invalid.
pub fn parseRequiredId(args: []const [:0]const u8, comptime entity: []const u8) ?u64 {
    if (args.len == 0) {
        const output = @import("output.zig");
        output.printError("{s} ID required", .{entity});
        return null;
    }
    const id_str = std.mem.sliceTo(args[0], 0);
    return std.fmt.parseInt(u64, id_str, 10) catch {
        const output = @import("output.zig");
        output.printError("Invalid {s} ID: {s}", .{ entity, id_str });
        return null;
    };
}

/// Parse enum from string (case-insensitive).
pub fn parseEnum(comptime E: type, text: []const u8) ?E {
    inline for (@typeInfo(E).@"enum".fields) |field| {
        if (std.ascii.eqlIgnoreCase(text, field.name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}

/// Get all enum field names as a comma-separated string (comptime).
pub fn enumNames(comptime E: type) []const u8 {
    comptime var result: []const u8 = "";
    const fields = @typeInfo(E).@"enum".fields;
    inline for (fields, 0..) |field, i| {
        result = result ++ field.name;
        if (i < fields.len - 1) result = result ++ ", ";
    }
    return result;
}

test "matchesAny helper function" {
    try std.testing.expect(matchesAny("help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("--help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("-h", &.{ "help", "--help", "-h" }));
    try std.testing.expect(!matchesAny("invalid", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("test", &[_][]const u8{"test"}));
    try std.testing.expect(!matchesAny("test", &[_][]const u8{"other"}));
}

test "containsHelpArgs" {
    const args_list = [_][:0]const u8{ "generate", "--help" };
    try std.testing.expect(containsHelpArgs(&args_list));
}

test "suggestCommand" {
    const commands = [_][]const u8{ "llm", "train", "embed" };
    const suggestion = suggestCommand("lm", &commands);
    try std.testing.expect(suggestion != null);
    try std.testing.expectEqualStrings("llm", suggestion.?);
}

test "suggestCommand: fuzzy match via edit distance" {
    const commands = [_][]const u8{ "config", "network", "train", "embed" };
    // "trian" doesn't match any prefix/substring, but is 2 edits from "train"
    const suggestion = suggestCommand("trian", &commands);
    try std.testing.expect(suggestion != null);
    try std.testing.expectEqualStrings("train", suggestion.?);
}

test "suggestCommand: no match when too distant" {
    const commands = [_][]const u8{ "config", "network", "train" };
    // "xyz" is too far from anything (distance > 3)
    const suggestion = suggestCommand("xyz", &commands);
    try std.testing.expect(suggestion == null);
}

test "ArgParser basic operations" {
    const allocator = std.testing.allocator;

    // Create test args
    const args = [_][:0]const u8{ "--name", "test", "--count", "42", "--verbose" };

    var parser = ArgParser.init(allocator, &args);

    // Test current/next
    try std.testing.expectEqualStrings("--name", parser.current().?);
    try std.testing.expectEqualStrings("--name", parser.next().?);
    try std.testing.expectEqualStrings("test", parser.current().?);
}

test "ArgParser consumeOption" {
    const allocator = std.testing.allocator;

    const args = [_][:0]const u8{ "--name", "myvalue", "--other" };
    var parser = ArgParser.init(allocator, &args);

    const value = parser.consumeOption(&[_][]const u8{ "--name", "-n" });
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("myvalue", value.?);

    // Should be at --other now
    try std.testing.expectEqualStrings("--other", parser.current().?);
}

test "ArgParser consumeInt" {
    const allocator = std.testing.allocator;

    const args = [_][:0]const u8{ "--count", "123", "--other" };
    var parser = ArgParser.init(allocator, &args);

    const count = parser.consumeInt(u32, &[_][]const u8{ "--count", "-c" }, 0);
    try std.testing.expectEqual(@as(u32, 123), count);
}

test "ArgParser consumeFlag" {
    const allocator = std.testing.allocator;

    const args = [_][:0]const u8{ "--verbose", "--name", "test" };
    var parser = ArgParser.init(allocator, &args);

    const verbose = parser.consumeFlag(&[_][]const u8{ "--verbose", "-v" });
    try std.testing.expect(verbose);

    // Should be at --name now
    try std.testing.expectEqualStrings("--name", parser.current().?);
}

test "parseEnum" {
    const TestEnum = enum { foo, bar, baz };

    try std.testing.expectEqual(TestEnum.foo, parseEnum(TestEnum, "foo").?);
    try std.testing.expectEqual(TestEnum.bar, parseEnum(TestEnum, "BAR").?);
    try std.testing.expectEqual(TestEnum.baz, parseEnum(TestEnum, "Baz").?);
    try std.testing.expect(parseEnum(TestEnum, "invalid") == null);
}

test "enumNames" {
    const TestEnum = enum { quick, medium, thorough };
    const names = enumNames(TestEnum);
    try std.testing.expectEqualStrings("quick, medium, thorough", names);
}

test "CliError formatting" {
    const err = cliErrorWithSuggestion(
        error.FileNotFound,
        "Could not open config file",
        "Check that the file exists and is readable",
    );
    try std.testing.expectEqual(error.FileNotFound, err.code);
    try std.testing.expectEqualStrings("Could not open config file", err.context);
}
