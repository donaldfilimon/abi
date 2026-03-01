//! Help text builder utilities for CLI commands.
//!
//! Provides a consistent way to generate help text across all CLI commands,
//! reducing duplication and ensuring uniform formatting.

const std = @import("std");

/// Option definition for help text generation.
pub const Option = struct {
    short: ?[]const u8 = null,
    long: []const u8,
    arg: ?[]const u8 = null,
    description: []const u8,

    /// Custom formatter used with the `{f}` format specifier.
    /// Renders as "-s, --long <ARG>  Description".
    pub fn format(self: Option, writer: anytype) !void {
        // Short option
        if (self.short) |s| {
            try writer.print("  {s}, ", .{s});
        } else {
            try writer.writeAll("      ");
        }

        // Long option with arg
        if (self.arg) |a| {
            try writer.print("{s} <{s}>", .{ self.long, a });
        } else {
            try writer.print("{s}", .{self.long});
        }

        // Padding and description
        const opt_len = if (self.short != null) @as(usize, 4) else @as(usize, 6);
        const long_len = self.long.len + if (self.arg) |a| a.len + 3 else 0;
        const total_len = opt_len + long_len;
        const padding = if (total_len < 28) 28 - total_len else 2;
        try writer.splatByteAll(' ', padding);
        try writer.print("{s}", .{self.description});
    }
};

/// Subcommand definition for help text generation.
pub const Subcommand = struct {
    name: []const u8,
    description: []const u8,

    /// Custom formatter used with the `{f}` format specifier.
    pub fn format(self: Subcommand, writer: anytype) !void {
        try writer.print("  {s}", .{self.name});
        const padding = if (self.name.len < 16) 16 - self.name.len else 2;
        try writer.splatByteAll(' ', padding);
        try writer.print("{s}", .{self.description});
    }
};

/// Help text builder with fluent API.
pub const HelpBuilder = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    last_error: ?anyerror = null,

    pub fn init(allocator: std.mem.Allocator) HelpBuilder {
        return .{
            .buffer = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HelpBuilder) void {
        self.buffer.deinit(self.allocator);
    }

    fn captureError(self: *HelpBuilder, result: anytype) void {
        _ = result catch |err| {
            self.last_error = err;
        };
    }

    /// Add usage line.
    pub fn usage(self: *HelpBuilder, command: []const u8, args: []const u8) *HelpBuilder {
        self.captureError(self.writeFmt("Usage: {s} {s}\n\n", .{ command, args }));
        return self;
    }

    /// Add description paragraph.
    pub fn description(self: *HelpBuilder, desc: []const u8) *HelpBuilder {
        self.captureError(self.writeFmt("{s}\n\n", .{desc}));
        return self;
    }

    /// Add section header.
    pub fn section(self: *HelpBuilder, title: []const u8) *HelpBuilder {
        self.captureError(self.writeFmt("{s}:\n", .{title}));
        return self;
    }

    /// Add option.
    pub fn option(self: *HelpBuilder, opt: Option) *HelpBuilder {
        self.captureError(self.writeOption(opt));
        return self;
    }

    fn writeOption(self: *HelpBuilder, opt: Option) !void {
        // Short option
        if (opt.short) |s| {
            try self.buffer.appendSlice(self.allocator, "  ");
            try self.buffer.appendSlice(self.allocator, s);
            try self.buffer.appendSlice(self.allocator, ", ");
        } else {
            try self.buffer.appendSlice(self.allocator, "      ");
        }

        // Long option with arg
        try self.buffer.appendSlice(self.allocator, opt.long);
        if (opt.arg) |a| {
            try self.buffer.appendSlice(self.allocator, " <");
            try self.buffer.appendSlice(self.allocator, a);
            try self.buffer.append(self.allocator, '>');
        }

        // Padding and description
        const opt_len = if (opt.short != null) @as(usize, 4) else @as(usize, 6);
        const long_len = opt.long.len + if (opt.arg) |a| a.len + 3 else 0;
        const total_len = opt_len + long_len;
        const padding = if (total_len < 28) 28 - total_len else 2;
        for (0..padding) |_| {
            try self.buffer.append(self.allocator, ' ');
        }
        try self.buffer.appendSlice(self.allocator, opt.description);
        try self.buffer.append(self.allocator, '\n');
    }

    /// Add multiple options.
    pub fn options(self: *HelpBuilder, opts: []const Option) *HelpBuilder {
        for (opts) |opt| {
            _ = self.option(opt);
        }
        return self;
    }

    /// Add subcommand.
    pub fn subcommand(self: *HelpBuilder, cmd: Subcommand) *HelpBuilder {
        self.captureError(self.writeSubcommand(cmd));
        return self;
    }

    fn writeSubcommand(self: *HelpBuilder, cmd: Subcommand) !void {
        try self.buffer.appendSlice(self.allocator, "  ");
        try self.buffer.appendSlice(self.allocator, cmd.name);
        const padding = if (cmd.name.len < 16) 16 - cmd.name.len else 2;
        for (0..padding) |_| {
            try self.buffer.append(self.allocator, ' ');
        }
        try self.buffer.appendSlice(self.allocator, cmd.description);
        try self.buffer.append(self.allocator, '\n');
    }

    /// Add multiple subcommands.
    pub fn subcommands(self: *HelpBuilder, cmds: []const Subcommand) *HelpBuilder {
        for (cmds) |cmd| {
            _ = self.subcommand(cmd);
        }
        return self;
    }

    /// Add raw text.
    pub fn text(self: *HelpBuilder, t: []const u8) *HelpBuilder {
        self.captureError(self.writeFmt("{s}", .{t}));
        return self;
    }

    /// Add newline.
    pub fn newline(self: *HelpBuilder) *HelpBuilder {
        self.captureError(self.writeFmt("\n", .{}));
        return self;
    }

    /// Add example.
    pub fn example(self: *HelpBuilder, cmd: []const u8, desc: []const u8) *HelpBuilder {
        self.captureError(self.writeFmt("  {s}\n", .{cmd}));
        if (desc.len > 0) {
            self.captureError(self.writeFmt("    # {s}\n", .{desc}));
        }
        return self;
    }

    /// Build and print help text.
    pub fn print(self: *HelpBuilder) void {
        const output_mod = @import("output.zig");
        output_mod.print("{s}", .{self.buffer.items});
    }

    /// Build and return help text (caller must free).
    pub fn build(self: *HelpBuilder) ![]const u8 {
        return try self.allocator.dupe(u8, self.buffer.items);
    }

    fn writeFmt(self: *HelpBuilder, comptime fmt: []const u8, args: anytype) !void {
        var tmp_buf: [4096]u8 = undefined;
        const formatted = std.fmt.bufPrint(&tmp_buf, fmt, args) catch |err| switch (err) {
            error.NoSpaceLeft => {
                // Buffer too small, append what we can
                try self.buffer.appendSlice(self.allocator, &tmp_buf);
                return;
            },
        };
        try self.buffer.appendSlice(self.allocator, formatted);
    }
};

/// Common options used across multiple commands.
pub const common_options = struct {
    pub const help = Option{
        .short = "-h",
        .long = "--help",
        .description = "Show this help message",
    };

    pub const verbose = Option{
        .short = "-v",
        .long = "--verbose",
        .description = "Enable verbose output",
    };

    pub const quiet = Option{
        .short = "-q",
        .long = "--quiet",
        .description = "Suppress non-essential output",
    };

    pub const output = Option{
        .short = "-o",
        .long = "--output",
        .arg = "FILE",
        .description = "Output file path",
    };

    pub const format = Option{
        .short = "-f",
        .long = "--format",
        .arg = "FORMAT",
        .description = "Output format (json, yaml, human)",
    };
};

/// Print a simple help message with title, usage, and options.
pub fn printSimpleHelp(
    command: []const u8,
    usage_args: []const u8,
    desc: []const u8,
    opts: []const Option,
) void {
    const output_mod = @import("output.zig");
    output_mod.println("Usage: {s} {s}", .{ command, usage_args });
    output_mod.println("", .{});
    output_mod.println("{s}", .{desc});
    output_mod.println("", .{});
    if (opts.len > 0) {
        output_mod.println("Options:", .{});
        for (opts) |opt| {
            output_mod.println("{f}", .{opt});
        }
    }
}

/// Compute the Levenshtein edit distance between two strings.
/// Uses a single-row buffer with a stack allocation for strings up to 64 chars.
/// Returns the minimum number of single-character edits (insertions, deletions,
/// substitutions) needed to transform `a` into `b`.
pub fn editDistance(a: []const u8, b: []const u8) usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;
    if (std.mem.eql(u8, a, b)) return 0;

    // Use a single-row DP approach. We need (b.len + 1) entries.
    var stack_buf: [65]usize = undefined;
    const row = stack_buf[0 .. b.len + 1];

    // Initialize row: row[j] = j (cost of inserting j chars)
    for (row, 0..) |*cell, j| {
        cell.* = j;
    }

    for (a, 0..) |ca, i| {
        var prev = i; // row[0] before update = i
        row[0] = i + 1;

        for (b, 0..) |cb, j| {
            const cost: usize = if (ca == cb) 0 else 1;
            const delete = row[j + 1] + 1; // deletion
            const insert = row[j] + 1; // insertion
            const substitute = prev + cost; // substitution

            prev = row[j + 1]; // save before overwrite
            row[j + 1] = @min(delete, @min(insert, substitute));
        }
    }

    return row[b.len];
}

// ============================================================================
// Tests
// ============================================================================

test "Option: formatting with all fields" {
    const opt = Option{
        .short = "-n",
        .long = "--name",
        .arg = "NAME",
        .description = "Set the name",
    };

    // Buffer sized for test data: ~65 chars max (short + long + arg + padding + desc).
    // The format output for this test option is approximately:
    // "  -n, --name <NAME>          Set the name" = ~42 chars
    var buf: [128]u8 = undefined;
    const result = try std.fmt.bufPrint(&buf, "{f}", .{opt});
    try std.testing.expect(std.mem.indexOf(u8, result, "-n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "--name") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<NAME>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Set the name") != null);
}

test "Option: formatting without short" {
    const opt = Option{
        .long = "--verbose",
        .description = "Enable verbose output",
    };

    // Buffer sized for test data: ~55 chars max (padding + long + padding + desc).
    // The format output for this test option is approximately:
    // "      --verbose              Enable verbose output" = ~50 chars
    var buf: [128]u8 = undefined;
    const result = try std.fmt.bufPrint(&buf, "{f}", .{opt});
    try std.testing.expect(std.mem.indexOf(u8, result, "--verbose") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Enable verbose") != null);
}

test "Subcommand: formatting" {
    const cmd = Subcommand{
        .name = "run",
        .description = "Run the command",
    };

    // Buffer sized for test data: ~40 chars max (indent + name + padding + desc).
    // The format output for this test subcommand is approximately:
    // "  run             Run the command" = ~34 chars
    var buf: [128]u8 = undefined;
    const result = try std.fmt.bufPrint(&buf, "{f}", .{cmd});
    try std.testing.expect(std.mem.indexOf(u8, result, "run") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Run the command") != null);
}

test "HelpBuilder: basic usage" {
    const allocator = std.testing.allocator;

    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi test", "[options]")
        .description("Run tests.")
        .section("Options")
        .option(common_options.help)
        .option(common_options.verbose)
        .newline()
        .section("Examples")
        .example("abi test --verbose", "Run with verbose output");

    const result = try builder.build();
    defer allocator.free(result);

    try std.testing.expect(result.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result, "Usage:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Options:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Examples:") != null);
}

test "common_options: all defined" {
    try std.testing.expect(common_options.help.long.len > 0);
    try std.testing.expect(common_options.verbose.long.len > 0);
    try std.testing.expect(common_options.quiet.long.len > 0);
    try std.testing.expect(common_options.output.long.len > 0);
    try std.testing.expect(common_options.format.long.len > 0);
}

test "editDistance: identical strings" {
    try std.testing.expectEqual(@as(usize, 0), editDistance("hello", "hello"));
    try std.testing.expectEqual(@as(usize, 0), editDistance("", ""));
}

test "editDistance: empty vs non-empty" {
    try std.testing.expectEqual(@as(usize, 5), editDistance("", "hello"));
    try std.testing.expectEqual(@as(usize, 5), editDistance("hello", ""));
}

test "editDistance: single edit" {
    // substitution
    try std.testing.expectEqual(@as(usize, 1), editDistance("cat", "bat"));
    // insertion
    try std.testing.expectEqual(@as(usize, 1), editDistance("cat", "cats"));
    // deletion
    try std.testing.expectEqual(@as(usize, 1), editDistance("cats", "cat"));
}

test "editDistance: common typos" {
    // "trian" -> "train" (transposition = 2 edits in Levenshtein)
    try std.testing.expectEqual(@as(usize, 2), editDistance("trian", "train"));
    // "nework" -> "network" (missing 't' = 1)
    try std.testing.expectEqual(@as(usize, 1), editDistance("nework", "network"));
    // "cofig" -> "config" (missing 'n' = 1)
    try std.testing.expectEqual(@as(usize, 1), editDistance("cofig", "config"));
}

test "editDistance: completely different" {
    try std.testing.expectEqual(@as(usize, 3), editDistance("abc", "xyz"));
}

test {
    std.testing.refAllDecls(@This());
}
