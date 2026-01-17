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

    /// Format as "-s, --long <ARG>  Description"
    pub fn format(
        self: Option,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
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
        try writer.writeByteNTimes(' ', padding);
        try writer.print("{s}", .{self.description});
    }
};

/// Subcommand definition for help text generation.
pub const Subcommand = struct {
    name: []const u8,
    description: []const u8,

    pub fn format(
        self: Subcommand,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("  {s}", .{self.name});
        const padding = if (self.name.len < 16) 16 - self.name.len else 2;
        try writer.writeByteNTimes(' ', padding);
        try writer.print("{s}", .{self.description});
    }
};

/// Help text builder with fluent API.
pub const HelpBuilder = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HelpBuilder {
        return .{
            .buffer = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HelpBuilder) void {
        self.buffer.deinit(self.allocator);
    }

    /// Add usage line.
    pub fn usage(self: *HelpBuilder, command: []const u8, args: []const u8) *HelpBuilder {
        self.writeFmt("Usage: {s} {s}\n\n", .{ command, args }) catch {};
        return self;
    }

    /// Add description paragraph.
    pub fn description(self: *HelpBuilder, desc: []const u8) *HelpBuilder {
        self.writeFmt("{s}\n\n", .{desc}) catch {};
        return self;
    }

    /// Add section header.
    pub fn section(self: *HelpBuilder, title: []const u8) *HelpBuilder {
        self.writeFmt("{s}:\n", .{title}) catch {};
        return self;
    }

    /// Add option.
    pub fn option(self: *HelpBuilder, opt: Option) *HelpBuilder {
        self.writeFmt("{}\n", .{opt}) catch {};
        return self;
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
        self.writeFmt("{}\n", .{cmd}) catch {};
        return self;
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
        self.writeFmt("{s}", .{t}) catch {};
        return self;
    }

    /// Add newline.
    pub fn newline(self: *HelpBuilder) *HelpBuilder {
        self.writeFmt("\n", .{}) catch {};
        return self;
    }

    /// Add example.
    pub fn example(self: *HelpBuilder, cmd: []const u8, desc: []const u8) *HelpBuilder {
        self.writeFmt("  {s}\n", .{cmd}) catch {};
        if (desc.len > 0) {
            self.writeFmt("    # {s}\n", .{desc}) catch {};
        }
        return self;
    }

    /// Build and print help text.
    pub fn print(self: *HelpBuilder) void {
        std.debug.print("{s}", .{self.buffer.items});
    }

    /// Build and return help text (caller must free).
    pub fn build(self: *HelpBuilder) ![]const u8 {
        return try self.allocator.dupe(u8, self.buffer.items);
    }

    fn writeFmt(self: *HelpBuilder, comptime fmt: []const u8, args: anytype) !void {
        const writer = self.buffer.writer(self.allocator);
        try writer.print(fmt, args);
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
    std.debug.print("Usage: {s} {s}\n\n", .{ command, usage_args });
    std.debug.print("{s}\n\n", .{desc});
    if (opts.len > 0) {
        std.debug.print("Options:\n", .{});
        for (opts) |opt| {
            std.debug.print("{}\n", .{opt});
        }
    }
    std.debug.print("\n", .{});
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

    var buf: [128]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, "{}", .{opt}) catch unreachable;
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

    var buf: [128]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, "{}", .{opt}) catch unreachable;
    try std.testing.expect(std.mem.indexOf(u8, result, "--verbose") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Enable verbose") != null);
}

test "Subcommand: formatting" {
    const cmd = Subcommand{
        .name = "run",
        .description = "Run the command",
    };

    var buf: [128]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, "{}", .{cmd}) catch unreachable;
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
