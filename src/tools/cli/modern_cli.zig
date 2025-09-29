//! Modern CLI Framework for ABI
//!
//! A production-ready command-line interface built with Zig 0.16 patterns:
//! - Structured argument parsing with validation
//! - Hierarchical subcommands with help generation
//! - Type-safe options and flags
//! - Rich error reporting with context
//! - Auto-generated help and usage text
//! - Plugin architecture for extensibility

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const Allocator = std.mem.Allocator;

/// CLI framework errors
pub const CliError = error{
    InvalidArgument,
    MissingArgument,
    UnknownCommand,
    UnknownFlag,
    ValidationFailed,
    HelpRequested,
    VersionRequested,
    ConfigurationError,
    ExecutionFailed,
};

/// Argument types supported by the CLI
pub const ArgType = enum {
    string,
    integer,
    float,
    boolean,
    path,
    url,
    email,
};

/// Command line option definition
pub const Option = struct {
    name: []const u8,
    short: ?u8 = null,
    long: []const u8,
    description: []const u8,
    arg_type: ArgType = .string,
    required: bool = false,
    default_value: ?[]const u8 = null,
    validator: ?*const fn ([]const u8) bool = null,
    env_var: ?[]const u8 = null,
};

/// Command line argument definition
pub const Argument = struct {
    name: []const u8,
    description: []const u8,
    arg_type: ArgType = .string,
    required: bool = true,
    validator: ?*const fn ([]const u8) bool = null,
};

/// Command handler function signature
pub const CommandHandler = *const fn (*Context, *ParsedArgs) anyerror!void;

/// Command definition
pub const Command = struct {
    name: []const u8,
    description: []const u8,
    usage: ?[]const u8 = null,
    options: []const Option = &.{},
    arguments: []const Argument = &.{},
    subcommands: []const *const Command = &.{},
    handler: ?CommandHandler = null,
    aliases: []const []const u8 = &.{},
    hidden: bool = false,
    category: ?[]const u8 = null,
    examples: []const []const u8 = &.{},

    /// Check if this command matches a given name or alias
    pub fn matches(self: *const Command, name: []const u8) bool {
        if (std.mem.eql(u8, self.name, name)) return true;
        for (self.aliases) |alias| {
            if (std.mem.eql(u8, alias, name)) return true;
        }
        return false;
    }

    /// Find subcommand by name
    pub fn findSubcommand(self: *const Command, name: []const u8) ?*const Command {
        for (self.subcommands) |sub| {
            if (sub.matches(name)) return sub;
        }
        return null;
    }

    /// Generate usage string for this command
    pub fn generateUsage(self: *const Command, writer: anytype, parent_path: []const u8) !void {
        if (parent_path.len > 0) {
            try writer.print("{s} {s}", .{ parent_path, self.name });
        } else {
            try writer.print("{s}", .{self.name});
        }

        // Add options
        if (self.options.len > 0) {
            try writer.writeAll(" [OPTIONS]");
        }

        // Add arguments
        for (self.arguments) |arg| {
            if (arg.required) {
                try writer.print(" <{s}>", .{arg.name});
            } else {
                try writer.print(" [{s}]", .{arg.name});
            }
        }

        // Add subcommands placeholder
        if (self.subcommands.len > 0) {
            try writer.writeAll(" [COMMAND]");
        }
    }
};

/// Parsed command line values
pub const ParsedValue = union(ArgType) {
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
    path: []const u8,
    url: []const u8,
    email: []const u8,
};

/// Container for parsed arguments and options
pub const ParsedArgs = struct {
    allocator: Allocator,
    command_path: std.ArrayList([]const u8),
    options: std.StringHashMap(ParsedValue),
    arguments: std.ArrayList(ParsedValue),
    raw_args: []const []const u8,

    pub fn init(allocator: Allocator) ParsedArgs {
        return .{
            .allocator = allocator,
            .command_path = std.ArrayList([]const u8).init(allocator),
            .options = std.StringHashMap(ParsedValue).init(allocator),
            .arguments = std.ArrayList(ParsedValue).init(allocator),
            .raw_args = &.{},
        };
    }

    pub fn deinit(self: *ParsedArgs) void {
        self.command_path.deinit(self.allocator);
        self.options.deinit();
        self.arguments.deinit(self.allocator);
    }

    /// Get option value by name
    pub fn getOption(self: *const ParsedArgs, name: []const u8) ?ParsedValue {
        return self.options.get(name);
    }

    /// Get argument by index
    pub fn getArgument(self: *const ParsedArgs, index: usize) ?ParsedValue {
        if (index >= self.arguments.items.len) return null;
        return self.arguments.items[index];
    }

    /// Check if a boolean flag is set
    pub fn hasFlag(self: *const ParsedArgs, name: []const u8) bool {
        if (self.getOption(name)) |value| {
            return switch (value) {
                .boolean => |b| b,
                else => false,
            };
        }
        return false;
    }

    /// Get string option with default
    pub fn getString(self: *const ParsedArgs, name: []const u8, default: []const u8) []const u8 {
        if (self.getOption(name)) |value| {
            return switch (value) {
                .string => |s| s,
                .path => |p| p,
                .url => |u| u,
                .email => |e| e,
                else => default,
            };
        }
        return default;
    }

    /// Get integer option with default
    pub fn getInteger(self: *const ParsedArgs, name: []const u8, default: i64) i64 {
        if (self.getOption(name)) |value| {
            return switch (value) {
                .integer => |i| i,
                else => default,
            };
        }
        return default;
    }
};

/// CLI execution context
pub const Context = struct {
    allocator: Allocator,
    program_name: []const u8,
    version: []const u8,
    author: []const u8,
    description: []const u8,
    root_command: *const Command,
    color_mode: ColorMode = .auto,
    verbosity: u8 = 0,
    config: ?std.StringHashMap([]const u8) = null,

    pub const ColorMode = enum { auto, always, never };

    pub fn init(allocator: Allocator, root: *const Command) Context {
        return .{
            .allocator = allocator,
            .program_name = "abi",
            .version = "0.1.0",
            .author = "ABI Team",
            .description = "High-performance AI framework and vector database",
            .root_command = root,
        };
    }

    /// Print colored text if color mode allows
    pub fn printColored(self: *const Context, writer: anytype, comptime color: []const u8, text: []const u8) !void {
        if (self.shouldUseColor()) {
            try writer.print("\x1b[{s}m{s}\x1b[0m", .{ color, text });
        } else {
            try writer.writeAll(text);
        }
    }

    fn shouldUseColor(self: *const Context) bool {
        return switch (self.color_mode) {
            .always => true,
            .never => false,
            .auto => std.io.tty.detectConfig(std.io.getStdOut()).escape_codes != .no_color,
        };
    }
};

/// CLI parser with comprehensive error handling
pub const Parser = struct {
    allocator: Allocator,
    context: *const Context,

    pub fn init(allocator: Allocator, context: *const Context) Parser {
        return .{
            .allocator = allocator,
            .context = context,
        };
    }

    /// Parse command line arguments
    pub fn parse(self: *Parser, args: []const []const u8) !ParsedArgs {
        var parsed = ParsedArgs.init(self.allocator);
        errdefer parsed.deinit();

        parsed.raw_args = args;

        if (args.len == 0) {
            return parsed;
        }

        var current_cmd = self.context.root_command;
        var arg_index: usize = 0;

        // Parse commands
        while (arg_index < args.len) {
            const arg = args[arg_index];

            // Check for global flags first
            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                return CliError.HelpRequested;
            }
            if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-V")) {
                return CliError.VersionRequested;
            }

            // Check for subcommand
            if (current_cmd.findSubcommand(arg)) |sub_cmd| {
                try parsed.command_path.append(parsed.allocator, arg);
                current_cmd = sub_cmd;
                arg_index += 1;
                continue;
            }

            // Not a subcommand, must be options or arguments
            break;
        }

        // Parse options and arguments for the current command
        try self.parseOptionsAndArgs(current_cmd, args[arg_index..], &parsed);

        return parsed;
    }

    fn parseOptionsAndArgs(self: *Parser, cmd: *const Command, args: []const []const u8, parsed: *ParsedArgs) !void {
        var arg_index: usize = 0;
        var positional_index: usize = 0;

        while (arg_index < args.len) {
            const arg = args[arg_index];

            if (std.mem.startsWith(u8, arg, "--")) {
                // Long option
                const opt_name = arg[2..];
                const option = self.findOptionByLong(cmd, opt_name) orelse {
                    std.debug.print("Unknown option: --{s}\n", .{opt_name});
                    return CliError.UnknownFlag;
                };

                if (option.arg_type == .boolean) {
                    try parsed.options.put(option.long, .{ .boolean = true });
                } else {
                    arg_index += 1;
                    if (arg_index >= args.len) {
                        std.debug.print("Option --{s} requires a value\n", .{opt_name});
                        return CliError.MissingArgument;
                    }
                    const value = try self.parseValue(args[arg_index], option.arg_type);
                    try parsed.options.put(option.long, value);
                }
            } else if (std.mem.startsWith(u8, arg, "-") and arg.len > 1) {
                // Short option(s)
                const short_opts = arg[1..];
                for (short_opts, 0..) |short_char, i| {
                    const option = self.findOptionByShort(cmd, short_char) orelse {
                        std.debug.print("Unknown option: -{c}\n", .{short_char});
                        return CliError.UnknownFlag;
                    };

                    if (option.arg_type == .boolean) {
                        try parsed.options.put(option.long, .{ .boolean = true });
                    } else if (i == short_opts.len - 1) {
                        // Last short option, can take a value
                        arg_index += 1;
                        if (arg_index >= args.len) {
                            std.debug.print("Option -{c} requires a value\n", .{short_char});
                            return CliError.MissingArgument;
                        }
                        const value = try self.parseValue(args[arg_index], option.arg_type);
                        try parsed.options.put(option.long, value);
                    } else {
                        std.debug.print("Option -{c} requires a value but is not the last in a group\n", .{short_char});
                        return CliError.InvalidArgument;
                    }
                }
            } else {
                // Positional argument
                if (positional_index >= cmd.arguments.len) {
                    std.debug.print("Unexpected argument: {s}\n", .{arg});
                    return CliError.InvalidArgument;
                }

                const arg_def = cmd.arguments[positional_index];
                const value = try self.parseValue(arg, arg_def.arg_type);
                try parsed.arguments.append(parsed.allocator, value);
                positional_index += 1;
            }

            arg_index += 1;
        }

        // Validate required arguments
        for (cmd.arguments, 0..) |arg_def, i| {
            if (arg_def.required and i >= parsed.arguments.items.len) {
                std.debug.print("Missing required argument: {s}\n", .{arg_def.name});
                return CliError.MissingArgument;
            }
        }

        // Validate required options and set defaults
        for (cmd.options) |option| {
            if (!parsed.options.contains(option.long)) {
                if (option.required) {
                    std.debug.print("Missing required option: --{s}\n", .{option.long});
                    return CliError.MissingArgument;
                } else if (option.default_value) |default| {
                    const value = try self.parseValue(default, option.arg_type);
                    try parsed.options.put(option.long, value);
                }
            }
        }
    }

    fn findOptionByLong(self: *Parser, cmd: *const Command, name: []const u8) ?Option {
        _ = self;
        for (cmd.options) |option| {
            if (std.mem.eql(u8, option.long, name)) {
                return option;
            }
        }
        return null;
    }

    fn findOptionByShort(self: *Parser, cmd: *const Command, short: u8) ?Option {
        _ = self;
        for (cmd.options) |option| {
            if (option.short) |s| {
                if (s == short) return option;
            }
        }
        return null;
    }

    fn parseValue(self: *Parser, text: []const u8, arg_type: ArgType) !ParsedValue {
        _ = self;
        return switch (arg_type) {
            .string => .{ .string = text },
            .integer => .{ .integer = std.fmt.parseInt(i64, text, 10) catch |err| {
                std.debug.print("Invalid integer: {s} (error: {s})\n", .{ text, @errorName(err) });
                return CliError.InvalidArgument;
            } },
            .float => .{ .float = std.fmt.parseFloat(f64, text) catch |err| {
                std.debug.print("Invalid float: {s} (error: {s})\n", .{ text, @errorName(err) });
                return CliError.InvalidArgument;
            } },
            .boolean => .{ .boolean = parseBool(text) catch |err| {
                std.debug.print("Invalid boolean: {s} (error: {s})\n", .{ text, @errorName(err) });
                return CliError.InvalidArgument;
            } },
            .path => .{ .path = text },
            .url => blk: {
                if (!isValidUrl(text)) {
                    std.debug.print("Invalid URL: {s}\n", .{text});
                    return CliError.InvalidArgument;
                }
                break :blk .{ .url = text };
            },
            .email => blk: {
                if (!isValidEmail(text)) {
                    std.debug.print("Invalid email: {s}\n", .{text});
                    return CliError.InvalidArgument;
                }
                break :blk .{ .email = text };
            },
        };
    }
};

/// Help formatter for generating usage and help text
pub const HelpFormatter = struct {
    context: *const Context,

    pub fn init(context: *const Context) HelpFormatter {
        return .{
            .context = context,
        };
    }

    /// Print help for a command
    pub fn printHelp(self: *HelpFormatter, writer: anytype, cmd: *const Command, command_path: []const []const u8) !void {
        // Program header
        try self.context.printColored(writer, "1;36", self.context.program_name);
        try writer.print(" v{s}\n", .{self.context.version});
        try writer.print("{s}\n\n", .{self.context.description});

        // Usage
        try self.context.printColored(writer, "1;33", "USAGE:");
        try writer.writeAll("\n    ");
        try self.printCommandPath(writer, command_path);
        try cmd.generateUsage(writer, "");
        try writer.writeAll("\n\n");

        // Description
        if (cmd.description.len > 0) {
            try writer.print("{s}\n\n", .{cmd.description});
        }

        // Arguments
        if (cmd.arguments.len > 0) {
            try self.context.printColored(writer, "1;33", "ARGUMENTS:");
            try writer.writeAll("\n");
            for (cmd.arguments) |arg| {
                try writer.print("    ");
                try self.context.printColored(writer, "1;32", arg.name);
                try writer.print("    {s}", .{arg.description});
                if (!arg.required) {
                    try writer.writeAll(" [optional]");
                }
                try writer.writeAll("\n");
            }
            try writer.writeAll("\n");
        }

        // Options
        if (cmd.options.len > 0) {
            try self.context.printColored(writer, "1;33", "OPTIONS:");
            try writer.writeAll("\n");
            for (cmd.options) |option| {
                try writer.writeAll("    ");
                if (option.short) |short| {
                    try writer.print("-{c}, ", .{short});
                }
                const option_name = try std.fmt.allocPrint(self.context.allocator, "--{s}", .{option.long});
                defer self.context.allocator.free(option_name);
                try self.context.printColored(writer, "1;32", option_name);

                if (option.arg_type != .boolean) {
                    try writer.print(" <{s}>", .{@tagName(option.arg_type)});
                }

                try writer.print("    {s}", .{option.description});

                if (option.default_value) |default| {
                    try writer.print(" [default: {s}]", .{default});
                }

                if (option.required) {
                    try writer.writeAll(" [required]");
                }

                try writer.writeAll("\n");
            }
            try writer.writeAll("\n");
        }

        // Subcommands
        if (cmd.subcommands.len > 0) {
            try self.context.printColored(writer, "1;33", "COMMANDS:");
            try writer.writeAll("\n");

            // Group commands by category
            var categories = std.StringHashMap(std.ArrayList(*const Command)).init(self.context.allocator);
            defer {
                var it = categories.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit();
                }
                categories.deinit();
            }

            for (cmd.subcommands) |sub| {
                if (sub.hidden) continue;

                const category = sub.category orelse "General";
                var list = categories.get(category) orelse std.ArrayList(*const Command).init(self.context.allocator);
                try list.append(self.context.allocator, sub);
                try categories.put(category, list);
            }

            var cat_it = categories.iterator();
            while (cat_it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.*, "General")) {
                    try writer.print("\n  {s}:\n", .{entry.key_ptr.*});
                }

                for (entry.value_ptr.items) |sub| {
                    try writer.writeAll("    ");
                    try self.context.printColored(writer, "1;32", sub.name);
                    try writer.print("    {s}", .{sub.description});
                    if (sub.aliases.len > 0) {
                        try writer.writeAll(" (aliases: ");
                        for (sub.aliases, 0..) |alias, i| {
                            if (i > 0) try writer.writeAll(", ");
                            try writer.writeAll(alias);
                        }
                        try writer.writeAll(")");
                    }
                    try writer.writeAll("\n");
                }
            }
            try writer.writeAll("\n");
        }

        // Examples
        if (cmd.examples.len > 0) {
            try self.context.printColored(writer, "1;33", "EXAMPLES:");
            try writer.writeAll("\n");
            for (cmd.examples) |example| {
                try writer.print("    {s}\n", .{example});
            }
            try writer.writeAll("\n");
        }
    }

    fn printCommandPath(self: *HelpFormatter, writer: anytype, path: []const []const u8) !void {
        try writer.writeAll(self.context.program_name);
        for (path) |cmd| {
            try writer.print(" {s}", .{cmd});
        }
    }

    /// Print version information
    pub fn printVersion(self: *HelpFormatter, writer: anytype) !void {
        try writer.print("{s} {s}\n", .{ self.context.program_name, self.context.version });
        try writer.print("Author: {s}\n", .{self.context.author});

        // Build information
        try writer.print("Built with Zig {s}\n", .{builtin.zig_version_string});
        try writer.print("Target: {s}\n", .{@tagName(builtin.target.cpu.arch)});
        try writer.print("Mode: {s}\n", .{@tagName(builtin.mode)});
    }
};

/// Utility functions
fn parseBool(text: []const u8) !bool {
    if (std.ascii.eqlIgnoreCase(text, "true") or std.mem.eql(u8, text, "1") or std.ascii.eqlIgnoreCase(text, "yes") or std.ascii.eqlIgnoreCase(text, "on")) {
        return true;
    } else if (std.ascii.eqlIgnoreCase(text, "false") or std.mem.eql(u8, text, "0") or std.ascii.eqlIgnoreCase(text, "no") or std.ascii.eqlIgnoreCase(text, "off")) {
        return false;
    } else {
        return error.InvalidBoolean;
    }
}

fn isValidUrl(url: []const u8) bool {
    return std.mem.startsWith(u8, url, "http://") or
        std.mem.startsWith(u8, url, "https://") or
        std.mem.startsWith(u8, url, "ftp://");
}

fn isValidEmail(email: []const u8) bool {
    const at_pos = std.mem.indexOf(u8, email, "@") orelse return false;
    if (at_pos == 0 or at_pos == email.len - 1) return false;

    const domain = email[at_pos + 1 ..];
    return std.mem.indexOf(u8, domain, ".") != null;
}

// Tests
test "CLI argument parsing" {
    const testing = std.testing;

    // Test command definition
    const test_cmd = Command{
        .name = "test",
        .description = "Test command",
        .options = &.{
            .{
                .name = "verbose",
                .long = "verbose",
                .short = 'v',
                .description = "Verbose output",
                .arg_type = .boolean,
            },
            .{
                .name = "count",
                .long = "count",
                .short = 'c',
                .description = "Number of items",
                .arg_type = .integer,
                .default_value = "10",
            },
        },
        .arguments = &.{
            .{
                .name = "input",
                .description = "Input file",
                .arg_type = .path,
            },
        },
    };

    var ctx = Context.init(testing.allocator, &test_cmd);
    var parser = Parser.init(testing.allocator, &ctx);

    const args = [_][]const u8{ "--verbose", "--count", "25", "input.txt" };
    var parsed = try parser.parse(&args);
    defer parsed.deinit();

    try testing.expect(parsed.hasFlag("verbose"));
    try testing.expectEqual(@as(i64, 25), parsed.getInteger("count", 0));
    try testing.expectEqualStrings("input.txt", parsed.getArgument(0).?.path);
}

test "help generation" {
    const testing = std.testing;

    const root_cmd = Command{
        .name = "abi",
        .description = "ABI AI Framework CLI",
        .options = &.{
            .{
                .name = "verbose",
                .long = "verbose",
                .short = 'v',
                .description = "Enable verbose output",
                .arg_type = .boolean,
            },
        },
        .subcommands = &.{
            &Command{
                .name = "chat",
                .description = "Interactive chat with AI agent",
                .category = "AI",
            },
            &Command{
                .name = "server",
                .description = "Start HTTP server",
                .category = "Network",
            },
        },
    };

    var ctx = Context.init(testing.allocator, &root_cmd);
    var formatter = HelpFormatter.init(&ctx);

    var output = std.ArrayList(u8).init(testing.allocator);
    defer output.deinit();

    var writer = output.writer();
    try formatter.printHelp(writer.any(), &root_cmd, &.{});

    const help_text = output.items;
    try testing.expect(root_cmd.subcommands.len == 2);
    try testing.expect(std.mem.indexOf(u8, help_text, "ABI AI Framework CLI") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "USAGE:") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "OPTIONS:") != null);
    try testing.expect(std.mem.indexOf(u8, help_text, "chat") != null);
}
