//! CLI Module
//!
//! This module provides the command-line interface for the WDBX database.

const std = @import("std");
const core = @import("../../core/mod.zig");

pub const commands = @import("commands.zig");
pub const parser = @import("parser.zig");

/// CLI structure
pub const CLI = struct {
    allocator: std.mem.Allocator,
    config: Config,
    database: ?*core.Database,

    const Self = @This();

    /// CLI configuration
    pub const Config = struct {
        /// Database path
        db_path: ?[]const u8 = null,
        /// Verbose output
        verbose: bool = false,
        /// Quiet mode
        quiet: bool = false,
        /// Output format
        output_format: OutputFormat = .text,
        /// Log level
        log_level: LogLevel = .info,
    };

    /// Output format options
    pub const OutputFormat = enum {
        text,
        json,
        csv,
        yaml,
    };

    /// Log level options
    pub const LogLevel = enum {
        trace,
        debug,
        info,
        warn,
        err,
        fatal,

        pub fn toInt(self: LogLevel) u8 {
            return switch (self) {
                .trace => 0,
                .debug => 1,
                .info => 2,
                .warn => 3,
                .err => 4,
                .fatal => 5,
            };
        }
    };

    /// Initialize CLI
    pub fn init(allocator: std.mem.Allocator, config: Config) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .database = null,
        };
        return self;
    }

    /// Deinitialize CLI
    pub fn deinit(self: *Self) void {
        if (self.database) |db| {
            db.close();
        }
        self.allocator.destroy(self);
    }

    /// Run CLI with arguments
    pub fn run(self: *Self, args: []const []const u8) !void {
        if (args.len == 0) {
            try self.showHelp();
            return;
        }

        const command = try commands.parseCommand(args[0]);
        const cmd_args = if (args.len > 1) args[1..] else &[_][]const u8{};

        try command.execute(self, cmd_args);
    }

    /// Show help message
    fn showHelp(self: *Self) !void {
        const help_text =
            \\WDBX Vector Database CLI
            \\
            \\Usage: wdbx <command> [options]
            \\
            \\Commands:
            \\  help           Show this help message
            \\  version        Show version information
            \\  init           Initialize a new database
            \\  add            Add vectors to database
            \\  search         Search for similar vectors
            \\  stats          Show database statistics
            \\  optimize       Optimize database performance
            \\  export         Export database data
            \\  import         Import data into database
            \\
            \\Options:
            \\  --db <path>    Database file path
            \\  --verbose      Enable verbose output
            \\  --quiet        Suppress output
            \\  --format       Output format (text, json, csv, yaml)
            \\
            \\Examples:
            \\  wdbx init --db vectors.wdbx --dimensions 384
            \\  wdbx add --db vectors.wdbx --vector "1.0,2.0,3.0"
            \\  wdbx search --db vectors.wdbx --query "1.1,2.1,3.1" --k 5
            \\
        ;
        try self.output(help_text);
    }

    /// Output message based on configuration
    pub fn output(self: *Self, message: []const u8) !void {
        if (!self.config.quiet) {
            const stdout = std.io.getStdOut().writer();
            try stdout.print("{s}\n", .{message});
        }
    }

    /// Log message based on level
    pub fn log(self: *Self, level: LogLevel, comptime fmt: []const u8, args: anytype) !void {
        if (level.toInt() >= self.config.log_level.toInt() and !self.config.quiet) {
            const stderr = std.io.getStdErr().writer();
            const level_str = switch (level) {
                .trace => "TRACE",
                .debug => "DEBUG",
                .info => "INFO",
                .warn => "WARN",
                .err => "ERROR",
                .fatal => "FATAL",
            };
            try stderr.print("[{s}] ", .{level_str});
            try stderr.print(fmt, args);
            try stderr.print("\n", .{});
        }
    }

    /// Open database
    pub fn openDatabase(self: *Self, path: []const u8, writable: bool) !void {
        if (self.database != null) {
            return error.DatabaseAlreadyOpen;
        }

        self.database = try core.Database.open(self.allocator, path, writable);
    }

    /// Get database (must be opened first)
    pub fn getDatabase(self: *Self) !*core.Database {
        return self.database orelse error.DatabaseNotOpen;
    }
};

/// Main entry point for CLI
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var arg_list = std.ArrayList([]const u8).init(allocator);
    defer arg_list.deinit();

    _ = args.next(); // Skip executable name
    while (args.next()) |arg| {
        try arg_list.append(arg);
    }

    // Parse configuration
    const config = try parser.parseConfig(allocator, arg_list.items);
    defer config.deinit(allocator);

    // Create and run CLI
    const cli = try CLI.init(allocator, config.cli_config);
    defer cli.deinit();

    try cli.run(config.command_args);
}