//! ABI Framework CLI
//!
//! Modern command-line interface for the ABI framework

const std = @import("std");
const abi = @import("../src/mod.zig");

/// Exit codes for the CLI
pub const ExitCode = enum(u8) {
    success = 0,
    usage_error = 1,
    config_error = 2,
    runtime_error = 3,
    io_error = 4,
    feature_error = 5,
};

/// CLI configuration
pub const CliConfig = struct {
    json_output: bool = false,
    verbose: bool = false,
    log_level: LogLevel = .info,
    
    pub const LogLevel = enum {
        debug,
        info,
        warn,
        error,
    };
};

/// Main CLI application
pub const Cli = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: CliConfig,
    framework: abi.Framework,
    
    pub fn init(allocator: std.mem.Allocator, config: CliConfig) !Self {
        const framework_config = abi.framework.defaultConfig();
        const framework = try abi.createFramework(allocator, framework_config);
        
        return Self{
            .allocator = allocator,
            .config = config,
            .framework = framework,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.framework.deinit();
    }
    
    /// Main entry point for CLI commands
    pub fn run(self: *Self, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printHelp();
            return .usage_error;
        }
        
        const command = args[0];
        const command_args = args[1..];
        
        if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
            try self.printHelp();
            return .success;
        }
        
        if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version")) {
            try self.printVersion();
            return .success;
        }
        
        if (std.mem.eql(u8, command, "features")) {
            return try self.handleFeatures(command_args);
        }
        
        if (std.mem.eql(u8, command, "framework")) {
            return try self.handleFramework(command_args);
        }
        
        if (std.mem.eql(u8, command, "ai")) {
            return try self.handleAi(command_args);
        }
        
        if (std.mem.eql(u8, command, "database")) {
            return try self.handleDatabase(command_args);
        }
        
        try self.printError("Unknown command: {s}", .{command});
        try self.printHelp();
        return .usage_error;
    }
    
    fn printHelp(self: *Self) !void {
        const help_text = 
            \\ABI Framework CLI
            \\
            \\Usage: abi <command> [options]
            \\
            \\Commands:
            \\  help        Show this help message
            \\  version     Show version information
            \\  features    Manage framework features
            \\  framework   Framework management commands
            \\  ai          AI/ML operations
            \\  database    Database operations
            \\
            \\Options:
            \\  --json      Output in JSON format
            \\  --verbose   Enable verbose output
            \\  --log-level <level>  Set log level (debug, info, warn, error)
            \\
        ;
        
        if (self.config.json_output) {
            try self.printJson("{s}", .{help_text});
        } else {
            try self.printInfo("{s}", .{help_text});
        }
    }
    
    fn printVersion(self: *Self) !void {
        const version = abi.version();
        if (self.config.json_output) {
            try self.printJson("{{\"version\":\"{s}\"}}", .{version});
        } else {
            try self.printInfo("ABI Framework version: {s}\n", .{version});
        }
    }
    
    fn handleFeatures(self: *Self, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printError("Features subcommand required");
            return .usage_error;
        }
        
        const subcommand = args[0];
        const sub_args = args[1..];
        
        if (std.mem.eql(u8, subcommand, "list")) {
            return try self.listFeatures();
        }
        
        if (std.mem.eql(u8, subcommand, "enable")) {
            return try self.enableFeatures(sub_args);
        }
        
        if (std.mem.eql(u8, subcommand, "disable")) {
            return try self.disableFeatures(sub_args);
        }
        
        try self.printError("Unknown features subcommand: {s}", .{subcommand});
        return .usage_error;
    }
    
    fn listFeatures(self: *Self) !ExitCode {
        const features = [_]abi.features.FeatureTag{ .ai, .gpu, .database, .web, .monitoring, .connectors };
        
        if (self.config.json_output) {
            try self.printJson("{{\"features\":{{", .{});
            var first = true;
            for (features) |feature| {
                if (!first) try self.printJson(",", .{});
                first = false;
                const enabled = self.framework.isFeatureEnabled(feature);
                try self.printJson("\"{s}\":{s}", .{ 
                    abi.features.config.getName(feature),
                    if (enabled) "true" else "false"
                });
            }
            try self.printJson("}}}", .{});
        } else {
            try self.printInfo("Enabled features:\n", .{});
            for (features) |feature| {
                const enabled = self.framework.isFeatureEnabled(feature);
                const status = if (enabled) "enabled" else "disabled";
                try self.printInfo("  - {s}: {s}\n", .{ 
                    abi.features.config.getName(feature),
                    status
                });
            }
        }
        
        return .success;
    }
    
    fn enableFeatures(self: *Self, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printError("Specify at least one feature to enable");
            return .usage_error;
        }
        
        var enabled_count: usize = 0;
        for (args) |feature_name| {
            const feature = self.parseFeature(feature_name) orelse {
                try self.printError("Unknown feature: {s}", .{feature_name});
                return .feature_error;
            };
            
            self.framework.enableFeature(feature);
            enabled_count += 1;
        }
        
        if (self.config.json_output) {
            try self.printJson("{{\"status\":\"success\",\"enabled\":{d}}}", .{enabled_count});
        } else {
            try self.printInfo("Enabled {d} feature(s)\n", .{enabled_count});
        }
        
        return .success;
    }
    
    fn disableFeatures(self: *Self, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printError("Specify at least one feature to disable");
            return .usage_error;
        }
        
        var disabled_count: usize = 0;
        for (args) |feature_name| {
            const feature = self.parseFeature(feature_name) orelse {
                try self.printError("Unknown feature: {s}", .{feature_name});
                return .feature_error;
            };
            
            self.framework.disableFeature(feature);
            disabled_count += 1;
        }
        
        if (self.config.json_output) {
            try self.printJson("{{\"status\":\"success\",\"disabled\":{d}}}", .{disabled_count});
        } else {
            try self.printInfo("Disabled {d} feature(s)\n", .{disabled_count});
        }
        
        return .success;
    }
    
    fn handleFramework(self: *Self, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printError("Framework subcommand required");
            return .usage_error;
        }
        
        const subcommand = args[0];
        
        if (std.mem.eql(u8, subcommand, "status")) {
            return try self.frameworkStatus();
        }
        
        if (std.mem.eql(u8, subcommand, "start")) {
            return try self.startFramework();
        }
        
        if (std.mem.eql(u8, subcommand, "stop")) {
            return try self.stopFramework();
        }
        
        try self.printError("Unknown framework subcommand: {s}", .{subcommand});
        return .usage_error;
    }
    
    fn frameworkStatus(self: *Self) !ExitCode {
        if (self.config.json_output) {
            const stats = self.framework.getStats();
            try self.printJson(
                "{{\"status\":\"{s}\",\"components\":{d},\"uptime\":{d},\"features\":{d}}}",
                .{
                    if (self.framework.isRunning()) "running" else "stopped",
                    stats.total_components,
                    stats.uptime(),
                    stats.enabled_features,
                }
            );
        } else {
            try self.framework.writeSummary(std.io.getStdOut().writer());
        }
        
        return .success;
    }
    
    fn startFramework(self: *Self) !ExitCode {
        if (self.framework.isRunning()) {
            try self.printError("Framework is already running");
            return .runtime_error;
        }
        
        try self.framework.start();
        
        if (self.config.json_output) {
            try self.printJson("{{\"status\":\"success\",\"message\":\"Framework started\"}}", .{});
        } else {
            try self.printInfo("Framework started successfully\n", .{});
        }
        
        return .success;
    }
    
    fn stopFramework(self: *Self) !ExitCode {
        if (!self.framework.isRunning()) {
            try self.printError("Framework is not running");
            return .runtime_error;
        }
        
        self.framework.stop();
        
        if (self.config.json_output) {
            try self.printJson("{{\"status\":\"success\",\"message\":\"Framework stopped\"}}", .{});
        } else {
            try self.printInfo("Framework stopped successfully\n", .{});
        }
        
        return .success;
    }
    
    fn handleAi(self: *Self, args: [][]const u8) !ExitCode {
        if (!self.framework.isFeatureEnabled(.ai)) {
            try self.printError("AI feature is not enabled");
            return .feature_error;
        }
        
        try self.printInfo("AI operations not yet implemented\n", .{});
        return .success;
    }
    
    fn handleDatabase(self: *Self, args: [][]const u8) !ExitCode {
        if (!self.framework.isFeatureEnabled(.database)) {
            try self.printError("Database feature is not enabled");
            return .feature_error;
        }
        
        try self.printInfo("Database operations not yet implemented\n", .{});
        return .success;
    }
    
    fn parseFeature(self: *Self, name: []const u8) ?abi.features.FeatureTag {
        const features = [_]abi.features.FeatureTag{ .ai, .gpu, .database, .web, .monitoring, .connectors };
        
        for (features) |feature| {
            if (std.mem.eql(u8, name, abi.features.config.getName(feature))) {
                return feature;
            }
        }
        
        return null;
    }
    
    fn printInfo(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        if (@intFromEnum(self.config.log_level) <= @intFromEnum(.info)) {
            try std.io.getStdOut().writer().print(fmt, args);
        }
    }
    
    fn printError(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        try std.io.getStdErr().writer().print("Error: " ++ fmt ++ "\n", args);
    }
    
    fn printJson(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        try std.io.getStdOut().writer().print(fmt, args);
    }
};

/// Parse command line arguments
fn parseArgs(allocator: std.mem.Allocator) !struct { config: CliConfig, args: [][]const u8 } {
    const raw_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, raw_args);
    
    if (raw_args.len == 0) {
        return .{ .config = CliConfig{}, .args = &[_][]const u8{} };
    }
    
    var config = CliConfig{};
    var args_start: usize = 1;
    
    // Parse global options
    for (raw_args[1..], 1..) |arg, i| {
        if (std.mem.eql(u8, arg, "--json")) {
            config.json_output = true;
            args_start = i + 2;
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            config.verbose = true;
            args_start = i + 2;
        } else if (std.mem.startsWith(u8, arg, "--log-level=")) {
            const level_str = arg["--log-level=".len..];
            config.log_level = std.meta.stringToEnum(CliConfig.LogLevel, level_str) orelse {
                try std.io.getStdErr().writer().print("Invalid log level: {s}\n", .{level_str});
                std.process.exit(@intFromEnum(ExitCode.usage_error));
            };
            args_start = i + 2;
        } else {
            break;
        }
    }
    
    const args = if (args_start < raw_args.len) raw_args[args_start..] else &[_][]const u8{};
    
    return .{ .config = config, .args = args };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    
    const allocator = gpa.allocator();
    
    const parsed = try parseArgs(allocator);
    defer std.process.argsFree(allocator, parsed.args);
    
    var cli = try Cli.init(allocator, parsed.config);
    defer cli.deinit();
    
    const exit_code = cli.run(parsed.args) catch |err| {
        try std.io.getStdErr().writer().print("Fatal error: {s}\n", .{@errorName(err)});
        return err;
    };
    
    if (exit_code != .success) {
        std.process.exit(@intFromEnum(exit_code));
    }
}

test "CLI initialization" {
    var cli = try Cli.init(std.testing.allocator, CliConfig{});
    defer cli.deinit();
    
    try std.testing.expect(!cli.framework.isRunning());
}

test "feature parsing" {
    var cli = try Cli.init(std.testing.allocator, CliConfig{});
    defer cli.deinit();
    
    try std.testing.expectEqual(abi.features.FeatureTag.ai, cli.parseFeature("ai").?);
    try std.testing.expectEqual(abi.features.FeatureTag.database, cli.parseFeature("database").?);
    try std.testing.expect(cli.parseFeature("unknown") == null);
}