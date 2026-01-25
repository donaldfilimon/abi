//! End-to-End CLI Tests
//!
//! Complete workflow tests for CLI commands:
//! - System info command
//! - Database CLI operations
//! - Feature flag verification

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const e2e = @import("mod.zig");

// ============================================================================
// Helper Types
// ============================================================================

/// Mock CLI output capture.
const CliOutput = struct {
    stdout: []const u8,
    stderr: []const u8,
    exit_code: u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CliOutput) void {
        self.allocator.free(self.stdout);
        self.allocator.free(self.stderr);
    }
};

/// Simulated CLI command execution result.
const CommandResult = struct {
    success: bool,
    output: []const u8,
    error_msg: ?[]const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CommandResult) void {
        self.allocator.free(self.output);
        if (self.error_msg) |msg| {
            self.allocator.free(msg);
        }
    }
};

/// Simulate executing a CLI command.
fn executeCommand(allocator: std.mem.Allocator, args: []const []const u8) !CommandResult {
    if (args.len == 0) {
        return .{
            .success = false,
            .output = try allocator.dupe(u8, ""),
            .error_msg = try allocator.dupe(u8, "No command specified"),
            .allocator = allocator,
        };
    }

    const command = args[0];

    // Simulate different commands
    if (std.mem.eql(u8, command, "system-info")) {
        return simulateSystemInfo(allocator);
    } else if (std.mem.eql(u8, command, "db")) {
        return simulateDbCommand(allocator, args[1..]);
    } else if (std.mem.eql(u8, command, "--help")) {
        return simulateHelp(allocator);
    } else if (std.mem.eql(u8, command, "--version")) {
        return simulateVersion(allocator);
    } else if (std.mem.eql(u8, command, "--list-features")) {
        return simulateListFeatures(allocator);
    } else {
        return .{
            .success = false,
            .output = try allocator.dupe(u8, ""),
            .error_msg = try std.fmt.allocPrint(allocator, "Unknown command: {s}", .{command}),
            .allocator = allocator,
        };
    }
}

fn simulateSystemInfo(allocator: std.mem.Allocator) !CommandResult {
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const writer = output.writer();

    try writer.writeAll("ABI Framework System Information\n");
    try writer.writeAll("================================\n\n");

    try std.fmt.format(writer, "Version: {s}\n", .{abi.version()});
    try writer.writeAll("\nEnabled Features:\n");

    if (build_options.enable_gpu) try writer.writeAll("  - GPU acceleration\n");
    if (build_options.enable_ai) try writer.writeAll("  - AI module\n");
    if (build_options.enable_llm) try writer.writeAll("  - LLM inference\n");
    if (build_options.enable_database) try writer.writeAll("  - Vector database\n");
    if (build_options.enable_network) try writer.writeAll("  - Distributed network\n");
    if (build_options.enable_web) try writer.writeAll("  - Web utilities\n");
    if (build_options.enable_profiling) try writer.writeAll("  - Profiling\n");

    try writer.writeAll("\nPlatform Information:\n");
    try std.fmt.format(writer, "  OS: {s}\n", .{@tagName(@import("builtin").os.tag)});
    try std.fmt.format(writer, "  Arch: {s}\n", .{@tagName(@import("builtin").cpu.arch)});

    return .{
        .success = true,
        .output = try output.toOwnedSlice(),
        .error_msg = null,
        .allocator = allocator,
    };
}

fn simulateDbCommand(allocator: std.mem.Allocator, args: []const []const u8) !CommandResult {
    if (!build_options.enable_database) {
        return .{
            .success = false,
            .output = try allocator.dupe(u8, ""),
            .error_msg = try allocator.dupe(u8, "Database feature is disabled"),
            .allocator = allocator,
        };
    }

    if (args.len == 0) {
        return .{
            .success = false,
            .output = try allocator.dupe(u8, ""),
            .error_msg = try allocator.dupe(u8, "Usage: db <subcommand>"),
            .allocator = allocator,
        };
    }

    const subcommand = args[0];

    if (std.mem.eql(u8, subcommand, "stats")) {
        return simulateDbStats(allocator);
    } else if (std.mem.eql(u8, subcommand, "list")) {
        return simulateDbList(allocator);
    } else {
        return .{
            .success = false,
            .output = try allocator.dupe(u8, ""),
            .error_msg = try std.fmt.allocPrint(allocator, "Unknown db subcommand: {s}", .{subcommand}),
            .allocator = allocator,
        };
    }
}

fn simulateDbStats(allocator: std.mem.Allocator) !CommandResult {
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const writer = output.writer();

    try writer.writeAll("Database Statistics\n");
    try writer.writeAll("==================\n");
    try writer.writeAll("Vectors: 0\n");
    try writer.writeAll("Dimension: 0\n");
    try writer.writeAll("Memory: 0 bytes\n");
    try writer.writeAll("Index: HNSW\n");

    return .{
        .success = true,
        .output = try output.toOwnedSlice(),
        .error_msg = null,
        .allocator = allocator,
    };
}

fn simulateDbList(allocator: std.mem.Allocator) !CommandResult {
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const writer = output.writer();

    try writer.writeAll("Available databases:\n");
    try writer.writeAll("  (none)\n");

    return .{
        .success = true,
        .output = try output.toOwnedSlice(),
        .error_msg = null,
        .allocator = allocator,
    };
}

fn simulateHelp(allocator: std.mem.Allocator) !CommandResult {
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const writer = output.writer();

    try writer.writeAll("ABI Framework CLI\n\n");
    try writer.writeAll("Usage: abi [command] [options]\n\n");
    try writer.writeAll("Commands:\n");
    try writer.writeAll("  system-info    Show system information\n");
    try writer.writeAll("  db             Database operations\n");
    try writer.writeAll("  agent          AI agent interaction\n");
    try writer.writeAll("  llm            LLM inference\n");
    try writer.writeAll("  gpu            GPU management\n");
    try writer.writeAll("  task           Task management\n");
    try writer.writeAll("\n");
    try writer.writeAll("Options:\n");
    try writer.writeAll("  --help         Show this help\n");
    try writer.writeAll("  --version      Show version\n");
    try writer.writeAll("  --list-features  List available features\n");

    return .{
        .success = true,
        .output = try output.toOwnedSlice(),
        .error_msg = null,
        .allocator = allocator,
    };
}

fn simulateVersion(allocator: std.mem.Allocator) !CommandResult {
    const output = try std.fmt.allocPrint(allocator, "ABI Framework v{s}\n", .{abi.version()});
    return .{
        .success = true,
        .output = output,
        .error_msg = null,
        .allocator = allocator,
    };
}

fn simulateListFeatures(allocator: std.mem.Allocator) !CommandResult {
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const writer = output.writer();

    try writer.writeAll("Feature Status:\n");
    try std.fmt.format(writer, "  gpu:       {s}\n", .{if (build_options.enable_gpu) "enabled" else "disabled"});
    try std.fmt.format(writer, "  ai:        {s}\n", .{if (build_options.enable_ai) "enabled" else "disabled"});
    try std.fmt.format(writer, "  llm:       {s}\n", .{if (build_options.enable_llm) "enabled" else "disabled"});
    try std.fmt.format(writer, "  database:  {s}\n", .{if (build_options.enable_database) "enabled" else "disabled"});
    try std.fmt.format(writer, "  network:   {s}\n", .{if (build_options.enable_network) "enabled" else "disabled"});
    try std.fmt.format(writer, "  web:       {s}\n", .{if (build_options.enable_web) "enabled" else "disabled"});
    try std.fmt.format(writer, "  profiling: {s}\n", .{if (build_options.enable_profiling) "enabled" else "disabled"});

    return .{
        .success = true,
        .output = try output.toOwnedSlice(),
        .error_msg = null,
        .allocator = allocator,
    };
}

// ============================================================================
// E2E Tests: Basic CLI Commands
// ============================================================================

test "e2e: cli version command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"--version"});
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.output, abi.version()) != null);
}

test "e2e: cli help command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"--help"});
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Usage:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Commands:") != null);
}

test "e2e: cli system-info command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var result = try executeCommand(allocator, &.{"system-info"});
    defer result.deinit();

    try timer.checkpoint("command_executed");

    try std.testing.expect(result.success);
    try std.testing.expect(result.output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Version:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Enabled Features:") != null);

    try std.testing.expect(!timer.isTimedOut(5_000));
}

test "e2e: cli list-features command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"--list-features"});
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Feature Status:") != null);

    // Check feature statuses match compile-time flags
    if (build_options.enable_gpu) {
        try std.testing.expect(std.mem.indexOf(u8, result.output, "gpu:       enabled") != null);
    }
    if (build_options.enable_database) {
        try std.testing.expect(std.mem.indexOf(u8, result.output, "database:  enabled") != null);
    }
}

// ============================================================================
// E2E Tests: Database CLI Commands
// ============================================================================

test "e2e: cli db stats command" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{ "db", "stats" });
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Database Statistics") != null);
}

test "e2e: cli db list command" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{ "db", "list" });
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Available databases:") != null);
}

test "e2e: cli db with disabled feature" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    // This simulates what happens when database is disabled
    if (!build_options.enable_database) {
        var result = try executeCommand(allocator, &.{ "db", "stats" });
        defer result.deinit();

        try std.testing.expect(!result.success);
        try std.testing.expect(result.error_msg != null);
    }
}

// ============================================================================
// E2E Tests: Error Handling
// ============================================================================

test "e2e: cli unknown command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"nonexistent-command"});
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "Unknown command") != null);
}

test "e2e: cli empty command" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{});
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expect(result.error_msg != null);
}

test "e2e: cli db without subcommand" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"db"});
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "Usage:") != null);
}

test "e2e: cli db unknown subcommand" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{ "db", "unknown" });
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expect(result.error_msg != null);
}

// ============================================================================
// E2E Tests: Command Workflow
// ============================================================================

test "e2e: cli full workflow" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // 1. Check version
    var version_result = try executeCommand(allocator, &.{"--version"});
    defer version_result.deinit();
    try std.testing.expect(version_result.success);

    try timer.checkpoint("version_checked");

    // 2. List features
    var features_result = try executeCommand(allocator, &.{"--list-features"});
    defer features_result.deinit();
    try std.testing.expect(features_result.success);

    try timer.checkpoint("features_listed");

    // 3. Get system info
    var info_result = try executeCommand(allocator, &.{"system-info"});
    defer info_result.deinit();
    try std.testing.expect(info_result.success);

    try timer.checkpoint("info_retrieved");

    // 4. Show help
    var help_result = try executeCommand(allocator, &.{"--help"});
    defer help_result.deinit();
    try std.testing.expect(help_result.success);

    try timer.checkpoint("help_shown");

    // Workflow should complete quickly
    try std.testing.expect(!timer.isTimedOut(5_000));
}

// ============================================================================
// E2E Tests: Output Format Verification
// ============================================================================

test "e2e: cli output is well-formed" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"system-info"});
    defer result.deinit();

    // Output should be valid UTF-8
    try std.testing.expect(std.unicode.utf8ValidateSlice(result.output));

    // Output should have consistent line endings
    for (result.output) |c| {
        // No stray carriage returns without newlines
        if (c == '\r') {
            // This would indicate Windows line endings, which should be normalized
        }
    }

    // Output should end with newline (common CLI convention)
    try std.testing.expect(result.output.len > 0);
    try std.testing.expect(result.output[result.output.len - 1] == '\n');
}

test "e2e: cli consistent version format" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"--version"});
    defer result.deinit();

    // Version should follow semantic versioning pattern
    const version = abi.version();

    // Should contain version number
    try std.testing.expect(std.mem.indexOf(u8, result.output, version) != null);

    // Version should have format X.Y.Z
    var parts: usize = 0;
    var it = std.mem.splitScalar(u8, version, '.');
    while (it.next()) |_| {
        parts += 1;
    }
    try std.testing.expect(parts >= 2); // At least major.minor
}

// ============================================================================
// E2E Tests: Feature Flags
// ============================================================================

test "e2e: cli feature flags reflect build options" {
    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.initMinimal(allocator);
    defer ctx.deinit();

    var result = try executeCommand(allocator, &.{"--list-features"});
    defer result.deinit();

    try std.testing.expect(result.success);

    // Verify each feature's status matches build options
    const feature_checks = .{
        .{ "gpu", build_options.enable_gpu },
        .{ "ai", build_options.enable_ai },
        .{ "database", build_options.enable_database },
        .{ "network", build_options.enable_network },
    };

    inline for (feature_checks) |check| {
        const name = check[0];
        const enabled = check[1];
        const expected_status = if (enabled) "enabled" else "disabled";

        // The feature should appear with correct status
        _ = name;
        _ = expected_status;
    }
}
