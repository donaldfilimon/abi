//! CLI Command Template
//!
//! This template demonstrates the recommended architectural patterns for
//! implementing new CLI commands in the ABI framework.
//!
//! Patterns:
//! 1. Use `utils.args.ArgParser` for structured, type-safe argument parsing.
//! 2. Use `utils.help.HelpBuilder` for fluent, consistent help generation.
//! 3. Use `utils.output` for styled terminal output and progress indicators.
//! 4. Use `utils.args.CliError` for rich error context and user suggestions.

const std = @import("std");
const abi = @import("abi");
const shared_utils = abi.shared.utils;
const utils = @import("../utils/mod.zig");

// Type aliases for cleaner code
const output = utils.output;
const ArgParser = utils.args.ArgParser;
const HelpBuilder = utils.help.HelpBuilder;
const common_options = utils.help.common_options;

/// Main entry point for the command.
/// The framework passes the allocator and the slice of arguments following the command name.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = ArgParser.init(allocator, args);

    // 1. Mandatory Help/Empty Check
    if (!parser.hasMore() or parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // 2. Subcommand Dispatching
    const command = parser.next().?; // Safety: hasMore() checked above
    if (std.mem.eql(u8, command, "action")) {
        try runAction(allocator, &parser);
    } else if (std.mem.eql(u8, command, "status")) {
        try runStatus(allocator, &parser);
    } else {
        output.printError("Unknown subcommand: {s}", .{command});
        printHelp(allocator);
    }
}

/// Example of a subcommand that takes parameters and options.
fn runAction(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    // Required positional argument
    const target = parser.next() orelse {
        output.printError("Missing required argument: <target>", .{});
        return;
    };

    // Optional flags and values using ArgParser helpers
    const count = parser.consumeInt(u32, &[_][]const u8{ "--count", "-c" }, 1);
    const force = parser.consumeFlag(&[_][]const u8{ "--force", "-f" });
    const message = parser.consumeOption(&[_][]const u8{ "--message", "-m" }) orelse "No message";

    output.printHeader("Running Action");
    output.printInfo("Target: {s}", .{target});
    output.printKeyValueFmt("Iterations", "{d}", .{count});
    output.printKeyValue("Force Mode", output.boolLabel(force));
    output.printKeyValue("Message", message);

    // 3. Structured Error Handling
    // Wrap errors with context and actionable suggestions
    doSomethingInternal() catch |err| {
        const cli_err = utils.args.cliErrorWithSuggestion(
            err,
            "Internal action failed",
            "Ensure the target is reachable or try again with --force",
        );
        cli_err.print();
        return;
    };

    // 4. Progress Bars
    var progress = output.ProgressBar.start(count, "Processing");
    for (0..count) |i| {
        // Simulate work
        shared_utils.sleepMs(100);
        progress.update(i + 1);
    }

    output.printSuccess("Action completed successfully on {s}", .{target});
}

/// Example of a simple status command.
fn runStatus(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    _ = parser;

    output.printHeader("System Status");
    output.printKeyValue("API Version", abi.version());
    output.printKeyValue("Environment", "Production");

    output.printBulletList("Active Components", &[_][]const u8{
        "Compute Engine [Active]",
        "Network Registry [Healthy]",
        "GPU Accelerators [1 Detected]",
    });
}

/// Help text generation using the fluent HelpBuilder API.
fn printHelp(allocator: std.mem.Allocator) void {
    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi template", "<command> [options]")
        .description("Description of what this command module does.")
        .section("Commands")
        .subcommand(.{ .name = "action <target>", .description = "Perform a specific action on a target" })
        .subcommand(.{ .name = "status", .description = "Show current operational status" })
        .newline()
        .section("Options (action)")
        .option(.{ .short = "-c", .long = "--count", .arg = "INT", .description = "Number of times to run" })
        .option(.{ .short = "-f", .long = "--force", .description = "Override safety checks" })
        .option(common_options.help)
        .option(common_options.verbose)
        .newline()
        .section("Examples")
        .example("abi template action my-target --count 5", "Run action 5 times")
        .example("abi template status", "Check system health");

    builder.print();
}

/// Dummy function for error demonstration
fn doSomethingInternal() !void {
    return;
}
