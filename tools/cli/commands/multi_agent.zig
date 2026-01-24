//! Multi‑Agent CLI command
//!
//! Provides a simple interface to the `MultiAgentCoordinator` API.
//! Currently supports a single sub‑command:
//!   * `info` – shows whether the AI feature is enabled and the number of
//!     registered agents (always zero in this lightweight example).
//! Additional functionality can be added later following the same patterns.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Entry point for the `multi-agent` command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    // Show help if no sub‑command or help flag is present.
    if (!parser.hasMore() or parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    const sub = parser.next().?; // safe after hasMore check
    if (std.mem.eql(u8, sub, "info")) {
        try runInfo(allocator);
        return;
    }

    utils.output.printError("unknown subcommand: {s}", .{sub});
    printHelp(allocator);
}

fn runInfo(allocator: std.mem.Allocator) !void {
    // Initialise the framework to access runtime feature matrix.
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    utils.output.printHeader("Multi‑Agent Coordinator");

    // Feature gating – only meaningful when AI is enabled.
    const ai_enabled = framework.isEnabled(.ai);
    utils.output.printKeyValue("AI Feature", utils.output.boolLabel(ai_enabled));

    if (!ai_enabled) {
        // When disabled, the stub returns error.AgentDisabled.
        utils.output.printInfo("Coordinator is unavailable (AI disabled)", .{});
        return;
    }

    // Use the public API re‑exported via `abi.ai`.
    const Coordinator = abi.ai.MultiAgentCoordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    const count = coord.agents.items.len;
    const count_str = try std.fmt.allocPrint(allocator, "{d}", .{count});
    defer allocator.free(count_str);
    utils.output.printKeyValue("Registered Agents", count_str);
    // No further actions – just illustrative output.
    utils.output.printSuccess("Coordinator ready.", .{});
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi multi-agent <subcommand>", "")
        .description("Interact with the Multi‑Agent Coordinator (AI feature gated).")
        .section("Subcommands")
        .subcommand(.{ .name = "info", .description = "Show coordinator status and feature gating" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help);

    builder.print();
}
