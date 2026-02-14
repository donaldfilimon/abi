//! Network CLI command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const subcommand = @import("../utils/subcommand.zig");

const ArgParser = utils.args.ArgParser;

/// Run the network command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    abi.network.init(allocator) catch |err| switch (err) {
        error.NetworkDisabled => {
            utils.output.printWarning("Network support disabled at build time.", .{});
            return;
        },
        else => return err,
    };
    defer abi.network.deinit();

    const commands = [_]subcommand.Command{
        .{ .names = &.{"status"}, .run = runStatusSubcommand },
        .{ .names = &.{ "list", "nodes" }, .run = runListSubcommand },
        .{ .names = &.{"register"}, .run = runRegisterSubcommand },
        .{ .names = &.{"unregister"}, .run = runUnregisterSubcommand },
        .{ .names = &.{"touch"}, .run = runTouchSubcommand },
        .{ .names = &.{"set-status"}, .run = runSetStatusSubcommand },
    };

    try subcommand.runSubcommand(
        allocator,
        &parser,
        &commands,
        runStatusSubcommand,
        printHelp,
        onUnknownCommand,
    );
}

/// Print a short network summary for system-info.
pub fn printSummary() void {
    if (!abi.network.isEnabled()) {
        std.debug.print("  Network: disabled\n", .{});
        return;
    }
    if (!abi.network.isInitialized()) {
        std.debug.print("  Network: enabled (not initialized)\n", .{});
        return;
    }
    if (abi.network.defaultConfig()) |config| {
        const registry = abi.network.defaultRegistry() catch null;
        const node_count = if (registry) |reg| reg.list().len else 0;
        std.debug.print("  Network: {s} ({d} nodes)\n", .{ config.cluster_id, node_count });
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi network", "<command>")
        .description("Manage network nodes and cluster status.")
        .section("Commands")
        .subcommand(.{ .name = "status", .description = "Show network config and node count" })
        .subcommand(.{ .name = "list | nodes", .description = "List registered nodes" })
        .subcommand(.{ .name = "register <id> <a>", .description = "Register or update a node" })
        .subcommand(.{ .name = "unregister <id>", .description = "Remove a node" })
        .subcommand(.{ .name = "touch <id>", .description = "Update node heartbeat timestamp" })
        .subcommand(.{ .name = "set-status <id> <s>", .description = "Set status (healthy, degraded, offline)" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi network status", "Show cluster info")
        .example("abi network list", "List all nodes")
        .example("abi network register node-1 127.0.0.1:8080", "Add a node");

    builder.print();
}

fn printStatus() !void {
    if (!abi.network.isEnabled()) {
        utils.output.printWarning("Network: disabled", .{});
        return;
    }
    const config = abi.network.defaultConfig();
    if (config == null) {
        utils.output.printInfo("Network: enabled (not initialized)", .{});
        return;
    }
    const registry = try abi.network.defaultRegistry();
    utils.output.printHeader("Network Status");
    utils.output.printKeyValue("Cluster ID", config.?.cluster_id);
    utils.output.printKeyValueFmt("Node Count", "{d}", .{registry.list().len});
}

fn runStatusSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    _ = parser;
    try printStatus();
}

fn runListSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = parser;
    try printNodes(allocator);
}

fn runRegisterSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    const id = parser.next() orelse {
        utils.output.printError("Missing node ID", .{});
        utils.output.printInfo("Usage: abi network register <id> <address>", .{});
        return;
    };
    const address = parser.next() orelse {
        utils.output.printError("Missing node address", .{});
        utils.output.printInfo("Usage: abi network register <id> <address>", .{});
        return;
    };
    const registry = try abi.network.defaultRegistry();
    try registry.register(id, address);
    utils.output.printSuccess("Registered {s} at {s}", .{ id, address });
}

fn runUnregisterSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    const id = parser.next() orelse {
        utils.output.printError("Missing node ID", .{});
        utils.output.printInfo("Usage: abi network unregister <id>", .{});
        return;
    };
    const registry = try abi.network.defaultRegistry();
    const removed = registry.unregister(id);
    if (removed) {
        utils.output.printSuccess("Unregistered {s}", .{id});
    } else {
        utils.output.printWarning("Node {s} not found", .{id});
    }
}

fn runTouchSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    const id = parser.next() orelse {
        utils.output.printError("Missing node ID", .{});
        utils.output.printInfo("Usage: abi network touch <id>", .{});
        return;
    };
    const registry = try abi.network.defaultRegistry();
    if (registry.touch(id)) {
        utils.output.printSuccess("Touched {s}", .{id});
    } else {
        utils.output.printWarning("Node {s} not found", .{id});
    }
}

fn runSetStatusSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    _ = allocator;
    const id = parser.next() orelse {
        utils.output.printError("Missing node ID", .{});
        utils.output.printInfo("Usage: abi network set-status <id> <status>", .{});
        return;
    };
    const status_text = parser.next() orelse {
        utils.output.printError("Missing status value", .{});
        utils.output.printInfo("Usage: abi network set-status <id> <status>", .{});
        return;
    };
    const status = utils.args.parseNodeStatus(status_text) orelse {
        utils.output.printError("Invalid status: {s}", .{status_text});
        utils.output.printInfo("Valid statuses: healthy, degraded, offline", .{});
        return;
    };
    const registry = try abi.network.defaultRegistry();
    if (registry.setStatus(id, status)) {
        utils.output.printSuccess("Updated {s} to {t}", .{ id, status });
    } else {
        utils.output.printWarning("Node {s} not found", .{id});
    }
}

fn onUnknownCommand(command: []const u8) void {
    utils.output.printError("Unknown network command: {s}", .{command});
}

fn printNodes(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const registry = try abi.network.defaultRegistry();
    const nodes = registry.list();
    if (nodes.len == 0) {
        utils.output.printInfo("No nodes registered.", .{});
        return;
    }
    utils.output.printHeaderFmt("Registered Nodes ({d})", .{nodes.len});
    for (nodes) |node| {
        std.debug.print("  " ++ utils.output.color.green ++ "•" ++ utils.output.color.reset ++ " {s: <15} {s: <20} ({t}) seen {d}ms ago\n", .{ node.id, node.address, node.status, node.last_seen_ms });
    }
}
