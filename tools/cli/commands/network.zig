//! Network CLI command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the network command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    abi.network.init(allocator) catch |err| switch (err) {
        error.NetworkDisabled => {
            std.debug.print("Network support disabled at build time.\n", .{});
            return;
        },
        else => return err,
    };

    if (args.len == 0) {
        try printStatus();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(command, &[_][]const u8{ "help", "--help" })) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printStatus();
        return;
    }

    if (utils.args.matchesAny(command, &[_][]const u8{ "list", "nodes" })) {
        try printNodes();
        return;
    }

    if (std.mem.eql(u8, command, "register")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi network register <id> <address>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const address = std.mem.sliceTo(args[2], 0);
        const registry = try abi.network.defaultRegistry();
        try registry.register(id, address);
        std.debug.print("Registered {s} at {s}\n", .{ id, address });
        return;
    }

    if (std.mem.eql(u8, command, "unregister")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi network unregister <id>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const registry = try abi.network.defaultRegistry();
        const removed = registry.unregister(id);
        if (removed) {
            std.debug.print("Unregistered {s}\n", .{id});
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    if (std.mem.eql(u8, command, "touch")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi network touch <id>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const registry = try abi.network.defaultRegistry();
        if (registry.touch(id)) {
            std.debug.print("Touched {s}\n", .{id});
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    if (std.mem.eql(u8, command, "set-status")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi network set-status <id> <status>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const status_text = std.mem.sliceTo(args[2], 0);
        const status = parseNodeStatus(status_text) orelse {
            std.debug.print("Invalid status: {s}\n", .{status_text});
            return;
        };
        const registry = try abi.network.defaultRegistry();
        if (registry.setStatus(id, status)) {
            std.debug.print("Updated {s} to {t}\n", .{ id, status });
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    std.debug.print("Unknown network command: {s}\n", .{command});
    printHelp();
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

fn printHelp() void {
    const text =
        "Usage: abi network <command>\n\n" ++
        "Commands:\n" ++
        "  status                    Show network config and node count\n" ++
        "  list | nodes               List registered nodes\n" ++
        "  register <id> <address>    Register or update a node\n" ++
        "  unregister <id>            Remove a node\n" ++
        "  touch <id>                 Update node heartbeat timestamp\n" ++
        "  set-status <id> <status>   Set status (healthy, degraded, offline)\n";
    std.debug.print("{s}", .{text});
}

fn printStatus() !void {
    if (!abi.network.isEnabled()) {
        std.debug.print("Network: disabled\n", .{});
        return;
    }
    const config = abi.network.defaultConfig();
    if (config == null) {
        std.debug.print("Network: enabled (not initialized)\n", .{});
        return;
    }
    const registry = try abi.network.defaultRegistry();
    std.debug.print(
        "Network cluster: {s}\nNodes: {d}\n",
        .{ config.?.cluster_id, registry.list().len },
    );
}

fn printNodes() !void {
    const registry = try abi.network.defaultRegistry();
    const nodes = registry.list();
    if (nodes.len == 0) {
        std.debug.print("No nodes registered.\n", .{});
        return;
    }
    std.debug.print("Nodes:\n", .{});
    for (nodes) |node| {
        std.debug.print(
            "  {s} {s} ({t}) last_seen_ms={d}\n",
            .{ node.id, node.address, node.status, node.last_seen_ms },
        );
    }
}

fn parseNodeStatus(text: []const u8) ?abi.network.NodeStatus {
    if (std.ascii.eqlIgnoreCase(text, "healthy")) return .healthy;
    if (std.ascii.eqlIgnoreCase(text, "degraded")) return .degraded;
    if (std.ascii.eqlIgnoreCase(text, "offline")) return .offline;
    return null;
}
