//! Network CLI command.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const ArgParser = utils.args.ArgParser;

// Wrapper functions for comptime children dispatch.
// Each wrapper performs network init/deinit (previously done in run()).
fn wrapStatus(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    try printStatus();
}
fn wrapList(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    var parser = ArgParser.init(allocator, args);
    try runListSubcommand(allocator, &parser);
}
fn wrapRegister(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    var parser = ArgParser.init(allocator, args);
    try runRegisterSubcommand(allocator, &parser);
}
fn wrapUnregister(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    var parser = ArgParser.init(allocator, args);
    try runUnregisterSubcommand(allocator, &parser);
}
fn wrapTouch(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    var parser = ArgParser.init(allocator, args);
    try runTouchSubcommand(allocator, &parser);
}
fn wrapSetStatus(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    var parser = ArgParser.init(allocator, args);
    try runSetStatusSubcommand(allocator, &parser);
}
fn wrapRaft(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    printRaftStatus(allocator);
}
fn wrapDiscovery(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    printDiscoveryStatus(allocator);
}
fn wrapBalancer(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    printBalancerStatus(allocator);
}
fn wrapHealth(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try initNetwork(allocator);
    defer abi.network.deinit();
    printHealthStatus(allocator);
}

fn initNetwork(allocator: std.mem.Allocator) !void {
    abi.network.init(allocator) catch |err| switch (err) {
        error.NetworkDisabled => {
            utils.output.printError("Network features are disabled.", .{});
            utils.output.printInfo("Rebuild with: zig build -Denable-network=true", .{});
            return err;
        },
        else => return err,
    };
}

pub const meta: command_mod.Meta = .{
    .name = "network",
    .description = "Network and distributed systems management",
    .kind = .group,
    .subcommands = &.{ "status", "list", "nodes", "register", "unregister", "touch", "set-status", "raft", "discovery", "balancer", "health" },
    .children = &.{
        .{ .name = "status", .description = "Show network config and node count", .handler = wrapStatus },
        .{ .name = "list", .description = "List registered nodes", .handler = wrapList },
        .{ .name = "nodes", .description = "List registered nodes", .handler = wrapList },
        .{ .name = "register", .description = "Register or update a node", .handler = wrapRegister },
        .{ .name = "unregister", .description = "Remove a node", .handler = wrapUnregister },
        .{ .name = "touch", .description = "Update node heartbeat timestamp", .handler = wrapTouch },
        .{ .name = "set-status", .description = "Set status (healthy, degraded, offline)", .handler = wrapSetStatus },
        .{ .name = "raft", .description = "Show Raft consensus state", .handler = wrapRaft },
        .{ .name = "discovery", .description = "Show service discovery status", .handler = wrapDiscovery },
        .{ .name = "balancer", .description = "Show load balancer status", .handler = wrapBalancer },
        .{ .name = "health", .description = "Show cluster health evaluation", .handler = wrapHealth },
    },
};

const network_subcommands = [_][]const u8{
    "status", "list", "nodes", "register", "unregister", "touch", "set-status", "raft", "discovery", "balancer", "health", "help",
};

/// Run the network command with the provided arguments.
/// Only reached when no child matches (help / unknown).
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        printHelp(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown network command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &network_subcommands)) |suggestion| {
        utils.output.println("Did you mean: {s}", .{suggestion});
    }
}

/// Print a short network summary for system-info.
pub fn printSummary() void {
    if (!abi.network.isEnabled()) {
        utils.output.println("  Network: disabled (rebuild with -Denable-network=true)", .{});
        return;
    }
    if (!abi.network.isInitialized()) {
        utils.output.println("  Network: enabled (run 'abi network status' to initialize)", .{});
        return;
    }
    if (abi.network.defaultConfig()) |config| {
        const registry = abi.network.defaultRegistry() catch null;
        const node_count = if (registry) |reg| reg.list().len else 0;
        utils.output.println("  Network: {s} ({d} nodes)", .{ config.cluster_id, node_count });
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi network", "<command>")
        .description("Manage network nodes and cluster status.")
        .section("Node Management")
        .subcommand(.{ .name = "status", .description = "Show network config and node count" })
        .subcommand(.{ .name = "list | nodes", .description = "List registered nodes" })
        .subcommand(.{ .name = "register <id> <a>", .description = "Register or update a node" })
        .subcommand(.{ .name = "unregister <id>", .description = "Remove a node" })
        .subcommand(.{ .name = "touch <id>", .description = "Update node heartbeat timestamp" })
        .subcommand(.{ .name = "set-status <id> <s>", .description = "Set status (healthy, degraded, offline)" })
        .section("Distributed Systems")
        .subcommand(.{ .name = "raft", .description = "Show Raft consensus state" })
        .subcommand(.{ .name = "discovery", .description = "Show service discovery status" })
        .subcommand(.{ .name = "balancer", .description = "Show load balancer status" })
        .subcommand(.{ .name = "health", .description = "Show cluster health evaluation" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi network status", "Show cluster info")
        .example("abi network list", "List all nodes")
        .example("abi network register node-1 127.0.0.1:8080", "Add a node")
        .example("abi network raft", "Show Raft consensus state")
        .example("abi network health", "Show cluster health");

    builder.print();
}

fn printStatus() !void {
    if (!abi.network.isEnabled()) {
        utils.output.printError("Network features are disabled.", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-network=true", .{});
        return;
    }
    const config = abi.network.defaultConfig();
    if (config == null) {
        utils.output.printInfo("Network: enabled but no cluster configured yet.", .{});
        utils.output.printInfo("Use 'abi network register <id> <address>' to add nodes.", .{});
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
        utils.output.println("  {s}•{s} {s: <15} {s: <20} ({t}) seen {d}ms ago", .{ utils.output.Color.green(), utils.output.Color.reset(), node.id, node.address, node.status, node.last_seen_ms });
    }
}

// ============================================================================
// Raft consensus status
// ============================================================================

fn printRaftStatus(allocator: std.mem.Allocator) void {
    if (!abi.network.isEnabled()) {
        utils.output.printWarning("Network feature disabled", .{});
        return;
    }
    utils.output.printHeader("Raft Consensus");

    // Create a temporary Raft node to show default configuration
    var node = abi.network.RaftNode.init(allocator, "local", .{}) catch {
        utils.output.printInfo("No active Raft node (showing defaults)", .{});
        utils.output.printKeyValue("State", "follower");
        utils.output.printKeyValueFmt("Term", "{d}", .{@as(u64, 0)});
        utils.output.printKeyValueFmt("Commit", "{d}", .{@as(u64, 0)});
        utils.output.printKeyValueFmt("Peers", "{d}", .{@as(usize, 0)});
        return;
    };
    defer node.deinit();

    const stats = node.getStats();
    utils.output.printKeyValue("State", stats.state.toString());
    utils.output.printKeyValueFmt("Term", "{d}", .{stats.current_term});
    utils.output.printKeyValueFmt("Commit", "{d}", .{stats.commit_index});
    utils.output.printKeyValueFmt("Peers", "{d}", .{stats.peer_count});
    utils.output.printKeyValueFmt("Log Length", "{d}", .{stats.log_length});
    utils.output.printKeyValueFmt("Last Applied", "{d}", .{stats.last_applied});
    if (stats.leader_id) |leader| {
        utils.output.printKeyValue("Leader", leader);
    } else {
        utils.output.printKeyValue("Leader", "none");
    }
}

// ============================================================================
// Service discovery status
// ============================================================================

fn printDiscoveryStatus(allocator: std.mem.Allocator) void {
    if (!abi.network.isEnabled()) {
        utils.output.printWarning("Network feature disabled", .{});
        return;
    }
    utils.output.printHeader("Service Discovery");

    // Create a temporary discovery instance to show configuration
    var disc = abi.network.ServiceDiscovery.init(allocator, .{}) catch {
        utils.output.printInfo("Service discovery not available", .{});
        return;
    };
    defer disc.deinit();

    utils.output.printKeyValueFmt("Backend", "{t}", .{disc.config.backend});
    utils.output.printKeyValue("Service Name", disc.config.service_name);
    utils.output.printKeyValue("Endpoint", disc.config.endpoint);
    utils.output.printKeyValueFmt("Port", "{d}", .{disc.config.service_port});
    utils.output.printKeyValueFmt("Health Interval", "{d}ms", .{disc.config.health_check_interval_ms});
    utils.output.printKeyValueFmt("TTL", "{d}s", .{disc.config.ttl_seconds});
    utils.output.printKeyValueFmt("Registered", "{s}", .{utils.output.boolLabel(disc.registered)});
    utils.output.printKeyValueFmt("Cached Services", "{d}", .{disc.cached_services.items.len});
}

// ============================================================================
// Load balancer status
// ============================================================================

fn printBalancerStatus(allocator: std.mem.Allocator) void {
    if (!abi.network.isEnabled()) {
        utils.output.printWarning("Network feature disabled", .{});
        return;
    }
    utils.output.printHeader("Load Balancer");

    var lb = abi.network.LoadBalancer.init(allocator, .{});
    defer lb.deinit();

    // Sync nodes from the registry if available
    if (abi.network.defaultRegistry()) |reg| {
        lb.syncFromRegistry(reg) catch {};
    } else |_| {}

    utils.output.printKeyValueFmt("Strategy", "{t}", .{lb.config.strategy});
    utils.output.printKeyValueFmt("Nodes", "{d}", .{lb.nodes.items.len});
    utils.output.printKeyValueFmt("Health Interval", "{d}ms", .{lb.config.health_check_interval_ms});
    utils.output.printKeyValueFmt("Sticky Sessions", "{s}", .{utils.output.boolLabel(lb.config.sticky_sessions)});
    utils.output.printKeyValueFmt("Max Retries", "{d}", .{lb.config.max_retries});

    // Show per-node stats if any nodes present
    if (lb.nodes.items.len > 0) {
        utils.output.println("", .{});
        for (lb.nodes.items) |*node| {
            const health_indicator = if (node.is_healthy)
                @as([]const u8, "healthy")
            else
                @as([]const u8, "unhealthy");
            const health_color = if (node.is_healthy) utils.output.Color.green() else utils.output.Color.red();
            utils.output.println("  {s: <15} w={d: <4} conns={d: <4} ({s}{s}{s})", .{
                node.id,
                node.weight,
                node.current_connections.load(.monotonic),
                health_color,
                health_indicator,
                utils.output.Color.reset(),
            });
        }
    }
}

// ============================================================================
// Cluster health status
// ============================================================================

fn printHealthStatus(allocator: std.mem.Allocator) void {
    if (!abi.network.isEnabled()) {
        utils.output.printWarning("Network feature disabled", .{});
        return;
    }
    utils.output.printHeader("Cluster Health");

    var hc = abi.network.HealthCheck.init(allocator, .{}) catch {
        utils.output.printInfo("Health check not available", .{});
        return;
    };
    defer hc.deinit();

    // Populate from registry if available
    var total_nodes: usize = 0;
    if (abi.network.defaultRegistry()) |reg| {
        const nodes = reg.list();
        total_nodes = nodes.len;
        for (nodes) |node| {
            hc.addNode(node.id) catch continue;
            const is_healthy = node.status != .offline;
            hc.reportHealth(.{
                .node_id = node.id,
                .healthy = is_healthy,
                .response_time_ms = 0,
                .error_message = null,
            }) catch continue;
        }
    } else |_| {}

    utils.output.printKeyValueFmt("Cluster", "{t}", .{hc.cluster_state});
    utils.output.printKeyValueFmt("Failover Policy", "{t}", .{hc.config.failover_policy});

    // Count healthy nodes
    var healthy = hc.getHealthyNodes() catch {
        utils.output.printKeyValueFmt("Healthy", "0/{d} nodes", .{total_nodes});
        utils.output.printKeyValue("Primary", if (hc.primary_node) |p| p else "none");
        return;
    };
    defer healthy.deinit(allocator);

    utils.output.printKeyValueFmt("Healthy", "{d}/{d} nodes", .{ healthy.items.len, total_nodes });
    utils.output.printKeyValue("Primary", if (hc.primary_node) |p| p else "none");
    utils.output.printKeyValueFmt("Check Interval", "{d}ms", .{hc.config.health_check_interval_ms});
    utils.output.printKeyValueFmt("Check Timeout", "{d}ms", .{hc.config.health_check_timeout_ms});
}

test {
    std.testing.refAllDecls(@This());
}
