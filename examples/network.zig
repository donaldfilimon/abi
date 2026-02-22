//! Distributed Network Example
//!
//! Demonstrates the distributed compute network with node registration,
//! cluster management, and Raft consensus coordination.
//!
//! Run with: `zig build run-network`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Network feature enabled flag: {}\n", .{abi.network.isEnabled()});
    if (!abi.network.isEnabled()) {
        std.debug.print("Network feature is disabled. Enable with -Denable-network=true\n", .{});
        return;
    }

    // Initialize module-level network state used by defaultRegistry().
    try abi.network.initWithConfig(allocator, .{
        .cluster_id = "example-cluster",
        .heartbeat_timeout_ms = 30_000,
        .max_nodes = 32,
    });
    defer abi.network.deinit();

    // The network registry is available after network module initialization.
    const registry = abi.network.defaultRegistry() catch |err| {
        std.debug.print("Failed to get default registry: {t}\n", .{err});
        return err;
    };

    // Register nodes
    registry.register("node-1", "localhost:8080") catch |err| {
        std.debug.print("Failed to register node-1: {t}\n", .{err});
        return err;
    };
    registry.register("node-2", "localhost:8081") catch |err| {
        std.debug.print("Failed to register node-2: {t}\n", .{err});
        return err;
    };

    // Update node status
    _ = registry.touch("node-1");
    _ = registry.setStatus("node-2", .degraded);

    const nodes = registry.list();
    std.debug.print("Network registry contains {} nodes:\n", .{nodes.len});
    for (nodes) |node| {
        std.debug.print("  Node '{s}' at {s} - Status: {t}\n", .{ node.id, node.address, node.status });
    }

    _ = registry.touch("node-1");
    _ = registry.setStatus("node-2", .degraded);
}
