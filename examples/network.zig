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

    std.debug.print("Network feature enabled flag: {}\n", .{abi.features.network.isEnabled()});
    if (!abi.features.network.isEnabled()) {
        std.debug.print("Network feature is disabled. Enable with -Denable-network=true\n", .{});
        return;
    }

    var builder = abi.App.builder(allocator);
    _ = builder.withDefault(.network);
    var framework = builder.build() catch |err| {
        std.debug.print("Failed to initialize network framework: {t}\n", .{err});
        return err;
    };
    defer framework.deinit();

    // Explicitly initialize the network subsystem before accessing the registry.
    abi.features.network.init(allocator) catch |err| {
        std.debug.print("Network init failed: {t}\n", .{err});
        return err;
    };
    defer abi.features.network.deinit();

    // The network registry is available after network module initialization.
    const registry = abi.features.network.defaultRegistry() catch |err| {
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
