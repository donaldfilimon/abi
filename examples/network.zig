const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (!abi.network.isEnabled()) {
        std.debug.print("Network feature is disabled. Enable with -Denable-network=true\n", .{});
        return;
    }

    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_network = true,
        .enable_gpu = false,
    }) catch |err| {
        std.debug.print("Failed to initialize network framework: {}\n", .{err});
        return err;
    };
    defer abi.shutdown(&framework);

    const registry = abi.network.defaultRegistry() catch |err| {
        std.debug.print("Failed to get default registry: {}\n", .{err});
        return err;
    };

    // Register nodes
    registry.register("node-1", "localhost:8080") catch |err| {
        std.debug.print("Failed to register node-1: {}\n", .{err});
        return err;
    };
    registry.register("node-2", "localhost:8081") catch |err| {
        std.debug.print("Failed to register node-2: {}\n", .{err});
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
