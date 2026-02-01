const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Network feature enabled flag: {}\n", .{abi.network.isEnabled()});
    if (!abi.network.isEnabled()) {
        std.debug.print("Network feature is disabled. Enable with -Denable-network=true\n", .{});
        return;
    }

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withNetworkDefaults()
        .build();
    defer framework.deinit();

    // Explicitly initialize the network subsystem before accessing the registry.
    abi.network.init(allocator) catch |err| {
        std.debug.print("Network init failed: {t}\n", .{err});
        return err;
    };
    defer abi.network.deinit();

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
