const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_network = true,
    });
    defer abi.shutdown(&framework);

    const registry = try abi.network.defaultRegistry();
    try registry.register("node-1", "localhost:8080");
    try registry.register("node-2", "localhost:8081");

    const nodes = try registry.listNodes(allocator);
    defer allocator.free(nodes);

    std.debug.print("Registered nodes:\n", .{});
    for (nodes) |node| {
        std.debug.print("  {s}: {s} ({t})\n", .{ node.id, node.address, node.status });
    }

    _ = registry.touch("node-1");
    _ = registry.setStatus("node-2", .degraded);
}
