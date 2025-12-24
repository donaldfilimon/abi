//! Tests for network module
//!
//! Tests only compiled when enable_network is true.

const std = @import("std");
const network = @import("network/mod.zig");

test "NetworkConfig defaults" {
    const config = network.DEFAULT_NETWORK_CONFIG;

    try std.testing.expectEqualStrings("0.0.0.0", config.listen_address);
    try std.testing.expectEqual(@as(u16, 8080), config.listen_port);
    try std.testing.expect(config.discovery_enabled);
    try std.testing.expectEqual(@as(u32, 32), config.max_connections);
}

test "SerializationFormat enum" {
    try std.testing.expectEqual(network.SerializationFormat.binary, network.SerializationFormat.binary);
    try std.testing.expectEqual(network.SerializationFormat.json, network.SerializationFormat.json);
}

test "NetworkEngine initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = network.DEFAULT_NETWORK_CONFIG;
    const engine = try network.NetworkEngine.init(gpa.allocator(), config);
    defer engine.deinit();

    try std.testing.expect(!engine.running.load(.acquire));
}

test "NodeRegistry add and remove" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var registry = try network.NodeRegistry.init(gpa.allocator(), 10);
    defer registry.deinit(gpa.allocator());

    const node = network.NodeInfo{
        .address = "192.168.1.100",
        .port = 8080,
        .cpu_count = 8,
        .total_memory_bytes = 16 * 1024 * 1024 * 1024,
        .current_task_count = 0,
        .last_seen_timestamp_ns = std.time.nanoTimestamp(),
    };

    try registry.addNode(node);
    try std.testing.expectEqual(@as(usize, 1), registry.nodes.count());

    registry.removeNode("192.168.1.100");
    try std.testing.expectEqual(@as(usize, 0), registry.nodes.count());
}

test "NodeInfo struct layout" {
    const node = network.NodeInfo{
        .address = "192.168.1.1",
        .port = 9000,
        .cpu_count = 4,
        .total_memory_bytes = 8 * 1024 * 1024 * 1024,
        .current_task_count = 5,
        .last_seen_timestamp_ns = 123456789,
    };

    try std.testing.expectEqualStrings("192.168.1.1", node.address);
    try std.testing.expectEqual(@as(u16, 9000), node.port);
    try std.testing.expectEqual(@as(u32, 4), node.cpu_count);
}

test "TaskMessage struct" {
    const message = network.TaskMessage{
        .task_id = 12345,
        .payload_type = "matrix_mult",
        .payload_data = "data",
        .hints = .{ .cpu_affinity = null, .estimated_duration_us = null },
    };

    try std.testing.expectEqual(@as(u64, 12345), message.task_id);
    try std.testing.expectEqualStrings("matrix_mult", message.payload_type);
}

test "ResultMessage struct" {
    const result = network.ResultMessage{
        .task_id = 12345,
        .success = true,
        .payload_data = "result_data",
        .error_message = null,
    };

    try std.testing.expectEqual(@as(u64, 12345), result.task_id);
    try std.testing.expect(result.success);
    try std.testing.expect(result.error_message == null);
}
