//! Integration tests for Phase 5 features
//!
//! Tests GPU execution, network serialization, and profiling integration.

const std = @import("std");
const abi = @import("abi");

const build_options = @import("build_options");

test "integration: GPU workload hints in WorkItem" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const hints = abi.abi.compute.WorkloadHints{
        .cpu_affinity = null,
        .estimated_duration_us = null,
        .prefers_gpu = true,
        .requires_gpu = false,
    };

    try std.testing.expect(hints.prefers_gpu);
    try std.testing.expect(!hints.requires_gpu);
}

test "integration: GPU buffer allocation and data transfer" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const flags = abi.compute.BufferFlags{
        .read_only = false,
        .write_only = false,
        .host_visible = true,
        .device_local = false,
    };

    var buffer = try abi.compute.GPUBuffer.init(gpa.allocator(), 1024, flags);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 1024), buffer.size);

    const data = "Hello, GPU!";
    try buffer.writeFromHost(data);

    const read_data = try buffer.readToHost(0, data.len);
    try std.testing.expectEqualStrings(data, read_data);
}

test "integration: GPU memory pool management" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var pool = abi.compute.GPUMemoryPool.init(gpa.allocator(), 10 * 1024);
    defer pool.deinit();

    const buffer1 = try pool.allocate(4096, .{});
    try std.testing.expectEqual(@as(usize, 4096), buffer1.size);

    const buffer2 = try pool.allocate(2048, .{});
    try std.testing.expectEqual(@as(usize, 2048), buffer2.size);

    const usage = pool.getUsage();
    try std.testing.expect(usage > 0.0 and usage < 1.0);

    pool.free(buffer1);
    const usage_after = pool.getUsage();
    try std.testing.expect(usage_after < usage);
}

test "integration: Network task serialization roundtrip" {
    if (!build_options.enable_network) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const hints = abi.abi.compute.WorkloadHints{
        .cpu_affinity = null,
        .estimated_duration_us = null,
        .prefers_gpu = true,
        .requires_gpu = false,
    };

    const item = abi.compute.WorkItem{
        .id = 12345,
        .user = undefined,
        .vtable = undefined,
        .priority = 0.5,
        .hints = hints,
        .gpu_vtable = null,
    };

    const payload_type = "matrix_multiply";
    const user_data = "test_data";

    const serialized = try abi.compute.network.serializeTask(gpa.allocator(), &item, payload_type, user_data);
    defer gpa.allocator().free(serialized);

    try std.testing.expect(serialized.len > 0);

    const deserialized = try abi.compute.network.deserializeTask(gpa.allocator(), serialized);
    defer {
        gpa.allocator().free(deserialized.payload_type);
        gpa.allocator().free(deserialized.user_data);
    }

    try std.testing.expectEqual(item.id, deserialized.item.id);
    try std.testing.expectEqualStrings(payload_type, deserialized.payload_type);
    try std.testing.expectEqualStrings(user_data, deserialized.user_data);
    try std.testing.expectEqual(@as(?u32, 2), deserialized.item.hints.cpu_affinity);
}

test "integration: Network result serialization roundtrip" {
    if (!build_options.enable_network) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const task_id: u64 = 99999;
    const payload_data = "result_data_here";
    const error_msg: ?[]const u8 = null;

    const serialized = try abi.compute.network.serializeResult(gpa.allocator(), task_id, undefined, true, error_msg, payload_data);
    defer gpa.allocator().free(serialized);

    try std.testing.expect(serialized.len > 0);

    const deserialized = try abi.compute.network.deserializeResult(gpa.allocator(), serialized);
    defer {
        if (deserialized.error_message) |msg| gpa.allocator().free(msg);
        gpa.allocator().free(deserialized.payload_data);
    }

    try std.testing.expectEqual(task_id, deserialized.task_id);
    try std.testing.expect(deserialized.success);
    try std.testing.expectEqualStrings(payload_data, deserialized.payload_data);
}

test "integration: Network result with error serialization" {
    if (!build_options.enable_network) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const task_id: u64 = 88888;
    const payload_data = "";
    const error_msg = "GPU out of memory";

    const serialized = try abi.compute.network.serializeResult(gpa.allocator(), task_id, undefined, false, error_msg, payload_data);
    defer gpa.allocator().free(serialized);

    const deserialized = try abi.compute.network.deserializeResult(gpa.allocator(), serialized);
    defer {
        if (deserialized.error_message) |msg| gpa.allocator().free(msg);
        gpa.allocator().free(deserialized.payload_data);
    }

    try std.testing.expectEqual(task_id, deserialized.task_id);
    try std.testing.expect(!deserialized.success);
    try std.testing.expectEqualStrings(error_msg, deserialized.error_message.?);
}

test "integration: Profiling metrics collection" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = abi.compute.profiling.DEFAULT_METRICS_CONFIG;
    const collector = try abi.compute.profiling.MetricsCollector.init(gpa.allocator(), config, 4);
    defer collector.deinit();

    collector.recordTaskExecution(0, 1000);
    collector.recordTaskExecution(0, 2000);
    collector.recordTaskExecution(1, 1500);

    const stats0 = collector.getWorkerStats(0).?;
    try std.testing.expectEqual(@as(u64, 2), stats0.tasks_executed);
    try std.testing.expectEqual(@as(u64, 3000), stats0.total_execution_ns);

    const stats1 = collector.getWorkerStats(1).?;
    try std.testing.expectEqual(@as(u64, 1), stats1.tasks_executed);
}

test "integration: Profiling histogram statistics" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var histogram = try abi.compute.profiling.Histogram.init(gpa.allocator(), &[_]u64{ 100, 500, 1000, 5000 });
    defer histogram.deinit(gpa.allocator());

    histogram.record(50);
    histogram.record(200);
    histogram.record(750);
    histogram.record(3000);
    histogram.record(10000);

    try std.testing.expectEqual(@as(u64, 5), histogram.total);
}

test "integration: Profiling summary calculation" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = abi.compute.profiling.DEFAULT_METRICS_CONFIG;
    const collector = try abi.compute.profiling.MetricsCollector.init(gpa.allocator(), config, 2);
    defer collector.deinit();

    collector.recordTaskExecution(0, 1000);
    collector.recordTaskExecution(1, 2000);
    collector.recordTaskExecution(0, 3000);

    const summary = collector.getSummary();

    try std.testing.expectEqual(@as(u64, 3), summary.total_tasks);
    try std.testing.expectEqual(@as(u64, 6000), summary.total_execution_ns);
    try std.testing.expectEqual(@as(u64, 2000), summary.avg_execution_ns);
    try std.testing.expectEqual(@as(u64, 1000), summary.min_execution_ns);
    try std.testing.expectEqual(@as(u64, 3000), summary.max_execution_ns);
}

test "integration: Node registry management" {
    if (!build_options.enable_network) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var registry = try abi.compute.network.NodeRegistry.init(gpa.allocator(), 5);
    defer registry.deinit(gpa.allocator());

    const node1 = abi.compute.network.NodeInfo{
        .address = "192.168.1.10",
        .port = 8080,
        .cpu_count = 8,
        .total_memory_bytes = 16 * 1024 * 1024 * 1024,
        .current_task_count = 2,
        .last_seen_timestamp_ns = std.time.nanoTimestamp(),
    };

    const node2 = abi.compute.network.NodeInfo{
        .address = "192.168.1.11",
        .port = 8080,
        .cpu_count = 4,
        .total_memory_bytes = 8 * 1024 * 1024 * 1024,
        .current_task_count = 1,
        .last_seen_timestamp_ns = std.time.nanoTimestamp(),
    };

    try registry.addNode(node1);
    try registry.addNode(node2);
    try std.testing.expectEqual(@as(usize, 2), registry.nodes.count());

    const best_node = registry.getBestNode(1000);
    try std.testing.expect(best_node != null);

    registry.removeNode("192.168.1.10");
    try std.testing.expectEqual(@as(usize, 1), registry.nodes.count());
}
