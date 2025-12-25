const std = @import("std");
const abi = @import("abi");

test "framework starts with default options" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.createDefaultFramework(gpa.allocator());
    defer framework.deinit();

    try std.testing.expect(framework.isRunning());
    try std.testing.expect(framework.isFeatureEnabled(.ai));
}

test "database insert and search" {
    var db = try abi.database.database.Database.init(std.testing.allocator, "test");
    defer db.deinit();

    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "first");
    try db.insert(2, &.{ 0.2, 0.2, 0.2 }, "second");

    const query = &.{ 0.1, 0.2, 0.3 };
    const results = try db.search(std.testing.allocator, query, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "compute engine returns results" {
    var engine = try abi.compute.runtime.engine.DistributedComputeEngine.init(
        std.testing.allocator,
        .{ .max_tasks = 4 },
    );
    defer engine.deinit();

    const task_id = try engine.submit_task(u64, sampleComputeTask);
    const result = try engine.wait_for_result(u64, task_id, 0);
    try std.testing.expectEqual(@as(u64, 99), result);
}

fn sampleComputeTask(_: std.mem.Allocator) !u64 {
    return 99;
}
