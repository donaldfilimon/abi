const std = @import("std");
const pitr = @import("../pitr.zig");

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

test "PitrManager initialization" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .retention_hours = 24,
    });
    defer manager.deinit();

    try std.testing.expectEqual(@as(u64, 0), manager.getCurrentSequence());
}

test "PitrManager capture and checkpoint" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try manager.captureOperation(.insert, "key1", "value1", null);
    try manager.captureOperation(.update, "key1", "value2", "value1");
    try manager.captureOperation(.delete, "key2", null, "old_value");

    const seq = try manager.createCheckpoint();
    try std.testing.expect(seq > 0);

    const points = manager.getRecoveryPoints();
    try std.testing.expectEqual(@as(usize, 1), points.len);
    try std.testing.expectEqual(@as(u64, 3), points[0].operation_count);
    try std.testing.expectEqual(@as(u64, 3), manager.getOperationLogLen());
}

test "recoverToTimestamp filters operations correctly" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    var i: i64 = 0;
    while (i < 10) : (i += 1) {
        var buf: [8]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "key{d}", .{i});
        try manager.captureOperationWithTimestamp(.insert, key, "val", null, 100 + i);
    }

    var result = try manager.recoverToTimestamp(104);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 5), result.operations_replayed);
    try std.testing.expectEqual(@as(u64, 10), result.total_in_log);
    try std.testing.expectEqual(@as(usize, 5), result.operations.len);

    for (result.operations, 0..) |op, idx| {
        try std.testing.expectEqual(@as(i64, 100 + @as(i64, @intCast(idx))), op.timestamp);
    }
}

test "recoverToSequence filters operations correctly" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var buf: [8]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "key{d}", .{i});
        try manager.captureOperationWithTimestamp(.insert, key, "val", null, @intCast(1000 + i));
    }

    var result = try manager.recoverToSequence(7);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 7), result.operations_replayed);
    try std.testing.expectEqual(@as(u64, 10), result.total_in_log);
    try std.testing.expectEqual(@as(usize, 7), result.operations.len);

    for (result.operations, 0..) |op, idx| {
        try std.testing.expectEqual(@as(u64, idx + 1), op.sequence_number);
    }
}

test "save and load operation log roundtrip" {
    const test_path = "test_pitr_oplog.bin";

    defer {
        var cleanup_io = initIoBackend(std.testing.allocator);
        defer cleanup_io.deinit();
        std.Io.Dir.cwd().deleteFile(cleanup_io.io(), test_path) catch {};
    }

    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try manager.captureOperationWithTimestamp(.insert, "alpha", "val_a", null, 500);
    try manager.captureOperationWithTimestamp(.update, "beta", "val_b2", "val_b1", 501);
    try manager.captureOperationWithTimestamp(.delete, "gamma", null, "val_g", 502);
    try manager.captureOperationWithTimestamp(.truncate, "delta", null, null, 503);

    try manager.saveOperationLog(test_path);

    var manager2 = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager2.deinit();

    try manager2.loadOperationLog(test_path);
    try std.testing.expectEqual(@as(u64, 4), manager2.getOperationLogLen());

    var result = try manager2.recoverToTimestamp(999);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 4), result.operations_replayed);
    try std.testing.expectEqualStrings("alpha", result.operations[0].key);
    try std.testing.expectEqualStrings("val_a", result.operations[0].value.?);
    try std.testing.expect(result.operations[0].previous_value == null);
    try std.testing.expectEqual(pitr.OperationType.insert, result.operations[0].type);

    try std.testing.expectEqualStrings("beta", result.operations[1].key);
    try std.testing.expectEqualStrings("val_b2", result.operations[1].value.?);
    try std.testing.expectEqualStrings("val_b1", result.operations[1].previous_value.?);

    try std.testing.expectEqualStrings("gamma", result.operations[2].key);
    try std.testing.expect(result.operations[2].value == null);
    try std.testing.expectEqualStrings("val_g", result.operations[2].previous_value.?);

    try std.testing.expectEqualStrings("delta", result.operations[3].key);
    try std.testing.expect(result.operations[3].value == null);
    try std.testing.expect(result.operations[3].previous_value == null);
}

test "empty log recovery returns NoRecoveryPoint" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try std.testing.expectError(error.NoRecoveryPoint, manager.recoverToTimestamp(999));
}

test "empty log recoverToSequence returns SequenceNotFound" {
    var manager = pitr.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try std.testing.expectError(error.SequenceNotFound, manager.recoverToSequence(5));
}

test {
    std.testing.refAllDecls(@This());
}
