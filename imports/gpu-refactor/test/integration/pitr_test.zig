//! Integration Tests: PITR public surface

const std = @import("std");
const abi = @import("abi");

const ha = abi.ha;

test "pitr: public types are available" {
    _ = ha.PitrManager;
    _ = ha.PitrConfig;
    _ = ha.RecoveryPoint;
    _ = ha.pitr.RecoveryResult;
    _ = ha.pitr.Operation;
    _ = ha.pitr.OperationType;
}

test "pitr: capture and checkpoint via public ha surface" {
    var manager = ha.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try manager.captureOperation(.insert, "public-key", "value", null);

    const seq = try manager.createCheckpoint();
    try std.testing.expect(seq > 0);
    try std.testing.expectEqual(@as(usize, 1), manager.getRecoveryPoints().len);
}

test "pitr: recover to sequence via public ha surface" {
    var manager = ha.PitrManager.init(std.testing.allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    try manager.captureOperationWithTimestamp(.insert, "a", "1", null, 10);
    try manager.captureOperationWithTimestamp(.insert, "b", "2", null, 11);
    try manager.captureOperationWithTimestamp(.insert, "c", "3", null, 12);

    var result = try manager.recoverToSequence(2);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.operations_replayed);
    try std.testing.expectEqualStrings("a", result.operations[0].key);
    try std.testing.expectEqualStrings("b", result.operations[1].key);
}

test {
    std.testing.refAllDecls(@This());
}
