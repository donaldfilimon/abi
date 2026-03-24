const std = @import("std");

pub fn Recovery(
    comptime Manager: type,
    comptime Operation: type,
    comptime RecoveryPoint: type,
    comptime RecoveryResult: type,
    comptime PitrEvent: type,
) type {
    return struct {
        const helpers = @import("common.zig").Helpers(Operation);
        const retention = @import("retention.zig").Retention(Manager, PitrEvent);

        pub fn getRecoveryPoints(self: *Manager) []const RecoveryPoint {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.recovery_points.items;
        }

        pub fn findNearestRecoveryPoint(self: *Manager, timestamp: i64) ?RecoveryPoint {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (timestamp < 0) return null;

            var nearest: ?RecoveryPoint = null;
            var min_diff: u64 = std.math.maxInt(u64);
            const ts: u64 = @intCast(timestamp);

            for (self.recovery_points.items) |point| {
                if (point.timestamp <= ts) {
                    const diff = ts - point.timestamp;
                    if (diff < min_diff) {
                        min_diff = diff;
                        nearest = point;
                    }
                }
            }

            return nearest;
        }

        pub fn getOperationLogLen(self: *Manager) u64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.operation_log.items.len;
        }

        pub fn recoverToTimestamp(self: *Manager, timestamp: i64) !RecoveryResult {
            self.mutex.lock();
            defer self.mutex.unlock();

            retention.emitEvent(self, .{ .recovery_started = .{ .target_timestamp = timestamp } });

            const total_in_log: u64 = self.operation_log.items.len;

            var count: usize = 0;
            for (self.operation_log.items) |op| {
                if (op.timestamp <= timestamp) {
                    count += 1;
                }
            }

            if (count == 0) {
                retention.emitEvent(self, .{ .recovery_failed = .{
                    .reason = "No operations at or before target timestamp",
                } });
                return error.NoRecoveryPoint;
            }

            const ops = try copyOperations(self, count, timestamp, .timestamp);

            retention.emitEvent(self, .{ .recovery_completed = .{
                .target_timestamp = timestamp,
                .operations_replayed = @intCast(count),
            } });

            return .{
                .operations = ops,
                .operations_replayed = @intCast(count),
                .total_in_log = total_in_log,
                .allocator = self.allocator,
            };
        }

        pub fn recoverToSequence(self: *Manager, sequence: u64) !RecoveryResult {
            self.mutex.lock();
            defer self.mutex.unlock();

            const total_in_log: u64 = self.operation_log.items.len;

            var count: usize = 0;
            for (self.operation_log.items) |op| {
                if (op.sequence_number <= sequence) {
                    count += 1;
                }
            }

            if (count == 0) {
                retention.emitEvent(self, .{ .recovery_failed = .{
                    .reason = "No operations found at or before target sequence",
                } });
                return error.SequenceNotFound;
            }

            var last_ts: i64 = 0;
            for (self.operation_log.items) |op| {
                if (op.sequence_number <= sequence) {
                    last_ts = op.timestamp;
                }
            }
            retention.emitEvent(self, .{ .recovery_started = .{ .target_timestamp = last_ts } });

            const ops = try copyOperations(self, count, sequence, .sequence);

            retention.emitEvent(self, .{ .recovery_completed = .{
                .target_timestamp = last_ts,
                .operations_replayed = @intCast(count),
            } });

            return .{
                .operations = ops,
                .operations_replayed = @intCast(count),
                .total_in_log = total_in_log,
                .allocator = self.allocator,
            };
        }

        const FilterMode = enum { timestamp, sequence };

        fn copyOperations(self: *Manager, count: usize, target: anytype, mode: FilterMode) ![]Operation {
            const ops = try self.allocator.alloc(Operation, count);
            var idx: usize = 0;
            errdefer {
                for (ops[0..idx]) |op| {
                    helpers.freeOperation(self.allocator, op);
                }
                self.allocator.free(ops);
            }

            for (self.operation_log.items) |op| {
                const include = switch (mode) {
                    .timestamp => op.timestamp <= target,
                    .sequence => op.sequence_number <= target,
                };
                if (!include) continue;

                ops[idx] = try helpers.cloneOperation(self.allocator, op);
                idx += 1;
            }

            return ops;
        }
    };
}
