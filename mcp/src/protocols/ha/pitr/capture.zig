const std = @import("std");
const time = @import("../../../foundation/mod.zig").time;

pub fn Capture(
    comptime Manager: type,
    comptime Operation: type,
    comptime OperationType: type,
    comptime RecoveryPoint: type,
    comptime CheckpointHeader: type,
    comptime PitrEvent: type,
) type {
    return struct {
        const helpers = @import("common.zig").Helpers(Operation);
        const retention = @import("retention.zig").Retention(Manager, PitrEvent);

        pub fn captureOperation(
            self: *Manager,
            op_type: OperationType,
            key: []const u8,
            value: ?[]const u8,
            previous_value: ?[]const u8,
        ) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const op = try buildOperation(self, op_type, key, value, previous_value, @intCast(time.timestampSec()));
            try self.pending_operations.append(self.allocator, op);
            try self.operation_log.append(self.allocator, try helpers.cloneOperation(self.allocator, op));

            if (shouldCheckpoint(self)) {
                _ = try createCheckpointLocked(self);
            }
        }

        pub fn captureOperationWithTimestamp(
            self: *Manager,
            op_type: OperationType,
            key: []const u8,
            value: ?[]const u8,
            previous_value: ?[]const u8,
            timestamp: i64,
        ) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const op = try buildOperation(self, op_type, key, value, previous_value, timestamp);
            try self.pending_operations.append(self.allocator, op);
            try self.operation_log.append(self.allocator, try helpers.cloneOperation(self.allocator, op));
        }

        pub fn createCheckpoint(self: *Manager) !u64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            return createCheckpointLocked(self);
        }

        pub fn createCheckpointLocked(self: *Manager) !u64 {
            if (self.pending_operations.items.len == 0) {
                return self.current_sequence;
            }

            self.current_sequence += 1;
            const sequence = self.current_sequence;

            var total_size: u64 = @sizeOf(CheckpointHeader);
            for (self.pending_operations.items) |op| {
                total_size += @sizeOf(OperationType) + 8;
                total_size += 4 + op.key.len;
                total_size += 4 + (if (op.value) |v| v.len else 0);
                total_size += 4 + (if (op.previous_value) |v| v.len else 0);
            }

            var checksum: [32]u8 = undefined;
            @memset(&checksum, 0);

            const recovery_point = RecoveryPoint{
                .sequence = sequence,
                .timestamp = @as(u64, time.timestampSec()),
                .size_bytes = total_size,
                .operation_count = self.pending_operations.items.len,
                .checksum = checksum,
            };

            try self.recovery_points.append(self.allocator, recovery_point);

            for (self.pending_operations.items) |op| {
                helpers.freeOperation(self.allocator, op);
            }
            self.pending_operations.clearRetainingCapacity();

            self.last_checkpoint_time = time.timestampSec();

            retention.emitEvent(self, .{ .checkpoint_created = .{
                .sequence = sequence,
                .size_bytes = total_size,
            } });

            return sequence;
        }

        pub fn shouldCheckpoint(self: *Manager) bool {
            const now = time.timestampSec();
            const interval = @as(u64, self.config.checkpoint_interval_sec);

            if (now - self.last_checkpoint_time >= interval) {
                return true;
            }

            var total_size: u64 = 0;
            for (self.pending_operations.items) |op| {
                total_size += op.key.len;
                if (op.value) |v| total_size += v.len;
                if (op.previous_value) |v| total_size += v.len;
            }

            return total_size >= self.config.max_checkpoint_size;
        }

        fn buildOperation(
            self: *Manager,
            op_type: OperationType,
            key: []const u8,
            value: ?[]const u8,
            previous_value: ?[]const u8,
            timestamp: i64,
        ) !Operation {
            const seq = self.next_op_sequence;
            self.next_op_sequence += 1;

            const key_copy = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(key_copy);

            var value_copy: ?[]const u8 = null;
            if (value) |v| {
                value_copy = try self.allocator.dupe(u8, v);
            }
            errdefer if (value_copy) |v| self.allocator.free(v);

            var prev_copy: ?[]const u8 = null;
            if (previous_value) |v| {
                prev_copy = try self.allocator.dupe(u8, v);
            }
            errdefer if (prev_copy) |v| self.allocator.free(v);

            return .{
                .type = op_type,
                .timestamp = timestamp,
                .sequence_number = seq,
                .key = key_copy,
                .value = value_copy,
                .previous_value = prev_copy,
            };
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
