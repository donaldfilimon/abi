//! Point-in-Time Recovery (PITR) Manager
//!
//! Enables recovery to any point in time within the retention window:
//! - Write-ahead log (WAL) style checkpoints
//! - Continuous change capture
//! - Binary checkpoint format
//! - Efficient recovery with minimal data loss
//! - Operation replay for timestamp and sequence-based recovery
//! - Persistent operation log serialization

const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;

const Mutex = sync.Mutex;

pub const PitrConfig = struct {
    retention_hours: u32 = 168,
    checkpoint_interval_sec: u32 = 60,
    max_checkpoint_size: u64 = 10 * 1024 * 1024,
    compression: bool = true,
    storage_path: []const u8 = "pitr/",
    on_event: ?*const fn (PitrEvent) void = null,
};

pub const PitrEvent = union(enum) {
    checkpoint_created: struct { sequence: u64, size_bytes: u64 },
    checkpoint_pruned: struct { sequence: u64 },
    recovery_started: struct { target_timestamp: i64 },
    recovery_completed: struct { target_timestamp: i64, operations_replayed: u64 },
    recovery_failed: struct { reason: []const u8 },
    retention_applied: struct { pruned_count: u32, freed_bytes: u64 },
};

pub const RecoveryPoint = struct {
    sequence: u64,
    timestamp: u64,
    size_bytes: u64,
    operation_count: u64,
    checksum: [32]u8,
};

pub const CheckpointHeader = extern struct {
    magic: u32 = 0x50495452,
    version: u16 = 1,
    flags: u16 = 0,
    sequence: u64,
    timestamp: i64,
    operation_count: u64,
    data_size: u64,
    checksum: [32]u8,
    reserved: [32]u8 = [_]u8{0} ** 32,
};

pub const OperationType = enum(u8) {
    insert = 1,
    update = 2,
    delete = 3,
    truncate = 4,
};

pub const Operation = struct {
    type: OperationType,
    timestamp: i64,
    sequence_number: u64,
    key: []const u8,
    value: ?[]const u8,
    previous_value: ?[]const u8,
};

pub const RecoveryResult = struct {
    operations: []Operation,
    operations_replayed: u64,
    total_in_log: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RecoveryResult) void {
        for (self.operations) |op| {
            self.allocator.free(op.key);
            if (op.value) |v| self.allocator.free(v);
            if (op.previous_value) |v| self.allocator.free(v);
        }
        self.allocator.free(self.operations);
        self.* = undefined;
    }
};

pub const PitrManager = struct {
    allocator: std.mem.Allocator,
    config: PitrConfig,
    current_sequence: u64,
    next_op_sequence: u64,
    last_checkpoint_time: u64,
    pending_operations: std.ArrayListUnmanaged(Operation),
    operation_log: std.ArrayListUnmanaged(Operation),
    recovery_points: std.ArrayListUnmanaged(RecoveryPoint),
    mutex: Mutex,

    const capture_impl = @import("pitr/capture.zig").Capture(
        @This(),
        Operation,
        OperationType,
        RecoveryPoint,
        CheckpointHeader,
        PitrEvent,
    );
    const recovery_impl = @import("pitr/recovery.zig").Recovery(
        @This(),
        Operation,
        RecoveryPoint,
        RecoveryResult,
        PitrEvent,
    );
    const persistence_impl = @import("pitr/persistence.zig").Persistence(
        @This(),
        OperationType,
    );
    const retention_impl = @import("pitr/retention.zig").Retention(@This(), PitrEvent);

    pub fn init(allocator: std.mem.Allocator, config: PitrConfig) PitrManager {
        return .{
            .allocator = allocator,
            .config = config,
            .current_sequence = 0,
            .next_op_sequence = 1,
            .last_checkpoint_time = 0,
            .pending_operations = .empty,
            .operation_log = .empty,
            .recovery_points = .empty,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PitrManager) void {
        const helpers = @import("pitr/common.zig").Helpers(Operation);

        for (self.pending_operations.items) |op| {
            helpers.freeOperation(self.allocator, op);
        }
        self.pending_operations.deinit(self.allocator);

        for (self.operation_log.items) |op| {
            helpers.freeOperation(self.allocator, op);
        }
        self.operation_log.deinit(self.allocator);

        self.recovery_points.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn getCurrentSequence(self: *PitrManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_sequence;
    }

    pub fn captureOperation(
        self: *PitrManager,
        op_type: OperationType,
        key: []const u8,
        value: ?[]const u8,
        previous_value: ?[]const u8,
    ) !void {
        try capture_impl.captureOperation(self, op_type, key, value, previous_value);
    }

    pub fn captureOperationWithTimestamp(
        self: *PitrManager,
        op_type: OperationType,
        key: []const u8,
        value: ?[]const u8,
        previous_value: ?[]const u8,
        timestamp: i64,
    ) !void {
        try capture_impl.captureOperationWithTimestamp(self, op_type, key, value, previous_value, timestamp);
    }

    pub fn createCheckpoint(self: *PitrManager) !u64 {
        return capture_impl.createCheckpoint(self);
    }

    pub fn getRecoveryPoints(self: *PitrManager) []const RecoveryPoint {
        return recovery_impl.getRecoveryPoints(self);
    }

    pub fn findNearestRecoveryPoint(self: *PitrManager, timestamp: i64) ?RecoveryPoint {
        return recovery_impl.findNearestRecoveryPoint(self, timestamp);
    }

    pub fn getOperationLogLen(self: *PitrManager) u64 {
        return recovery_impl.getOperationLogLen(self);
    }

    pub fn recoverToTimestamp(self: *PitrManager, timestamp: i64) !RecoveryResult {
        return recovery_impl.recoverToTimestamp(self, timestamp);
    }

    pub fn recoverToSequence(self: *PitrManager, sequence: u64) !RecoveryResult {
        return recovery_impl.recoverToSequence(self, sequence);
    }

    pub fn saveOperationLog(self: *PitrManager, path: []const u8) !void {
        try persistence_impl.saveOperationLog(self, path);
    }

    pub fn saveCheckpoints(self: *PitrManager, path: []const u8) !void {
        try persistence_impl.saveCheckpoints(self, path);
    }

    pub fn loadCheckpoints(self: *PitrManager, path: []const u8) !void {
        try persistence_impl.loadCheckpoints(self, path);
    }

    pub fn loadOperationLog(self: *PitrManager, path: []const u8) !void {
        try persistence_impl.loadOperationLog(self, path);
    }

    pub fn applyRetention(self: *PitrManager) !void {
        try retention_impl.applyRetention(self);
    }
};

test {
    std.testing.refAllDecls(@This());
}
