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
const time = @import("../../foundation/mod.zig").time;

const sync = @import("../../foundation/mod.zig").sync;
const Mutex = sync.Mutex;

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Atomic file write: write to tmp, fsync, rename. Old file intact on crash.
fn atomicWriteFile(allocator: std.mem.Allocator, path: []const u8, data: []const u8) !void {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{path});
    defer allocator.free(tmp_path);

    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    {
        var file = try std.Io.Dir.cwd().createFile(io, tmp_path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, data);
        file.sync(io) catch {};
    }

    const cwd = std.Io.Dir.cwd();
    cwd.rename(tmp_path, cwd, path, io) catch {
        cwd.deleteFile(io, tmp_path) catch {};
        return error.PersistFailed;
    };
}

/// PITR configuration
pub const PitrConfig = struct {
    /// Retention period in hours
    retention_hours: u32 = 168, // 7 days
    /// Checkpoint interval in seconds
    checkpoint_interval_sec: u32 = 60,
    /// Maximum checkpoint size in bytes
    max_checkpoint_size: u64 = 10 * 1024 * 1024, // 10 MB
    /// Enable compression
    compression: bool = true,
    /// Storage path for checkpoints
    storage_path: []const u8 = "pitr/",
    /// Callback for PITR events
    on_event: ?*const fn (PitrEvent) void = null,
};

/// PITR events
pub const PitrEvent = union(enum) {
    checkpoint_created: struct { sequence: u64, size_bytes: u64 },
    checkpoint_pruned: struct { sequence: u64 },
    recovery_started: struct { target_timestamp: i64 },
    recovery_completed: struct { target_timestamp: i64, operations_replayed: u64 },
    recovery_failed: struct { reason: []const u8 },
    retention_applied: struct { pruned_count: u32, freed_bytes: u64 },
};

/// Recovery point information
pub const RecoveryPoint = struct {
    sequence: u64,
    timestamp: u64,
    size_bytes: u64,
    operation_count: u64,
    checksum: [32]u8,
};

/// Checkpoint header (binary format)
pub const CheckpointHeader = extern struct {
    magic: u32 = 0x50495452, // "PITR"
    version: u16 = 1,
    flags: u16 = 0,
    sequence: u64,
    timestamp: i64,
    operation_count: u64,
    data_size: u64,
    checksum: [32]u8,
    reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Operation types for change capture
pub const OperationType = enum(u8) {
    insert = 1,
    update = 2,
    delete = 3,
    truncate = 4,
};

/// Captured operation
pub const Operation = struct {
    type: OperationType,
    timestamp: i64,
    sequence_number: u64,
    key: []const u8,
    value: ?[]const u8,
    previous_value: ?[]const u8,
};

/// Result of a recovery operation, containing the filtered operations to replay.
pub const RecoveryResult = struct {
    operations: []Operation,
    operations_replayed: u64,
    total_in_log: u64,
    allocator: std.mem.Allocator,

    /// Free all operation data owned by this result.
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

/// PITR manager
pub const PitrManager = struct {
    allocator: std.mem.Allocator,
    config: PitrConfig,

    // State
    current_sequence: u64,
    next_op_sequence: u64,
    last_checkpoint_time: u64,
    pending_operations: std.ArrayListUnmanaged(Operation),

    // Full operation log (retained for recovery)
    operation_log: std.ArrayListUnmanaged(Operation),

    // Recovery points
    recovery_points: std.ArrayListUnmanaged(RecoveryPoint),

    // Synchronization
    mutex: Mutex,

    /// Initialize the PITR manager
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

    /// Deinitialize the PITR manager
    pub fn deinit(self: *PitrManager) void {
        // Free pending operation data
        for (self.pending_operations.items) |op| {
            self.allocator.free(op.key);
            if (op.value) |v| self.allocator.free(v);
            if (op.previous_value) |v| self.allocator.free(v);
        }
        self.pending_operations.deinit(self.allocator);

        // Free operation log data
        for (self.operation_log.items) |op| {
            self.allocator.free(op.key);
            if (op.value) |v| self.allocator.free(v);
            if (op.previous_value) |v| self.allocator.free(v);
        }
        self.operation_log.deinit(self.allocator);

        self.recovery_points.deinit(self.allocator);
        self.* = undefined;
    }

    /// Get current sequence number
    pub fn getCurrentSequence(self: *PitrManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_sequence;
    }

    /// Capture an operation
    pub fn captureOperation(
        self: *PitrManager,
        op_type: OperationType,
        key: []const u8,
        value: ?[]const u8,
        previous_value: ?[]const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const seq = self.next_op_sequence;
        self.next_op_sequence += 1;

        // Duplicate data for pending operations
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

        const op = Operation{
            .type = op_type,
            .timestamp = @intCast(time.timestampSec()),
            .sequence_number = seq,
            .key = key_copy,
            .value = value_copy,
            .previous_value = prev_copy,
        };

        try self.pending_operations.append(self.allocator, op);

        // Also duplicate into the operation log for recovery
        const log_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(log_key);

        var log_value: ?[]const u8 = null;
        if (value) |v| {
            log_value = try self.allocator.dupe(u8, v);
        }
        errdefer if (log_value) |v| self.allocator.free(v);

        var log_prev: ?[]const u8 = null;
        if (previous_value) |v| {
            log_prev = try self.allocator.dupe(u8, v);
        }
        errdefer if (log_prev) |v| self.allocator.free(v);

        const log_op = Operation{
            .type = op_type,
            .timestamp = op.timestamp,
            .sequence_number = seq,
            .key = log_key,
            .value = log_value,
            .previous_value = log_prev,
        };

        try self.operation_log.append(self.allocator, log_op);

        // Check if checkpoint is needed
        if (self.shouldCheckpoint()) {
            _ = try self.createCheckpointLocked();
        }
    }

    /// Capture an operation with an explicit timestamp (for testing and log loading).
    pub fn captureOperationWithTimestamp(
        self: *PitrManager,
        op_type: OperationType,
        key: []const u8,
        value: ?[]const u8,
        previous_value: ?[]const u8,
        timestamp: i64,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const seq = self.next_op_sequence;
        self.next_op_sequence += 1;

        // Duplicate data for pending operations
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

        const op = Operation{
            .type = op_type,
            .timestamp = timestamp,
            .sequence_number = seq,
            .key = key_copy,
            .value = value_copy,
            .previous_value = prev_copy,
        };

        try self.pending_operations.append(self.allocator, op);

        // Also duplicate into the operation log
        const log_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(log_key);

        var log_value: ?[]const u8 = null;
        if (value) |v| {
            log_value = try self.allocator.dupe(u8, v);
        }
        errdefer if (log_value) |v| self.allocator.free(v);

        var log_prev: ?[]const u8 = null;
        if (previous_value) |v| {
            log_prev = try self.allocator.dupe(u8, v);
        }
        errdefer if (log_prev) |v| self.allocator.free(v);

        const log_op = Operation{
            .type = op_type,
            .timestamp = timestamp,
            .sequence_number = seq,
            .key = log_key,
            .value = log_value,
            .previous_value = log_prev,
        };

        try self.operation_log.append(self.allocator, log_op);
    }

    fn shouldCheckpoint(self: *PitrManager) bool {
        const now = time.timestampSec();
        const interval = @as(u64, self.config.checkpoint_interval_sec);

        // Time-based trigger
        if (now - self.last_checkpoint_time >= interval) {
            return true;
        }

        // Size-based trigger
        var total_size: u64 = 0;
        for (self.pending_operations.items) |op| {
            total_size += op.key.len;
            if (op.value) |v| total_size += v.len;
            if (op.previous_value) |v| total_size += v.len;
        }

        return total_size >= self.config.max_checkpoint_size;
    }

    /// Create a checkpoint
    pub fn createCheckpoint(self: *PitrManager) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.createCheckpointLocked();
    }

    fn createCheckpointLocked(self: *PitrManager) !u64 {
        if (self.pending_operations.items.len == 0) {
            return self.current_sequence;
        }

        self.current_sequence += 1;
        const sequence = self.current_sequence;

        // Calculate checkpoint size and checksum
        var total_size: u64 = @sizeOf(CheckpointHeader);
        for (self.pending_operations.items) |op| {
            total_size += @sizeOf(OperationType) + 8; // type + timestamp
            total_size += 4 + op.key.len; // length prefix + key
            total_size += 4 + (if (op.value) |v| v.len else 0);
            total_size += 4 + (if (op.previous_value) |v| v.len else 0);
        }

        var checksum: [32]u8 = undefined;
        @memset(&checksum, 0);
        // In real implementation, compute SHA-256 of checkpoint data

        const recovery_point = RecoveryPoint{
            .sequence = sequence,
            .timestamp = @as(u64, time.timestampSec()),
            .size_bytes = total_size,
            .operation_count = self.pending_operations.items.len,
            .checksum = checksum,
        };

        try self.recovery_points.append(self.allocator, recovery_point);

        // Free pending operations (the log retains its own copies)
        for (self.pending_operations.items) |op| {
            self.allocator.free(op.key);
            if (op.value) |v| self.allocator.free(v);
            if (op.previous_value) |v| self.allocator.free(v);
        }
        self.pending_operations.clearRetainingCapacity();

        self.last_checkpoint_time = time.timestampSec();

        self.emitEvent(.{ .checkpoint_created = .{
            .sequence = sequence,
            .size_bytes = total_size,
        } });

        return sequence;
    }

    /// Get available recovery points
    pub fn getRecoveryPoints(self: *PitrManager) []const RecoveryPoint {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.recovery_points.items;
    }

    /// Find nearest recovery point to timestamp
    /// Find the nearest recovery point **not after** the given timestamp.
    /// Returns `null` when the timestamp is negative or no suitable point exists.
    pub fn findNearestRecoveryPoint(self: *PitrManager, timestamp: i64) ?RecoveryPoint {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Guard against negative timestamps which cannot be represented as unsigned.
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

    /// Get the total number of operations in the log.
    pub fn getOperationLogLen(self: *PitrManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.operation_log.items.len;
    }

    /// Recover to a specific timestamp. Returns a RecoveryResult containing
    /// copies of all operations with timestamp <= target_timestamp, in order.
    /// The caller owns the returned result and must call `deinit()` on it.
    pub fn recoverToTimestamp(self: *PitrManager, timestamp: i64) !RecoveryResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.emitEvent(.{ .recovery_started = .{ .target_timestamp = timestamp } });

        const total_in_log: u64 = self.operation_log.items.len;

        // Count matching operations first
        var count: usize = 0;
        for (self.operation_log.items) |op| {
            if (op.timestamp <= timestamp) {
                count += 1;
            }
        }

        if (count == 0) {
            self.emitEvent(.{ .recovery_failed = .{
                .reason = "No operations at or before target timestamp",
            } });
            return error.NoRecoveryPoint;
        }

        // Allocate result array
        const ops = try self.allocator.alloc(Operation, count);
        var idx: usize = 0;
        errdefer {
            // Free all already-copied entries on error
            for (ops[0..idx]) |op| {
                self.allocator.free(op.key);
                if (op.value) |v| self.allocator.free(v);
                if (op.previous_value) |v| self.allocator.free(v);
            }
            self.allocator.free(ops);
        }

        // Copy matching operations (they are already in order)
        for (self.operation_log.items) |op| {
            if (op.timestamp <= timestamp) {
                const key_copy = try self.allocator.dupe(u8, op.key);
                errdefer self.allocator.free(key_copy);

                var val_copy: ?[]const u8 = null;
                if (op.value) |v| {
                    val_copy = try self.allocator.dupe(u8, v);
                }
                errdefer if (val_copy) |v| self.allocator.free(v);

                var prev_copy: ?[]const u8 = null;
                if (op.previous_value) |v| {
                    prev_copy = try self.allocator.dupe(u8, v);
                }

                ops[idx] = .{
                    .type = op.type,
                    .timestamp = op.timestamp,
                    .sequence_number = op.sequence_number,
                    .key = key_copy,
                    .value = val_copy,
                    .previous_value = prev_copy,
                };
                idx += 1;
            }
        }

        self.emitEvent(.{ .recovery_completed = .{
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

    /// Recover to a specific sequence number. Returns a RecoveryResult containing
    /// copies of all operations with sequence_number <= target_sequence, in order.
    /// The caller owns the returned result and must call `deinit()` on it.
    pub fn recoverToSequence(self: *PitrManager, sequence: u64) !RecoveryResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const total_in_log: u64 = self.operation_log.items.len;

        // Count matching operations
        var count: usize = 0;
        for (self.operation_log.items) |op| {
            if (op.sequence_number <= sequence) {
                count += 1;
            }
        }

        if (count == 0) {
            self.emitEvent(.{ .recovery_failed = .{
                .reason = "No operations found at or before target sequence",
            } });
            return error.SequenceNotFound;
        }

        // Emit start event using timestamp of last matching op, or 0
        var last_ts: i64 = 0;
        for (self.operation_log.items) |op| {
            if (op.sequence_number <= sequence) {
                last_ts = op.timestamp;
            }
        }
        self.emitEvent(.{ .recovery_started = .{ .target_timestamp = last_ts } });

        // Allocate result array
        const ops = try self.allocator.alloc(Operation, count);
        var idx: usize = 0;
        errdefer {
            for (ops[0..idx]) |op| {
                self.allocator.free(op.key);
                if (op.value) |v| self.allocator.free(v);
                if (op.previous_value) |v| self.allocator.free(v);
            }
            self.allocator.free(ops);
        }

        for (self.operation_log.items) |op| {
            if (op.sequence_number <= sequence) {
                const key_copy = try self.allocator.dupe(u8, op.key);
                errdefer self.allocator.free(key_copy);

                var val_copy: ?[]const u8 = null;
                if (op.value) |v| {
                    val_copy = try self.allocator.dupe(u8, v);
                }
                errdefer if (val_copy) |v| self.allocator.free(v);

                var prev_copy: ?[]const u8 = null;
                if (op.previous_value) |v| {
                    prev_copy = try self.allocator.dupe(u8, v);
                }

                ops[idx] = .{
                    .type = op.type,
                    .timestamp = op.timestamp,
                    .sequence_number = op.sequence_number,
                    .key = key_copy,
                    .value = val_copy,
                    .previous_value = prev_copy,
                };
                idx += 1;
            }
        }

        self.emitEvent(.{ .recovery_completed = .{
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

    /// Serialize the operation log to a file in binary format.
    /// Format: count (u64) + N * (timestamp: i64, sequence: u64, type: u8, key_len: u32, key, val_len: u32, val, prev_len: u32, prev)
    pub fn saveOperationLog(self: *PitrManager, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        const count: u64 = self.operation_log.items.len;
        try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u64, count)));

        for (self.operation_log.items) |op| {
            try buf.appendSlice(self.allocator, &std.mem.toBytes(op.timestamp));
            try buf.appendSlice(self.allocator, &std.mem.toBytes(op.sequence_number));
            try buf.append(self.allocator, @intFromEnum(op.type));

            const key_len: u32 = @intCast(op.key.len);
            try buf.appendSlice(self.allocator, &std.mem.toBytes(key_len));
            try buf.appendSlice(self.allocator, op.key);

            // u32 max sentinel encodes null
            if (op.value) |v| {
                const val_len: u32 = @intCast(v.len);
                try buf.appendSlice(self.allocator, &std.mem.toBytes(val_len));
                try buf.appendSlice(self.allocator, v);
            } else {
                try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, std.math.maxInt(u32))));
            }

            if (op.previous_value) |v| {
                const prev_len: u32 = @intCast(v.len);
                try buf.appendSlice(self.allocator, &std.mem.toBytes(prev_len));
                try buf.appendSlice(self.allocator, v);
            } else {
                try buf.appendSlice(self.allocator, &std.mem.toBytes(@as(u32, std.math.maxInt(u32))));
            }
        }

        try atomicWriteFile(self.allocator, path, buf.items);
    }

    /// Persist recovery point checkpoints to disk.
    /// Format: count (u64) + N * (sequence: u64, timestamp: u64, size_bytes: u64, op_count: u64, checksum: [32]u8)
    pub fn saveCheckpoints(self: *PitrManager, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        const count: u64 = self.recovery_points.items.len;
        try buf.appendSlice(self.allocator, &std.mem.toBytes(count));

        for (self.recovery_points.items) |rp| {
            try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.sequence));
            try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.timestamp));
            try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.size_bytes));
            try buf.appendSlice(self.allocator, &std.mem.toBytes(rp.operation_count));
            try buf.appendSlice(self.allocator, &rp.checksum);
        }

        try atomicWriteFile(self.allocator, path, buf.items);
    }

    /// Load recovery point checkpoints from disk.
    pub fn loadCheckpoints(self: *PitrManager, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const max_size: usize = 16 * 1024 * 1024; // 16MB limit
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(max_size));
        defer self.allocator.free(data);

        self.recovery_points.clearRetainingCapacity();

        var pos: usize = 0;
        if (data.len < 8) return error.UnexpectedEndOfFile;
        const count = std.mem.readInt(u64, data[pos..][0..8], .little);
        pos += 8;

        const entry_size: usize = 8 + 8 + 8 + 8 + 32; // sequence + timestamp + size_bytes + op_count + checksum
        var i: u64 = 0;
        while (i < count) : (i += 1) {
            if (pos + entry_size > data.len) return error.UnexpectedEndOfFile;

            const sequence = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            const timestamp = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            const size_bytes = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            const op_count = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            var checksum: [32]u8 = undefined;
            @memcpy(&checksum, data[pos..][0..32]);
            pos += 32;

            try self.recovery_points.append(self.allocator, .{
                .sequence = sequence,
                .timestamp = timestamp,
                .size_bytes = size_bytes,
                .operation_count = op_count,
                .checksum = checksum,
            });
        }
    }

    /// Load an operation log from a binary file, replacing the current log.
    pub fn loadOperationLog(self: *PitrManager, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const max_size = 256 * 1024 * 1024; // 256 MB
        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(max_size));
        defer self.allocator.free(data);

        // Free existing log
        for (self.operation_log.items) |op| {
            self.allocator.free(op.key);
            if (op.value) |v| self.allocator.free(v);
            if (op.previous_value) |v| self.allocator.free(v);
        }
        self.operation_log.clearRetainingCapacity();

        var pos: usize = 0;

        if (data.len < 8) return error.UnexpectedEndOfFile;
        const count = std.mem.readInt(u64, data[pos..][0..8], .little);
        pos += 8;

        var max_seq: u64 = 0;

        var i: u64 = 0;
        while (i < count) : (i += 1) {
            if (pos + 17 > data.len) return error.UnexpectedEndOfFile; // i64 + u64 + u8
            const timestamp = std.mem.readInt(i64, data[pos..][0..8], .little);
            pos += 8;
            const seq = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            const op_type: OperationType = @enumFromInt(data[pos]);
            pos += 1;

            if (seq > max_seq) max_seq = seq;

            // Key
            if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
            const key_len = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            if (pos + key_len > data.len) return error.UnexpectedEndOfFile;
            const key = try self.allocator.dupe(u8, data[pos .. pos + key_len]);
            errdefer self.allocator.free(key);
            pos += key_len;

            // Value
            if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
            const val_sentinel = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            var val: ?[]const u8 = null;
            if (val_sentinel != std.math.maxInt(u32)) {
                if (pos + val_sentinel > data.len) return error.UnexpectedEndOfFile;
                val = try self.allocator.dupe(u8, data[pos .. pos + val_sentinel]);
                pos += val_sentinel;
            }
            errdefer if (val) |v| self.allocator.free(v);

            // Previous value
            if (pos + 4 > data.len) return error.UnexpectedEndOfFile;
            const prev_sentinel = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;
            var prev: ?[]const u8 = null;
            if (prev_sentinel != std.math.maxInt(u32)) {
                if (pos + prev_sentinel > data.len) return error.UnexpectedEndOfFile;
                prev = try self.allocator.dupe(u8, data[pos .. pos + prev_sentinel]);
                pos += prev_sentinel;
            }
            errdefer if (prev) |v| self.allocator.free(v);

            try self.operation_log.append(self.allocator, .{
                .type = op_type,
                .timestamp = timestamp,
                .sequence_number = seq,
                .key = key,
                .value = val,
                .previous_value = prev,
            });
        }

        // Update next_op_sequence to be after the highest loaded sequence
        if (max_seq >= self.next_op_sequence) {
            self.next_op_sequence = max_seq + 1;
        }
    }

    /// Apply retention policy
    pub fn applyRetention(self: *PitrManager) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.timestampSec();
        const retention_sec = @as(u64, self.config.retention_hours) * 3600;
        const cutoff = now - retention_sec;

        var pruned_count: u32 = 0;
        var freed_bytes: u64 = 0;

        var i: usize = 0;
        while (i < self.recovery_points.items.len) {
            const point = self.recovery_points.items[i];
            if (point.timestamp < cutoff) {
                freed_bytes += point.size_bytes;
                pruned_count += 1;

                self.emitEvent(.{ .checkpoint_pruned = .{ .sequence = point.sequence } });
                _ = self.recovery_points.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        if (pruned_count > 0) {
            self.emitEvent(.{ .retention_applied = .{
                .pruned_count = pruned_count,
                .freed_bytes = freed_bytes,
            } });
        }
    }

    fn emitEvent(self: *PitrManager, event: PitrEvent) void {
        if (self.config.on_event) |callback| {
            callback(event);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "PitrManager initialization" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .retention_hours = 24,
    });
    defer manager.deinit();

    try std.testing.expectEqual(@as(u64, 0), manager.getCurrentSequence());
}

test "PitrManager capture and checkpoint" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600, // Disable auto-checkpoint
    });
    defer manager.deinit();

    // Capture some operations
    try manager.captureOperation(.insert, "key1", "value1", null);
    try manager.captureOperation(.update, "key1", "value2", "value1");
    try manager.captureOperation(.delete, "key2", null, "old_value");

    // Create checkpoint
    const seq = try manager.createCheckpoint();
    try std.testing.expect(seq > 0);

    // Verify recovery point was created
    const points = manager.getRecoveryPoints();
    try std.testing.expectEqual(@as(usize, 1), points.len);
    try std.testing.expectEqual(@as(u64, 3), points[0].operation_count);

    // Verify operation log still has the operations
    try std.testing.expectEqual(@as(u64, 3), manager.getOperationLogLen());
}

test "recoverToTimestamp filters operations correctly" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    // Record 10 operations with explicit timestamps 100..109
    var i: i64 = 0;
    while (i < 10) : (i += 1) {
        var buf: [8]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "key{d}", .{i}) catch unreachable;
        try manager.captureOperationWithTimestamp(.insert, key, "val", null, 100 + i);
    }

    // Recover to timestamp 104 => operations at 100,101,102,103,104 = 5
    var result = try manager.recoverToTimestamp(104);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 5), result.operations_replayed);
    try std.testing.expectEqual(@as(u64, 10), result.total_in_log);
    try std.testing.expectEqual(@as(usize, 5), result.operations.len);

    // Verify ordering: timestamps should be 100..104
    for (result.operations, 0..) |op, idx| {
        try std.testing.expectEqual(@as(i64, 100 + @as(i64, @intCast(idx))), op.timestamp);
    }
}

test "recoverToSequence filters operations correctly" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    // Record 10 operations; sequences will be 1..10
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var buf: [8]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "key{d}", .{i}) catch unreachable;
        try manager.captureOperationWithTimestamp(.insert, key, "val", null, @intCast(1000 + i));
    }

    // Recover to sequence 7 => operations 1..7
    var result = try manager.recoverToSequence(7);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 7), result.operations_replayed);
    try std.testing.expectEqual(@as(u64, 10), result.total_in_log);
    try std.testing.expectEqual(@as(usize, 7), result.operations.len);

    // Verify sequence numbers are 1..7
    for (result.operations, 0..) |op, idx| {
        try std.testing.expectEqual(@as(u64, idx + 1), op.sequence_number);
    }
}

test "save and load operation log roundtrip" {
    const allocator = std.testing.allocator;

    const test_path = "test_pitr_oplog.bin";

    // Clean up test file at the end
    defer {
        var cleanup_io = initIoBackend(allocator);
        defer cleanup_io.deinit();
        std.Io.Dir.cwd().deleteFile(cleanup_io.io(), test_path) catch {};
    }

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    // Record operations with various types and null/non-null values
    try manager.captureOperationWithTimestamp(.insert, "alpha", "val_a", null, 500);
    try manager.captureOperationWithTimestamp(.update, "beta", "val_b2", "val_b1", 501);
    try manager.captureOperationWithTimestamp(.delete, "gamma", null, "val_g", 502);
    try manager.captureOperationWithTimestamp(.truncate, "delta", null, null, 503);

    // Save
    try manager.saveOperationLog(test_path);

    // Load into a fresh manager
    var manager2 = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager2.deinit();

    try manager2.loadOperationLog(test_path);

    // Verify count
    try std.testing.expectEqual(@as(u64, 4), manager2.getOperationLogLen());

    // Recover all and verify contents
    var result = try manager2.recoverToTimestamp(999);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 4), result.operations_replayed);

    // Check first op
    try std.testing.expectEqualStrings("alpha", result.operations[0].key);
    try std.testing.expectEqualStrings("val_a", result.operations[0].value.?);
    try std.testing.expect(result.operations[0].previous_value == null);
    try std.testing.expectEqual(OperationType.insert, result.operations[0].type);

    // Check second op (update with previous value)
    try std.testing.expectEqualStrings("beta", result.operations[1].key);
    try std.testing.expectEqualStrings("val_b2", result.operations[1].value.?);
    try std.testing.expectEqualStrings("val_b1", result.operations[1].previous_value.?);

    // Check third op (delete, null value)
    try std.testing.expectEqualStrings("gamma", result.operations[2].key);
    try std.testing.expect(result.operations[2].value == null);
    try std.testing.expectEqualStrings("val_g", result.operations[2].previous_value.?);

    // Check fourth op (truncate, both null)
    try std.testing.expectEqualStrings("delta", result.operations[3].key);
    try std.testing.expect(result.operations[3].value == null);
    try std.testing.expect(result.operations[3].previous_value == null);
}

test "empty log recovery returns NoRecoveryPoint" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    // Recover from empty log should return error
    try std.testing.expectError(error.NoRecoveryPoint, manager.recoverToTimestamp(999));
}

test "empty log recoverToSequence returns SequenceNotFound" {
    const allocator = std.testing.allocator;

    var manager = PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600,
    });
    defer manager.deinit();

    // Recover sequence from empty log should return error
    try std.testing.expectError(error.SequenceNotFound, manager.recoverToSequence(5));
}

test {
    std.testing.refAllDecls(@This());
}
