//! Write-Ahead Log (WAL) for WDBX Distributed
//!
//! Sequential append-only log of mutations applied to the vector database.
//! WAL entries are replicated via Raft AppendEntries before being applied
//! to the in-memory database. Compaction truncates the log after snapshot.
//!
//! Storage format uses unified storage wal_entry block type (0x05).

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

pub const WalEntryType = enum(u8) {
    insert = 0x01,
    delete = 0x02,
    update = 0x03,
    batch_insert = 0x04,
    snapshot_marker = 0x10,
    compaction_marker = 0x11,
};

pub const WalEntry = struct {
    sequence: u64,
    entry_type: WalEntryType,
    timestamp: i64,
    // Payload
    vector_id: u64,
    dimension: u32,
    data_offset: u32, // Offset into data buffer
    data_len: u32,
    // Checksums
    crc32: u32,
};

pub const WalHeader = struct {
    magic: [4]u8 = .{ 'W', 'A', 'L', 'X' },
    version: u16 = 1,
    node_id: u64 = 0,
    created_at: i64 = 0,
    entry_count: u64 = 0,
    last_sequence: u64 = 0,
};

// ============================================================================
// WAL Writer
// ============================================================================

pub const WalWriter = struct {
    allocator: std.mem.Allocator,
    header: WalHeader,
    entries: std.ArrayListUnmanaged(WalEntry),
    data_buf: std.ArrayListUnmanaged(u8),
    path: [512]u8,
    path_len: usize,
    dirty: bool,

    pub fn init(allocator: std.mem.Allocator, path: []const u8, node_id: u64) WalWriter {
        var self: WalWriter = undefined;
        self.allocator = allocator;
        self.header = .{ .node_id = node_id };
        self.entries = .empty;
        self.data_buf = .empty;
        self.path = [_]u8{0} ** 512;
        self.path_len = @min(path.len, 511);
        @memcpy(self.path[0..self.path_len], path[0..self.path_len]);
        self.dirty = false;

        // Set creation timestamp
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        self.header.created_at = @intCast(ts.sec);

        return self;
    }

    pub fn deinit(self: *WalWriter) void {
        self.entries.deinit(self.allocator);
        self.data_buf.deinit(self.allocator);
    }

    /// Append an insert operation to the WAL.
    pub fn appendInsert(self: *WalWriter, vector_id: u64, dimension: u32, vector_data: []const f32) !void {
        const data_bytes = std.mem.sliceAsBytes(vector_data);
        const offset: u32 = @intCast(self.data_buf.items.len);

        try self.data_buf.appendSlice(self.allocator, data_bytes);

        self.header.last_sequence += 1;
        try self.entries.append(self.allocator, .{
            .sequence = self.header.last_sequence,
            .entry_type = .insert,
            .timestamp = currentTimestamp(),
            .vector_id = vector_id,
            .dimension = dimension,
            .data_offset = offset,
            .data_len = @intCast(data_bytes.len),
            .crc32 = std.hash.Crc32.hash(data_bytes),
        });

        self.header.entry_count += 1;
        self.dirty = true;
    }

    /// Append a delete operation to the WAL.
    pub fn appendDelete(self: *WalWriter, vector_id: u64) !void {
        self.header.last_sequence += 1;
        try self.entries.append(self.allocator, .{
            .sequence = self.header.last_sequence,
            .entry_type = .delete,
            .timestamp = currentTimestamp(),
            .vector_id = vector_id,
            .dimension = 0,
            .data_offset = 0,
            .data_len = 0,
            .crc32 = 0,
        });
        self.header.entry_count += 1;
        self.dirty = true;
    }

    /// Append an update operation to the WAL.
    pub fn appendUpdate(self: *WalWriter, vector_id: u64, dimension: u32, vector_data: []const f32) !void {
        const data_bytes = std.mem.sliceAsBytes(vector_data);
        const offset: u32 = @intCast(self.data_buf.items.len);

        try self.data_buf.appendSlice(self.allocator, data_bytes);

        self.header.last_sequence += 1;
        try self.entries.append(self.allocator, .{
            .sequence = self.header.last_sequence,
            .entry_type = .update,
            .timestamp = currentTimestamp(),
            .vector_id = vector_id,
            .dimension = dimension,
            .data_offset = offset,
            .data_len = @intCast(data_bytes.len),
            .crc32 = std.hash.Crc32.hash(data_bytes),
        });
        self.header.entry_count += 1;
        self.dirty = true;
    }

    /// Mark a snapshot point — entries before this can be compacted.
    pub fn appendSnapshotMarker(self: *WalWriter) !void {
        self.header.last_sequence += 1;
        try self.entries.append(self.allocator, .{
            .sequence = self.header.last_sequence,
            .entry_type = .snapshot_marker,
            .timestamp = currentTimestamp(),
            .vector_id = 0,
            .dimension = 0,
            .data_offset = 0,
            .data_len = 0,
            .crc32 = 0,
        });
        self.header.entry_count += 1;
        self.dirty = true;
    }

    /// Compact the WAL: remove all entries before the last snapshot marker.
    pub fn compact(self: *WalWriter) void {
        // Find last snapshot marker
        var last_snapshot: ?usize = null;
        for (self.entries.items, 0..) |entry, i| {
            if (entry.entry_type == .snapshot_marker) last_snapshot = i;
        }

        if (last_snapshot) |idx| {
            if (idx + 1 < self.entries.items.len) {
                // Keep entries after snapshot
                const remaining = self.entries.items.len - idx - 1;
                std.mem.copyForwards(WalEntry, self.entries.items[0..remaining], self.entries.items[idx + 1 ..]);
                self.entries.items.len = remaining;
                self.header.entry_count = remaining;
            } else {
                self.entries.items.len = 0;
                self.header.entry_count = 0;
            }
            self.dirty = true;
        }
    }

    /// Get entries since a given sequence number (for replication).
    pub fn entriesSince(self: *const WalWriter, since_sequence: u64) []const WalEntry {
        for (self.entries.items, 0..) |entry, i| {
            if (entry.sequence > since_sequence) {
                return self.entries.items[i..];
            }
        }
        return &[_]WalEntry{};
    }

    /// Get current entry count.
    pub fn entryCount(self: *const WalWriter) u64 {
        return self.header.entry_count;
    }

    /// Get last sequence number.
    pub fn lastSequence(self: *const WalWriter) u64 {
        return self.header.last_sequence;
    }

    /// Serialize the WAL to a byte buffer for disk persistence.
    pub fn serialize(self: *const WalWriter, allocator: std.mem.Allocator) ![]u8 {
        const header_size = @sizeOf(WalHeader);
        const entry_size = @sizeOf(WalEntry);
        const entries_size = self.entries.items.len * entry_size;
        const total = header_size + entries_size + self.data_buf.items.len;

        const buffer = try allocator.alloc(u8, total);
        var offset: usize = 0;

        // Write header
        const header_bytes = std.mem.asBytes(&self.header);
        @memcpy(buffer[offset .. offset + header_size], header_bytes);
        offset += header_size;

        // Write entries
        for (self.entries.items) |entry| {
            const entry_bytes = std.mem.asBytes(&entry);
            @memcpy(buffer[offset .. offset + entry_size], entry_bytes);
            offset += entry_size;
        }

        // Write data buffer
        if (self.data_buf.items.len > 0) {
            @memcpy(buffer[offset .. offset + self.data_buf.items.len], self.data_buf.items);
        }

        return buffer;
    }

    /// Deserialize WAL from a byte buffer (recovery).
    pub fn deserialize(self: *WalWriter, buffer: []const u8) !void {
        const header_size = @sizeOf(WalHeader);
        const entry_size = @sizeOf(WalEntry);

        if (buffer.len < header_size) return error.InvalidData;

        // Read header
        self.header = std.mem.bytesToValue(WalHeader, buffer[0..header_size]);

        // Validate magic
        if (!std.mem.eql(u8, &self.header.magic, "WALX")) return error.InvalidData;

        var offset: usize = header_size;

        // Read entries
        self.entries.items.len = 0;
        var entries_read: u64 = 0;
        while (entries_read < self.header.entry_count and offset + entry_size <= buffer.len) {
            const entry = std.mem.bytesToValue(WalEntry, buffer[offset..][0..entry_size]);
            try self.entries.append(self.allocator, entry);
            offset += entry_size;
            entries_read += 1;
        }

        // Read data buffer
        if (offset < buffer.len) {
            self.data_buf.items.len = 0;
            try self.data_buf.appendSlice(self.allocator, buffer[offset..]);
        }

        self.dirty = false;
    }

    /// Compute CRC32 for an entry's data payload.
    pub fn computeEntryCrc(data: []const u8) u32 {
        return std.hash.Crc32.hash(data);
    }

    /// Truncate entries before a given sequence number (compaction).
    pub fn truncateBeforeSequence(self: *WalWriter, seq: u64) void {
        var keep_start: usize = self.entries.items.len;
        for (self.entries.items, 0..) |entry, i| {
            if (entry.sequence >= seq) {
                keep_start = i;
                break;
            }
        }
        if (keep_start > 0 and keep_start < self.entries.items.len) {
            const remaining = self.entries.items.len - keep_start;
            std.mem.copyForwards(WalEntry, self.entries.items[0..remaining], self.entries.items[keep_start..]);
            self.entries.items.len = remaining;
            self.header.entry_count = remaining;
            self.dirty = true;
        }
    }
};

// ============================================================================
// WAL Reader (for recovery)
// ============================================================================

pub const WalReader = struct {
    entries: []const WalEntry,
    data: []const u8,
    position: usize,

    pub fn init(entries: []const WalEntry, data: []const u8) WalReader {
        return .{ .entries = entries, .data = data, .position = 0 };
    }

    pub fn next(self: *WalReader) ?WalEntry {
        if (self.position >= self.entries.len) return null;
        const entry = self.entries[self.position];
        self.position += 1;
        return entry;
    }

    pub fn getVectorData(self: *const WalReader, entry: WalEntry) ?[]const f32 {
        if (entry.data_len == 0) return null;
        const start = entry.data_offset;
        const end = start + entry.data_len;
        if (end > self.data.len) return null;
        const bytes = self.data[start..end];
        return std.mem.bytesAsSlice(f32, @alignCast(bytes));
    }

    pub fn reset(self: *WalReader) void {
        self.position = 0;
    }
};

// ============================================================================
// Error types
// ============================================================================

pub const WalError = error{
    InvalidData,
    CorruptedEntry,
    SequenceGap,
};

// ============================================================================
// Replication state tracking
// ============================================================================

/// Replication state tracking per-peer.
pub const ReplicationState = struct {
    peer_offsets: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    pending_acks: std.AutoHashMapUnmanaged(u64, u32) = .empty,

    pub fn init() ReplicationState {
        return .{};
    }

    pub fn deinit(self: *ReplicationState, allocator: std.mem.Allocator) void {
        self.peer_offsets.deinit(allocator);
        self.pending_acks.deinit(allocator);
    }

    pub fn recordAck(self: *ReplicationState, allocator: std.mem.Allocator, peer_id: u64, sequence: u64) !void {
        try self.peer_offsets.put(allocator, peer_id, sequence);
        // Decrement pending ack count
        if (self.pending_acks.getPtr(sequence)) |count| {
            if (count.* > 0) count.* -= 1;
        }
    }

    pub fn getReplicationLag(self: *const ReplicationState, peer_id: u64, current_seq: u64) u64 {
        const peer_seq = self.peer_offsets.get(peer_id) orelse 0;
        if (current_seq > peer_seq) return current_seq - peer_seq;
        return 0;
    }

    pub fn isFullyReplicated(self: *const ReplicationState, sequence: u64) bool {
        const count = self.pending_acks.get(sequence) orelse return true;
        return count == 0;
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn currentTimestamp() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

// ============================================================================
// Tests
// ============================================================================

test "WalWriter basic operations" {
    const allocator = std.testing.allocator;
    var wal = WalWriter.init(allocator, "/tmp/test.wal", 1);
    defer wal.deinit();

    try wal.appendInsert(100, 3, &[_]f32{ 1.0, 2.0, 3.0 });
    try wal.appendInsert(101, 3, &[_]f32{ 4.0, 5.0, 6.0 });
    try wal.appendDelete(100);

    try std.testing.expectEqual(@as(u64, 3), wal.entryCount());
    try std.testing.expectEqual(@as(u64, 3), wal.lastSequence());
}

test "WalWriter compact" {
    const allocator = std.testing.allocator;
    var wal = WalWriter.init(allocator, "/tmp/test.wal", 1);
    defer wal.deinit();

    try wal.appendInsert(1, 2, &[_]f32{ 1.0, 2.0 });
    try wal.appendInsert(2, 2, &[_]f32{ 3.0, 4.0 });
    try wal.appendSnapshotMarker();
    try wal.appendInsert(3, 2, &[_]f32{ 5.0, 6.0 });

    wal.compact();
    try std.testing.expectEqual(@as(u64, 1), wal.entryCount());
}

test "WalWriter entriesSince" {
    const allocator = std.testing.allocator;
    var wal_writer = WalWriter.init(allocator, "/tmp/test.wal", 1);
    defer wal_writer.deinit();

    try wal_writer.appendInsert(1, 2, &[_]f32{ 1.0, 2.0 });
    try wal_writer.appendInsert(2, 2, &[_]f32{ 3.0, 4.0 });
    try wal_writer.appendInsert(3, 2, &[_]f32{ 5.0, 6.0 });

    const since = wal_writer.entriesSince(1);
    try std.testing.expectEqual(@as(usize, 2), since.len);
}

test "WalWriter serialize/deserialize round-trip" {
    const allocator = std.testing.allocator;
    var wal_writer = WalWriter.init(allocator, "/tmp/test.wal", 42);
    defer wal_writer.deinit();

    try wal_writer.appendInsert(100, 3, &[_]f32{ 1.0, 2.0, 3.0 });
    try wal_writer.appendInsert(101, 3, &[_]f32{ 4.0, 5.0, 6.0 });

    const serialized = try wal_writer.serialize(allocator);
    defer allocator.free(serialized);

    // Deserialize into a fresh WAL
    var wal2 = WalWriter.init(allocator, "/tmp/test2.wal", 0);
    defer wal2.deinit();

    try wal2.deserialize(serialized);

    try std.testing.expectEqual(@as(u64, 42), wal2.header.node_id);
    try std.testing.expectEqual(@as(u64, 2), wal2.header.entry_count);
    try std.testing.expectEqual(@as(u64, 2), wal2.lastSequence());
    try std.testing.expectEqual(@as(usize, 2), wal2.entries.items.len);
    try std.testing.expectEqual(@as(u64, 100), wal2.entries.items[0].vector_id);
    try std.testing.expectEqual(@as(u64, 101), wal2.entries.items[1].vector_id);
}

test "WalWriter CRC validation" {
    const data = "hello world";
    const crc = WalWriter.computeEntryCrc(data);
    const crc2 = WalWriter.computeEntryCrc(data);
    try std.testing.expectEqual(crc, crc2);

    // Different data should produce different CRC
    const crc3 = WalWriter.computeEntryCrc("different data");
    try std.testing.expect(crc != crc3);
}

test "WalWriter truncateBeforeSequence" {
    const allocator = std.testing.allocator;
    var wal_writer = WalWriter.init(allocator, "/tmp/test.wal", 1);
    defer wal_writer.deinit();

    try wal_writer.appendInsert(1, 2, &[_]f32{ 1.0, 2.0 });
    try wal_writer.appendInsert(2, 2, &[_]f32{ 3.0, 4.0 });
    try wal_writer.appendInsert(3, 2, &[_]f32{ 5.0, 6.0 });
    try wal_writer.appendInsert(4, 2, &[_]f32{ 7.0, 8.0 });

    // Truncate entries before sequence 3 (keep seq 3 and 4)
    wal_writer.truncateBeforeSequence(3);

    try std.testing.expectEqual(@as(usize, 2), wal_writer.entries.items.len);
    try std.testing.expectEqual(@as(u64, 3), wal_writer.entries.items[0].sequence);
    try std.testing.expectEqual(@as(u64, 4), wal_writer.entries.items[1].sequence);
    try std.testing.expect(wal_writer.dirty);
}

test "ReplicationState ack tracking" {
    const allocator = std.testing.allocator;
    var state = ReplicationState.init();
    defer state.deinit(allocator);

    // Record acks from peers
    try state.recordAck(allocator, 1, 5);
    try state.recordAck(allocator, 2, 3);

    // Verify peer offsets
    try std.testing.expectEqual(@as(u64, 5), state.peer_offsets.get(1).?);
    try std.testing.expectEqual(@as(u64, 3), state.peer_offsets.get(2).?);

    // Check fully replicated (no pending acks registered)
    try std.testing.expect(state.isFullyReplicated(5));
}

test "ReplicationState lag calculation" {
    const allocator = std.testing.allocator;
    var state = ReplicationState.init();
    defer state.deinit(allocator);

    try state.recordAck(allocator, 1, 5);
    try state.recordAck(allocator, 2, 3);

    // Lag for peer 1 at current seq 10 => 10 - 5 = 5
    try std.testing.expectEqual(@as(u64, 5), state.getReplicationLag(1, 10));
    // Lag for peer 2 at current seq 10 => 10 - 3 = 7
    try std.testing.expectEqual(@as(u64, 7), state.getReplicationLag(2, 10));
    // Lag for unknown peer => full lag (current_seq - 0)
    try std.testing.expectEqual(@as(u64, 10), state.getReplicationLag(99, 10));
}
