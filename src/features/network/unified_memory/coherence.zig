//! Memory Coherence Protocol
//!
//! Implements cache coherence protocols for distributed unified memory.
//! Supports MESI, MOESI, and directory-based coherence for different
//! cluster sizes and topologies.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const shared_utils = @import("../../../services/shared/utils.zig");
const memory_region = @import("memory_region.zig");

const RegionId = memory_region.RegionId;
const RegionState = memory_region.RegionState;

/// Coherence state for cache lines / pages.
pub const CoherenceState = enum(u8) {
    /// Invalid - data not present or stale.
    invalid = 0,
    /// Shared - clean copy, others may have copies.
    shared = 1,
    /// Exclusive - only copy, clean.
    exclusive = 2,
    /// Modified - only copy, dirty.
    modified = 3,
    /// Owned (MOESI) - dirty copy, others may have shared copies.
    owned = 4,
    /// Forward (directory) - designated responder for requests.
    forward = 5,

    /// Check if state allows reads.
    pub fn allowsRead(self: CoherenceState) bool {
        return self != .invalid;
    }

    /// Check if state allows writes.
    pub fn allowsWrite(self: CoherenceState) bool {
        return self == .exclusive or self == .modified;
    }

    /// Check if data is present.
    pub fn hasData(self: CoherenceState) bool {
        return self != .invalid;
    }

    /// Check if we own the data (responsible for writebacks).
    pub fn isOwner(self: CoherenceState) bool {
        return self == .modified or self == .owned or self == .exclusive;
    }
};

/// Types of coherence messages.
pub const CoherenceMessageType = enum(u8) {
    // Read operations
    bus_rd, // Read request (want shared copy)
    bus_rdx, // Read exclusive (want to write)
    bus_upgr, // Upgrade (shared -> exclusive)

    // Response messages
    data_reply, // Data response
    ack, // Acknowledgment
    nack, // Negative acknowledgment

    // Invalidation
    inv, // Invalidate
    inv_ack, // Invalidation acknowledgment

    // Writeback
    bus_wb, // Writeback (dirty data to memory/owner)
    flush, // Flush and invalidate
    flush_opt, // Flush but keep in shared state

    // Directory-specific
    dir_fwd, // Directory forward request
    dir_inv, // Directory-initiated invalidation
    dir_wb, // Directory writeback request

    // Ownership transfer
    owner_transfer, // Transfer ownership to another node
};

/// Coherence message for inter-node communication.
pub const CoherenceMessage = struct {
    /// Message type.
    msg_type: CoherenceMessageType,

    /// Source node ID.
    source_node: u64,

    /// Destination node ID (0 = broadcast).
    dest_node: u64,

    /// Region being accessed.
    region_id: RegionId,

    /// Page/block index within region.
    block_index: usize,

    /// Block size.
    block_size: usize,

    /// Data payload (for data replies and writebacks).
    data: ?[]const u8,

    /// Request ID for matching responses.
    request_id: u64,

    /// Timestamp for ordering.
    timestamp: i64,

    /// Number of expected acknowledgments.
    ack_count: u32,

    /// Additional flags.
    flags: MessageFlags,

    pub const MessageFlags = struct {
        urgent: bool = false,
        exclusive_request: bool = false,
        writeback_required: bool = false,
        data_included: bool = false,
        broadcast: bool = false,
    };

    /// Create a read request message.
    pub fn readRequest(source: u64, region_id: RegionId, block_idx: usize) CoherenceMessage {
        return .{
            .msg_type = .bus_rd,
            .source_node = source,
            .dest_node = 0, // Broadcast
            .region_id = region_id,
            .block_index = block_idx,
            .block_size = 4096,
            .data = null,
            .request_id = generateRequestId(),
            .timestamp = shared_utils.unixMs(),
            .ack_count = 0,
            .flags = .{ .broadcast = true },
        };
    }

    /// Create a read exclusive request.
    pub fn readExclusiveRequest(source: u64, region_id: RegionId, block_idx: usize) CoherenceMessage {
        return .{
            .msg_type = .bus_rdx,
            .source_node = source,
            .dest_node = 0,
            .region_id = region_id,
            .block_index = block_idx,
            .block_size = 4096,
            .data = null,
            .request_id = generateRequestId(),
            .timestamp = shared_utils.unixMs(),
            .ack_count = 0,
            .flags = .{ .broadcast = true, .exclusive_request = true },
        };
    }

    /// Create an invalidation request.
    pub fn invalidation(source: u64, dest: u64, region_id: RegionId, block_idx: usize) CoherenceMessage {
        return .{
            .msg_type = .inv,
            .source_node = source,
            .dest_node = dest,
            .region_id = region_id,
            .block_index = block_idx,
            .block_size = 4096,
            .data = null,
            .request_id = generateRequestId(),
            .timestamp = shared_utils.unixMs(),
            .ack_count = 0,
            .flags = .{},
        };
    }

    /// Create a data reply.
    pub fn dataReply(request_id: u64, source: u64, dest: u64, data: []const u8) CoherenceMessage {
        return .{
            .msg_type = .data_reply,
            .source_node = source,
            .dest_node = dest,
            .region_id = 0,
            .block_index = 0,
            .block_size = data.len,
            .data = data,
            .request_id = request_id,
            .timestamp = shared_utils.unixMs(),
            .ack_count = 0,
            .flags = .{ .data_included = true },
        };
    }

    var request_counter: u64 = 0;
    fn generateRequestId() u64 {
        return @atomicRmw(u64, &request_counter, .Add, 1, .seq_cst);
    }
};

/// Invalidation request for cache coherence.
pub const InvalidationRequest = struct {
    /// Region to invalidate.
    region_id: RegionId,

    /// Start block index.
    start_block: usize,

    /// Number of blocks to invalidate.
    block_count: usize,

    /// Requesting node.
    requestor: u64,

    /// Nodes that need to be invalidated.
    target_nodes: []u64,

    /// Number of acknowledgments received.
    acks_received: u32,

    /// Number of acknowledgments expected.
    acks_expected: u32,

    /// Request completed.
    completed: bool,

    /// Completion callback (if any).
    on_complete: ?*const fn (*InvalidationRequest) void,
};

/// Directory entry for directory-based coherence.
pub const DirectoryEntry = struct {
    /// Region/block identifier.
    region_id: RegionId,
    block_index: usize,

    /// Current owner node (has exclusive/modified copy).
    owner: ?u64,

    /// Set of nodes with shared copies.
    sharers: std.AutoHashMapUnmanaged(u64, void),

    /// Current state.
    state: DirectoryState,

    /// Lock for concurrent access.
    lock: sync.Mutex,

    pub const DirectoryState = enum {
        uncached, // No cached copies
        shared, // One or more shared copies
        exclusive, // Single exclusive owner
    };

    pub fn init(region_id: RegionId, block_index: usize) DirectoryEntry {
        return .{
            .region_id = region_id,
            .block_index = block_index,
            .owner = null,
            .sharers = .{},
            .state = .uncached,
            .lock = .{},
        };
    }

    pub fn deinit(self: *DirectoryEntry, allocator: std.mem.Allocator) void {
        self.sharers.deinit(allocator);
    }

    pub fn addSharer(self: *DirectoryEntry, allocator: std.mem.Allocator, node_id: u64) !void {
        self.lock.lock();
        defer self.lock.unlock();
        try self.sharers.put(allocator, node_id, {});
        if (self.state == .uncached) {
            self.state = .shared;
        }
    }

    pub fn removeSharer(self: *DirectoryEntry, node_id: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        _ = self.sharers.remove(node_id);
        if (self.sharers.count() == 0 and self.owner == null) {
            self.state = .uncached;
        }
    }

    pub fn setExclusive(self: *DirectoryEntry, node_id: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.owner = node_id;
        self.sharers.clearRetainingCapacity();
        self.state = .exclusive;
    }

    pub fn getSharerList(self: *DirectoryEntry, allocator: std.mem.Allocator) ![]u64 {
        self.lock.lock();
        defer self.lock.unlock();

        var list = std.ArrayListUnmanaged(u64){};
        var it = self.sharers.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }
        return list.toOwnedSlice(allocator);
    }
};

/// Coherence protocol implementation.
pub const CoherenceProtocol = struct {
    allocator: std.mem.Allocator,

    /// Protocol type.
    protocol_type: ProtocolType,

    /// Local node ID.
    local_node_id: u64,

    /// Per-block coherence states.
    block_states: std.AutoHashMapUnmanaged(BlockKey, CoherenceState),

    /// Directory entries (for directory-based protocol).
    directory: ?std.AutoHashMapUnmanaged(BlockKey, *DirectoryEntry),

    /// Pending invalidation requests.
    pending_invalidations: std.ArrayListUnmanaged(*InvalidationRequest),

    /// Pending read requests waiting for data.
    pending_reads: std.AutoHashMapUnmanaged(u64, PendingRead),

    /// Statistics.
    stats: ProtocolStats,

    /// Lock for thread safety.
    mutex: sync.Mutex,

    pub const ProtocolType = enum {
        mesi,
        moesi,
        directory,
        none,
    };

    pub const BlockKey = struct {
        region_id: RegionId,
        block_index: usize,

        pub fn hash(self: BlockKey) u64 {
            return self.region_id ^ (@as(u64, self.block_index) << 32);
        }
    };

    pub const PendingRead = struct {
        request_id: u64,
        region_id: RegionId,
        block_index: usize,
        requestor: u64,
        timestamp: i64,
        data_received: bool,
        completion_event: std.Thread.ResetEvent,
    };

    pub const ProtocolStats = struct {
        read_hits: u64 = 0,
        read_misses: u64 = 0,
        write_hits: u64 = 0,
        write_misses: u64 = 0,
        invalidations_sent: u64 = 0,
        invalidations_received: u64 = 0,
        data_replies_sent: u64 = 0,
        data_replies_received: u64 = 0,
        upgrades: u64 = 0,
        writebacks: u64 = 0,
        directory_lookups: u64 = 0,
    };

    /// Initialize coherence protocol.
    pub fn init(
        allocator: std.mem.Allocator,
        protocol_type: anytype,
    ) !*CoherenceProtocol {
        const cp = try allocator.create(CoherenceProtocol);
        errdefer allocator.destroy(cp);

        const pt: ProtocolType = switch (@TypeOf(protocol_type)) {
            @TypeOf(@import("mod.zig").UnifiedMemoryConfig.CoherenceProtocolType) => switch (protocol_type) {
                .mesi => .mesi,
                .moesi => .moesi,
                .directory => .directory,
                .none => .none,
            },
            else => protocol_type,
        };

        cp.* = .{
            .allocator = allocator,
            .protocol_type = pt,
            .local_node_id = 0,
            .block_states = .{},
            .directory = if (pt == .directory) std.AutoHashMapUnmanaged(BlockKey, *DirectoryEntry){} else null,
            .pending_invalidations = .{},
            .pending_reads = .{},
            .stats = .{},
            .mutex = .{},
        };

        return cp;
    }

    /// Deinitialize protocol.
    pub fn deinit(self: *CoherenceProtocol) void {
        self.block_states.deinit(self.allocator);

        if (self.directory) |*dir| {
            var it = dir.valueIterator();
            while (it.next()) |entry| {
                entry.*.deinit(self.allocator);
                self.allocator.destroy(entry.*);
            }
            dir.deinit(self.allocator);
        }

        for (self.pending_invalidations.items) |inv| {
            self.allocator.free(inv.target_nodes);
            self.allocator.destroy(inv);
        }
        self.pending_invalidations.deinit(self.allocator);
        self.pending_reads.deinit(self.allocator);
    }

    /// Get the coherence state of a block.
    pub fn getState(self: *CoherenceProtocol, region_id: RegionId, block_index: usize) CoherenceState {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        return self.block_states.get(key) orelse .invalid;
    }

    /// Set the coherence state of a block.
    pub fn setState(self: *CoherenceProtocol, region_id: RegionId, block_index: usize, state: CoherenceState) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        try self.block_states.put(self.allocator, key, state);
    }

    /// Request read access to a block.
    pub fn requestRead(self: *CoherenceProtocol, region_id: RegionId, block_index: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        const current_state = self.block_states.get(key) orelse .invalid;

        if (current_state.allowsRead()) {
            self.stats.read_hits += 1;
            return;
        }

        // Need to fetch data - send BusRd
        self.stats.read_misses += 1;

        // State transition: Invalid -> Shared (after receiving data)
        try self.block_states.put(self.allocator, key, .shared);
    }

    /// Request write access to a block.
    pub fn requestWrite(self: *CoherenceProtocol, region_id: RegionId, block_index: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        const current_state = self.block_states.get(key) orelse .invalid;

        if (current_state.allowsWrite()) {
            self.stats.write_hits += 1;
            try self.block_states.put(self.allocator, key, .modified);
            return;
        }

        self.stats.write_misses += 1;

        // Need exclusive access
        if (current_state == .shared) {
            // Upgrade: Shared -> Modified
            self.stats.upgrades += 1;
        }

        // Invalidate other copies and get exclusive access
        try self.block_states.put(self.allocator, key, .modified);
    }

    /// Handle receiving a coherence message.
    pub fn handleMessage(self: *CoherenceProtocol, msg: CoherenceMessage) !?CoherenceMessage {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = msg.region_id, .block_index = msg.block_index };
        const current_state = self.block_states.get(key) orelse .invalid;

        switch (msg.msg_type) {
            .bus_rd => {
                // Someone wants to read - we can share
                if (current_state == .modified or current_state == .exclusive) {
                    // Downgrade to shared, send data
                    try self.block_states.put(self.allocator, key, .shared);
                    self.stats.data_replies_sent += 1;
                    // Would return data reply here
                }
            },
            .bus_rdx => {
                // Someone wants exclusive access - invalidate our copy
                if (current_state != .invalid) {
                    try self.block_states.put(self.allocator, key, .invalid);
                    self.stats.invalidations_received += 1;
                    // Send data if we have it
                    if (current_state == .modified) {
                        self.stats.writebacks += 1;
                    }
                }
            },
            .inv => {
                // Invalidation request
                try self.block_states.put(self.allocator, key, .invalid);
                self.stats.invalidations_received += 1;
                // Return ack
                return CoherenceMessage{
                    .msg_type = .inv_ack,
                    .source_node = self.local_node_id,
                    .dest_node = msg.source_node,
                    .region_id = msg.region_id,
                    .block_index = msg.block_index,
                    .block_size = msg.block_size,
                    .data = null,
                    .request_id = msg.request_id,
                    .timestamp = shared_utils.unixMs(),
                    .ack_count = 0,
                    .flags = .{},
                };
            },
            .inv_ack => {
                // Track acknowledgment for pending invalidation
            },
            .data_reply => {
                self.stats.data_replies_received += 1;
            },
            else => {},
        }

        return null;
    }

    /// Invalidate a region across all nodes.
    pub fn invalidateRegion(self: *CoherenceProtocol, region_id: RegionId) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Find all blocks for this region and invalidate
        var keys_to_remove = std.ArrayListUnmanaged(BlockKey){};
        defer keys_to_remove.deinit(self.allocator);

        var it = self.block_states.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.region_id == region_id) {
                try keys_to_remove.append(self.allocator, entry.key_ptr.*);
            }
        }

        for (keys_to_remove.items) |key| {
            _ = self.block_states.remove(key);
        }

        self.stats.invalidations_sent += 1;
    }

    /// Get protocol statistics.
    pub fn getStats(self: *CoherenceProtocol) ProtocolStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Perform a writeback for dirty data.
    pub fn writeback(self: *CoherenceProtocol, region_id: RegionId, block_index: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        const current_state = self.block_states.get(key) orelse return;

        if (current_state == .modified) {
            // Transition: Modified -> Exclusive (after writeback)
            try self.block_states.put(self.allocator, key, .exclusive);
            self.stats.writebacks += 1;
        }
    }

    /// Evict a block from the cache.
    pub fn evict(self: *CoherenceProtocol, region_id: RegionId, block_index: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = BlockKey{ .region_id = region_id, .block_index = block_index };
        const current_state = self.block_states.get(key) orelse return;

        if (current_state == .modified) {
            // Need to writeback before eviction
            self.stats.writebacks += 1;
        }

        _ = self.block_states.remove(key);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CoherenceState access checks" {
    try std.testing.expect(CoherenceState.shared.allowsRead());
    try std.testing.expect(!CoherenceState.shared.allowsWrite());
    try std.testing.expect(CoherenceState.exclusive.allowsRead());
    try std.testing.expect(CoherenceState.exclusive.allowsWrite());
    try std.testing.expect(CoherenceState.modified.isOwner());
    try std.testing.expect(!CoherenceState.invalid.hasData());
}

test "CoherenceProtocol state transitions" {
    const allocator = std.testing.allocator;

    const protocol = try CoherenceProtocol.init(allocator, CoherenceProtocol.ProtocolType.mesi);
    defer {
        protocol.deinit();
        allocator.destroy(protocol);
    }

    // Initial state should be invalid
    try std.testing.expectEqual(CoherenceState.invalid, protocol.getState(1, 0));

    // Request read - should transition to shared
    try protocol.requestRead(1, 0);
    try std.testing.expectEqual(CoherenceState.shared, protocol.getState(1, 0));

    // Request write - should transition to modified
    try protocol.requestWrite(1, 0);
    try std.testing.expectEqual(CoherenceState.modified, protocol.getState(1, 0));

    // Writeback - should transition to exclusive
    try protocol.writeback(1, 0);
    try std.testing.expectEqual(CoherenceState.exclusive, protocol.getState(1, 0));
}

test "CoherenceMessage creation" {
    const msg = CoherenceMessage.readRequest(1, 100, 5);
    try std.testing.expectEqual(CoherenceMessageType.bus_rd, msg.msg_type);
    try std.testing.expectEqual(@as(u64, 1), msg.source_node);
    try std.testing.expectEqual(@as(RegionId, 100), msg.region_id);
    try std.testing.expectEqual(@as(usize, 5), msg.block_index);
    try std.testing.expect(msg.flags.broadcast);
}

test "DirectoryEntry operations" {
    const allocator = std.testing.allocator;

    var entry = DirectoryEntry.init(1, 0);
    defer entry.deinit(allocator);

    try std.testing.expectEqual(DirectoryEntry.DirectoryState.uncached, entry.state);

    try entry.addSharer(allocator, 1);
    try entry.addSharer(allocator, 2);
    try std.testing.expectEqual(DirectoryEntry.DirectoryState.shared, entry.state);

    entry.setExclusive(3);
    try std.testing.expectEqual(DirectoryEntry.DirectoryState.exclusive, entry.state);
    try std.testing.expectEqual(@as(?u64, 3), entry.owner);
    try std.testing.expectEqual(@as(usize, 0), entry.sharers.count());
}
