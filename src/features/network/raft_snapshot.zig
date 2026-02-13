//! Raft Snapshots and Membership Changes
//!
//! Provides log compaction via snapshots and dynamic membership
//! changes for the Raft consensus protocol.

const std = @import("std");
const raft = @import("raft.zig");
const RaftNode = raft.RaftNode;
const LogEntry = raft.LogEntry;
const initIoBackend = raft.initIoBackend;

/// Snapshot metadata.
pub const SnapshotMetadata = struct {
    /// Last included index in the snapshot.
    last_included_index: u64,
    /// Term of last included index.
    last_included_term: u64,
    /// Configuration at snapshot time (serialized peer list).
    config_size: u32,
};

/// Snapshot file magic number.
const SNAPSHOT_MAGIC: u32 = 0x534E4150; // "SNAP"
/// Snapshot format version.
const SNAPSHOT_VERSION: u16 = 1;

/// Raft snapshot manager for log compaction.
pub const RaftSnapshotManager = struct {
    allocator: std.mem.Allocator,
    snapshot_dir: []const u8,
    /// Minimum log entries to keep before compaction.
    min_log_entries: usize,
    /// Threshold for triggering automatic snapshot.
    snapshot_threshold: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        snapshot_dir: []const u8,
        config: SnapshotConfig,
    ) !RaftSnapshotManager {
        return .{
            .allocator = allocator,
            .snapshot_dir = try allocator.dupe(u8, snapshot_dir),
            .min_log_entries = config.min_log_entries,
            .snapshot_threshold = config.snapshot_threshold,
        };
    }

    pub fn deinit(self: *RaftSnapshotManager) void {
        self.allocator.free(self.snapshot_dir);
    }

    /// Create a snapshot of the current state machine.
    pub fn createSnapshot(
        self: *RaftSnapshotManager,
        node: *RaftNode,
        state_machine_data: []const u8,
    ) !void {
        if (node.commit_index == 0) return; // Nothing to snapshot

        const last_included_index = node.commit_index;
        const last_included_term = if (last_included_index > 0 and last_included_index <= node.log.items.len)
            node.log.items[last_included_index - 1].term
        else
            0;

        // Build configuration data (peer list)
        var config_data = std.ArrayListUnmanaged(u8){};
        defer config_data.deinit(self.allocator);

        var iter = node.peers.keyIterator();
        while (iter.next()) |key| {
            try config_data.appendSlice(self.allocator, key.*);
            try config_data.append(self.allocator, '\n');
        }

        // Calculate total size
        const total_size = 4 + 2 + @sizeOf(SnapshotMetadata) + config_data.items.len + state_machine_data.len;

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);

        var offset: usize = 0;

        // Write magic
        std.mem.writeInt(u32, buffer[offset..][0..4], SNAPSHOT_MAGIC, .little);
        offset += 4;

        // Write version
        std.mem.writeInt(u16, buffer[offset..][0..2], SNAPSHOT_VERSION, .little);
        offset += 2;

        // Write metadata
        const metadata = SnapshotMetadata{
            .last_included_index = last_included_index,
            .last_included_term = last_included_term,
            .config_size = @intCast(config_data.items.len),
        };
        const metadata_bytes = std.mem.asBytes(&metadata);
        @memcpy(buffer[offset..][0..metadata_bytes.len], metadata_bytes);
        offset += metadata_bytes.len;

        // Write config
        @memcpy(buffer[offset..][0..config_data.items.len], config_data.items);
        offset += config_data.items.len;

        // Write state machine data
        @memcpy(buffer[offset..][0..state_machine_data.len], state_machine_data);

        // Generate snapshot filename
        var filename_buf: [256]u8 = undefined;
        const filename = std.fmt.bufPrint(
            &filename_buf,
            "{s}/snapshot-{d}-{d}.snap",
            .{ self.snapshot_dir, last_included_index, last_included_term },
        ) catch return error.PathTooLong;

        // Write to file
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Ensure directory exists
        std.Io.Dir.cwd().createDirPath(io, self.snapshot_dir) catch |err| {
            std.log.warn("Failed to create snapshot directory '{s}': {t}", .{ self.snapshot_dir, err });
        };

        var file = try std.Io.Dir.cwd().createFile(io, filename, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, buffer[0..offset]);

        // Compact the log
        try self.compactLog(node, last_included_index);
    }

    /// Compact the log by removing entries before the snapshot index.
    fn compactLog(self: *RaftSnapshotManager, node: *RaftNode, snapshot_index: u64) !void {
        if (snapshot_index == 0 or snapshot_index > node.log.items.len) return;

        // Keep at least min_log_entries
        const entries_to_remove = if (snapshot_index > self.min_log_entries)
            snapshot_index - self.min_log_entries
        else
            0;

        if (entries_to_remove == 0) return;

        // Free data for removed entries
        for (node.log.items[0..entries_to_remove]) |entry| {
            node.allocator.free(entry.data);
        }

        // Shift remaining entries
        const remaining = node.log.items.len - entries_to_remove;
        std.mem.copyForwards(
            LogEntry,
            node.log.items[0..remaining],
            node.log.items[entries_to_remove..],
        );
        node.log.shrinkRetainingCapacity(remaining);
    }

    /// Load the latest snapshot.
    pub fn loadLatestSnapshot(
        self: *RaftSnapshotManager,
        node: *RaftNode,
    ) !?[]u8 {
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Find latest snapshot file
        var latest_index: u64 = 0;
        var latest_term: u64 = 0;
        var latest_filename: ?[]const u8 = null;

        var dir = std.Io.Dir.cwd().openDir(io, self.snapshot_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return null;
            return err;
        };
        defer dir.close(io);

        var dir_iter = dir.iterate();
        while (try dir_iter.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".snap")) {
                // Parse index and term from filename
                if (parseSnapshotFilename(entry.name)) |parsed| {
                    if (parsed.index > latest_index or
                        (parsed.index == latest_index and parsed.term > latest_term))
                    {
                        latest_index = parsed.index;
                        latest_term = parsed.term;
                        const new_filename = try self.allocator.dupe(u8, entry.name);
                        if (latest_filename) |f| self.allocator.free(f);
                        latest_filename = new_filename;
                    }
                }
            }
        }

        if (latest_filename == null) return null;
        defer self.allocator.free(latest_filename.?);

        // Load the snapshot
        var path_buf: [512]u8 = undefined;
        const full_path = std.fmt.bufPrint(
            &path_buf,
            "{s}/{s}",
            .{ self.snapshot_dir, latest_filename.? },
        ) catch return error.PathTooLong;

        const buffer = try std.Io.Dir.cwd().readFileAlloc(
            io,
            full_path,
            self.allocator,
            .limited(256 * 1024 * 1024),
        );
        defer self.allocator.free(buffer);

        if (buffer.len < 6 + @sizeOf(SnapshotMetadata)) return error.InvalidSnapshot;

        var offset: usize = 0;

        // Verify magic
        const magic = std.mem.readInt(u32, buffer[offset..][0..4], .little);
        if (magic != SNAPSHOT_MAGIC) return error.InvalidSnapshot;
        offset += 4;

        // Verify version
        const version = std.mem.readInt(u16, buffer[offset..][0..2], .little);
        if (version != SNAPSHOT_VERSION) return error.UnsupportedVersion;
        offset += 2;

        // Read metadata (copy to aligned struct to avoid alignment issues)
        var snap_metadata: SnapshotMetadata = undefined;
        const snap_metadata_bytes = std.mem.asBytes(&snap_metadata);
        if (buffer.len < offset + snap_metadata_bytes.len) return error.InvalidSnapshot;
        @memcpy(snap_metadata_bytes, buffer[offset..][0..snap_metadata_bytes.len]);
        offset += @sizeOf(SnapshotMetadata);

        // Skip config for now
        offset += snap_metadata.config_size;

        // Extract state machine data
        const state_data_len = buffer.len - offset;
        const state_data = try self.allocator.dupe(u8, buffer[offset..][0..state_data_len]);

        // Update node state
        node.commit_index = @max(node.commit_index, snap_metadata.last_included_index);
        node.last_applied = @max(node.last_applied, snap_metadata.last_included_index);

        return state_data;
    }

    /// Check if snapshot is needed based on log size.
    pub fn shouldSnapshot(self: *const RaftSnapshotManager, node: *const RaftNode) bool {
        return node.log.items.len >= self.snapshot_threshold;
    }

    /// Get list of available snapshot files.
    pub fn listSnapshots(self: *RaftSnapshotManager) !std.ArrayListUnmanaged(SnapshotInfo) {
        var snapshots = std.ArrayListUnmanaged(SnapshotInfo){};
        errdefer {
            for (snapshots.items) |s| self.allocator.free(s.filename);
            snapshots.deinit(self.allocator);
        }

        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        var dir = std.Io.Dir.cwd().openDir(io, self.snapshot_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return snapshots;
            return err;
        };
        defer dir.close(io);

        var dir_iter = dir.iterate();
        while (try dir_iter.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".snap")) {
                if (parseSnapshotFilename(entry.name)) |parsed| {
                    const fname = try self.allocator.dupe(u8, entry.name);
                    errdefer self.allocator.free(fname);
                    try snapshots.append(self.allocator, .{
                        .filename = fname,
                        .last_included_index = parsed.index,
                        .last_included_term = parsed.term,
                    });
                }
            }
        }

        return snapshots;
    }

    const ParsedSnapshot = struct { index: u64, term: u64 };

    fn parseSnapshotFilename(name: []const u8) ?ParsedSnapshot {
        // Expected format: snapshot-{index}-{term}.snap
        if (!std.mem.startsWith(u8, name, "snapshot-")) return null;
        if (!std.mem.endsWith(u8, name, ".snap")) return null;

        const content = name["snapshot-".len .. name.len - ".snap".len];
        var parts = std.mem.splitScalar(u8, content, '-');

        const index_str = parts.next() orelse return null;
        const term_str = parts.next() orelse return null;

        const index = std.fmt.parseInt(u64, index_str, 10) catch return null;
        const term = std.fmt.parseInt(u64, term_str, 10) catch return null;

        return .{ .index = index, .term = term };
    }

    pub const SnapshotError = error{
        InvalidSnapshot,
        UnsupportedVersion,
        PathTooLong,
    } || std.mem.Allocator.Error || std.Io.Dir.OpenDirError || std.Io.Dir.MakePathError ||
        std.Io.Dir.CreateFileError || std.Io.Dir.ReadFileAllocError;
};

/// Snapshot configuration options.
pub const SnapshotConfig = struct {
    /// Minimum number of log entries to keep after compaction.
    min_log_entries: usize = 100,
    /// Log size threshold to trigger automatic snapshot.
    snapshot_threshold: usize = 10000,
};

/// Information about a snapshot file.
pub const SnapshotInfo = struct {
    filename: []const u8,
    last_included_index: u64,
    last_included_term: u64,
};

// ============================================================================
// InstallSnapshot RPC for Raft
// ============================================================================

/// InstallSnapshot RPC request.
pub const InstallSnapshotRequest = struct {
    /// Leader's term.
    term: u64,
    /// Leader ID so follower can redirect clients.
    leader_id: []const u8,
    /// Last included index in snapshot.
    last_included_index: u64,
    /// Term of last included index.
    last_included_term: u64,
    /// Byte offset in snapshot chunk.
    offset: u64,
    /// Snapshot chunk data.
    data: []const u8,
    /// True if this is the last chunk.
    done: bool,
};

/// InstallSnapshot RPC response.
pub const InstallSnapshotResponse = struct {
    /// Current term for leader to update.
    term: u64,
    /// True if chunk was accepted.
    success: bool,
    /// Follower's current offset for next chunk.
    next_offset: u64,
};

// ============================================================================
// Membership Change Protocol
// ============================================================================

/// Configuration change entry types.
pub const ConfigChangeType = enum(u8) {
    add_node = 0,
    remove_node = 1,
    promote_learner = 2, // Learner -> Voting member
};

/// Configuration change request.
pub const ConfigChangeRequest = struct {
    change_type: ConfigChangeType,
    node_id: []const u8,
    node_address: ?[]const u8,
};

/// Apply a configuration change to the Raft node.
pub fn applyConfigChange(node: *RaftNode, change: ConfigChangeRequest) !void {
    switch (change.change_type) {
        .add_node => {
            try node.addPeer(change.node_id);
        },
        .remove_node => {
            node.removePeer(change.node_id);
        },
        .promote_learner => {
            // Learner promotion would update peer flags
            // For now, learners are not tracked separately
        },
    }
}
