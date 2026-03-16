//! Raft Persistence Layer
//!
//! Provides durable state storage for Raft consensus nodes.
//! Handles saving and loading of persistent state (current term, voted_for)
//! and log entries to/from disk.

const std = @import("std");
const raft = @import("raft.zig");
const RaftNode = raft.RaftNode;
const LogEntry = raft.LogEntry;
const initIoBackend = raft.initIoBackend;

/// Persistent state that must survive restarts.
pub const PersistentState = struct {
    /// Current term.
    current_term: u64,
    /// Candidate that received vote in current term (null-terminated).
    voted_for_len: u32,
    voted_for: [256]u8,
    /// Number of log entries.
    log_count: u32,
};

/// Log entry header for persistence.
pub const PersistentLogEntry = struct {
    term: u64,
    index: u64,
    entry_type: u8,
    data_len: u32,
};

/// File format magic number.
const RAFT_MAGIC: u32 = 0x52414654; // "RAFT"
/// Current format version.
const RAFT_VERSION: u16 = 1;

/// Raft persistence manager for durable state storage.
pub const RaftPersistence = struct {
    allocator: std.mem.Allocator,
    path: []const u8,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !RaftPersistence {
        return .{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn deinit(self: *RaftPersistence) void {
        self.allocator.free(self.path);
    }

    /// Save Raft node state to disk.
    pub fn save(self: *RaftPersistence, node: *const RaftNode) !void {
        // Calculate total size needed
        var total_size: usize = 4 + 2 + @sizeOf(PersistentState); // magic + version + state
        for (node.log.items) |entry| {
            total_size += @sizeOf(PersistentLogEntry) + entry.data.len;
        }

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);

        var offset: usize = 0;

        // Write magic
        std.mem.writeInt(u32, buffer[offset..][0..4], RAFT_MAGIC, .little);
        offset += 4;

        // Write version
        std.mem.writeInt(u16, buffer[offset..][0..2], RAFT_VERSION, .little);
        offset += 2;

        // Write persistent state
        var state = PersistentState{
            .current_term = node.current_term,
            .voted_for_len = 0,
            .voted_for = undefined,
            .log_count = @intCast(node.log.items.len),
        };
        @memset(&state.voted_for, 0);
        if (node.voted_for) |vf| {
            const len = @min(vf.len, state.voted_for.len);
            @memcpy(state.voted_for[0..len], vf[0..len]);
            state.voted_for_len = @intCast(len);
        }

        const state_bytes = std.mem.asBytes(&state);
        @memcpy(buffer[offset..][0..state_bytes.len], state_bytes);
        offset += state_bytes.len;

        // Write log entries
        for (node.log.items) |entry| {
            const entry_header = PersistentLogEntry{
                .term = entry.term,
                .index = entry.index,
                .entry_type = @intFromEnum(entry.entry_type),
                .data_len = @intCast(entry.data.len),
            };
            const header_bytes = std.mem.asBytes(&entry_header);
            @memcpy(buffer[offset..][0..header_bytes.len], header_bytes);
            offset += header_bytes.len;

            @memcpy(buffer[offset..][0..entry.data.len], entry.data);
            offset += entry.data.len;
        }

        // Write to file using std.Io
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = try std.Io.Dir.cwd().createFile(io, self.path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, buffer[0..offset]);
    }

    /// Load Raft node state from disk.
    pub fn load(self: *RaftPersistence, node: *RaftNode) !void {
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const buffer = std.Io.Dir.cwd().readFileAlloc(
            io,
            self.path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            if (err == error.FileNotFound) return; // No state to load
            return err;
        };
        defer self.allocator.free(buffer);

        if (buffer.len < 6) return error.InvalidFormat;

        var offset: usize = 0;

        // Verify magic
        const magic = std.mem.readInt(u32, buffer[offset..][0..4], .little);
        if (magic != RAFT_MAGIC) return error.InvalidFormat;
        offset += 4;

        // Verify version
        const version = std.mem.readInt(u16, buffer[offset..][0..2], .little);
        if (version != RAFT_VERSION) return error.UnsupportedVersion;
        offset += 2;

        // Read persistent state
        if (buffer.len < offset + @sizeOf(PersistentState)) return error.InvalidFormat;
        const state: *const PersistentState = @ptrCast(@alignCast(buffer[offset..].ptr));
        offset += @sizeOf(PersistentState);

        // Apply state
        node.current_term = state.current_term;

        // Free old voted_for if any
        if (node.voted_for) |vf| {
            node.allocator.free(vf);
            node.voted_for = null;
        }
        if (state.voted_for_len > 0) {
            node.voted_for = try node.allocator.dupe(u8, state.voted_for[0..state.voted_for_len]);
        }

        // Clear existing log
        for (node.log.items) |entry| {
            node.allocator.free(entry.data);
        }
        node.log.clearRetainingCapacity();

        // Read log entries
        var i: u32 = 0;
        while (i < state.log_count) : (i += 1) {
            if (buffer.len < offset + @sizeOf(PersistentLogEntry)) return error.InvalidFormat;
            const entry_header: *const PersistentLogEntry = @ptrCast(@alignCast(buffer[offset..].ptr));
            offset += @sizeOf(PersistentLogEntry);

            if (buffer.len < offset + entry_header.data_len) return error.InvalidFormat;
            const data = try node.allocator.dupe(u8, buffer[offset..][0..entry_header.data_len]);
            errdefer node.allocator.free(data);
            offset += entry_header.data_len;

            try node.log.append(node.allocator, LogEntry{
                .term = entry_header.term,
                .index = entry_header.index,
                .entry_type = @enumFromInt(entry_header.entry_type),
                .data = data,
            });
        }
    }

    /// Check if persistence file exists.
    pub fn exists(self: *RaftPersistence) bool {
        var io_backend = initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        if (std.Io.Dir.cwd().openFile(io, self.path, .{})) |file| {
            file.close(io);
            return true;
        } else |_| {
            return false;
        }
    }

    pub const LoadError = error{
        InvalidFormat,
        UnsupportedVersion,
    } || std.mem.Allocator.Error || std.Io.Dir.ReadFileAllocError;
};

test {
    std.testing.refAllDecls(@This());
}
