//! Cluster Bootstrap & Membership — Auto-join and node management for WDBX.
//!
//! Handles cluster formation, node discovery, health checking, and role
//! assignment. Nodes can be primary (accept writes), replica (serve reads),
//! or observer (monitoring only).
//!
//! Transport abstraction supports TCP (default), TLS (WAN), and local
//! high-speed paths (Thunderbolt/shared memory on macOS).

const std = @import("std");
const shard_manager = @import("shard_manager.zig");

// ============================================================================
// Types
// ============================================================================

pub const NodeRole = enum {
    primary, // Accept writes, coordinate replication
    replica, // Serve reads, receive replicated data
    observer, // Monitor only, no data
};

pub const NodeState = enum {
    joining, // Discovering cluster, syncing state
    active, // Fully operational
    draining, // Preparing to leave, transferring shards
    failed, // Health check failed, pending recovery
    removed, // Decommissioned
};

pub const TransportType = enum {
    tcp, // Default, works everywhere
    tls, // TLS-encrypted TCP for WAN
    thunderbolt, // Local high-speed (macOS/Windows)
    auto, // Auto-detect best available
};

pub const NodeInfo = struct {
    node_id: u64,
    address: [64]u8,
    address_len: u8,
    port: u16,
    role: NodeRole,
    state: NodeState,
    transport: TransportType,
    last_heartbeat: i64,
    shard_count: u32,
    vector_count: u64,
};

pub const ClusterConfig = struct {
    node_id: u64 = 0,
    listen_port: u16 = 9200,
    transport: TransportType = .tcp,
    replication_factor: u8 = 3,
    heartbeat_interval_ms: u32 = 1000,
    failure_timeout_ms: u32 = 5000,
    auto_rebalance: bool = true,
    max_nodes: u16 = 256,
    // Bootstrap peers (comma-separated "host:port" list)
    bootstrap_peers: [512]u8 = [_]u8{0} ** 512,
    bootstrap_peers_len: u16 = 0,
};

pub const ClusterStatus = struct {
    node_count: u16,
    active_nodes: u16,
    total_shards: u32,
    total_vectors: u64,
    replication_health: f32, // 0.0–1.0
    leader_id: u64,
    self_role: NodeRole,
};

// ============================================================================
// Cluster Manager
// ============================================================================

pub const ClusterManager = struct {
    allocator: std.mem.Allocator,
    config: ClusterConfig,
    self_info: NodeInfo,
    peers: std.ArrayListUnmanaged(NodeInfo),
    started: bool,
    leader_id: u64,

    pub fn init(allocator: std.mem.Allocator, config: ClusterConfig) ClusterManager {
        var self_info: NodeInfo = undefined;
        self_info.node_id = config.node_id;
        self_info.address = [_]u8{0} ** 64;
        self_info.address_len = 0;
        self_info.port = config.listen_port;
        self_info.role = .primary; // Default until cluster negotiation
        self_info.state = .joining;
        self_info.transport = config.transport;
        self_info.last_heartbeat = 0;
        self_info.shard_count = 0;
        self_info.vector_count = 0;

        return .{
            .allocator = allocator,
            .config = config,
            .self_info = self_info,
            .peers = .empty,
            .started = false,
            .leader_id = 0,
        };
    }

    pub fn deinit(self: *ClusterManager) void {
        self.peers.deinit(self.allocator);
    }

    /// Start the cluster manager — begin discovery and join process.
    pub fn start(self: *ClusterManager) !void {
        if (self.started) return;

        // If no bootstrap peers, start as standalone primary
        if (self.config.bootstrap_peers_len == 0) {
            self.self_info.state = .active;
            self.self_info.role = .primary;
            self.leader_id = self.config.node_id;
            self.started = true;
            return;
        }

        // Parse bootstrap peers and initiate join
        try self.parseBootstrapPeers();
        self.self_info.state = .joining;
        self.started = true;
    }

    /// Stop the cluster manager — drain and leave gracefully.
    pub fn stop(self: *ClusterManager) void {
        if (!self.started) return;
        self.self_info.state = .draining;
        // In production: transfer shards, notify peers, wait for confirmation
        self.self_info.state = .removed;
        self.started = false;
    }

    /// Process a heartbeat from a peer.
    pub fn onHeartbeat(self: *ClusterManager, node_id: u64, vector_count: u64) void {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const now: i64 = @intCast(ts.sec);

        for (self.peers.items) |*peer| {
            if (peer.node_id == node_id) {
                peer.last_heartbeat = now;
                peer.vector_count = vector_count;
                if (peer.state == .failed) peer.state = .active;
                return;
            }
        }
    }

    /// Check for failed nodes (no heartbeat within timeout).
    pub fn checkHealth(self: *ClusterManager) void {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const now: i64 = @intCast(ts.sec);
        const timeout_sec = @as(i64, @intCast(self.config.failure_timeout_ms / 1000));

        for (self.peers.items) |*peer| {
            if (peer.state == .active and peer.last_heartbeat > 0) {
                if (now - peer.last_heartbeat > timeout_sec) {
                    peer.state = .failed;
                }
            }
        }
    }

    /// Add a peer to the cluster.
    pub fn addPeer(self: *ClusterManager, info: NodeInfo) !void {
        // Check for duplicate
        for (self.peers.items) |peer| {
            if (peer.node_id == info.node_id) return;
        }
        try self.peers.append(self.allocator, info);
    }

    /// Remove a peer from the cluster.
    pub fn removePeer(self: *ClusterManager, node_id: u64) void {
        var i: usize = 0;
        while (i < self.peers.items.len) {
            if (self.peers.items[i].node_id == node_id) {
                _ = self.peers.swapRemove(i);
                return;
            }
            i += 1;
        }
    }

    /// Get current cluster status.
    pub fn getStatus(self: *const ClusterManager) ClusterStatus {
        var active: u16 = 0;
        var total_vecs: u64 = self.self_info.vector_count;
        var total_shards: u32 = self.self_info.shard_count;

        if (self.self_info.state == .active) active += 1;

        for (self.peers.items) |peer| {
            if (peer.state == .active) {
                active += 1;
                total_vecs += peer.vector_count;
                total_shards += peer.shard_count;
            }
        }

        const node_count: u16 = @intCast(self.peers.items.len + 1);
        const health: f32 = if (node_count > 0)
            @as(f32, @floatFromInt(active)) / @as(f32, @floatFromInt(node_count))
        else
            0;

        return .{
            .node_count = node_count,
            .active_nodes = active,
            .total_shards = total_shards,
            .total_vectors = total_vecs,
            .replication_health = health,
            .leader_id = self.leader_id,
            .self_role = self.self_info.role,
        };
    }

    /// Get active node count.
    pub fn activeNodeCount(self: *const ClusterManager) u16 {
        return self.getStatus().active_nodes;
    }

    // ================================================================
    // Internal
    // ================================================================

    /// Elect a leader: highest active node_id becomes leader.
    pub fn electLeader(self: *ClusterManager) void {
        var highest_id: u64 = self.self_info.node_id;

        // Consider self if active
        if (self.self_info.state != .active) {
            highest_id = 0;
        }

        for (self.peers.items) |peer| {
            if (peer.state == .active and peer.node_id > highest_id) {
                highest_id = peer.node_id;
            }
        }

        self.leader_id = highest_id;

        // Update self role based on leader election
        if (self.self_info.node_id == highest_id) {
            self.self_info.role = .primary;
        } else {
            self.self_info.role = .replica;
        }
    }

    /// Create a heartbeat message with self_info.
    pub fn serializeHeartbeat(self: *const ClusterManager) !ClusterMessage {
        var msg = ClusterMessage{
            .msg_type = .heartbeat,
            .sender_id = self.self_info.node_id,
        };

        // Pack vector_count into payload as little-endian u64
        std.mem.writeInt(u64, msg.payload[0..8], self.self_info.vector_count, .little);
        // Pack shard_count as u32
        std.mem.writeInt(u32, msg.payload[8..12], self.self_info.shard_count, .little);
        msg.payload_len = 12;

        return msg;
    }

    fn parseBootstrapPeers(self: *ClusterManager) !void {
        const peers_str = self.config.bootstrap_peers[0..self.config.bootstrap_peers_len];
        var iter = std.mem.splitScalar(u8, peers_str, ',');
        var peer_id: u64 = 1000;

        while (iter.next()) |peer_addr| {
            const trimmed = std.mem.trim(u8, peer_addr, " ");
            if (trimmed.len == 0) continue;

            var info: NodeInfo = undefined;
            info.node_id = peer_id;
            info.address = [_]u8{0} ** 64;
            const copy_len = @min(trimmed.len, 64);
            @memcpy(info.address[0..copy_len], trimmed[0..copy_len]);
            info.address_len = @intCast(copy_len);
            info.port = 9200;
            info.role = .replica;
            info.state = .joining;
            info.transport = self.config.transport;
            info.last_heartbeat = 0;
            info.shard_count = 0;
            info.vector_count = 0;

            try self.peers.append(self.allocator, info);
            peer_id += 1;
        }
    }
};

// ============================================================================
// Error types
// ============================================================================

pub const ClusterError = error{
    BufferTooSmall,
    InvalidMessage,
    PeerNotFound,
};

// ============================================================================
// Wire protocol types
// ============================================================================

/// Message types for cluster communication protocol.
pub const MessageType = enum(u8) {
    heartbeat = 1,
    join_request = 2,
    join_response = 3,
    shard_transfer = 4,
    leader_announce = 5,
};

/// Wire message for cluster communication.
pub const ClusterMessage = struct {
    msg_type: MessageType,
    sender_id: u64,
    payload: [1024]u8 = [_]u8{0} ** 1024,
    payload_len: u32 = 0,

    pub fn serialize(self: *const ClusterMessage, buffer: []u8) !usize {
        if (buffer.len < 13) return error.BufferTooSmall;
        buffer[0] = @intFromEnum(self.msg_type);
        std.mem.writeInt(u64, buffer[1..9], self.sender_id, .little);
        std.mem.writeInt(u32, buffer[9..13], self.payload_len, .little);
        const total = 13 + self.payload_len;
        if (buffer.len < total) return error.BufferTooSmall;
        if (self.payload_len > 0) {
            @memcpy(buffer[13..total], self.payload[0..self.payload_len]);
        }
        return total;
    }

    pub fn deserialize(buffer: []const u8) !ClusterMessage {
        if (buffer.len < 13) return error.BufferTooSmall;
        var msg = ClusterMessage{
            .msg_type = @enumFromInt(buffer[0]),
            .sender_id = std.mem.readInt(u64, buffer[1..9], .little),
            .payload_len = std.mem.readInt(u32, buffer[9..13], .little),
        };
        if (msg.payload_len > 0 and buffer.len >= 13 + msg.payload_len) {
            @memcpy(msg.payload[0..msg.payload_len], buffer[13 .. 13 + msg.payload_len]);
        }
        return msg;
    }
};

/// Parsed peer address from bootstrap config.
pub const PeerAddress = struct {
    host: [64]u8 = [_]u8{0} ** 64,
    host_len: u8 = 0,
    port: u16 = 9200,

    pub fn fromString(addr_str: []const u8) PeerAddress {
        var result = PeerAddress{};
        // Try to parse "host:port" format
        if (std.mem.indexOfScalar(u8, addr_str, ':')) |colon_idx| {
            const host_part = addr_str[0..colon_idx];
            const port_part = addr_str[colon_idx + 1 ..];
            result.host_len = @intCast(@min(host_part.len, 64));
            @memcpy(result.host[0..result.host_len], host_part[0..result.host_len]);
            result.port = std.fmt.parseInt(u16, port_part, 10) catch 9200;
        } else {
            result.host_len = @intCast(@min(addr_str.len, 64));
            @memcpy(result.host[0..result.host_len], addr_str[0..result.host_len]);
        }
        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ClusterManager standalone init" {
    const allocator = std.testing.allocator;
    var cm = ClusterManager.init(allocator, .{ .node_id = 1 });
    defer cm.deinit();

    try cm.start();
    try std.testing.expect(cm.started);
    try std.testing.expectEqual(NodeState.active, cm.self_info.state);
    try std.testing.expectEqual(NodeRole.primary, cm.self_info.role);
}

test "ClusterManager status" {
    const allocator = std.testing.allocator;
    var cm = ClusterManager.init(allocator, .{ .node_id = 1 });
    defer cm.deinit();

    try cm.start();
    const status = cm.getStatus();
    try std.testing.expectEqual(@as(u16, 1), status.node_count);
    try std.testing.expectEqual(@as(u16, 1), status.active_nodes);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), status.replication_health, 0.01);
}

test "ClusterManager add/remove peers" {
    const allocator = std.testing.allocator;
    var cm = ClusterManager.init(allocator, .{ .node_id = 1 });
    defer cm.deinit();

    var peer: NodeInfo = undefined;
    peer.node_id = 2;
    peer.address = [_]u8{0} ** 64;
    peer.address_len = 0;
    peer.port = 9200;
    peer.role = .replica;
    peer.state = .active;
    peer.transport = .tcp;
    peer.last_heartbeat = 0;
    peer.shard_count = 0;
    peer.vector_count = 100;

    try cm.addPeer(peer);
    try std.testing.expectEqual(@as(usize, 1), cm.peers.items.len);

    cm.removePeer(2);
    try std.testing.expectEqual(@as(usize, 0), cm.peers.items.len);
}

test "ClusterMessage serialize/deserialize round-trip" {
    var original = ClusterMessage{
        .msg_type = .heartbeat,
        .sender_id = 42,
    };
    original.payload[0] = 0xAB;
    original.payload[1] = 0xCD;
    original.payload_len = 2;

    var buffer: [2048]u8 = undefined;
    const written = try original.serialize(&buffer);
    try std.testing.expectEqual(@as(usize, 15), written); // 13 header + 2 payload

    const decoded = try ClusterMessage.deserialize(buffer[0..written]);
    try std.testing.expectEqual(MessageType.heartbeat, decoded.msg_type);
    try std.testing.expectEqual(@as(u64, 42), decoded.sender_id);
    try std.testing.expectEqual(@as(u32, 2), decoded.payload_len);
    try std.testing.expectEqual(@as(u8, 0xAB), decoded.payload[0]);
    try std.testing.expectEqual(@as(u8, 0xCD), decoded.payload[1]);
}

test "PeerAddress.fromString parsing" {
    const addr = PeerAddress.fromString("192.168.1.1:8080");
    try std.testing.expectEqual(@as(u16, 8080), addr.port);
    try std.testing.expectEqual(@as(u8, 13), addr.host_len);
    try std.testing.expect(std.mem.eql(u8, "192.168.1.1", addr.host[0..11]));
}

test "PeerAddress without port defaults to 9200" {
    const addr = PeerAddress.fromString("myhost");
    try std.testing.expectEqual(@as(u16, 9200), addr.port);
    try std.testing.expectEqual(@as(u8, 6), addr.host_len);
    try std.testing.expect(std.mem.eql(u8, "myhost", addr.host[0..6]));
}

test "ClusterManager electLeader selects highest node_id" {
    const allocator = std.testing.allocator;
    var cm = ClusterManager.init(allocator, .{ .node_id = 5 });
    defer cm.deinit();

    try cm.start();

    // Add peers with higher IDs
    var peer1: NodeInfo = undefined;
    peer1.node_id = 10;
    peer1.address = [_]u8{0} ** 64;
    peer1.address_len = 0;
    peer1.port = 9200;
    peer1.role = .replica;
    peer1.state = .active;
    peer1.transport = .tcp;
    peer1.last_heartbeat = 0;
    peer1.shard_count = 0;
    peer1.vector_count = 0;

    var peer2: NodeInfo = undefined;
    peer2.node_id = 20;
    peer2.address = [_]u8{0} ** 64;
    peer2.address_len = 0;
    peer2.port = 9201;
    peer2.role = .replica;
    peer2.state = .active;
    peer2.transport = .tcp;
    peer2.last_heartbeat = 0;
    peer2.shard_count = 0;
    peer2.vector_count = 0;

    try cm.addPeer(peer1);
    try cm.addPeer(peer2);

    cm.electLeader();
    try std.testing.expectEqual(@as(u64, 20), cm.leader_id);
    // Self (node_id=5) should be replica since 20 > 5
    try std.testing.expectEqual(NodeRole.replica, cm.self_info.role);
}

test "ClusterManager serializeHeartbeat creates valid message" {
    const allocator = std.testing.allocator;
    var cm = ClusterManager.init(allocator, .{ .node_id = 7 });
    defer cm.deinit();
    cm.self_info.vector_count = 1000;
    cm.self_info.shard_count = 4;

    const msg = try cm.serializeHeartbeat();
    try std.testing.expectEqual(MessageType.heartbeat, msg.msg_type);
    try std.testing.expectEqual(@as(u64, 7), msg.sender_id);
    try std.testing.expectEqual(@as(u32, 12), msg.payload_len);

    // Verify payload contents
    const vec_count = std.mem.readInt(u64, msg.payload[0..8], .little);
    const shard_count = std.mem.readInt(u32, msg.payload[8..12], .little);
    try std.testing.expectEqual(@as(u64, 1000), vec_count);
    try std.testing.expectEqual(@as(u32, 4), shard_count);
}
