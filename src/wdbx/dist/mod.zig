//! Allows growth from single-node to clustered deployment.

const std = @import("std");

pub const NodeState = struct {
    id: u32,
    address: []const u8,
    is_healthy: bool = true,
    last_seen: i64 = 0,
    load_factor: f32 = 0.0,
    
    // FIXME: implement heartbeat and health-check state machine
};

pub const Coordinator = struct {
    nodes: std.ArrayListUnmanaged(NodeState) = .empty,
    shard_map: std.AutoHashMapUnmanaged(u32, u32) = .empty, // shard_id -> node_id

    pub fn deinit(self: *Coordinator, allocator: std.mem.Allocator) void {
        for (self.nodes.items) |node| {
            allocator.free(node.address);
        }
        self.nodes.deinit(allocator);
        self.shard_map.deinit(allocator);
    }

    pub fn registerNode(self: *Coordinator, allocator: std.mem.Allocator, address: []const u8) !u32 {
        const id = @as(u32, @truncate(self.nodes.items.len));
        try self.nodes.append(allocator, .{
            .id = id,
            .address = try allocator.dupe(u8, address),
            .last_seen = std.time.timestamp(),
        });
        return id;
    }

    pub fn updateHeartbeat(self: *Coordinator, node_id: u32, load_factor: f32) void {
        if (node_id < self.nodes.items.len) {
            var node = &self.nodes.items[node_id];
            node.last_seen = std.time.timestamp();
            node.load_factor = load_factor;
            node.is_healthy = true;
        }
    }

    // FIXME: implement shard ownership, replication protocols, and query fan-out
};
