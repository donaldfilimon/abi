//! Raft cluster management for testing.

const std = @import("std");
const node_mod = @import("node.zig");
const types = @import("types.zig");

pub const RaftNode = node_mod.RaftNode;
pub const RaftConfig = types.RaftConfig;

/// Create a Raft cluster for testing/simulation.
pub fn createCluster(allocator: std.mem.Allocator, node_ids: []const []const u8, config: RaftConfig) !std.ArrayListUnmanaged(*RaftNode) {
    var nodes = std.ArrayListUnmanaged(*RaftNode).empty;
    errdefer {
        for (nodes.items) |node| {
            node.deinit();
            allocator.destroy(node);
        }
        nodes.deinit(allocator);
    }

    // Create nodes
    for (node_ids) |id| {
        const node = try allocator.create(RaftNode);
        errdefer allocator.destroy(node);
        node.* = try RaftNode.init(allocator, id, config);
        try nodes.append(allocator, node);
    }

    // Connect peers
    for (nodes.items) |node| {
        for (node_ids) |peer_id| {
            if (!std.mem.eql(u8, peer_id, node.node_id)) {
                try node.addPeer(peer_id);
            }
        }
    }

    return nodes;
}
