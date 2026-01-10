//! Node registry for tracking distributed compute cluster members.
//!
//! Maintains a list of cluster nodes with health status, addresses, and last-seen
//! timestamps for network coordination.

const std = @import("std");

const time = @import("../../shared/utils/time.zig");

pub const NodeStatus = enum {
    healthy,
    degraded,
    offline,
};

pub const NodeInfo = struct {
    id: []const u8,
    address: []const u8,
    status: NodeStatus = .healthy,
    last_seen_ms: i64,
};

pub const NodeRegistry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .{},
    /// O(1) lookup index: node id string -> array index
    id_index: std.StringHashMapUnmanaged(usize) = .{},

    pub fn init(allocator: std.mem.Allocator) NodeRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *NodeRegistry) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.id);
            self.allocator.free(node.address);
        }
        self.nodes.deinit(self.allocator);
        self.id_index.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn register(self: *NodeRegistry, id: []const u8, address: []const u8) !void {
        if (self.findIndex(id)) |index| {
            var node = &self.nodes.items[index];
            if (!std.mem.eql(u8, node.address, address)) {
                self.allocator.free(node.address);
                node.address = try self.allocator.dupe(u8, address);
            }
            node.last_seen_ms = time.nowMilliseconds();
            node.status = .healthy;
            return;
        }

        const new_index = self.nodes.items.len;
        const id_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_copy);
        try self.nodes.append(self.allocator, .{
            .id = id_copy,
            .address = try self.allocator.dupe(u8, address),
            .status = .healthy,
            .last_seen_ms = time.nowMilliseconds(),
        });
        // Maintain O(1) lookup index - use the id stored in nodes (already allocated)
        try self.id_index.put(self.allocator, self.nodes.items[new_index].id, new_index);
    }

    pub fn unregister(self: *NodeRegistry, id: []const u8) bool {
        const index = self.findIndex(id) orelse return false;
        const node = self.nodes.swapRemove(index);
        // Remove from O(1) index
        _ = self.id_index.remove(node.id);
        // If swapRemove moved the last element to fill the gap, update its index
        if (index < self.nodes.items.len) {
            const moved_id = self.nodes.items[index].id;
            self.id_index.putAssumeCapacity(moved_id, index);
        }
        self.allocator.free(node.id);
        self.allocator.free(node.address);
        return true;
    }

    pub fn touch(self: *NodeRegistry, id: []const u8) bool {
        const index = self.findIndex(id) orelse return false;
        self.nodes.items[index].last_seen_ms = time.nowMilliseconds();
        return true;
    }

    pub fn setStatus(self: *NodeRegistry, id: []const u8, status: NodeStatus) bool {
        const index = self.findIndex(id) orelse return false;
        self.nodes.items[index].status = status;
        self.nodes.items[index].last_seen_ms = time.nowMilliseconds();
        return true;
    }

    pub fn list(self: *NodeRegistry) []const NodeInfo {
        return self.nodes.items;
    }

    /// O(1) lookup using hash index instead of O(n) linear scan
    fn findIndex(self: *NodeRegistry, id: []const u8) ?usize {
        return self.id_index.get(id);
    }
};
