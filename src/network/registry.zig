//! Node registry for tracking distributed compute cluster members.
//!
//! Maintains a list of cluster nodes with health status, addresses, and last-seen
//! timestamps for network coordination.

const std = @import("std");

const time = @import("../shared/utils.zig");

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

// ============================================================================
// Tests
// ============================================================================

test "node registry init and deinit" {
    const allocator = std.testing.allocator;
    var registry = NodeRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.nodes.items.len);
}

test "node registry register" {
    const allocator = std.testing.allocator;
    var registry = NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("node-1", "192.168.1.1:8080");
    try std.testing.expectEqual(@as(usize, 1), registry.nodes.items.len);
    try std.testing.expectEqualStrings("node-1", registry.nodes.items[0].id);
    try std.testing.expectEqual(NodeStatus.healthy, registry.nodes.items[0].status);
}

test "node registry unregister" {
    const allocator = std.testing.allocator;
    var registry = NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("node-1", "192.168.1.1:8080");
    try std.testing.expectEqual(@as(usize, 1), registry.nodes.items.len);

    const removed = registry.unregister("node-1");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 0), registry.nodes.items.len);

    const not_found = registry.unregister("nonexistent");
    try std.testing.expect(!not_found);
}

test "node registry setStatus" {
    const allocator = std.testing.allocator;
    var registry = NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("node-1", "192.168.1.1:8080");
    try std.testing.expectEqual(NodeStatus.healthy, registry.nodes.items[0].status);

    const updated = registry.setStatus("node-1", .degraded);
    try std.testing.expect(updated);
    try std.testing.expectEqual(NodeStatus.degraded, registry.nodes.items[0].status);

    const not_found = registry.setStatus("nonexistent", .offline);
    try std.testing.expect(!not_found);
}

test "node registry list" {
    const allocator = std.testing.allocator;
    var registry = NodeRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("node-1", "192.168.1.1:8080");
    try registry.register("node-2", "192.168.1.2:8080");

    const nodes = registry.list();
    try std.testing.expectEqual(@as(usize, 2), nodes.len);
}
