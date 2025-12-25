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

    pub fn init(allocator: std.mem.Allocator) NodeRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *NodeRegistry) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.id);
            self.allocator.free(node.address);
        }
        self.nodes.deinit(self.allocator);
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

        try self.nodes.append(self.allocator, .{
            .id = try self.allocator.dupe(u8, id),
            .address = try self.allocator.dupe(u8, address),
            .status = .healthy,
            .last_seen_ms = time.nowMilliseconds(),
        });
    }

    pub fn unregister(self: *NodeRegistry, id: []const u8) bool {
        const index = self.findIndex(id) orelse return false;
        const node = self.nodes.swapRemove(index);
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

    fn findIndex(self: *NodeRegistry, id: []const u8) ?usize {
        for (self.nodes.items, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, id)) return i;
        }
        return null;
    }
};
