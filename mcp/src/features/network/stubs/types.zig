const std = @import("std");

pub const Error = error{
    NetworkDisabled,
    ConnectionFailed,
    NodeNotFound,
    ConsensusFailed,
    Timeout,
};

pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};

pub const NetworkState = enum { disconnected, connected };
pub const Node = struct {};
pub const NodeStatus = enum { healthy, degraded, offline };

pub const NodeInfo = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    status: NodeStatus = .healthy,
    last_seen_ms: i64 = 0,
};

pub const NodeRegistry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .empty,
    id_index: std.StringHashMapUnmanaged(usize) = .empty,

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

    pub fn register(self: *NodeRegistry, id: []const u8, address: []const u8) Error!void {
        if (self.findIndex(id)) |index| {
            var node = &self.nodes.items[index];
            if (!std.mem.eql(u8, node.address, address)) {
                self.allocator.free(node.address);
                node.address = self.allocator.dupe(u8, address) catch return error.ConnectionFailed;
            }
            node.last_seen_ms += 1;
            node.status = .healthy;
            return;
        }

        const id_copy = self.allocator.dupe(u8, id) catch return error.ConnectionFailed;
        errdefer self.allocator.free(id_copy);
        const address_copy = self.allocator.dupe(u8, address) catch return error.ConnectionFailed;
        errdefer self.allocator.free(address_copy);

        self.nodes.append(self.allocator, .{
            .id = id_copy,
            .address = address_copy,
            .status = .healthy,
            .last_seen_ms = 1,
        }) catch return error.ConnectionFailed;
        self.id_index.put(self.allocator, id_copy, self.nodes.items.len - 1) catch return error.ConnectionFailed;
    }

    pub fn unregister(self: *NodeRegistry, id: []const u8) bool {
        const index = self.findIndex(id) orelse return false;
        const node = self.nodes.swapRemove(index);
        _ = self.id_index.remove(node.id);
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
        self.nodes.items[index].last_seen_ms += 1;
        return true;
    }

    pub fn setStatus(self: *NodeRegistry, id: []const u8, status: NodeStatus) bool {
        const index = self.findIndex(id) orelse return false;
        self.nodes.items[index].status = status;
        self.nodes.items[index].last_seen_ms += 1;
        return true;
    }

    pub fn list(self: *NodeRegistry) []const NodeInfo {
        return self.nodes.items;
    }

    fn findIndex(self: *NodeRegistry, id: []const u8) ?usize {
        return self.id_index.get(id);
    }
};

test {
    std.testing.refAllDecls(@This());
}
