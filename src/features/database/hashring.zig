//! Consistent hashing ring for distributed database sharding.
const std = @import("std");
const crypto = std.crypto;
const mem = std.mem;

pub const HashRingError = error{
    EmptyRing,
    NoReplicas,
    InvalidWeight,
};

pub const NodeWeight = struct {
    node_id: []const u8,
    weight: u32,
    virtual_nodes: u32,
};

pub const RingNode = struct {
    node_id: []const u8,
    hash: u128,
    weight: u32,
    owner: *const HashRing,
};

pub const HashRing = struct {
    allocator: mem.Allocator,
    nodes: std.ArrayListUnmanaged(RingNode),
    sorted_hashes: std.ArrayListUnmanaged(u128),
    total_weight: u32,
    replica_count: u32,
    version: u64,

    const virtual_node_multiplier = 100;

    pub fn init(allocator: mem.Allocator, replica_count: u32) HashRing {
        return .{
            .allocator = allocator,
            .nodes = std.ArrayListUnmanaged(RingNode).empty,
            .sorted_hashes = std.ArrayListUnmanaged(u128).empty,
            .total_weight = 0,
            .replica_count = replica_count,
            .version = 0,
        };
    }

    pub fn deinit(self: *HashRing) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.node_id);
        }
        self.nodes.deinit(self.allocator);
        self.sorted_hashes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addNode(self: *HashRing, node_id: []const u8, weight: u32) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);

        const virtual_nodes = weight * virtual_node_multiplier;
        const node = RingNode{
            .node_id = id_copy,
            .hash = 0,
            .weight = weight,
            .owner = self,
        };

        try self.nodes.append(self.allocator, node);
        self.total_weight += weight;

        var i: u32 = 0;
        while (i < virtual_nodes) : (i += 1) {
            const vnode_key = try self.allocator.alloc(u8, node_id.len + 8);
            defer self.allocator.free(vnode_key);
            @memcpy(vnode_key[0..node_id.len], node_id);
            mem.writeInt(u64, vnode_key[node_id.len..], i, .big);
            const vhash = hashKey(vnode_key);
            try self.sorted_hashes.append(self.allocator, vhash);
        }

        self.rebuildSortedHashes();
        self.version += 1;
    }

    pub fn removeNode(self: *HashRing, node_id: []const u8) bool {
        var removed = false;
        var i: usize = 0;
        while (i < self.nodes.items.len) {
            if (mem.eql(u8, self.nodes.items[i].node_id, node_id)) {
                self.allocator.free(self.nodes.items[i].node_id);
                _ = self.nodes.swapRemove(i);
                removed = true;
                break;
            } else {
                i += 1;
            }
        }

        if (removed) {
            self.rebuildSortedHashes();
            self.total_weight = 0;
            for (self.nodes.items) |node| {
                self.total_weight += node.weight;
            }
            self.version += 1;
        }

        return removed;
    }

    pub fn getNode(self: *const HashRing, key: []const u8) !?[]const u8 {
        if (self.sorted_hashes.items.len == 0) return HashRingError.EmptyRing;
        if (self.nodes.items.len == 0) return HashRingError.NoReplicas;

        const key_hash = hashKey(key);
        const idx = self.findPosition(key_hash);
        const node_hash = self.sorted_hashes.items[idx];
        return self.getNodeByHash(node_hash);
    }

    pub fn getNodes(
        self: *const HashRing,
        key: []const u8,
        count: usize,
    ) ![]const []const u8 {
        if (self.sorted_hashes.items.len == 0) return HashRingError.EmptyRing;
        if (self.nodes.items.len == 0) return HashRingError.NoReplicas;

        const key_hash = hashKey(key);
        const start_idx = self.findPosition(key_hash);

        const result = try self.allocator.alloc([]const u8, count);
        errdefer self.allocator.free(result);

        const seen = try self.allocator.alloc(bool, self.nodes.items.len);
        defer self.allocator.free(seen);
        @memset(seen, false);

        var added: usize = 0;
        var idx = start_idx;
        while (added < count) {
            const node_hash = self.sorted_hashes.items[idx % self.sorted_hashes.items.len];
            if (self.getNodeByHash(node_hash)) |node_id| {
                const node_idx = self.findNodeIndex(node_id);
                if (node_idx < seen.len and !seen[node_idx]) {
                    seen[node_idx] = true;
                    result[added] = node_id;
                    added += 1;
                }
            }
            idx += 1;
            if (idx >= self.sorted_hashes.items.len) idx = 0;
            if (idx == start_idx) break;
        }

        if (added < count) {
            self.allocator.free(result);
            return error.NotEnoughNodes;
        }

        return result;
    }

    fn findPosition(self: *const HashRing, hash: u128) usize {
        const hashes = self.sorted_hashes.items;
        var low: usize = 0;
        var high = hashes.len;

        while (low < high) {
            const mid = (low + high) / 2;
            if (hashes[mid] <= hash) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        if (low >= hashes.len) return 0;
        return low;
    }

    fn getNodeByHash(self: *const HashRing, hash: u128) ?[]const u8 {
        for (self.nodes.items) |node| {
            const vnode_start = hashNode(node.node_id, 0);
            const vnode_end = hashNode(node.node_id, node.weight * virtual_node_multiplier);
            if (hash >= vnode_start and hash < vnode_end) {
                return node.node_id;
            }
        }
        return null;
    }

    fn findNodeIndex(self: *const HashRing, node_id: []const u8) usize {
        for (self.nodes.items, 0..) |node, i| {
            if (mem.eql(u8, node.node_id, node_id)) return i;
        }
        return self.nodes.items.len;
    }

    fn hashNode(node_id: []const u8, suffix: u32) u128 {
        var key_buf: [256]u8 = undefined;
        const key = if (node_id.len + 4 <= key_buf.len) blk: {
            @memcpy(key_buf[0..node_id.len], node_id);
            mem.writeInt(u32, key_buf[node_id.len .. node_id.len + 4], suffix, .big);
            break :blk key_buf[0 .. node_id.len + 4];
        } else node_id;

        return hashKey(key);
    }

    fn rebuildSortedHashes(self: *HashRing) void {
        self.sorted_hashes.clearRetainingCapacity();

        for (self.nodes.items) |node| {
            const virtual_nodes = node.weight * virtual_node_multiplier;
            var i: u32 = 0;
            while (i < virtual_nodes) : (i += 1) {
                const vhash = hashNode(node.node_id, i);
                const insert_pos = self.findInsertionPosition(vhash);
                self.sorted_hashes.insertAssumeCapacity(insert_pos, vhash);
            }
        }
    }

    fn findInsertionPosition(self: *const HashRing, hash: u128) usize {
        const hashes = self.sorted_hashes.items;
        var low: usize = 0;
        var high = hashes.len;

        while (low < high) {
            const mid = (low + high) / 2;
            if (hashes[mid] <= hash) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        return low;
    }

    pub fn getStats(self: *const HashRing) RingStats {
        return .{
            .node_count = self.nodes.items.len,
            .virtual_node_count = self.sorted_hashes.items.len,
            .total_weight = self.total_weight,
            .version = self.version,
        };
    }
};

pub const RingStats = struct {
    node_count: usize,
    virtual_node_count: usize,
    total_weight: u32,
    version: u64,
};

fn hashKey(key: []const u8) u128 {
    var state: crypto.hash.sha2.Sha256 = .{};
    state.update(key);
    var result: [32]u8 = undefined;
    state.final(result[0..]);
    return mem.readInt(u128, result[0..16], .big);
}

test "hash ring add and lookup" {
    const allocator = std.testing.allocator;
    var ring = HashRing.init(allocator, 1);
    defer ring.deinit();

    try ring.addNode("node1", 100);
    try ring.addNode("node2", 100);
    try ring.addNode("node3", 100);

    const node = try ring.getNode("key1");
    try std.testing.expect(node != null);
}

test "hash ring remove" {
    const allocator = std.testing.allocator;
    var ring = HashRing.init(allocator, 1);
    defer ring.deinit();

    try ring.addNode("node1", 100);
    try ring.addNode("node2", 100);

    const removed = ring.removeNode("node1");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 1), ring.nodes.items.len);
}

test "hash ring get multiple nodes" {
    const allocator = std.testing.allocator;
    var ring = HashRing.init(allocator, 1);
    defer ring.deinit();

    try ring.addNode("node1", 100);
    try ring.addNode("node2", 100);
    try ring.addNode("node3", 100);

    const nodes = try ring.getNodes("key1", 2);
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 2), nodes.len);
    try std.testing.expect(!mem.eql(u8, nodes[0], nodes[1]));
}
