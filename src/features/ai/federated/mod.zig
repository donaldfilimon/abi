const std = @import("std");
const time = @import("../../../shared/utils/time.zig");

pub const NodeInfo = struct {
    id: []const u8,
    last_update: i64,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .{},

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Registry) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.id);
        }
        self.nodes.deinit(self.allocator);
    }

    pub fn touch(self: *Registry, id: []const u8) !void {
        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, id)) {
                node.last_update = time.nowSeconds();
                return;
            }
        }
        const copy = try self.allocator.dupe(u8, id);
        try self.nodes.append(self.allocator, .{
            .id = copy,
            .last_update = time.nowSeconds(),
        });
    }

    pub fn remove(self: *Registry, id: []const u8) bool {
        for (self.nodes.items, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, id)) {
                const removed = self.nodes.swapRemove(i);
                self.allocator.free(removed.id);
                return true;
            }
        }
        return false;
    }

    pub fn list(self: *const Registry) []const NodeInfo {
        return self.nodes.items;
    }

    pub fn count(self: *const Registry) usize {
        return self.nodes.items.len;
    }

    pub fn prune(self: *Registry, max_age_seconds: i64) usize {
        if (max_age_seconds <= 0) return 0;
        const now = time.nowSeconds();
        var removed: usize = 0;
        var i: usize = 0;
        while (i < self.nodes.items.len) {
            const node = self.nodes.items[i];
            if (now - node.last_update > max_age_seconds) {
                const removed_node = self.nodes.swapRemove(i);
                self.allocator.free(removed_node.id);
                removed += 1;
                continue;
            }
            i += 1;
        }
        return removed;
    }
};

test "federated registry prune and remove" {
    var registry = Registry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.touch("node-a");
    try registry.touch("node-b");
    try std.testing.expectEqual(@as(usize, 2), registry.count());

    registry.nodes.items[0].last_update -= 120;
    const removed = registry.prune(60);
    try std.testing.expectEqual(@as(usize, 1), removed);
    try std.testing.expectEqual(@as(usize, 1), registry.count());

    try std.testing.expect(registry.remove("node-b"));
    try std.testing.expectEqual(@as(usize, 0), registry.count());
}
