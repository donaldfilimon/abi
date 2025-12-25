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
};
