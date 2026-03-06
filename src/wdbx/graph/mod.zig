//! Owns explicit relationships.

const std = @import("std");

pub const EdgeKind = enum {
    derived_from,
    summarized_by,
    contradicts,
    supports,
    authored_by,
    belongs_to_project,
    relevant_to_persona,
    references_artifact,
    follows_conversation_turn,
    supersedes,
};

pub const Edge = struct {
    kind: EdgeKind,
    target: [32]u8, // BlockId.id
    weight: f32 = 1.0,
};

pub const GraphStore = struct {
    // Adjacency list: Source BlockId -> List of Edges
    edges: std.AutoHashMapUnmanaged([32]u8, std.ArrayListUnmanaged(Edge)) = .empty,
    // Reverse adjacency list for back-traversal
    incoming: std.AutoHashMapUnmanaged([32]u8, std.ArrayListUnmanaged(Edge)) = .empty,

    pub fn deinit(self: *GraphStore, allocator: std.mem.Allocator) void {
        var it = self.edges.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        self.edges.deinit(allocator);

        var it2 = self.incoming.valueIterator();
        while (it2.next()) |list| {
            list.deinit(allocator);
        }
        self.incoming.deinit(allocator);
    }

    pub fn addEdge(self: *GraphStore, allocator: std.mem.Allocator, source: [32]u8, target: [32]u8, kind: EdgeKind) !void {
        {
            const gop = try self.edges.getOrPut(allocator, source);
            if (!gop.found_existing) {
                gop.value_ptr.* = .empty;
            }
            try gop.value_ptr.append(allocator, .{ .kind = kind, .target = target });
        }
        {
            // Reverse edge
            const gop = try self.incoming.getOrPut(allocator, target);
            if (!gop.found_existing) {
                gop.value_ptr.* = .empty;
            }
            try gop.value_ptr.append(allocator, .{ .kind = kind, .target = source });
        }
    }

    pub fn hasEdge(self: GraphStore, source: [32]u8, target: [32]u8) bool {
        const list = self.edges.get(source) orelse return false;
        for (list.items) |edge| {
            if (std.mem.eql(u8, &edge.target, &target)) return true;
        }
        return false;
    }

    pub fn removeEdge(self: *GraphStore, source: [32]u8, target: [32]u8) void {
        // Forward
        if (self.edges.getPtr(source)) |list| {
            var i: usize = 0;
            while (i < list.items.len) {
                if (std.mem.eql(u8, &list.items[i].target, &target)) {
                    _ = list.swapRemove(i);
                } else {
                    i += 1;
                }
            }
        }
        // Reverse
        if (self.incoming.getPtr(target)) |list| {
            var i: usize = 0;
            while (i < list.items.len) {
                if (std.mem.eql(u8, &list.items[i].target, &source)) {
                    _ = list.swapRemove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    pub fn scorePath(self: GraphStore, path: []const [32]u8) f32 {
        var total: f32 = 0;
        if (path.len < 2) return 0;
        for (0..path.len - 1) |i| {
            const list = self.edges.get(path[i]) orelse continue;
            for (list.items) |edge| {
                if (std.mem.eql(u8, &edge.target, &path[i + 1])) {
                    total += edge.weight;
                }
            }
        }
        return total;
    }
};
