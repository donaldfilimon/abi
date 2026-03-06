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

    pub fn deinit(self: *GraphStore, allocator: std.mem.Allocator) void {
        var it = self.edges.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        self.edges.deinit(allocator);
    }

    pub fn addEdge(self: *GraphStore, allocator: std.mem.Allocator, source: [32]u8, target: [32]u8, kind: EdgeKind) !void {
        const gop = try self.edges.getOrPut(allocator, source);
        if (!gop.found_existing) {
            gop.value_ptr.* = .empty;
        }
        try gop.value_ptr.append(allocator, .{ .kind = kind, .target = target });
    }

    pub fn hasEdge(self: GraphStore, source: [32]u8, target: [32]u8) bool {
        const list = self.edges.get(source) orelse return false;
        for (list.items) |edge| {
            if (std.mem.eql(u8, &edge.target, &target)) return true;
        }
        return false;
    }

    pub fn removeEdge(self: *GraphStore, source: [32]u8, target: [32]u8) void {
        const list = self.edges.getPtr(source) orelse return;
        var i: usize = 0;
        while (i < list.items.len) {
            if (std.mem.eql(u8, &list.items[i].target, &target)) {
                _ = list.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    // FIXME: implement reverse traversal and path scoring
};
