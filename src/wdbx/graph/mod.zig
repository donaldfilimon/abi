//! Owns explicit relationships.
//!
//! GraphStore uses a flat edge list plus index maps for cache-friendly
//! traversal and lower allocation count than per-node edge lists.

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

/// Single contiguous edge buffer; outgoing/incoming map node -> indices.
pub const GraphStore = struct {
    edges: std.ArrayListUnmanaged(Edge) = .empty,
    outgoing: std.AutoHashMapUnmanaged([32]u8, std.ArrayListUnmanaged(u32)) = .empty,
    incoming: std.AutoHashMapUnmanaged([32]u8, std.ArrayListUnmanaged(u32)) = .empty,

    pub fn deinit(self: *GraphStore, allocator: std.mem.Allocator) void {
        self.edges.deinit(allocator);
        var it = self.outgoing.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        self.outgoing.deinit(allocator);
        it = self.incoming.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        self.incoming.deinit(allocator);
    }

    /// Outgoing edges for `source`; slice valid until next mutation. Uses `arena` for the slice.
    pub fn getOutgoing(self: *const GraphStore, arena: std.mem.Allocator, source: [32]u8) ![]const Edge {
        const indices = self.outgoing.get(source) orelse return &.{};
        const out = try arena.alloc(Edge, indices.items.len);
        for (indices.items, out) |idx, *e| e.* = self.edges.items[idx];
        return out;
    }

    pub fn addEdge(self: *GraphStore, allocator: std.mem.Allocator, source: [32]u8, target: [32]u8, kind: EdgeKind) !void {
        const idx = @as(u32, @truncate(self.edges.items.len));
        try self.edges.append(allocator, .{ .kind = kind, .target = target });
        {
            const gop = try self.outgoing.getOrPut(allocator, source);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(allocator, idx);
        }
        {
            const gop = try self.incoming.getOrPut(allocator, target);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(allocator, idx);
        }
    }

    pub fn hasEdge(self: *const GraphStore, source: [32]u8, target: [32]u8) bool {
        const indices = self.outgoing.get(source) orelse return false;
        for (indices.items) |idx| {
            if (std.mem.eql(u8, &self.edges.items[idx].target, &target)) return true;
        }
        return false;
    }

    pub fn removeEdge(self: *GraphStore, source: [32]u8, target: [32]u8) void {
        const out_indices = self.outgoing.getPtr(source) orelse return;
        var edge_idx: ?u32 = null;
        for (out_indices.items) |idx| {
            if (std.mem.eql(u8, &self.edges.items[idx].target, &target)) {
                edge_idx = idx;
                break;
            }
        }
        const idx = edge_idx orelse return;
        // Remove index from outgoing[source]
        var i: usize = 0;
        while (i < out_indices.items.len) {
            if (out_indices.items[i] == idx) {
                _ = out_indices.swapRemove(i);
                break;
            }
            i += 1;
        }
        // Remove index from incoming[target]
        const inc = self.incoming.getPtr(target) orelse return;
        i = 0;
        while (i < inc.items.len) {
            if (inc.items[i] == idx) {
                _ = inc.swapRemove(i);
                break;
            }
            i += 1;
        }
        // Swap-remove edge; fix the index that moved from last to idx
        const last_idx = @as(u32, @truncate(self.edges.items.len - 1));
        if (idx != last_idx) {
            _ = self.edges.swapRemove(idx);
            var out_it = self.outgoing.valueIterator();
            while (out_it.next()) |list| {
                for (list.items) |*pi| {
                    if (pi.* == last_idx) pi.* = idx;
                }
            }
            var inc_it = self.incoming.valueIterator();
            while (inc_it.next()) |list| {
                for (list.items) |*pi| {
                    if (pi.* == last_idx) pi.* = idx;
                }
            }
        } else {
            _ = self.edges.pop();
        }
    }

    /// Breadth-first search from source, up to max_depth levels.
    /// Returns block IDs in level-order. Arena-allocated.
    pub fn bfs(self: *const GraphStore, arena: std.mem.Allocator, source: [32]u8, max_depth: u32) ![]const [32]u8 {
        var result: std.ArrayListUnmanaged([32]u8) = .empty;
        var visited: std.AutoHashMapUnmanaged([32]u8, void) = .empty;
        defer visited.deinit(arena);

        var queue: std.ArrayListUnmanaged(struct { id: [32]u8, depth: u32 }) = .empty;
        defer queue.deinit(arena);

        try visited.put(arena, source, {});
        try queue.append(arena, .{ .id = source, .depth = 0 });
        try result.append(arena, source);

        var head: usize = 0;
        while (head < queue.items.len) {
            const current = queue.items[head];
            head += 1;

            if (current.depth >= max_depth) continue;

            const neighbors = try self.getOutgoing(arena, current.id);
            for (neighbors) |edge| {
                const gop = try visited.getOrPut(arena, edge.target);
                if (!gop.found_existing) {
                    try result.append(arena, edge.target);
                    try queue.append(arena, .{ .id = edge.target, .depth = current.depth + 1 });
                }
            }
        }

        return result.items;
    }

    /// Depth-first search visitor callback type.
    pub const DfsVisitor = *const fn (node: [32]u8, depth: u32, user_data: ?*anyopaque) void;

    /// Stack-based depth-first search from source with visitor callback.
    pub fn dfs(self: *const GraphStore, arena: std.mem.Allocator, source: [32]u8, visitor: DfsVisitor, user_data: ?*anyopaque) !void {
        var visited: std.AutoHashMapUnmanaged([32]u8, void) = .empty;
        defer visited.deinit(arena);

        var stack: std.ArrayListUnmanaged(struct { id: [32]u8, depth: u32 }) = .empty;
        defer stack.deinit(arena);

        try stack.append(arena, .{ .id = source, .depth = 0 });

        while (stack.items.len > 0) {
            const current = stack.pop().?;

            const gop = try visited.getOrPut(arena, current.id);
            if (gop.found_existing) continue;

            visitor(current.id, current.depth, user_data);

            const neighbors = try self.getOutgoing(arena, current.id);
            var i: usize = neighbors.len;
            while (i > 0) {
                i -= 1;
                if (visited.get(neighbors[i].target) == null) {
                    try stack.append(arena, .{ .id = neighbors[i].target, .depth = current.depth + 1 });
                }
            }
        }
    }

    /// Find all paths from source to target up to max_depth.
    /// Returns a slice of paths, where each path is a slice of block IDs.
    pub fn findPaths(
        self: *const GraphStore,
        arena: std.mem.Allocator,
        source: [32]u8,
        target: [32]u8,
        max_depth: u32,
    ) ![]const []const [32]u8 {
        var all_paths: std.ArrayListUnmanaged([]const [32]u8) = .empty;

        // DFS with path tracking
        var path_stack: std.ArrayListUnmanaged([32]u8) = .empty;
        defer path_stack.deinit(arena);

        try self.findPathsRecurse(arena, source, target, max_depth, 0, &path_stack, &all_paths);

        return all_paths.items;
    }

    fn findPathsRecurse(
        self: *const GraphStore,
        arena: std.mem.Allocator,
        current: [32]u8,
        target: [32]u8,
        max_depth: u32,
        depth: u32,
        path: *std.ArrayListUnmanaged([32]u8),
        results: *std.ArrayListUnmanaged([]const [32]u8),
    ) !void {
        try path.append(arena, current);
        defer _ = path.pop();

        if (std.mem.eql(u8, &current, &target)) {
            // Found a path — copy it
            const path_copy = try arena.dupe([32]u8, path.items);
            try results.append(arena, path_copy);
            return;
        }

        if (depth >= max_depth) return;

        // Check for cycles in current path
        for (path.items[0 .. path.items.len - 1]) |node| {
            if (std.mem.eql(u8, &node, &current)) return;
        }

        const neighbors = try self.getOutgoing(arena, current);
        for (neighbors) |edge| {
            var in_path = false;
            for (path.items) |node| {
                if (std.mem.eql(u8, &node, &edge.target)) {
                    in_path = true;
                    break;
                }
            }
            if (!in_path) {
                try self.findPathsRecurse(arena, edge.target, target, max_depth, depth + 1, path, results);
            }
        }
    }

    pub fn scorePath(self: *const GraphStore, arena: std.mem.Allocator, path: []const [32]u8) f32 {
        var total: f32 = 0;
        if (path.len < 2) return 0;
        for (0..path.len - 1) |i| {
            const neighbors = self.getOutgoing(arena, path[i]) catch continue;
            for (neighbors) |edge| {
                if (std.mem.eql(u8, &edge.target, &path[i + 1])) {
                    total += edge.weight;
                }
            }
        }
        return total;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

fn makeId(comptime c: u8) [32]u8 {
    var id: [32]u8 = [_]u8{0} ** 32;
    id[0] = c;
    return id;
}

test "BFS: 5-node graph, level-order, max_depth respected" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);

    const a = makeId('A');
    const b = makeId('B');
    const c = makeId('C');
    const d = makeId('D');
    const e = makeId('E');

    // A -> B -> D
    // A -> C -> E
    try graph.addEdge(allocator, a, b, .derived_from);
    try graph.addEdge(allocator, a, c, .supports);
    try graph.addEdge(allocator, b, d, .derived_from);
    try graph.addEdge(allocator, c, e, .supports);

    // Full BFS from A
    const all = try graph.bfs(allocator, a, 10);
    try std.testing.expectEqual(@as(usize, 5), all.len);
    try std.testing.expect(std.mem.eql(u8, &all[0], &a)); // A first

    // Depth-limited BFS (depth=1: only A + direct neighbors)
    const shallow = try graph.bfs(allocator, a, 1);
    try std.testing.expectEqual(@as(usize, 3), shallow.len); // A, B, C
}

test "DFS: all reachable nodes visited" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);

    const a = makeId('A');
    const b = makeId('B');
    const c = makeId('C');

    try graph.addEdge(allocator, a, b, .derived_from);
    try graph.addEdge(allocator, b, c, .derived_from);

    const Counter = struct {
        var visited: u32 = 0;
        fn visit(_: [32]u8, _: u32, _: ?*anyopaque) void {
            visited += 1;
        }
    };
    Counter.visited = 0;

    try graph.dfs(allocator, a, Counter.visit, null);
    try std.testing.expectEqual(@as(u32, 3), Counter.visited);
}

test "findPaths: two known paths between A and D" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);

    const a = makeId('A');
    const b = makeId('B');
    const c = makeId('C');
    const d = makeId('D');

    try graph.addEdge(allocator, a, b, .derived_from);
    try graph.addEdge(allocator, a, c, .supports);
    try graph.addEdge(allocator, b, d, .derived_from);
    try graph.addEdge(allocator, c, d, .supports);

    const paths = try graph.findPaths(allocator, a, d, 5);
    try std.testing.expectEqual(@as(usize, 2), paths.len);

    for (paths) |path| {
        try std.testing.expectEqual(@as(usize, 3), path.len);
        try std.testing.expect(std.mem.eql(u8, &path[0], &a));
        try std.testing.expect(std.mem.eql(u8, &path[2], &d));
    }
}

test "removeEdge: hasEdge and bfs after removal" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);

    const a = makeId('A');
    const b = makeId('B');
    const c = makeId('C');

    try graph.addEdge(allocator, a, b, .derived_from);
    try graph.addEdge(allocator, a, c, .supports);
    try std.testing.expect(graph.hasEdge(a, b));
    try std.testing.expect(graph.hasEdge(a, c));

    graph.removeEdge(a, c);
    try std.testing.expect(graph.hasEdge(a, b));
    try std.testing.expect(!graph.hasEdge(a, c));

    const reached = try graph.bfs(allocator, a, 2);
    try std.testing.expectEqual(@as(usize, 2), reached.len); // a, b only
}

test "scorePath: sum of edge weights along path" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);

    const a = makeId('A');
    const b = makeId('B');
    const c = makeId('C');
    try graph.addEdge(allocator, a, b, .derived_from);
    try graph.addEdge(allocator, b, c, .supports);

    const path = [_][32]u8{ a, b, c };
    const score = graph.scorePath(allocator, &path);
    try std.testing.expect(score >= 1.0); // at least two edges with weight 1
}

test "GraphStore: many edges (stress)" {
    const allocator = std.testing.allocator;
    var graph: GraphStore = .{};
    defer graph.deinit(allocator);
    const n: usize = 64;
    var ids: [64][32]u8 = undefined;
    for (&ids, 0..) |*id, i| {
        @memset(id, 0);
        id[0] = @as(u8, @intCast(0x41 + (i % 26)));
    }
    for (0..n - 1) |i| {
        try graph.addEdge(allocator, ids[i], ids[i + 1], .derived_from);
    }
    const reached = try graph.bfs(allocator, ids[0], n);
    try std.testing.expect(reached.len >= 1 and reached.len <= n);
    try std.testing.expect(graph.hasEdge(ids[0], ids[1]));
    graph.removeEdge(ids[n / 2], ids[n / 2 + 1]);
    const after = try graph.bfs(allocator, ids[0], n);
    try std.testing.expect(after.len < reached.len or reached.len == 1);
}

test {
    std.testing.refAllDecls(@This());
}
