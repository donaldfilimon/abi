const std = @import("std");
const build_options = @import("build_options");
const sync = @import("../../foundation/sync.zig");
const memory = @import("../../core/memory.zig");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");
const types = @import("types.zig");
const storage = @import("hnsw_storage.zig");
const distance = @import("hnsw_distance.zig");

pub const MAX_LAYERS = types.MAX_LAYERS;
pub const M = 16;
pub const EF_CONSTRUCTION = 40;
pub const EF_SEARCH = 32;
pub const VectorStorage = storage.VectorStorage;
pub const Candidate = distance.Candidate;
pub const cosineDistanceSIMD = distance.cosineDistanceSIMD;

pub const HnswNode = struct {
    id: u32,
    level: usize,
    edges: [MAX_LAYERS]std.ArrayListUnmanaged(u32),
    lock: sync.SpinLock = .{},

    pub fn initEdges(allocator: std.mem.Allocator) [MAX_LAYERS]std.ArrayListUnmanaged(u32) {
        var arr: [MAX_LAYERS]std.ArrayListUnmanaged(u32) = undefined;
        var i: usize = 0;
        _ = allocator;
        while (i < MAX_LAYERS) : (i += 1) {
            arr[i] = .empty;
        }
        return arr;
    }

    pub fn deinit(self: *HnswNode, allocator: std.mem.Allocator) void {
        var i: usize = 0;
        while (i < MAX_LAYERS) : (i += 1) {
            self.edges[i].deinit(allocator);
        }
    }
};

pub fn HnswIndex(comptime D: usize) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        storage: VectorStorage,
        nodes: std.ArrayListUnmanaged(HnswNode),
        entry_node: ?u32 = null,
        lock: sync.RwLock = .{},
        max_level: usize = 0,
        rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),
        distance_ops: gpu.VectorOps,
        tracker: ?*memory.MemoryTracker = null,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .storage = VectorStorage.init(allocator, D, 64),
                .nodes = .empty,
                .distance_ops = gpu.vectorOps(),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.nodes.items) |*node| {
                if (self.tracker) |t| t.trackFreeNoTag((node.level + 1) * M * @sizeOf(u32));
                node.deinit(self.allocator);
            }
            self.nodes.deinit(self.allocator);
            self.storage.deinit();
        }

        pub fn setTracker(self: *Self, tracker: *memory.MemoryTracker) void {
            self.tracker = tracker;
            self.storage.setTracker(tracker);
        }

        fn randomLevel(self: *Self) usize {
            const p = 1.0 / @as(f64, @floatFromInt(M));
            var level: usize = 0;
            while (self.rng.random().float(f64) < p and level < MAX_LAYERS - 1) : (level += 1) {}
            return level;
        }

        fn cosineDistance(self: *const Self, a: []const f32, b: []const f32) f32 {
            return distance.cosineDistanceWithOps(self.distance_ops, a, b);
        }

        pub fn insert(self: *Self, id: u32, values: []const f32) !void {
            self.lock.lockWrite();
            defer self.lock.unlockWrite();

            if (values.len != D) return error.DimensionMismatch;

            try self.storage.insert(id, values);

            const level = self.randomLevel();

            const node = HnswNode{
                .id = id,
                .level = level,
                .edges = blk: {
                    var edges = HnswNode.initEdges(self.allocator);
                    var layer: usize = 0;
                    while (layer <= level) : (layer += 1) {
                        edges[layer] = try std.ArrayListUnmanaged(u32).initCapacity(self.allocator, M);
                    }
                    break :blk edges;
                },
            };
            try self.nodes.append(self.allocator, node);
            // Edge lists are the node's persistent allocation: (level+1) layers,
            // each reserved at capacity M and capped at M (no realloc). Track the
            // exact footprint; the matching free is accounted in deinit.
            if (self.tracker) |t| t.trackAllocNoTag((level + 1) * M * @sizeOf(u32));

            const node_idx = self.nodes.items.len - 1;

            if (self.entry_node == null) {
                self.entry_node = @as(u32, @intCast(node_idx));
                self.max_level = level;
                return;
            }

            var curr = self.entry_node.?;
            var curr_level = self.max_level;

            while (curr_level > level) : (curr_level -= 1) {
                var best_dist = self.cosineDistance(
                    self.storage.get(self.nodes.items[curr].id),
                    self.storage.get(id),
                );
                var changed = true;
                while (changed) {
                    changed = false;
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        const dist = self.cosineDistance(
                            self.storage.get(self.nodes.items[neighbor].id),
                            self.storage.get(id),
                        );
                        if (dist < best_dist) {
                            best_dist = dist;
                            curr = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            while (true) {
                try self.connectNodes(curr, node_idx, curr_level);
                var best_dist = self.cosineDistance(
                    self.storage.get(self.nodes.items[curr].id),
                    self.storage.get(id),
                );
                var changed = true;
                while (changed) {
                    changed = false;
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        const dist = self.cosineDistance(
                            self.storage.get(self.nodes.items[neighbor].id),
                            self.storage.get(id),
                        );
                        if (dist < best_dist) {
                            best_dist = dist;
                            curr = neighbor;
                            changed = true;
                        }
                    }
                }
                if (curr_level == 0) break;
                curr_level -= 1;
            }

            if (level > self.max_level) {
                self.max_level = level;
                self.entry_node = @as(u32, @intCast(node_idx));
            }
        }

        fn connectNodes(self: *Self, from: usize, to: usize, level: usize) !void {
            const from_edges = &self.nodes.items[from].edges[level];
            for (from_edges.items) |existing| {
                if (existing == to) return;
            }
            if (from_edges.items.len < M) {
                try from_edges.append(self.allocator, @as(u32, @intCast(to)));
            }

            const to_edges = &self.nodes.items[to].edges[level];
            for (to_edges.items) |existing| {
                if (existing == from) return;
            }
            if (to_edges.items.len < M) {
                try to_edges.append(self.allocator, @as(u32, @intCast(from)));
            }
        }

        pub fn search(self: *Self, query: []const f32, limit: usize) ![]types.SearchResult {
            if (query.len != D) return error.DimensionMismatch;

            self.lock.lockRead();
            defer self.lock.unlockRead();

            if (self.entry_node == null) return self.allocator.alloc(types.SearchResult, 0);

            // Per-query scratch arena: the candidate list, visited set, and batch
            // distance temporaries live and die within this call, so one arena
            // (freed on return via defer) removes per-append allocator churn and
            // cannot outlive the search — no use-after-free risk. The returned
            // results array escapes to the caller, so it uses the index allocator.
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const scratch = arena.allocator();

            var candidates: std.ArrayListUnmanaged(Candidate) = .empty;

            var visited = std.AutoHashMap(u32, void).init(scratch);

            try candidates.append(scratch, .{
                .id = self.entry_node.?,
                .distance = self.cosineDistance(
                    self.storage.get(self.nodes.items[self.entry_node.?].id),
                    query,
                ),
            });
            try visited.put(self.entry_node.?, {});

            var curr = self.entry_node.?;
            var curr_level = self.max_level;

            while (true) {
                var changed = true;
                while (changed) {
                    changed = false;
                    const curr_dist = self.cosineDistance(
                        self.storage.get(self.nodes.items[curr].id),
                        query,
                    );
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    var neighbor_ids: [M]u32 = undefined;
                    var neighbor_vectors: [M][]const f32 = undefined;
                    var neighbor_distances: [M]f32 = undefined;
                    var neighbor_count: usize = 0;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        if (visited.contains(neighbor)) continue;
                        try visited.put(neighbor, {});
                        neighbor_ids[neighbor_count] = neighbor;
                        neighbor_vectors[neighbor_count] = self.storage.get(self.nodes.items[neighbor].id);
                        neighbor_count += 1;
                    }
                    try distance.batchCosineDistancesWithOps(
                        scratch,
                        self.distance_ops,
                        query,
                        neighbor_vectors[0..neighbor_count],
                        neighbor_distances[0..neighbor_count],
                    );
                    for (neighbor_ids[0..neighbor_count], neighbor_distances[0..neighbor_count]) |neighbor, dist| {
                        if (dist < curr_dist) {
                            curr = neighbor;
                            changed = true;
                        }
                        try candidates.append(scratch, .{ .id = neighbor, .distance = dist });
                    }
                }
                if (curr_level == 0) break;
                curr_level -= 1;
            }

            std.mem.sort(Candidate, candidates.items, {}, distance.lessDistance);

            const result_count = @min(limit, candidates.items.len);
            const results = try self.allocator.alloc(types.SearchResult, result_count);
            for (results, 0..) |*res, idx| {
                res.* = .{
                    .id = self.nodes.items[candidates.items[idx].id].id,
                    .score = 1.0 - candidates.items[idx].distance,
                };
            }
            return results;
        }

        pub fn count(self: *const Self) usize {
            const lock = @constCast(&self.lock);
            lock.lockRead();
            defer lock.unlockRead();
            return self.nodes.items.len;
        }
    };
}

test "HnswIndex insert and search" {
    const Index = HnswIndex(4);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();

    try index.insert(1, &.{ 1.0, 0.0, 0.0, 0.0 });
    try index.insert(2, &.{ 0.9, 0.1, 0.0, 0.0 });
    try index.insert(3, &.{ 0.0, 1.0, 0.0, 0.0 });

    const results = try index.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expect(results.len == 2);
    try std.testing.expect(results[0].score >= results[1].score);
}

test "HnswIndex dimension mismatch" {
    const Index = HnswIndex(4);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();

    try std.testing.expectError(
        error.DimensionMismatch,
        index.insert(1, &.{ 1.0, 0.0 }),
    );
}

test "HnswIndex empty search" {
    const Index = HnswIndex(4);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();

    const results = try index.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len == 0);
}

test "HnswNode deinit" {
    var node = HnswNode{ .id = 42, .level = 2, .edges = HnswNode.initEdges(std.testing.allocator) };
    defer node.deinit(std.testing.allocator);

    try node.edges[0].append(std.testing.allocator, 1);
    try node.edges[1].append(std.testing.allocator, 2);
    try node.edges[2].append(std.testing.allocator, 3);

    try std.testing.expectEqual(@as(usize, 1), node.edges[0].items.len);
    try std.testing.expectEqual(@as(usize, 1), node.edges[1].items.len);
    try std.testing.expectEqual(@as(usize, 1), node.edges[2].items.len);
}

test "HnswIndex multiple inserts" {
    const Index = HnswIndex(4);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();

    var i: usize = 0;
    while (i < 20) : (i += 1) {
        const id: u32 = @intCast(i);
        const vals = [_]f32{
            @as(f32, @floatFromInt(i)) / 20.0,
            @as(f32, @floatFromInt(20 - i)) / 20.0,
            0.0,
            0.0,
        };
        try index.insert(id, &vals);
    }

    try std.testing.expectEqual(@as(usize, 20), index.count());

    const results = try index.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len == 5);
}

test "HnswIndex tracks edge-list memory and searches via scratch arena" {
    const Index = HnswIndex(4);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();

    var tracker = memory.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();
    index.setTracker(&tracker);

    var i: usize = 0;
    while (i < 12) : (i += 1) {
        const id: u32 = @intCast(i + 1);
        const vals = [_]f32{ @as(f32, @floatFromInt(i)) / 12.0, @as(f32, @floatFromInt(12 - i)) / 12.0, 0.0, 0.0 };
        try index.insert(id, &vals);
    }
    // Edge-list allocations were observed by the tracker.
    try std.testing.expect(tracker.getPeakUsage() > 0);

    // Search runs entirely on its per-query arena and returns owned results.
    const results = try index.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 5);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 5), results.len);
}

test {
    _ = @import("hnsw_distance.zig");
    _ = @import("hnsw_storage.zig");
    std.testing.refAllDecls(@This());
}
