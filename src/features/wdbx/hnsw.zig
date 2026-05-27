const std = @import("std");
const sync = @import("../../foundation/sync.zig");
const wdbx_mod = @import("mod.zig");

pub const MAX_LAYERS = wdbx_mod.MAX_LAYERS;
pub const M = 16;
pub const EF_CONSTRUCTION = 40;
pub const EF_SEARCH = 32;

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

pub const VectorStorage = struct {
    allocator: std.mem.Allocator,
    data: std.ArrayListUnmanaged(f32),
    present: std.AutoHashMap(u32, void),
    dimensions: usize = 0,
    capacity: usize = 0,

    pub fn init(allocator: std.mem.Allocator, dimensions: usize, initial_capacity: usize) VectorStorage {
        return .{
            .allocator = allocator,
            .data = .empty,
            .present = std.AutoHashMap(u32, void).init(allocator),
            .dimensions = dimensions,
            .capacity = initial_capacity,
        };
    }

    pub fn deinit(self: *VectorStorage) void {
        self.present.deinit();
        self.data.deinit(self.allocator);
    }

    pub fn insert(self: *VectorStorage, id: u32, values: []const f32) !void {
        if (values.len != self.dimensions) return error.DimensionMismatch;
        const needed = (id + 1) * self.dimensions;
        if (needed > self.data.items.len) {
            const old_len = self.data.items.len;
            const new_cap = @max(needed, self.data.items.len * 2 + 64);
            try self.data.resize(self.allocator, new_cap);
            @memset(self.data.items[old_len..new_cap], 0);
        }
        const offset = id * self.dimensions;
        @memcpy(self.data.items[offset .. offset + self.dimensions], values);
        try self.present.put(id, {});
    }

    pub fn get(self: *const VectorStorage, id: u32) []const f32 {
        const offset = id * self.dimensions;
        return self.data.items[offset .. offset + self.dimensions];
    }

    pub fn contains(self: *const VectorStorage, id: u32) bool {
        return self.present.contains(id);
    }
};

pub const Candidate = struct {
    id: u32,
    distance: f32,
};

fn greaterDistance(_: void, lhs: Candidate, rhs: Candidate) bool {
    return lhs.distance > rhs.distance;
}

fn lessDistance(_: void, lhs: Candidate, rhs: Candidate) bool {
    return lhs.distance < rhs.distance;
}

pub fn cosineDistanceSIMD(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 1.0;
    const len = a.len;
    if (len == 0) return 1.0;

    const simd_width = std.simd.suggestVectorLength(f32) orelse 4;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    const simd_len = (len / simd_width) * simd_width;

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        const va: @Vector(simd_width, f32) = a[i .. i + simd_width][0..simd_width].*;
        const vb: @Vector(simd_width, f32) = b[i .. i + simd_width][0..simd_width].*;
        dot += @reduce(.Add, va * vb);
        norm_a += @reduce(.Add, va * va);
        norm_b += @reduce(.Add, vb * vb);
    }

    while (i < len) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 1.0;
    return 1.0 - (dot / denom);
}

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

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .storage = VectorStorage.init(allocator, D, 64),
                .nodes = .empty,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.nodes.items) |*node| {
                node.deinit(self.allocator);
            }
            self.nodes.deinit(self.allocator);
            self.storage.deinit();
        }

        fn randomLevel(self: *Self) usize {
            const p = 1.0 / @as(f64, @floatFromInt(M));
            var level: usize = 0;
            while (self.rng.random().float(f64) < p and level < MAX_LAYERS - 1) : (level += 1) {}
            return level;
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

            const node_idx = self.nodes.items.len - 1;

            if (self.entry_node == null) {
                self.entry_node = @as(u32, @intCast(node_idx));
                self.max_level = level;
                return;
            }

            var curr = self.entry_node.?;
            var curr_level = self.max_level;

            while (curr_level > level) : (curr_level -= 1) {
                var best_dist = cosineDistanceSIMD(
                    self.storage.get(self.nodes.items[curr].id),
                    self.storage.get(id),
                );
                var changed = true;
                while (changed) {
                    changed = false;
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        const dist = cosineDistanceSIMD(
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
                var best_dist = cosineDistanceSIMD(
                    self.storage.get(self.nodes.items[curr].id),
                    self.storage.get(id),
                );
                var changed = true;
                while (changed) {
                    changed = false;
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        const dist = cosineDistanceSIMD(
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

        pub fn search(self: *Self, query: []const f32, limit: usize) ![]wdbx_mod.SearchResult {
            if (query.len != D) return error.DimensionMismatch;

            self.lock.lockRead();
            defer self.lock.unlockRead();

            if (self.entry_node == null) return self.allocator.alloc(wdbx_mod.SearchResult, 0);

            var candidates: std.ArrayListUnmanaged(Candidate) = .empty;
            defer candidates.deinit(self.allocator);

            var visited = std.AutoHashMap(u32, void).init(self.allocator);
            defer visited.deinit();

            try candidates.append(self.allocator, .{
                .id = self.entry_node.?,
                .distance = cosineDistanceSIMD(
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
                    const curr_dist = cosineDistanceSIMD(
                        self.storage.get(self.nodes.items[curr].id),
                        query,
                    );
                    const edges = self.nodes.items[curr].edges[curr_level].items;
                    for (edges) |neighbor| {
                        if (neighbor >= self.nodes.items.len) continue;
                        if (visited.contains(neighbor)) continue;
                        try visited.put(neighbor, {});
                        const dist = cosineDistanceSIMD(
                            self.storage.get(self.nodes.items[neighbor].id),
                            query,
                        );
                        if (dist < curr_dist) {
                            curr = neighbor;
                            changed = true;
                        }
                        try candidates.append(self.allocator, .{ .id = neighbor, .distance = dist });
                    }
                }
                if (curr_level == 0) break;
                curr_level -= 1;
            }

            std.mem.sort(Candidate, candidates.items, {}, lessDistance);

            const result_count = @min(limit, candidates.items.len);
            const results = try self.allocator.alloc(wdbx_mod.SearchResult, result_count);
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

test "cosineDistanceSIMD identical vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const dist = cosineDistanceSIMD(&a, &b);
    try std.testing.expect(dist < 0.001);
}

test "cosineDistanceSIMD orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const dist = cosineDistanceSIMD(&a, &b);
    try std.testing.expect(dist > 0.99);
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

test "VectorStorage insert and get" {
    var storage = VectorStorage.init(std.testing.allocator, 3, 8);
    defer storage.deinit();

    try storage.insert(0, &.{ 1.0, 2.0, 3.0 });
    try storage.insert(5, &.{ 4.0, 5.0, 6.0 });

    const v0 = storage.get(0);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0 }, v0);

    const v5 = storage.get(5);
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 5.0, 6.0 }, v5);

    try std.testing.expect(storage.contains(0));
    try std.testing.expect(storage.contains(5));
    try std.testing.expect(!storage.contains(10));
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

test {
    std.testing.refAllDecls(@This());
}
