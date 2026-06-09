//! Temporal + causal retrieval graph and hybrid ranking (Index Layer).
//!
//! Layers two relevance signals on top of HNSW semantic similarity:
//!   * temporal — exponential recency decay over record timestamps;
//!   * causal   — proximity over an explicit causal edge graph (BFS hops).
//!
//! The combined relevance follows the north-star scoring model:
//!   score = semantic × temporal × causal × persona
//! Every factor is kept in [0, 1] so the product is a well-behaved ranking key.

const std = @import("std");
const memory = @import("../../core/memory.zig");

pub const ScoreComponents = struct {
    semantic: f32,
    temporal: f32,
    causal: f32,
    persona: f32,

    pub fn combined(self: ScoreComponents) f32 {
        return self.semantic * self.temporal * self.causal * self.persona;
    }
};

/// Exponential recency weight: 1.0 at age 0, halving every `half_life_ms`.
/// Clamped to [0, 1]. A non-positive half-life disables decay (returns 1.0).
pub fn temporalWeight(now_ms: i64, ts_ms: i64, half_life_ms: i64) f32 {
    if (half_life_ms <= 0) return 1.0;
    const age: f32 = @floatFromInt(@max(@as(i64, 0), now_ms - ts_ms));
    const hl: f32 = @floatFromInt(half_life_ms);
    const w = std.math.pow(f32, 0.5, age / hl);
    return std.math.clamp(w, 0.0, 1.0);
}

pub const RankedNode = struct {
    id: u32,
    score: f32,
    components: ScoreComponents,
};

/// A semantic candidate to be re-ranked: an id plus its semantic similarity.
pub const Candidate = struct {
    id: u32,
    semantic: f32,
};

/// In-memory causal graph keyed by node id, with timestamps for temporal decay.
/// Causal edges are stored directed (cause → effect) but reachability for the
/// causal weight is evaluated undirected so related events on either side of a
/// link are surfaced.
pub const TemporalCausalGraph = struct {
    allocator: std.mem.Allocator,
    timestamps: std.AutoHashMapUnmanaged(u32, i64) = .empty,
    adjacency: std.AutoHashMapUnmanaged(u32, std.ArrayListUnmanaged(u32)) = .empty,
    /// Optional allocation observer. Tracks logical node/edge bytes per insert
    /// and frees the running total on deinit (precise pairing, not a heap-exact
    /// estimate of hashmap overhead).
    tracker: ?*memory.MemoryTracker = null,
    tracked_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator) TemporalCausalGraph {
        return .{ .allocator = allocator };
    }

    pub fn setTracker(self: *TemporalCausalGraph, t: *memory.MemoryTracker) void {
        self.tracker = t;
    }

    pub fn deinit(self: *TemporalCausalGraph) void {
        if (self.tracker) |t| t.trackFreeNoTag(self.tracked_bytes);
        self.tracked_bytes = 0;
        self.timestamps.deinit(self.allocator);
        var it = self.adjacency.valueIterator();
        while (it.next()) |list| list.deinit(self.allocator);
        self.adjacency.deinit(self.allocator);
    }

    pub fn addNode(self: *TemporalCausalGraph, id: u32, timestamp_ms: i64) !void {
        const gop = try self.timestamps.getOrPut(self.allocator, id);
        if (!gop.found_existing) {
            const est = @sizeOf(u32) + @sizeOf(i64);
            if (self.tracker) |t| t.trackAllocNoTag(est);
            self.tracked_bytes += est;
        }
        gop.value_ptr.* = timestamp_ms;
    }

    pub fn nodeCount(self: *const TemporalCausalGraph) usize {
        return self.timestamps.count();
    }

    pub fn timestampFor(self: *const TemporalCausalGraph, id: u32) ?i64 {
        return self.timestamps.get(id);
    }

    pub fn edgeCount(self: *const TemporalCausalGraph) usize {
        var count: usize = 0;
        var it = self.adjacency.iterator();
        while (it.next()) |entry| {
            const from = entry.key_ptr.*;
            for (entry.value_ptr.items) |to| {
                if (from < to) count += 1;
            }
        }
        return count;
    }

    fn link(self: *TemporalCausalGraph, from: u32, to: u32) !void {
        const gop = try self.adjacency.getOrPut(self.allocator, from);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        for (gop.value_ptr.items) |e| if (e == to) return;
        try gop.value_ptr.append(self.allocator, to);
        if (self.tracker) |t| t.trackAllocNoTag(@sizeOf(u32));
        self.tracked_bytes += @sizeOf(u32);
    }

    /// Record that `cause` causally precedes `effect`. Stored both directions
    /// for undirected reachability while keeping insertion intent explicit.
    pub fn addCausalEdge(self: *TemporalCausalGraph, cause: u32, effect: u32) !void {
        if (cause == effect) return error.SelfEdge;
        try self.link(cause, effect);
        try self.link(effect, cause);
    }

    /// BFS hop distance from `from` to `to` over causal edges, or null if
    /// unreachable within `max_hops`.
    pub fn hopDistance(self: *const TemporalCausalGraph, from: u32, to: u32, max_hops: u32) !?u32 {
        if (from == to) return 0;
        var visited: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer visited.deinit(self.allocator);
        var frontier: std.ArrayListUnmanaged(u32) = .empty;
        defer frontier.deinit(self.allocator);
        var next: std.ArrayListUnmanaged(u32) = .empty;
        defer next.deinit(self.allocator);

        try visited.put(self.allocator, from, {});
        try frontier.append(self.allocator, from);
        var depth: u32 = 0;
        while (frontier.items.len > 0 and depth < max_hops) {
            depth += 1;
            next.clearRetainingCapacity();
            for (frontier.items) |node| {
                const neighbors = self.adjacency.get(node) orelse continue;
                for (neighbors.items) |nb| {
                    if (nb == to) return depth;
                    if (visited.contains(nb)) continue;
                    try visited.put(self.allocator, nb, {});
                    try next.append(self.allocator, nb);
                }
            }
            std.mem.swap(std.ArrayListUnmanaged(u32), &frontier, &next);
        }
        return null;
    }
};

/// Combines semantic similarity with temporal and causal proximity. Defaults
/// give a one-day half-life and a causal decay of 0.6 per hop with a 0.25 floor
/// for unrelated nodes (so causality boosts but never zeroes a result).
pub const HybridScorer = struct {
    now_ms: i64,
    half_life_ms: i64 = 24 * 60 * 60 * 1000,
    causal_decay: f32 = 0.6,
    causal_floor: f32 = 0.25,
    max_hops: u32 = 4,

    pub fn causalWeight(self: HybridScorer, graph: *const TemporalCausalGraph, focus_id: u32, node_id: u32) !f32 {
        const hops = (try graph.hopDistance(focus_id, node_id, self.max_hops)) orelse return self.causal_floor;
        const w = std.math.pow(f32, self.causal_decay, @floatFromInt(hops));
        return @max(self.causal_floor, w);
    }

    /// Score one node. `semantic` and `persona` are caller-supplied in [0, 1]
    /// (semantic from HNSW cosine; persona from the router profile weight).
    pub fn score(self: HybridScorer, graph: *const TemporalCausalGraph, focus_id: u32, node_id: u32, semantic: f32, persona: f32) !ScoreComponents {
        const ts = graph.timestamps.get(node_id) orelse self.now_ms;
        return .{
            .semantic = std.math.clamp(semantic, 0.0, 1.0),
            .temporal = temporalWeight(self.now_ms, ts, self.half_life_ms),
            .causal = try self.causalWeight(graph, focus_id, node_id),
            .persona = std.math.clamp(persona, 0.0, 1.0),
        };
    }

    /// Re-rank semantic candidates by the full hybrid model, highest first.
    /// `personaFor` maps a node id to its persona weight in [0, 1].
    pub fn rank(
        self: HybridScorer,
        allocator: std.mem.Allocator,
        graph: *const TemporalCausalGraph,
        focus_id: u32,
        candidates: []const Candidate,
        personaFor: *const fn (u32) f32,
    ) ![]RankedNode {
        var out = try allocator.alloc(RankedNode, candidates.len);
        errdefer allocator.free(out);
        for (candidates, 0..) |c, i| {
            const comps = try self.score(graph, focus_id, c.id, c.semantic, personaFor(c.id));
            out[i] = .{ .id = c.id, .score = comps.combined(), .components = comps };
        }
        std.mem.sort(RankedNode, out, {}, struct {
            fn lessThan(_: void, a: RankedNode, b: RankedNode) bool {
                return a.score > b.score; // descending
            }
        }.lessThan);
        return out;
    }
};

test "temporal weight halves at each half-life and clamps" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), temporalWeight(1000, 1000, 1000), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), temporalWeight(2000, 1000, 1000), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), temporalWeight(3000, 1000, 1000), 1e-6);
    // Future timestamps do not exceed 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), temporalWeight(1000, 5000, 1000), 1e-6);
}

test "causal hop distance over an undirected reachability graph" {
    var g = TemporalCausalGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addNode(1, 0);
    try g.addNode(2, 0);
    try g.addNode(3, 0);
    try g.addCausalEdge(1, 2);
    try g.addCausalEdge(2, 3);

    try std.testing.expectEqual(@as(?u32, 0), try g.hopDistance(1, 1, 4));
    try std.testing.expectEqual(@as(?u32, 1), try g.hopDistance(1, 2, 4));
    try std.testing.expectEqual(@as(?u32, 2), try g.hopDistance(1, 3, 4));
    try std.testing.expectEqual(@as(?u32, null), try g.hopDistance(1, 99, 4));
    try std.testing.expectError(error.SelfEdge, g.addCausalEdge(1, 1));
}

test "hybrid score multiplies semantic × temporal × causal × persona" {
    var g = TemporalCausalGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addNode(1, 1000);
    try g.addNode(2, 0); // one half-life older
    try g.addCausalEdge(1, 2);

    const scorer = HybridScorer{ .now_ms = 1000, .half_life_ms = 1000, .causal_decay = 0.6, .causal_floor = 0.25, .max_hops = 4 };

    // Focus node itself: temporal=1, causal=1.
    const a = try scorer.score(&g, 1, 1, 0.9, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), a.combined(), 1e-6);

    // Node 2: one half-life old (temporal 0.5), one causal hop (0.6), persona 0.5.
    const b = try scorer.score(&g, 1, 2, 1.0, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), b.temporal, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), b.causal, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 * 0.5 * 0.6 * 0.5), b.combined(), 1e-6);
}

const test_persona = struct {
    fn weight(id: u32) f32 {
        return if (id == 2) 1.0 else 0.5;
    }
};

test "hybrid rank orders candidates by combined score" {
    var g = TemporalCausalGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addNode(1, 1000);
    try g.addNode(2, 1000);
    try g.addCausalEdge(1, 2);

    const scorer = HybridScorer{ .now_ms = 1000, .half_life_ms = 1000 };
    const cands = [_]Candidate{
        .{ .id = 1, .semantic = 0.6 },
        .{ .id = 2, .semantic = 0.6 },
    };
    const ranked = try scorer.rank(std.testing.allocator, &g, 1, &cands, test_persona.weight);
    defer std.testing.allocator.free(ranked);

    try std.testing.expectEqual(@as(usize, 2), ranked.len);
    // Node 2 has higher persona weight, so it outranks node 1 at equal semantic/temporal.
    try std.testing.expectEqual(@as(u32, 2), ranked[0].id);
    try std.testing.expect(ranked[0].score >= ranked[1].score);
}

test "temporal graph reports timestamps and unique undirected edge count" {
    var g = TemporalCausalGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addNode(1, 1000);
    try g.addNode(2, 2000);
    try g.addNode(3, 3000);
    try g.addCausalEdge(1, 2);
    try g.addCausalEdge(2, 1); // duplicate undirected relationship
    try g.addCausalEdge(2, 3);

    try std.testing.expectEqual(@as(?i64, 1000), g.timestampFor(1));
    try std.testing.expectEqual(@as(?i64, null), g.timestampFor(99));
    try std.testing.expectEqual(@as(usize, 2), g.edgeCount());
}

test {
    std.testing.refAllDecls(@This());
}
