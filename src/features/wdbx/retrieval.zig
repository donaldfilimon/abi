//! Hybrid retrieval over a WDBX Store (Query Layer).
//!
//! Wires the temporal/causal HybridScorer (`temporal.zig`) into a real query
//! path: run the HNSW semantic search, then re-rank the candidates by the
//! north-star scoring model `semantic × temporal × causal × persona`. The caller
//! supplies the temporal/causal graph (node timestamps + causal edges) and a
//! persona-weight lookup, so the ranker composes with whatever timestamp/persona
//! source the caller maintains (e.g. conversation-block timestamps) without the
//! Store needing to carry that data itself.

const std = @import("std");
const wdbx_mod = @import("mod.zig");
const temporal = @import("temporal.zig");

pub const RankedNode = temporal.RankedNode;

/// Semantic search + temporal/causal/persona re-ranking, highest combined score
/// first. `allocator` must be the Store's allocator (it owns/frees the search
/// results). Returns owned `RankedNode`s the caller frees.
pub fn hybridSearch(
    allocator: std.mem.Allocator,
    store: *wdbx_mod.Store,
    query_vec: []const f32,
    limit: usize,
    graph: *const temporal.TemporalCausalGraph,
    scorer: temporal.HybridScorer,
    focus_id: u32,
    personaFor: *const fn (u32) f32,
) ![]temporal.RankedNode {
    const results = try store.search(query_vec, limit);
    defer allocator.free(results);

    const cands = try allocator.alloc(temporal.Candidate, results.len);
    defer allocator.free(cands);
    for (results, 0..) |r, i| {
        cands[i] = .{ .id = r.id, .semantic = std.math.clamp(r.score, 0.0, 1.0) };
    }
    return scorer.rank(allocator, graph, focus_id, cands, personaFor);
}

const testing = std.testing;

fn personaEqual(id: u32) f32 {
    _ = id;
    return 0.5;
}

test "hybridSearch re-ranks equal-semantic candidates by recency + causality" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // Two identical vectors -> equal semantic similarity to the query.
    const id_old = try store.putVector(&.{ 1, 0, 0, 0 });
    const id_new = try store.putVector(&.{ 1, 0, 0, 0 });

    var graph = temporal.TemporalCausalGraph.init(allocator);
    defer graph.deinit();
    try graph.addNode(id_old, 0); // one half-life old
    try graph.addNode(id_new, 1000); // current

    const scorer = temporal.HybridScorer{ .now_ms = 1000, .half_life_ms = 1000 };
    const ranked = try hybridSearch(allocator, &store, &.{ 1, 0, 0, 0 }, 10, &graph, scorer, id_new, personaEqual);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    // Equal semantic + persona: the recent focus node (temporal=1, causal=1)
    // must outrank the older, causally-distant one — proving the hybrid factors
    // are actually applied on top of HNSW similarity.
    try testing.expectEqual(id_new, ranked[0].id);
    try testing.expect(ranked[0].score >= ranked[1].score);
}

test {
    testing.refAllDecls(@This());
}
