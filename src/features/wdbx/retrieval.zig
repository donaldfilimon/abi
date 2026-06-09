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
pub const PersonaWeightFn = *const fn (*const anyopaque, u32) f32;

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

/// Semantic search + hybrid re-ranking using a context-aware persona callback.
/// This is the default-query building block for long-lived stores where persona
/// labels live in WDBX metadata rather than a global lookup table.
pub fn hybridSearchWithPersonaContext(
    allocator: std.mem.Allocator,
    store: *wdbx_mod.Store,
    query_vec: []const f32,
    limit: usize,
    graph: *const temporal.TemporalCausalGraph,
    scorer: temporal.HybridScorer,
    focus_id: u32,
    persona_ctx: *const anyopaque,
    personaFor: PersonaWeightFn,
) ![]temporal.RankedNode {
    const results = try store.search(query_vec, limit);
    defer allocator.free(results);

    var out = try allocator.alloc(temporal.RankedNode, results.len);
    errdefer allocator.free(out);
    for (results, 0..) |r, i| {
        const comps = try scorer.score(graph, focus_id, r.id, r.score, personaFor(persona_ctx, r.id));
        out[i] = .{ .id = r.id, .score = comps.combined(), .components = comps };
    }
    std.mem.sort(temporal.RankedNode, out, {}, struct {
        fn lessThan(_: void, a: temporal.RankedNode, b: temporal.RankedNode) bool {
            return a.score > b.score;
        }
    }.lessThan);
    return out;
}

/// Predicate deciding whether vector `id` belongs to the scoped persona. The
/// caller supplies the membership test (e.g. resolving a persona label from its
/// own metadata convention), keeping persona semantics out of the storage layer.
pub const KeepFn = *const fn (*const anyopaque, u32) bool;

fn scopedPersona(id: u32) f32 {
    _ = id;
    // Within an isolated persona the persona factor is a full match: results are
    // already partitioned, so persona must not dampen the ranking.
    return 1.0;
}

/// Multi-persona memory ISOLATION: semantic search restricted to the vectors a
/// `keepFn` accepts, then temporal/causal re-ranked. Unlike
/// `hybridSearchWithPersonaContext` (which blends every persona by weight), this
/// returns ONLY the scoped persona's memories. Over-fetches a wider candidate
/// pool so filtering still yields up to `limit` results. Caller owns the result.
pub fn hybridSearchScoped(
    allocator: std.mem.Allocator,
    store: *wdbx_mod.Store,
    query_vec: []const f32,
    limit: usize,
    graph: *const temporal.TemporalCausalGraph,
    scorer: temporal.HybridScorer,
    focus_id: u32,
    keep_ctx: *const anyopaque,
    keepFn: KeepFn,
) ![]temporal.RankedNode {
    const have = store.vectorCount();
    const want = limit *| 8;
    const pool = if (have == 0) limit else if (want < have) want else have;

    const results = try store.search(query_vec, pool);
    defer allocator.free(results);

    var cands: std.ArrayListUnmanaged(temporal.Candidate) = .empty;
    defer cands.deinit(allocator);
    for (results) |r| {
        if (keepFn(keep_ctx, r.id)) {
            try cands.append(allocator, .{ .id = r.id, .semantic = std.math.clamp(r.score, 0.0, 1.0) });
        }
    }

    const ranked = try scorer.rank(allocator, graph, focus_id, cands.items, scopedPersona);
    if (ranked.len <= limit) return ranked;
    defer allocator.free(ranked);
    return allocator.dupe(temporal.RankedNode, ranked[0..limit]);
}

const testing = std.testing;

fn personaEqual(id: u32) f32 {
    _ = id;
    return 0.5;
}

const ScopeContext = struct {
    allowed: []const u32,
};

fn scopeKeep(ctx: *const anyopaque, id: u32) bool {
    const scope: *const ScopeContext = @ptrCast(@alignCast(ctx));
    for (scope.allowed) |a| {
        if (a == id) return true;
    }
    return false;
}

test "hybridSearchScoped returns only the scoped persona's vectors" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // Four distinct vectors; persona A owns ids 1 and 3, persona B owns 2 and 4.
    const id1 = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    const id2 = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    const id3 = try store.putVector(&.{ 0.9, 0.1, 0.0, 0.0 });
    const id4 = try store.putVector(&.{ 0.0, 0.0, 1.0, 0.0 });

    var graph = temporal.TemporalCausalGraph.init(allocator);
    defer graph.deinit();
    inline for (.{ id1, id2, id3, id4 }) |id| try graph.addNode(id, 1000);

    const scorer = temporal.HybridScorer{ .now_ms = 1000, .half_life_ms = 1000 };
    const scope = ScopeContext{ .allowed = &.{ id1, id3 } };
    const ranked = try hybridSearchScoped(allocator, &store, &.{ 1.0, 0.0, 0.0, 0.0 }, 10, &graph, scorer, id1, &scope, scopeKeep);
    defer allocator.free(ranked);

    // Isolation: persona B's vectors (id2, id4) never appear, even though id2 is
    // semantically closer to the query than id4.
    try testing.expect(ranked.len == 2);
    for (ranked) |r| try testing.expect(r.id == id1 or r.id == id3);
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

const PersonaContext = struct {
    boosted: u32,
};

fn contextualPersona(ctx: *const anyopaque, id: u32) f32 {
    const persona_ctx: *const PersonaContext = @ptrCast(@alignCast(ctx));
    return if (id == persona_ctx.boosted) 1.0 else 0.2;
}

test "hybridSearchWithPersonaContext applies metadata-backed persona weights" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    const low_persona = try store.putVector(&.{ 1, 0, 0, 0 });
    const high_persona = try store.putVector(&.{ 1, 0, 0, 0 });
    var graph = temporal.TemporalCausalGraph.init(allocator);
    defer graph.deinit();
    try graph.addNode(low_persona, 1000);
    try graph.addNode(high_persona, 1000);

    const scorer = temporal.HybridScorer{ .now_ms = 1000, .half_life_ms = 1000 };
    const persona_ctx = PersonaContext{ .boosted = high_persona };
    const ranked = try hybridSearchWithPersonaContext(allocator, &store, &.{ 1, 0, 0, 0 }, 10, &graph, scorer, high_persona, &persona_ctx, contextualPersona);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    try testing.expectEqual(high_persona, ranked[0].id);
    try testing.expect(ranked[0].components.persona > ranked[1].components.persona);
}

test {
    testing.refAllDecls(@This());
}
