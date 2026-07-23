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
const spatial_3d = @import("spatial_3d.zig");
const hnsw_distance = @import("hnsw_distance.zig");

pub const RankedNode = temporal.RankedNode;
pub const PersonaWeightFn = *const fn (*const anyopaque, u32) f32;

/// Attach zero-copy borrowed vector views to each ranked node via
/// `Store.getVector`. Slices alias store index backing and remain valid until
/// the next mutation that grows/frees that buffer. Safe to call repeatedly.
pub fn attachBorrowedVectors(store: *const wdbx_mod.Store, ranked: []temporal.RankedNode) void {
    for (ranked) |*r| {
        r.vector = store.getVector(r.id);
    }
}

/// Semantic search + temporal/causal/persona re-ranking, highest combined score
/// first. `allocator` must be the Store's allocator (it owns/frees the search
/// results). Returns owned `RankedNode`s the caller frees; each node carries a
/// borrowed `vector` view when the id is present in the store.
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
    const ranked = try scorer.rank(allocator, graph, focus_id, cands, personaFor);
    attachBorrowedVectors(store, ranked);
    return ranked;
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
    attachBorrowedVectors(store, out);
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
    if (ranked.len <= limit) {
        attachBorrowedVectors(store, ranked);
        return ranked;
    }
    defer allocator.free(ranked);
    const trimmed = try allocator.dupe(temporal.RankedNode, ranked[0..limit]);
    attachBorrowedVectors(store, trimmed);
    return trimmed;
}

/// One ranked result from `hybridSpatialSearch`: `point` and `payload` are
/// zero-copy borrowed views into the store's spatial index, valid until the
/// next mutation that grows/frees that index's backing storage (same
/// convention as `attachBorrowedVectors`).
pub const RankedSpatialResult = struct {
    id: u32,
    point: spatial_3d.Point3D,
    payload: []const u8,
    score: f32,
    semantic_score: f32,
    spatial_score: f32,
};

fn lessThanSpatialScore(_: void, a: RankedSpatialResult, b: RankedSpatialResult) bool {
    return a.score > b.score;
}

/// Semantic + 3D-spatial hybrid ranking: blends cosine similarity to
/// `query_vector` with proximity to `center` into one score, highest first.
/// `weight_semantic` and `weight_spatial` need not sum to 1 -- the caller
/// controls the blend; passing 1.0/0.0 recovers pure-semantic ranking over
/// the spatial candidate pool, and 0.0/1.0 recovers pure-spatial ranking.
///
/// Candidates are drawn from the store's 3D spatial index (up to
/// `limit *| 8` nearest points to `center`, euclidean metric). A candidate
/// whose id has no vector attached via `Store.putVector` is skipped -- a
/// hybrid score requires both signals. `allocator` must be the Store's
/// allocator. Caller owns and frees the returned slice.
pub fn hybridSpatialSearch(
    allocator: std.mem.Allocator,
    store: *const wdbx_mod.Store,
    query_vector: []const f32,
    center: spatial_3d.Point3D,
    limit: usize,
    weight_semantic: f32,
    weight_spatial: f32,
) ![]RankedSpatialResult {
    const pool = limit *| 8;
    const spatial_hits = try store.searchSpatial3D(center, pool, .euclidean);
    defer allocator.free(spatial_hits);

    if (spatial_hits.len == 0) return try allocator.alloc(RankedSpatialResult, 0);

    var max_distance: f32 = 0;
    for (spatial_hits) |hit| max_distance = @max(max_distance, hit.distance);

    var candidates: std.ArrayListUnmanaged(RankedSpatialResult) = .empty;
    defer candidates.deinit(allocator);

    for (spatial_hits) |hit| {
        const vector = store.getVector(hit.id) orelse continue;
        const semantic_score = 1.0 - hnsw_distance.cosineDistanceSIMD(query_vector, vector);
        const normalized_distance = if (max_distance == 0) 0 else hit.distance / max_distance;
        const spatial_score = 1.0 - normalized_distance;
        try candidates.append(allocator, .{
            .id = hit.id,
            .point = hit.point,
            .payload = hit.payload,
            .score = weight_semantic * semantic_score + weight_spatial * spatial_score,
            .semantic_score = semantic_score,
            .spatial_score = spatial_score,
        });
    }

    std.mem.sort(RankedSpatialResult, candidates.items, {}, lessThanSpatialScore);

    const out_len = @min(limit, candidates.items.len);
    return try allocator.dupe(RankedSpatialResult, candidates.items[0..out_len]);
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

test "hybridSearch attaches zero-copy borrowed vector views" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    const id = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    var graph = temporal.TemporalCausalGraph.init(allocator);
    defer graph.deinit();
    try graph.addNode(id, 1000);

    const scorer = temporal.HybridScorer{ .now_ms = 1000, .half_life_ms = 1000 };
    const ranked = try hybridSearch(allocator, &store, &.{ 1.0, 0.0, 0.0, 0.0 }, 5, &graph, scorer, id, personaEqual);
    defer allocator.free(ranked);

    try testing.expect(ranked.len >= 1);
    const view = ranked[0].vector orelse return error.MissingBorrowedVector;
    const direct = store.getVector(ranked[0].id) orelse return error.MissingDirectVector;
    try testing.expectEqual(direct.len, view.len);
    // Pointer alias: same backing buffer, no copy.
    try testing.expect(view.ptr == direct.ptr);
}

test "Store.search SearchResult carries borrowed vector view" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    const id = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    const results = try store.search(&.{ 0.0, 1.0, 0.0, 0.0 }, 5);
    defer allocator.free(results);

    try testing.expect(results.len >= 1);
    try testing.expectEqual(id, results[0].id);
    const view = results[0].vector orelse return error.MissingBorrowedVector;
    const direct = store.getVector(id) orelse return error.MissingDirectVector;
    try testing.expect(view.ptr == direct.ptr);
    try testing.expectEqual(direct.len, view.len);
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

test "hybridSpatialSearch weight=1.0/0.0 recovers pure-semantic ranking" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // id_semantic_match: vector identical to the query, but far away in space.
    // id_spatial_match: vector orthogonal to the query, but at the query center.
    const id_semantic_match = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_semantic_match, .{ .x = 100.0, .y = 100.0, .z = 100.0 }, "");

    const id_spatial_match = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    try store.putSpatial3D(id_spatial_match, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 1.0, 0.0);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    try testing.expectEqual(id_semantic_match, ranked[0].id);
}

test "hybridSpatialSearch weight=0.0/1.0 recovers pure-spatial ranking" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    const id_semantic_match = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_semantic_match, .{ .x = 100.0, .y = 100.0, .z = 100.0 }, "");

    const id_spatial_match = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    try store.putSpatial3D(id_spatial_match, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 0.0, 1.0);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    try testing.expectEqual(id_spatial_match, ranked[0].id);
}

test "hybridSpatialSearch skips candidates with no attached vector" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // Spatial-only point: no matching putVector call for this id.
    try store.putSpatial3D(999, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const id_with_vector = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_with_vector, .{ .x = 1.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 0.5, 0.5);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 1), ranked.len);
    try testing.expectEqual(id_with_vector, ranked[0].id);
}

test {
    testing.refAllDecls(@This());
}
