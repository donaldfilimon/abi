const std = @import("std");
const types = @import("types.zig");

/// Eight orthogonal scoring signals for SEA (design reference:
/// `docs/spec/sea-design-extract.mdx` §5). Each captures a distinct retrieval
/// dimension: semantic meaning, lexical overlap, structural relevance,
/// freshness, provenance/trust, graph-connectivity, explicit contradiction,
/// and task-alignment.
pub const SeaSignals = struct {
    semantic: f32 = 0,
    keyword: f32 = 0,
    metadata: f32 = 0,
    recency: f32 = 0,
    authority: f32 = 0,
    graph: f32 = 0,
    contradiction: f32 = 0,
    task_fit: f32 = 0,
};

/// Default signal-weight vector. The weights sum to 1.0 so the combined E-score
/// stays in `[0,1]`. Semantic similarity dominates (0.30) but can never solely
/// decide selection — lexical, structural, trust, freshness, graph-connectivity,
/// contradiction, and task-fit signals together carry the other 0.70.
pub const DEFAULT_SEA_WEIGHTS = SeaWeights{
    .semantic = 0.30,
    .keyword = 0.15,
    .metadata = 0.15,
    .recency = 0.10,
    .authority = 0.10,
    .graph = 0.10,
    .contradiction = 0.05,
    .task_fit = 0.05,
};

pub const SeaWeights = struct {
    semantic: f32 = 0.30,
    keyword: f32 = 0.15,
    metadata: f32 = 0.15,
    recency: f32 = 0.10,
    authority: f32 = 0.10,
    graph: f32 = 0.10,
    contradiction: f32 = 0.05,
    task_fit: f32 = 0.05,
};

/// A single candidate for SEA selection. Carries all eight sub-scores,
/// bookkeeping (`estimated_tokens`, `cluster_id`), and the computed
/// `final_score`.
pub const SeaCandidate = struct {
    record_id: u32,
    cluster_id: u8,
    estimated_tokens: usize,
    signals: SeaSignals,
    final_score: f32,
};

/// Budgets and weights for SEA selection. Defaults match the reference design.
pub const SeaOptions = struct {
    max_tokens: usize = 4096,
    max_records: usize = 16,
    per_cluster_limit: usize = 4,
    weights: SeaWeights = DEFAULT_SEA_WEIGHTS,
};

/// Combine eight signals into a single `[0,1]` final score via a weighted sum.
/// A plain weighted sum is the reference design combiner — no gating, no
/// non-linear transform (see `docs/spec/sea-design-extract.mdx` §5.2).
pub fn seaScore(signals: SeaSignals, weights: SeaWeights) f32 {
    const raw = signals.semantic * weights.semantic +
        signals.keyword * weights.keyword +
        signals.metadata * weights.metadata +
        signals.recency * weights.recency +
        signals.authority * weights.authority +
        signals.graph * weights.graph +
        signals.contradiction * weights.contradiction +
        signals.task_fit * weights.task_fit;
    return std.math.clamp(raw, 0.0, 1.0);
}

/// Per-candidate, per-query task-aware weight adjustments. Encodes that
/// different intents value different evidence. These are additive deltas on the
/// default weight vector; the final clamp absorbs off-1.0 sums.
pub fn adjustWeightsForTask(base: SeaWeights, task: u8) SeaWeights {
    var w = base;
    // 0 = general, 1 = implementation_design, 2 = code_repair, 3 = legal_review,
    // 4 = research_synthesis, 5 = project_recall, 6 = benchmark_review.
    if (task == 2) { // code_repair
        w.metadata += 0.05;
        w.task_fit += 0.05;
        w.recency += 0.05;
        w.semantic -= 0.05;
    } else if (task == 5) { // project_recall / exact_recall
        w.authority += 0.10;
        w.keyword += 0.05;
        w.semantic -= 0.10;
    } else if (task == 6) { // benchmark_review
        w.metadata += 0.05;
        w.task_fit += 0.10;
        w.semantic -= 0.05;
    }
    return w;
}

/// Budgeted greedy SEA selection: sort candidates by `final_score` descending,
/// then greedily admit candidates until a token, record-count, or per-cluster
/// budget is exceeded. An exceptionally high-scoring candidate (>= 0.92) is
/// admitted past the per-kind cap — the "escape hatch" that prevents a
/// near-perfect match from being dropped purely for diversity.
///
/// Returns `selected_ids` (admitted) and `rejected_ids` (budget-exceeded), plus
/// total estimated tokens and a human-readable reason. The caller owns the
/// returned slices.
pub const SeaSelection = struct {
    selected_ids: []u32,
    rejected_ids: []u32,
    total_estimated_tokens: usize,
    reason: []const u8,
};

pub fn selectSeaCandidates(
    allocator: std.mem.Allocator,
    candidates: []SeaCandidate,
    options: SeaOptions,
) !SeaSelection {
    if (candidates.len == 0) {
        return .{
            .selected_ids = try allocator.alloc(u32, 0),
            .rejected_ids = try allocator.alloc(u32, 0),
            .total_estimated_tokens = 0,
            .reason = "no candidates to select from",
        };
    }

    // Sort by final_score descending (stable sort is fine for the greedy walk).
    std.mem.sort(SeaCandidate, candidates, {}, seaCandidateDesc);

    var selected = std.ArrayListUnmanaged(u32).empty;
    var rejected = std.ArrayListUnmanaged(u32).empty;
    var used_tokens: usize = 0;
    var cluster_counts = std.mem.zeroes([9]usize);

    for (candidates) |c| {
        const count = cluster_counts[@as(usize, @min(c.cluster_id, 8))];
        const too_many_cluster = count >= options.per_cluster_limit and c.final_score < 0.92;
        const too_many_records = selected.items.len >= options.max_records;
        const too_many_tokens = used_tokens + c.estimated_tokens > options.max_tokens;

        if (too_many_cluster or too_many_records or too_many_tokens) {
            try rejected.append(allocator, c.record_id);
            continue;
        }

        try selected.append(allocator, c.record_id);
        cluster_counts[@as(usize, @min(c.cluster_id, 8))] += 1;
        used_tokens += c.estimated_tokens;
    }

    const reason = if (rejected.items.len > 0) "budget-limited" else "all candidates selected";
    return .{
        .selected_ids = try selected.toOwnedSlice(allocator),
        .rejected_ids = try rejected.toOwnedSlice(allocator),
        .total_estimated_tokens = used_tokens,
        .reason = reason,
    };
}

fn seaCandidateDesc(_: void, a: SeaCandidate, b: SeaCandidate) bool {
    return a.final_score > b.final_score;
}

/// Render selected records into a labeled text block — the evidence preamble
/// that would be handed to a model call. Each record is rendered as
/// `[id=<id>] kind=<kind_text> authority=<authority_text> score=<score>` plus
/// optional project/source/text lines if provided.
pub fn contextPack(
    allocator: std.mem.Allocator,
    selected: *const SeaSelection,
    candidates: []const SeaCandidate,
    kind_texts: []const []const u8,
    snippets: []const []const u8,
) ![]u8 {
    _ = kind_texts;
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "[SEA evidence]\n");
    for (selected.selected_ids) |id| {
        for (candidates, 0..) |c, ci| {
            if (c.record_id == id) {
                const snippet = if (ci < snippets.len) snippets[ci] else "";
                try buf.print(allocator, "- [id={d}] score={d:.3}\n  {s}\n", .{ id, c.final_score, snippet });
                break;
            }
        }
    }
    return buf.toOwnedSlice(allocator);
}

test "seaScore computes the weighted sum correctly" {
    const signals = SeaSignals{
        .semantic = 1.0,
        .keyword = 0.5,
        .metadata = 0.5,
        .recency = 1.0,
        .authority = 1.0,
        .graph = 0.5,
        .contradiction = 0.0,
        .task_fit = 1.0,
    };
    const w = DEFAULT_SEA_WEIGHTS;
    const expected = 1.0 * w.semantic + 0.5 * w.keyword + 0.5 * w.metadata +
        1.0 * w.recency + 1.0 * w.authority + 0.5 * w.graph +
        0.0 * w.contradiction + 1.0 * w.task_fit;
    const score = seaScore(signals, w);
    try std.testing.expectApproxEqAbs(expected, score, 1e-5);
}

test "seaScore clamps to [0,1]" {
    const high = seaScore(.{ .semantic = 2.0, .keyword = 2.0, .metadata = 2.0, .recency = 2.0, .authority = 2.0, .graph = 2.0, .contradiction = 2.0, .task_fit = 2.0 }, DEFAULT_SEA_WEIGHTS);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), high, 1e-5);

    const low = seaScore(.{ .semantic = -1.0, .keyword = -1.0, .metadata = -1.0, .recency = -1.0, .authority = -1.0, .graph = -1.0, .contradiction = -1.0, .task_fit = -1.0 }, DEFAULT_SEA_WEIGHTS);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), low, 1e-5);
}

test "default weights sum to 1.0" {
    const w = DEFAULT_SEA_WEIGHTS;
    const sum = w.semantic + w.keyword + w.metadata + w.recency + w.authority + w.graph + w.contradiction + w.task_fit;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "selectSeaCandidates respects budgets and cluster diversity" {
    var candidates: [6]SeaCandidate = .{
        .{ .record_id = 1, .cluster_id = 0, .estimated_tokens = 100, .signals = .{}, .final_score = 0.95 },
        .{ .record_id = 2, .cluster_id = 0, .estimated_tokens = 100, .signals = .{}, .final_score = 0.90 },
        .{ .record_id = 3, .cluster_id = 1, .estimated_tokens = 100, .signals = .{}, .final_score = 0.85 },
        .{ .record_id = 4, .cluster_id = 0, .estimated_tokens = 100, .signals = .{}, .final_score = 0.80 },
        .{ .record_id = 5, .cluster_id = 1, .estimated_tokens = 100, .signals = .{}, .final_score = 0.75 },
        .{ .record_id = 6, .cluster_id = 2, .estimated_tokens = 100, .signals = .{}, .final_score = 0.70 },
    };
    const options = SeaOptions{
        .max_tokens = 500,
        .max_records = 4,
        .per_cluster_limit = 2,
    };

    const selection = try selectSeaCandidates(std.testing.allocator, &candidates, options);
    defer std.testing.allocator.free(selection.selected_ids);
    defer std.testing.allocator.free(selection.rejected_ids);

    try std.testing.expect(selection.selected_ids.len <= 4);
    try std.testing.expect(selection.total_estimated_tokens <= 500);
}

test "selectSeaCandidates high-score escape hatch admits past cluster limit" {
    var candidates: [3]SeaCandidate = .{
        .{ .record_id = 1, .cluster_id = 0, .estimated_tokens = 100, .signals = .{}, .final_score = 0.93 },
        .{ .record_id = 2, .cluster_id = 0, .estimated_tokens = 100, .signals = .{}, .final_score = 0.99 },
        .{ .record_id = 3, .cluster_id = 1, .estimated_tokens = 100, .signals = .{}, .final_score = 0.50 },
    };
    const options = SeaOptions{
        .max_tokens = 500,
        .max_records = 4,
        .per_cluster_limit = 1,
    };

    const selection = try selectSeaCandidates(std.testing.allocator, &candidates, options);
    defer std.testing.allocator.free(selection.selected_ids);
    defer std.testing.allocator.free(selection.rejected_ids);

    // Both cluster-0 records should be admitted: the first fits the limit, the
    // second has final_score >= 0.92 so the escape hatch admits it past the cap.
    try std.testing.expectEqual(@as(usize, 3), selection.selected_ids.len);
}

test "selectSeaCandidates empty input returns empty selection" {
    const selection = try selectSeaCandidates(std.testing.allocator, &.{}, .{});
    defer std.testing.allocator.free(selection.selected_ids);
    defer std.testing.allocator.free(selection.rejected_ids);

    try std.testing.expectEqual(@as(usize, 0), selection.selected_ids.len);
    try std.testing.expectEqual(@as(usize, 0), selection.rejected_ids.len);
    try std.testing.expectEqualStrings("no candidates to select from", selection.reason);
}

test "contextPack renders selected records" {
    const allocator = std.testing.allocator;
    var candidates: [2]SeaCandidate = .{
        .{ .record_id = 10, .cluster_id = 0, .estimated_tokens = 50, .signals = .{}, .final_score = 0.90 },
        .{ .record_id = 20, .cluster_id = 1, .estimated_tokens = 100, .signals = .{}, .final_score = 0.80 },
    };
    const selection = SeaSelection{
        .selected_ids = try allocator.dupe(u32, &.{ 10, 20 }),
        .rejected_ids = try allocator.alloc(u32, 0),
        .total_estimated_tokens = 150,
        .reason = "all candidates selected",
    };
    defer allocator.free(selection.selected_ids);
    defer allocator.free(selection.rejected_ids);

    const context = try contextPack(allocator, &selection, &candidates, &.{}, &.{ "first snippet", "second snippet" });
    defer allocator.free(context);

    try std.testing.expect(std.mem.indexOf(u8, context, "first snippet") != null);
    try std.testing.expect(std.mem.indexOf(u8, context, "second snippet") != null);
}

test "adjustWeightsForTask code_repair nudges metadata, task_fit, recency" {
    const adjusted = adjustWeightsForTask(DEFAULT_SEA_WEIGHTS, 2);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.metadata + 0.05, adjusted.metadata, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.task_fit + 0.05, adjusted.task_fit, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.recency + 0.05, adjusted.recency, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.semantic - 0.05, adjusted.semantic, 1e-5);
}

test "adjustWeightsForTask project_recall boosts authority and keyword" {
    const adjusted = adjustWeightsForTask(DEFAULT_SEA_WEIGHTS, 5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.authority + 0.10, adjusted.authority, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.keyword + 0.05, adjusted.keyword, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.semantic - 0.10, adjusted.semantic, 1e-5);
}

test "adjustWeightsForTask general leaves weights unchanged" {
    const adjusted = adjustWeightsForTask(DEFAULT_SEA_WEIGHTS, 0);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.semantic, adjusted.semantic, 1e-5);
    try std.testing.expectApproxEqAbs(DEFAULT_SEA_WEIGHTS.keyword, adjusted.keyword, 1e-5);
}

test {
    std.testing.refAllDecls(@This());
}
