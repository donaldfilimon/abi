const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const helpers = @import("../ai/helpers.zig");
const query_plan = @import("query_plan.zig");
const ai_types = @import("../ai/types.zig");
const sea_types = @import("types.zig");

/// When a `QueryPlan` requests `exact_recall`, the recalled hit's semantic
/// similarity is blended with lexical (keyword) overlap so exact-wording matches
/// outrank fuzzy-but-distant ones. `0.5` weights the two evenly, shifting
/// retrieval toward provenance/exact wording per the SEA design (§5.3).
const EXACT_RECALL_KEYWORD_WEIGHT: f32 = 0.5;

/// Known persona labels. `profile_label` on an `EvidenceItem` is *borrowed*: it
/// always points at one of these static literals (or `unknown`), never at
/// freshly parsed/owned memory, so an item can be freed without touching it.
const known_profile_labels = ai_types.PROFILE_LABELS;
const unknown_profile_label = "unknown";

/// Upper bound on the augmented-prompt preamble. SEA prepends recalled snippets
/// as context; this cap keeps a runaway store from producing an unbounded prompt.
pub const MAX_PROMPT_BYTES: usize = 4096;

/// A single recalled record. `snippet` is owned (freed by `deinit`);
/// `profile_label` is borrowed (a static literal — not freed).
pub const EvidenceItem = struct {
    vector_id: u32,
    profile_label: []const u8,
    authority: sea_types.Authority,
    snippet: []u8,
    score: f32,

    pub fn deinit(self: *EvidenceItem, allocator: std.mem.Allocator) void {
        if (self.snippet.len > 0) {
            allocator.free(self.snippet);
            self.snippet = &.{};
        }
    }
};

/// Owned collection of recalled evidence. `deinit` frees every item snippet and
/// the backing slice.
pub const EvidenceContext = struct {
    items: []EvidenceItem,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *EvidenceContext) void {
        for (self.items) |*item| item.deinit(self.allocator);
        if (self.items.len > 0) self.allocator.free(self.items);
        self.items = &.{};
    }

    pub fn isEmpty(self: *const EvidenceContext) bool {
        return self.items.len == 0;
    }
};

/// Map a parsed metadata `profile` value onto a borrowed static label so the
/// item never owns it. Unrecognized values collapse to `unknown`.
fn staticProfileLabel(metadata: []const u8) []const u8 {
    for (known_profile_labels) |label| {
        // Match the JSON `"profile":"<label>"` field without a full parse.
        var needle_buf: [32]u8 = undefined;
        const needle = std.fmt.bufPrint(&needle_buf, "\"profile\":\"{s}\"", .{label}) catch continue;
        if (std.mem.indexOf(u8, metadata, needle) != null) return label;
    }
    return unknown_profile_label;
}

/// Resolve persisted provenance onto the SEA trust ladder. Missing or unknown
/// authority is treated as inferred, the least-trusted rung, rather than being
/// silently promoted.
fn staticAuthority(metadata: []const u8) sea_types.Authority {
    for (std.meta.tags(sea_types.Authority)) |authority| {
        var needle_buf: [48]u8 = undefined;
        const needle = std.fmt.bufPrint(&needle_buf, "\"authority\":\"{s}\"", .{authority.text()}) catch continue;
        if (std.mem.indexOf(u8, metadata, needle) != null) return authority;
    }
    return .inferred;
}

/// Gather evidence relevant to `input` from a durable store.
///
/// Convenience wrapper: infers a `QueryPlan` from `input` (deterministic keyword
/// heuristic, no model call) and delegates to `gatherEvidenceWithPlan`, so the
/// existing call sites become task-aware automatically.
pub fn gatherEvidence(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    limit: usize,
) !EvidenceContext {
    return gatherEvidenceWithPlan(allocator, store, input, limit, query_plan.infer(input));
}

/// Gather evidence relevant to `input` under an explicit `QueryPlan`.
///
/// Phase 1: embed the input via the shared AI embedding, run a vector search,
/// and read each hit's stored completion metadata (`completion:<id>`) as the
/// snippet. An empty store (no vectors) yields an empty context; search/metadata
/// failures degrade to skipping that hit rather than aborting the loop.
///
/// Task-awareness: when `plan.exact_recall` is set (e.g. a `project_recall`
/// task), each hit's semantic score is blended with its lexical keyword overlap
/// against the query and the results are re-sorted, so exact-wording matches
/// outrank fuzzy ones. Other plans keep the store's semantic ordering unchanged.
pub fn gatherEvidenceWithPlan(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    limit: usize,
    plan: query_plan.QueryPlan,
) !EvidenceContext {
    if (input.len == 0 or limit == 0 or store.vectorCount() == 0) {
        return .{ .items = &.{}, .allocator = allocator };
    }

    const embedding = helpers.textEmbedding(input);
    const hits = store.search(&embedding, limit) catch |err| {
        // Inference path: don't fail the whole completion, but don't swallow
        // silently either — a broken retrieval (dim mismatch, OOM, index error)
        // degrades SEA to zero evidence, which must leave a trace.
        std.log.scoped(.sea).warn("evidence retrieval failed ({s}); degrading to zero evidence", .{@errorName(err)});
        return .{ .items = &.{}, .allocator = allocator };
    };
    // `hits` is owned by the store's own allocator (the index allocates the
    // result slice with it), which can differ from `allocator` when the store
    // outlives the request — e.g. the long-lived MCP store backed by
    // `page_allocator` while each call passes a transient request allocator.
    // Free it with the owning allocator to avoid a cross-allocator free.
    defer if (comptime @hasField(wdbx.Store, "allocator")) store.allocator.free(hits) else allocator.free(hits);

    var items: std.ArrayListUnmanaged(EvidenceItem) = .empty;
    errdefer {
        for (items.items) |*item| item.deinit(allocator);
        items.deinit(allocator);
    }

    for (hits) |hit| {
        const key = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{hit.id});
        defer allocator.free(key);

        const metadata = store.get(key) orelse continue;
        const snippet = try allocator.dupe(u8, metadata);
        errdefer allocator.free(snippet);

        // Under exact_recall, blend the semantic score with lexical overlap so
        // exact-wording matches are favored; otherwise keep the pure semantic
        // score and the store's existing ordering.
        const authority = staticAuthority(metadata);
        const relevance = if (plan.exact_recall)
            (1.0 - EXACT_RECALL_KEYWORD_WEIGHT) * hit.score +
                EXACT_RECALL_KEYWORD_WEIGHT * keywordOverlap(input, metadata)
        else
            hit.score;
        // Provenance participates in ranking. Unlabeled/generated records stay
        // on the least-trusted rung; tool/file/system evidence is promoted only
        // when its metadata explicitly carries that authority.
        const score = relevance * authority.score();

        try items.append(allocator, .{
            .vector_id = hit.id,
            .profile_label = staticProfileLabel(metadata),
            .authority = authority,
            .snippet = snippet,
            .score = score,
        });
    }

    const owned = try items.toOwnedSlice(allocator);
    // Re-sort by the re-weighted score; only meaningful when scores were blended.
    if (plan.exact_recall) std.mem.sort(EvidenceItem, owned, {}, scoreDesc);
    return .{ .items = owned, .allocator = allocator };
}

fn scoreDesc(_: void, a: EvidenceItem, b: EvidenceItem) bool {
    return a.score > b.score;
}

/// Fraction of the query's significant tokens (>= 3 chars) that appear,
/// case-insensitively, in `text`. Returns `[0,1]`; `0` when the query has no
/// significant tokens. A deterministic, model-free lexical signal.
fn keywordOverlap(query: []const u8, text: []const u8) f32 {
    var total: usize = 0;
    var hits: usize = 0;
    var it = std.mem.tokenizeAny(u8, query, " \t\n\r.,;:!?\"'()[]{}<>/\\");
    while (it.next()) |tok| {
        if (tok.len < 3) continue;
        total += 1;
        if (containsIgnoreCase(text, tok)) hits += 1;
    }
    if (total == 0) return 0;
    return @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(total));
}

/// Case-insensitive ASCII substring search (local to avoid cross-feature import).
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Build an augmented prompt by prepending recalled snippets as a preamble,
/// capped at `MAX_PROMPT_BYTES`. With no evidence, returns a copy of `input`
/// unchanged. The caller owns the returned buffer.
pub fn augmentPrompt(allocator: std.mem.Allocator, input: []const u8, ctx: *const EvidenceContext) ![]u8 {
    if (ctx.isEmpty()) return allocator.dupe(u8, input);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "[SEA evidence]\n");
    for (ctx.items) |item| {
        // Stop appending preamble once the cap is reached; the original input is
        // always appended afterward so the prompt is never truncated mid-input.
        if (out.items.len >= MAX_PROMPT_BYTES) break;
        try out.print(allocator, "- (vec {d}, {s}, authority={s}): {s}\n", .{ item.vector_id, item.profile_label, item.authority.text(), item.snippet });
        if (out.items.len > MAX_PROMPT_BYTES) {
            out.shrinkRetainingCapacity(MAX_PROMPT_BYTES);
            break;
        }
    }
    try out.appendSlice(allocator, "[query]\n");
    try out.appendSlice(allocator, input);

    return out.toOwnedSlice(allocator);
}

test "gatherEvidence on an empty store returns an empty context" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var ctx = try gatherEvidence(allocator, &store, "hello", 5);
    defer ctx.deinit();
    try std.testing.expect(ctx.isEmpty());

    const prompt = try augmentPrompt(allocator, "hello", &ctx);
    defer allocator.free(prompt);
    try std.testing.expectEqualStrings("hello", prompt);
}

test "augmentPrompt caps the preamble at MAX_PROMPT_BYTES" {
    const allocator = std.testing.allocator;

    // Build a synthetic context whose snippets exceed the cap.
    const items = try allocator.alloc(EvidenceItem, 8);
    for (items, 0..) |*item, i| {
        const snippet = try allocator.alloc(u8, 1024);
        @memset(snippet, 'x');
        item.* = .{
            .vector_id = @intCast(i),
            .profile_label = "abbey",
            .authority = .inferred,
            .snippet = snippet,
            .score = 1.0,
        };
    }
    var ctx = EvidenceContext{ .items = items, .allocator = allocator };
    defer ctx.deinit();

    const prompt = try augmentPrompt(allocator, "the-query", &ctx);
    defer allocator.free(prompt);

    // Preamble is capped, but the original input is always present.
    try std.testing.expect(prompt.len <= MAX_PROMPT_BYTES + "[query]\n".len + "the-query".len);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "the-query"));
}

test "gatherEvidence recalls a stored completion with its resolved persona label" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // A stored turn: a vector plus its completion metadata under completion:<id>.
    const embedding = helpers.textEmbedding("aviva said hello");
    const id = try store.putVector(&embedding);
    const key = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{id});
    defer allocator.free(key);
    try store.store(key, "{\"profile\":\"aviva\",\"authority\":\"user_stated\",\"text\":\"hello there\"}");

    var ctx = try gatherEvidence(allocator, &store, "hello", 5);
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 1), ctx.items.len);
    try std.testing.expectEqualStrings("aviva", ctx.items[0].profile_label);
    try std.testing.expectEqual(sea_types.Authority.user_stated, ctx.items[0].authority);
    try std.testing.expectEqualStrings("{\"profile\":\"aviva\",\"authority\":\"user_stated\",\"text\":\"hello there\"}", ctx.items[0].snippet);
}

test "gatherEvidence maps an unrecognized profile to the unknown label" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const embedding = helpers.textEmbedding("a mystery turn");
    const id = try store.putVector(&embedding);
    const key = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{id});
    defer allocator.free(key);
    try store.store(key, "{\"profile\":\"nobody\"}");

    var ctx = try gatherEvidence(allocator, &store, "mystery", 5);
    defer ctx.deinit();
    try std.testing.expectEqual(@as(usize, 1), ctx.items.len);
    try std.testing.expectEqualStrings("unknown", ctx.items[0].profile_label);
    try std.testing.expectEqual(sea_types.Authority.inferred, ctx.items[0].authority);
}

test "gatherEvidence weights explicit provenance without promoting unlabeled records" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const low_embedding = helpers.textEmbedding("same provenance query");
    const low_id = try store.putVector(&low_embedding);
    const low_key = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{low_id});
    defer allocator.free(low_key);
    try store.store(low_key, "{\"profile\":\"abbey\",\"text\":\"same provenance query\"}");

    const high_embedding = helpers.textEmbedding("same provenance query");
    const high_id = try store.putVector(&high_embedding);
    const high_key = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{high_id});
    defer allocator.free(high_key);
    try store.store(high_key, "{\"profile\":\"abbey\",\"authority\":\"file_verified\",\"text\":\"same provenance query\"}");

    var ctx = try gatherEvidence(allocator, &store, "same provenance query", 5);
    defer ctx.deinit();
    try std.testing.expect(ctx.items.len >= 2);

    var low_score: ?f32 = null;
    var high_score: ?f32 = null;
    for (ctx.items) |item| {
        if (item.vector_id == low_id) low_score = item.score;
        if (item.vector_id == high_id) high_score = item.score;
    }
    try std.testing.expect(low_score != null and high_score != null);
    try std.testing.expect(high_score.? > low_score.?);
}

test "keywordOverlap counts significant query tokens present in text" {
    // Tokens shorter than 3 chars are ignored; both significant tokens present.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), keywordOverlap("paris france", "paris is in france"), 1e-5);
    // "paris" present, "berlin" absent -> 1/2.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), keywordOverlap("paris berlin", "paris is nice"), 1e-5);
    // All tokens < 3 chars -> no significant tokens -> 0.
    try std.testing.expectEqual(@as(f32, 0.0), keywordOverlap("a an of", "anything"));
}

test "exact_recall re-weights retrieval toward lexical overlap and re-sorts" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // Two stored turns. The query shares exact wording with the SECOND record's
    // metadata but the embeddings may rank the first higher semantically; with
    // exact_recall the lexically-matching record must surface at the top.
    const e1 = helpers.textEmbedding("alpha topic overview");
    const id1 = try store.putVector(&e1);
    const k1 = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{id1});
    defer allocator.free(k1);
    try store.store(k1, "{\"profile\":\"abbey\",\"text\":\"alpha topic overview\"}");

    const e2 = helpers.textEmbedding("the prior decision about widget pricing");
    const id2 = try store.putVector(&e2);
    const k2 = try std.fmt.allocPrint(allocator, ai_types.COMPLETION_KEY_FMT, .{id2});
    defer allocator.free(k2);
    try store.store(k2, "{\"profile\":\"aviva\",\"text\":\"the prior decision about widget pricing\"}");

    // "remember"/"prior"/"decision" -> project_recall -> exact_recall = true.
    const plan = query_plan.infer("remember the prior decision about widget pricing");
    try std.testing.expect(plan.exact_recall);

    var ctx = try gatherEvidenceWithPlan(allocator, &store, plan.query, 5, plan);
    defer ctx.deinit();

    try std.testing.expect(ctx.items.len >= 1);
    // The lexically-exact record (id2) ranks first under exact_recall.
    try std.testing.expectEqual(id2, ctx.items[0].vector_id);
}

test {
    std.testing.refAllDecls(@This());
}
