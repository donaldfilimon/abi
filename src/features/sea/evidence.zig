const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const helpers = @import("../ai/helpers.zig");

/// Known persona labels. `profile_label` on an `EvidenceItem` is *borrowed*: it
/// always points at one of these static literals (or `unknown`), never at
/// freshly parsed/owned memory, so an item can be freed without touching it.
const known_profile_labels = [_][]const u8{ "abbey", "aviva", "abi" };
const unknown_profile_label = "unknown";

/// Upper bound on the augmented-prompt preamble. SEA prepends recalled snippets
/// as context; this cap keeps a runaway store from producing an unbounded prompt.
pub const MAX_PROMPT_BYTES: usize = 4096;

/// A single recalled record. `snippet` is owned (freed by `deinit`);
/// `profile_label` is borrowed (a static literal — not freed).
pub const EvidenceItem = struct {
    vector_id: u32,
    profile_label: []const u8,
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

/// Gather evidence relevant to `input` from a durable store.
///
/// Phase 1: embed the input via the shared AI embedding, run a vector search,
/// and read each hit's stored completion metadata (`completion:<id>`) as the
/// snippet. An empty store (no vectors) yields an empty context; search/metadata
/// failures degrade to skipping that hit rather than aborting the loop.
pub fn gatherEvidence(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    limit: usize,
) !EvidenceContext {
    if (input.len == 0 or limit == 0 or store.vectorCount() == 0) {
        return .{ .items = &.{}, .allocator = allocator };
    }

    const embedding = helpers.textEmbedding(input);
    const hits = store.search(&embedding, limit) catch {
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
        const key = try std.fmt.allocPrint(allocator, "completion:{d}", .{hit.id});
        defer allocator.free(key);

        const metadata = store.get(key) orelse continue;
        const snippet = try allocator.dupe(u8, metadata);
        errdefer allocator.free(snippet);

        try items.append(allocator, .{
            .vector_id = hit.id,
            .profile_label = staticProfileLabel(metadata),
            .snippet = snippet,
            .score = hit.score,
        });
    }

    return .{ .items = try items.toOwnedSlice(allocator), .allocator = allocator };
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
        try out.print(allocator, "- (vec {d}, {s}): {s}\n", .{ item.vector_id, item.profile_label, item.snippet });
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

test {
    std.testing.refAllDecls(@This());
}
