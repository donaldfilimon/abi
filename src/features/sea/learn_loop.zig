const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const ai = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");
const router = @import("../ai/router.zig");
const evidence = @import("evidence.zig");

/// Configuration for one self-learning pass.
pub const LearnLoopConfig = struct {
    /// Number of prior records to recall as evidence.
    evidence_limit: usize = 5,
    /// Persist the completion (query/response vectors + block) into the store.
    persist: bool = true,
    /// Update + save the adaptive persona-router weights from this turn.
    adapt_router: bool = true,
    /// Hard cap on the augmented prompt (mirrors `evidence.MAX_PROMPT_BYTES`).
    max_prompt_bytes: usize = evidence.MAX_PROMPT_BYTES,
};

/// Result of `runLearnLoop`. Owns the underlying completion; `deinit` frees it.
pub const LearnLoopResult = struct {
    completion: ai.CompletionResult,
    evidence_count: usize,
    adapted: bool,

    pub fn deinit(self: *LearnLoopResult, allocator: std.mem.Allocator) void {
        self.completion.deinit(allocator);
    }
};

/// Run one SEA self-learning pass:
///   1. gather evidence relevant to `input`
///   2. augment the prompt with recalled snippets
///   3. load the adaptive persona-router weights
///   4. complete (persisting per `config.persist`)
///   5. update + save the router weights from this turn (per `config.adapt_router`)
///
/// Reuses the existing `ai.completeWithStore` and `router.AdaptiveModulator`;
/// no new ML is introduced.
pub fn runLearnLoop(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    model: []const u8,
    config: LearnLoopConfig,
) !LearnLoopResult {
    var ctx = try evidence.gatherEvidence(allocator, store, input, config.evidence_limit);
    defer ctx.deinit();
    const evidence_count = ctx.items.len;

    const augmented = try evidence.augmentPrompt(allocator, input, &ctx);
    defer allocator.free(augmented);

    var modulator = router.AdaptiveModulator.loadWeights(store);

    var completion = try ai.completeWithStore(allocator, store, .{
        .input = augmented,
        .model = model,
        .store_result = config.persist,
    });
    errdefer completion.deinit(allocator);

    var adapted = false;
    if (config.adapt_router) {
        modulator.update(router.analyzeSentiment(input));
        // A failed weight save must not discard the completion the caller owns;
        // surface it as "not adapted" rather than leaking or aborting.
        if (modulator.saveWeights(allocator, store)) |_| {
            adapted = true;
        } else |_| {
            adapted = false;
        }
    }

    return .{ .completion = completion, .evidence_count = evidence_count, .adapted = adapted };
}

test {
    std.testing.refAllDecls(@This());
}
