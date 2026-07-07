const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const ai = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");
const router = @import("../ai/router.zig");
const evidence = @import("evidence.zig");
const query_plan = @import("query_plan.zig");
const core_memory = @import("../../core/memory.zig");

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
    /// Optional memory tracker. When set, the adaptive-weight save's transient
    /// serialize buffer is routed through a `TrackingAllocator` so weight
    /// persistence is observable. The buffer is balanced (alloc+free inside
    /// `saveWeights`), so no caller-owned/escaping allocation is tracked.
    tracker: ?*core_memory.MemoryTracker = null,
};

/// Result of `runLearnLoop`. Owns the underlying completion; `deinit` frees it.
pub const LearnLoopResult = struct {
    completion: ai.CompletionResult,
    evidence_count: usize,
    adapted: bool,
    /// The task intent inferred from the input for this turn (drives task-aware
    /// retrieval). `.general` when nothing more specific matched.
    query_task: query_plan.TaskType = .general,

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
    // Infer a task-aware plan once and thread it through retrieval, so e.g. a
    // project_recall query shifts evidence weighting toward exact wording.
    const plan = query_plan.infer(input);
    var ctx = try evidence.gatherEvidenceWithPlan(allocator, store, input, config.evidence_limit, plan);
    defer ctx.deinit();
    const evidence_count = ctx.items.len;

    const augmented = try evidence.augmentPrompt(allocator, input, &ctx);
    defer allocator.free(augmented);

    var modulator = router.AdaptiveModulator.loadWeights(store);

    var completion = try ai.completeWithStoreAdaptive(allocator, store, .{
        .input = augmented,
        .model = model,
        .store_result = config.persist,
    });
    errdefer completion.deinit(allocator);

    var adapted = false;
    if (config.adapt_router) {
        modulator.update(router.analyzeSentiment(input));
        // Observe adaptive-weight persistence when a tracker is provided:
        // saveWeights' serialize buffer is alloc+freed internally (store.store
        // dupes it), so a TrackingAllocator records a balanced alloc/free without
        // touching any caller-owned buffer.
        var save_ta: ?core_memory.TrackingAllocator =
            if (config.tracker) |t| core_memory.TrackingAllocator.init(allocator, t) else null;
        const save_alloc = if (save_ta) |*ta| ta.allocator() else allocator;
        // A failed weight save must not discard the completion the caller owns;
        // surface it as "not adapted" rather than leaking or aborting.
        if (modulator.saveWeights(save_alloc, store)) |_| {
            adapted = true;
        } else |err| {
            std.log.warn("sea: router weight save failed: {s}", .{@errorName(err)});
            adapted = false;
        }
    }

    return .{
        .completion = completion,
        .evidence_count = evidence_count,
        .adapted = adapted,
        .query_task = plan.task,
    };
}

test "runLearnLoop tracks adaptive-weight persistence when a tracker is set" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var tracker = core_memory.MemoryTracker.init(allocator);
    defer tracker.deinit();

    var result = try runLearnLoop(allocator, &store, "hello world", "abi-local", .{
        .persist = false,
        .adapt_router = true,
        .tracker = &tracker,
    });
    defer result.deinit(allocator);

    // The weight-save routed its serialize buffer through the tracker, so the
    // adaptive-weight persistence is now observable — and it was balanced
    // (alloc+free inside saveWeights), leaving the tracker net-zero.
    try std.testing.expect(result.adapted);
    try std.testing.expect(tracker.getTotalAllocated() > 0);
    try std.testing.expectEqual(tracker.getTotalAllocated(), tracker.getTotalFreed());
}

test "runLearnLoop persists a turn that a later related turn recalls as evidence" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // First turn persisted (query/response vectors + completion metadata stored).
    var first = try runLearnLoop(allocator, &store, "the capital of france is paris", "abi-local", .{
        .persist = true,
        .adapt_router = false,
    });
    first.deinit(allocator);

    // A related second turn recalls the first turn as evidence (vectorCount>0 now).
    var second = try runLearnLoop(allocator, &store, "tell me about paris in france", "abi-local", .{
        .persist = true,
        .adapt_router = false,
    });
    defer second.deinit(allocator);

    try std.testing.expect(second.evidence_count > 0);
}

test "runLearnLoop surfaces the inferred task intent on the result" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var result = try runLearnLoop(allocator, &store, "remember the prior decision we made", "abi-local", .{
        .persist = false,
        .adapt_router = false,
    });
    defer result.deinit(allocator);

    try std.testing.expectEqual(query_plan.TaskType.project_recall, result.query_task);
}

test "runLearnLoop uses saved adaptive weights for later learned completions" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var modulator = router.AdaptiveModulator.deserialize("0.010000,0.980000,0.010000,8,0.050000");
    try modulator.saveWeights(allocator, &store);

    var result = try runLearnLoop(allocator, &store, "analyze the logical structure", "abi-local", .{
        .persist = false,
        .adapt_router = false,
    });
    defer result.deinit(allocator);

    try std.testing.expectEqual(ai.AgentProfile.aviva, result.completion.selected_profile);
    try std.testing.expect(std.mem.indexOf(u8, result.completion.output, "Aviva creative exploration") != null);
}

test {
    std.testing.refAllDecls(@This());
}
