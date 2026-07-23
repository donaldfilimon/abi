//! MCP AI and WDBX tool bodies.
//!
//! Handlers parse JSON-RPC arguments; this module owns the tool-specific work
//! and response formatting for AI completion/training and WDBX query/status.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const state = @import("state.zig");

const features = abi.features;

pub fn schedulerStatsText(allocator: std.mem.Allocator) ![]u8 {
    const sched = state.getScheduler();
    const s = sched.stats();
    return try std.fmt.allocPrint(
        allocator,
        "scheduler running={d} pending={d} completed={d} failed={d} cancelled={d} total_tasks={d} source=mcp-server",
        .{ s.running, s.pending, s.completed, s.failed, s.cancelled, s.total_tasks },
    );
}

pub fn runTraining(allocator: std.mem.Allocator, config: features.ai.TrainingConfig) ![]u8 {
    const ai_mod = features.ai;
    const store = try state.getWdbxStore();
    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    const before = store.stats();
    var result = blk: {
        var ctx = ai_mod.TrainingTaskContext{
            .allocator = allocator,
            .store = store,
            .config = config,
        };
        if (ai_mod.submitTrainingTask(state.getScheduler(), "train:mcp", &ctx)) |_| {
            try state.getScheduler().runAll();
            break :blk ctx.result orelse return error.MissingTrainingResult;
        } else |err| {
            if (!ai_mod.isFeatureDisabled(err)) return err;
            break :blk try ai_mod.train(allocator, config);
        }
    };
    defer result.deinit(allocator);
    const after = store.stats();

    const status: []const u8 = if (result.accepted) "training accepted" else "training disabled";
    return try std.fmt.allocPrint(
        allocator,
        "{s} profile={s} dataset={s} records={d} backend={s} wdbx_kv_entries={d} wdbx_vectors={d} wdbx_blocks={d} total_kv_entries={d} total_vectors={d} total_blocks={d}: {s}",
        .{
            status,
            result.profile,
            result.dataset_path,
            result.records_stored,
            result.acceleration_backend,
            state.statDelta(after.kv_entries, before.kv_entries),
            state.statDelta(after.vectors, before.vectors),
            state.statDelta(after.blocks, before.blocks),
            after.kv_entries,
            after.vectors,
            after.blocks,
            result.message,
        },
    );
}

pub fn runLocalCompletion(allocator: std.mem.Allocator, input: []const u8, model: []const u8) ![]u8 {
    const ai_mod = features.ai;
    const store = try state.getWdbxStore();

    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    const before = store.stats();
    var result = try ai_mod.completeWithScheduler(
        allocator,
        store,
        state.getScheduler(),
        "complete:mcp",
        .{ .input = input, .model = model, .store_result = true },
    );
    defer result.deinit(allocator);
    const stats = store.stats();
    const persisted = result.query_vector_id != null and result.response_vector_id != null and result.block_id != null;

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(
        allocator,
        "model={s} profile={s} audit_passed={s} audit_escore={d:.3} audit_vetoed={s} persisted={s} kv_entries={d} vectors={d} blocks={d} total_kv_entries={d} total_vectors={d} total_blocks={d}",
        .{
            result.model,
            result.selected_profile.label(),
            if (result.audit.passed) "true" else "false",
            result.audit.escore,
            if (result.audit.vetoed) "true" else "false",
            if (persisted) "true" else "false",
            state.statDelta(stats.kv_entries, before.kv_entries),
            state.statDelta(stats.vectors, before.vectors),
            state.statDelta(stats.blocks, before.blocks),
            stats.kv_entries,
            stats.vectors,
            stats.blocks,
        },
    );
    try appendCompletionTail(&out, allocator, &result, persisted, stats.acceleration.message);
    return try out.toOwnedSlice(allocator);
}

/// Append the persisted-record IDs and `: {output}` suffix shared by
/// `runLocalCompletion` and `runLearn`. Kept byte-identical to both call
/// sites' historical output (asserted by the MCP tool contract tests).
fn appendCompletionTail(
    out: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    completion: *const features.ai.CompletionResult,
    persisted: bool,
    wdbx_status: []const u8,
) !void {
    if (completion.query_vector_id) |qid| {
        try out.print(allocator, " query_vector_id={d} metadata_key=completion:{d}", .{ qid, qid });
    }
    if (completion.response_vector_id) |rid| try out.print(allocator, " response_vector_id={d}", .{rid});
    if (completion.block_id) |block_id| {
        const block_hex = std.fmt.bytesToHex(block_id, .lower);
        try out.print(allocator, " block_id={s}", .{&block_hex});
    }
    if (!persisted) try out.print(allocator, " wdbx_status={s}", .{wdbx_status});
    try out.print(allocator, ": {s}", .{completion.output});
}

/// `ai_learn`: run one SEA self-learning pass against the ambient MCP WDBX store
/// (the same long-lived store `ai_complete`/`ai_train` use). Mirrors
/// `runLocalCompletion`'s reporting shape and adds `evidence_count` (recalled
/// records) and `adapted` (router-weight update). `features.sea` is build-time
/// selected, so with `-Dfeat-sea=false` the stub degrades to a plain persisted
/// completion (evidence_count=0, adapted=false).
pub fn runLearn(allocator: std.mem.Allocator, input: []const u8, model: []const u8, evidence_limit: usize) ![]u8 {
    const store = try state.getWdbxStore();

    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    const before = store.stats();
    var result = try features.sea.runLearnLoop(allocator, store, input, model, .{ .evidence_limit = evidence_limit });
    defer result.deinit(allocator);

    const completion = result.completion;
    const stats = store.stats();
    const persisted = completion.query_vector_id != null and completion.response_vector_id != null and completion.block_id != null;

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(
        allocator,
        "model={s} profile={s} audit_passed={s} audit_escore={d:.3} audit_vetoed={s} persisted={s} evidence_count={d} adapted={s} kv_entries={d} vectors={d} blocks={d} total_kv_entries={d} total_vectors={d} total_blocks={d}",
        .{
            completion.model,
            completion.selected_profile.label(),
            if (completion.audit.passed) "true" else "false",
            completion.audit.escore,
            if (completion.audit.vetoed) "true" else "false",
            if (persisted) "true" else "false",
            result.evidence_count,
            if (result.adapted) "true" else "false",
            state.statDelta(stats.kv_entries, before.kv_entries),
            state.statDelta(stats.vectors, before.vectors),
            state.statDelta(stats.blocks, before.blocks),
            stats.kv_entries,
            stats.vectors,
            stats.blocks,
        },
    );
    try appendCompletionTail(&out, allocator, &completion, persisted, stats.acceleration.message);
    return try out.toOwnedSlice(allocator);
}

pub fn wdbxStatsText(allocator: std.mem.Allocator) ![]u8 {
    const store = try state.getWdbxStore();
    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    const s = store.stats();
    var dims_buf: [32]u8 = undefined;
    const dims_str = if (s.vector_dimensions) |d| try std.fmt.bufPrint(&dims_buf, "{d}", .{d}) else "null";

    return try std.fmt.allocPrint(
        allocator,
        "kv={d} vectors={d} blocks={d} spatial={d} dims={s} backend={s} source=mcp-store",
        .{ s.kv_entries, s.vectors, s.blocks, s.spatial_records, dims_str, features.gpu.backendName(s.acceleration.backend) },
    );
}

pub fn runLocalWdbxQuery(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    const ai_mod = features.ai;
    const store = try state.getWdbxStore();

    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    seedMcpProfileVectors(allocator, store) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };

    const query_vec = ai_mod.textEmbedding(query);
    if (comptime build_options.feat_wdbx) {
        const weights = ai_mod.profile.analyzeSentiment(query);
        const selected = ai_mod.profile.selectBestProfile(weights);
        const focus_id = profileVectorId(store, selected.label()) orelse 1;
        const query_ctx = QueryPersonaContext{ .store = store, .weights = weights };
        const scorer = features.wdbx.temporal.HybridScorer{ .now_ms = 1000, .half_life_ms = 24 * 60 * 60 * 1000 };
        const ranked = features.wdbx.retrieval.hybridSearchWithPersonaContext(
            std.heap.page_allocator,
            store,
            &query_vec,
            3,
            &store.temporal_graph,
            scorer,
            focus_id,
            &query_ctx,
            queryPersonaWeight,
        ) catch |err| {
            if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
            return err;
        };
        defer std.heap.page_allocator.free(ranked);

        if (ranked.len == 0) return try allocator.dupe(u8, "wdbx query returned no local matches");
        const stats = store.stats();
        return try std.fmt.allocPrint(
            allocator,
            "wdbx local match profile={s} vector_id={d} score={d:.3} semantic={d:.3} temporal={d:.3} causal={d:.3} persona={d:.3} total_vectors={d} total_blocks={d} ranking=hybrid",
            .{
                profileForVector(store, ranked[0].id),
                ranked[0].id,
                ranked[0].score,
                ranked[0].components.semantic,
                ranked[0].components.temporal,
                ranked[0].components.causal,
                ranked[0].components.persona,
                stats.vectors,
                stats.blocks,
            },
        );
    } else {
        const results = store.search(&query_vec, 1) catch |err| {
            if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
            return err;
        };
        defer std.heap.page_allocator.free(results);

        if (results.len == 0) return try allocator.dupe(u8, "wdbx query returned no local matches");
        const stats = store.stats();
        return try std.fmt.allocPrint(
            allocator,
            "wdbx local match profile={s} vector_id={d} score={d:.3} total_vectors={d} total_blocks={d}",
            .{ profileForVector(store, results[0].id), results[0].id, results[0].score, stats.vectors, stats.blocks },
        );
    }
}

const QueryPersonaContext = struct {
    store: *const features.wdbx.Store,
    weights: features.ai.profile.ProfileWeights,
};

fn queryPersonaWeight(ctx: *const anyopaque, id: u32) f32 {
    const query_ctx: *const QueryPersonaContext = @ptrCast(@alignCast(ctx));
    const label = profileForVector(query_ctx.store, id);
    if (std.mem.eql(u8, label, "abbey")) return query_ctx.weights.w_abbey;
    if (std.mem.eql(u8, label, "aviva")) return query_ctx.weights.w_aviva;
    if (std.mem.eql(u8, label, "abi")) return query_ctx.weights.w_abi;
    return 0.5;
}

fn seedMcpProfileVectors(allocator: std.mem.Allocator, store: *features.wdbx.Store) !void {
    if (store.get("wdbx:profiles_seeded") != null and store.temporalNodeCount() > 0) return;

    const labels = abi.features.ai.PROFILE_LABELS;

    var seeded_ids: [labels.len]u32 = undefined;
    for (labels, 0..) |label, i| {
        // Each persona's prototype is the embedding of its name, so it shares the
        // feature space and dimensionality of every other stored vector.
        const vector = features.ai.textEmbedding(label);
        const id = try store.putVector(&vector);
        seeded_ids[i] = id;
        const key = try std.fmt.allocPrint(allocator, "wdbx:profile:{d}", .{id});
        defer allocator.free(key);
        try store.store(key, label);
        try store.addTemporalNode(id, 1000);
    }
    try store.addTemporalEdge(seeded_ids[0], seeded_ids[1]);
    try store.addTemporalEdge(seeded_ids[1], seeded_ids[2]);
    try store.store("wdbx:profiles_seeded", "true");
}

fn profileForVector(store: *const features.wdbx.Store, id: u32) []const u8 {
    var key_buf: [64]u8 = undefined;
    const profile_key = std.fmt.bufPrint(&key_buf, "wdbx:profile:{d}", .{id}) catch return "unknown";
    if (store.get(profile_key)) |label| return label;

    const completion_key = std.fmt.bufPrint(&key_buf, abi.features.ai.COMPLETION_KEY_FMT, .{id}) catch return "unknown";
    const metadata = store.get(completion_key) orelse return "unknown";
    return profileFromCompletionMetadata(metadata);
}

fn profileVectorId(store: *const features.wdbx.Store, label: []const u8) ?u32 {
    const stats = store.stats();
    var id: u32 = 1;
    while (id < stats.next_vector_id) : (id += 1) {
        if (std.mem.eql(u8, profileForVector(store, id), label)) return id;
    }
    return null;
}

fn profileFromCompletionMetadata(metadata: []const u8) []const u8 {
    const needle = "\"profile\":";
    const start = std.mem.indexOf(u8, metadata, needle) orelse return "stored-completion";
    const after_key = metadata[start + needle.len ..];
    if (after_key.len == 0 or after_key[0] != '"') return "stored-completion";
    const value_start: usize = 1;
    const value_end = std.mem.indexOfScalar(u8, after_key[value_start..], '"') orelse return "stored-completion";
    return after_key[value_start .. value_start + value_end];
}

test "completion metadata profile parser finds profile value" {
    try std.testing.expectEqualStrings("abbey", profileFromCompletionMetadata("{\"profile\":\"abbey\"}"));
    try std.testing.expectEqualStrings("stored-completion", profileFromCompletionMetadata("{}"));
}

test "query persona weight follows stored profile labels" {
    if (!build_options.feat_wdbx) return;
    var store = features.wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    try store.store("wdbx:profile:7", "aviva");
    const weights = features.ai.profile.ProfileWeights{ .w_abbey = 0.1, .w_aviva = 0.8, .w_abi = 0.1 };
    const ctx = QueryPersonaContext{ .store = &store, .weights = weights };
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), queryPersonaWeight(&ctx, 7), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), queryPersonaWeight(&ctx, 99), 1e-6);
}

test {
    std.testing.refAllDecls(@This());
}
