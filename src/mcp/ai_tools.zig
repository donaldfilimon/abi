//! MCP AI and WDBX tool bodies.
//!
//! Handlers parse JSON-RPC arguments; this module owns the tool-specific work
//! and response formatting for AI completion/training and WDBX query/status.

const std = @import("std");
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
    const store = state.getWdbxStore();
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
    const store = state.getWdbxStore();

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
        "model={s} profile={s} audit_passed={s} persisted={s} kv_entries={d} vectors={d} blocks={d} total_kv_entries={d} total_vectors={d} total_blocks={d}",
        .{
            result.model,
            result.selected_profile.label(),
            if (result.audit.passed) "true" else "false",
            if (persisted) "true" else "false",
            state.statDelta(stats.kv_entries, before.kv_entries),
            state.statDelta(stats.vectors, before.vectors),
            state.statDelta(stats.blocks, before.blocks),
            stats.kv_entries,
            stats.vectors,
            stats.blocks,
        },
    );
    if (result.query_vector_id) |qid| {
        try out.print(allocator, " query_vector_id={d} metadata_key=completion:{d}", .{ qid, qid });
    }
    if (result.response_vector_id) |rid| try out.print(allocator, " response_vector_id={d}", .{rid});
    if (result.block_id) |block_id| {
        const block_hex = std.fmt.bytesToHex(block_id, .lower);
        try out.print(allocator, " block_id={s}", .{&block_hex});
    }
    if (!persisted) try out.print(allocator, " wdbx_status={s}", .{stats.acceleration.message});
    try out.print(allocator, ": {s}", .{result.output});
    return try out.toOwnedSlice(allocator);
}

pub fn wdbxStatsText(allocator: std.mem.Allocator) ![]u8 {
    const store = state.getWdbxStore();
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
    const store = state.getWdbxStore();

    state.lockWdbxStore();
    defer state.unlockWdbxStore();

    seedMcpProfileVectors(allocator, store) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };

    const query_vec = ai_mod.textEmbedding(query);
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

fn seedMcpProfileVectors(allocator: std.mem.Allocator, store: *features.wdbx.Store) !void {
    if (store.get("wdbx:profiles_seeded") != null) return;

    const profiles = [_]struct {
        label: []const u8,
        vector: [4]f32,
    }{
        .{ .label = "abbey", .vector = .{ 0.92, 0.48, 0.25, 0.76 } },
        .{ .label = "aviva", .vector = .{ 0.34, 0.94, 0.82, 0.41 } },
        .{ .label = "abi", .vector = .{ 0.71, 0.69, 0.88, 0.97 } },
    };

    for (profiles) |entry| {
        const id = try store.putVector(&entry.vector);
        const key = try std.fmt.allocPrint(allocator, "wdbx:profile:{d}", .{id});
        defer allocator.free(key);
        try store.store(key, entry.label);
    }
    try store.store("wdbx:profiles_seeded", "true");
}

fn profileForVector(store: *features.wdbx.Store, id: u32) []const u8 {
    var key_buf: [64]u8 = undefined;
    const profile_key = std.fmt.bufPrint(&key_buf, "wdbx:profile:{d}", .{id}) catch return "unknown";
    if (store.get(profile_key)) |label| return label;

    const completion_key = std.fmt.bufPrint(&key_buf, "completion:{d}", .{id}) catch return "unknown";
    const metadata = store.get(completion_key) orelse return "unknown";
    return profileFromCompletionMetadata(metadata);
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

test {
    std.testing.refAllDecls(@This());
}
