const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../../core/scheduler.zig");
const foundation_time = @import("../../foundation/time.zig");
const helpers = @import("helpers.zig");
const router = @import("router.zig");
const constitution = @import("constitution.zig");
const types = @import("types.zig");
const models = @import("models.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const telemetry = if (build_options.feat_telemetry) @import("../telemetry/mod.zig") else @import("../telemetry/stub.zig");

pub fn complete(allocator: std.mem.Allocator, request: types.CompletionRequest) !types.CompletionResult {
    if (request.input.len == 0) return error.InvalidCompletionInput;
    // Resolve catalog aliases (e.g. "fable-5" -> "claude-fable-5") to their
    // canonical id; unknown/freeform ids pass through unchanged. Both branches
    // yield a non-owned slice (static catalog literal or the caller's slice),
    // matching the existing borrowed-`model` lifetime — `deinit` never frees it.
    const resolved_model = models.resolve(request.model) orelse request.model;
    const weights = router.analyzeSentiment(request.input);
    const selected = router.selectBestProfile(weights);
    const response = try router.routeInput(allocator, request.input);
    errdefer allocator.free(response);
    const audit = constitution.Constitution.validate(response);
    // Per-turn governance telemetry: every evaluated turn is counted, and a
    // tripped hard safety veto is counted separately (fire-and-forget; a no-op
    // when `feat-telemetry` is compiled out). The float E-score itself is
    // surfaced on `CompletionResult.audit.escore` and in the persisted metadata
    // rather than the integer counter table.
    telemetry.record("ai.constitution.evaluated");
    if (audit.vetoed) telemetry.record("ai.constitution.vetoed");
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return .{
        .model = resolved_model,
        .selected_profile = selected,
        .output = response,
        .audit = audit,
    };
}

pub fn submitCompletionTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *types.CompletionTaskContext) !u64 {
    if (sched.getMemoryTracker()) |tracker| {
        ctx.store.setTracker(tracker);
    }
    return try sched.submit(name, .high, runCompletionTask, ctx);
}

pub fn completeWithScheduler(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    sched: *scheduler_mod.Scheduler,
    name: []const u8,
    request: types.CompletionRequest,
) !types.CompletionResult {
    var ctx = types.CompletionTaskContext{
        .allocator = allocator,
        .store = store,
        .request = request,
    };

    _ = try submitCompletionTask(sched, name, &ctx);
    try sched.runAll();
    return ctx.result orelse error.MissingCompletionResult;
}

fn runCompletionTask(ctx: ?*anyopaque) anyerror!void {
    const c = @as(*types.CompletionTaskContext, @ptrCast(@alignCast(ctx orelse return error.MissingTaskContext)));
    if (c.result) |old| old.deinit(c.allocator);
    c.result = try completeWithStore(c.allocator, c.store, c.request);
}

pub fn completeWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, request: types.CompletionRequest) !types.CompletionResult {
    var result = try complete(allocator, request);
    errdefer result.deinit(allocator);

    if (!request.store_result) return result;

    const query_vec = helpers.textEmbedding(request.input);
    const response_vec = helpers.textEmbedding(result.output);
    const query_id = store.putVector(&query_vec) catch |err| {
        if (isFeatureDisabled(err)) return result;
        return err;
    };
    const response_id = store.putVector(&response_vec) catch |err| {
        if (isFeatureDisabled(err)) return result;
        return err;
    };

    const metadata = try completionMetadataJson(allocator, request, result, query_id, response_id);
    defer allocator.free(metadata);

    const key = try completionMetadataKey(allocator, query_id);
    defer allocator.free(key);

    // The metadata JSON and key string are transient owned buffers (freed by the
    // defers above). Record them as a balanced alloc/free pair on the store's
    // tracker so the completion persistence step's own memory cost is observable
    // alongside the store's vector tracking, without registering a false leak.
    if (store.getTracker()) |t| {
        const transient = metadata.len + key.len;
        t.trackAllocNoTag(transient);
        t.trackFreeNoTag(transient);
    }

    try store.store(key, metadata);

    const block_id = try store.appendBlock(result.selected_profile.label(), query_id, response_id, metadata);
    result.query_vector_id = query_id;
    result.response_vector_id = response_id;
    result.block_id = block_id;
    return result;
}

fn completionMetadataJson(
    allocator: std.mem.Allocator,
    request: types.CompletionRequest,
    result: types.CompletionResult,
    query_id: u32,
    response_id: u32,
) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"kind\":\"completion\",\"model\":");
    try appendMetadataJsonString(&out, allocator, request.model);
    try out.appendSlice(allocator, ",\"profile\":");
    try appendMetadataJsonString(&out, allocator, result.selected_profile.label());
    const audit_passed = if (result.audit.passed) "true" else "false";
    const audit_vetoed = if (result.audit.vetoed) "true" else "false";
    try out.print(
        allocator,
        ",\"audit_passed\":{s},\"audit_vetoed\":{s},\"escore\":{d:.3},\"input_bytes\":{d},\"output_bytes\":{d},\"query_vector_id\":{d},\"response_vector_id\":{d}}}",
        .{ audit_passed, audit_vetoed, result.audit.escore, request.input.len, result.output.len, query_id, response_id },
    );

    return try out.toOwnedSlice(allocator);
}

fn appendMetadataJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, 0x22);
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => {
                if (byte < 0x20) {
                    try out.appendSlice(allocator, "\\u00");
                    try out.print(allocator, "{X:0>2}", .{byte});
                } else {
                    try out.append(allocator, byte);
                }
            },
        }
    }
    try out.append(allocator, 0x22);
}

/// completion_kv_delta is the documented number of KV entries written by
/// completeWithStore when store_result=true (per public-api contract captured to SCRATCH this turn; verified).
pub const completion_kv_delta: usize = 1;

/// completionMetadataKey returns the key for the completion metadata entry
/// per the committed contract (git show HEAD:docs/contracts/public-api.mdx):
/// "stores JSON completion metadata under `completion:<query_vector_id>`".
pub fn completionMetadataKey(allocator: std.mem.Allocator, query_id: u32) ![]const u8 {
    return std.fmt.allocPrint(allocator, "completion:{d}", .{query_id});
}

test "completionMetadataKey matches committed public-api contract" {
    const key = try completionMetadataKey(std.testing.allocator, 42);
    defer std.testing.allocator.free(key);
    try std.testing.expect(std.mem.startsWith(u8, key, "completion:"));
    try std.testing.expect(std.mem.indexOf(u8, key, "42") != null);
}

fn isFeatureDisabled(err: anyerror) bool {
    return err == error.FeatureDisabled;
}

test "completion rejects empty input" {
    try std.testing.expectError(error.InvalidCompletionInput, complete(std.testing.allocator, .{ .input = "" }));
}

test "complete surfaces the weighted E-score and emits per-turn governance telemetry" {
    if (!build_options.feat_telemetry) return;
    telemetry.reset();
    defer telemetry.reset();

    const before = telemetry.counterValue("ai.constitution.evaluated");
    var result = try complete(std.testing.allocator, .{ .input = "tell me something helpful" });
    defer result.deinit(std.testing.allocator);

    // E-score is surfaced on the completion result and stays in range.
    try std.testing.expect(result.audit.escore >= 0.0 and result.audit.escore <= 1.0);
    // The turn was counted exactly once through the existing telemetry path.
    try std.testing.expectEqual(before + 1, telemetry.counterValue("ai.constitution.evaluated"));
}

test "metadata JSON includes the escore and veto fields" {
    var result = types.CompletionResult{
        .model = "m",
        .selected_profile = .abbey,
        .output = try std.testing.allocator.dupe(u8, "out"),
        .audit = constitution.AuditResult.init(),
    };
    defer result.deinit(std.testing.allocator);

    const metadata = try completionMetadataJson(std.testing.allocator, .{ .input = "in", .model = "m" }, result, 1, 2);
    defer std.testing.allocator.free(metadata);
    try std.testing.expect(std.mem.indexOf(u8, metadata, "\"escore\":1.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, metadata, "\"audit_vetoed\":false") != null);
}

test "metadata JSON escapes model and profile fields" {
    var result = types.CompletionResult{
        .model = "m",
        .selected_profile = .abbey,
        .output = try std.testing.allocator.dupe(u8, "out"),
        .audit = constitution.AuditResult.init(),
    };
    defer result.deinit(std.testing.allocator);

    const metadata = try completionMetadataJson(std.testing.allocator, .{ .input = "in", .model = "m\"x" }, result, 1, 2);
    defer std.testing.allocator.free(metadata);
    try std.testing.expect(std.mem.indexOf(u8, metadata, "\"model\":\"m\\\"x\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, metadata, "\"profile\":\"abbey\"") != null);
}

test "completeWithStore tracks transient persistence memory and frees it" {
    if (!build_options.feat_wdbx) return;
    const memory = @import("../../core/memory.zig");
    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    var tracker = memory.MemoryTracker.init(allocator);
    defer tracker.deinit();
    store.setTracker(&tracker);

    const result = try completeWithStore(allocator, &store, .{ .input = "trace memory", .store_result = true });
    defer result.deinit(allocator);

    // The completion persistence step records its transient metadata + key
    // buffers as a balanced alloc/free pair. Vector inserts are persistent (never
    // freed until store.deinit), so a non-zero total-freed isolates and proves the
    // newly-wired AI-internal transient tracking actually fired and balanced.
    try std.testing.expect(tracker.getTotalAllocated() > 0);
    try std.testing.expect(tracker.getTotalFreed() > 0);
}

test {
    std.testing.refAllDecls(@This());
}
