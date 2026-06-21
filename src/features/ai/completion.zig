const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../../core/scheduler.zig");
const helpers = @import("helpers.zig");
const router = @import("router.zig");
const constitution = @import("constitution.zig");
const types = @import("types.zig");
const models = @import("models.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");

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

    const key = try std.fmt.allocPrint(allocator, "completion:{d}", .{query_id});
    defer allocator.free(key);
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
    try out.print(
        allocator,
        ",\"audit_passed\":{s},\"input_bytes\":{d},\"output_bytes\":{d},\"query_vector_id\":{d},\"response_vector_id\":{d}}}",
        .{ audit_passed, request.input.len, result.output.len, query_id, response_id },
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

fn isFeatureDisabled(err: anyerror) bool {
    return err == error.FeatureDisabled;
}

test "completion rejects empty input" {
    try std.testing.expectError(error.InvalidCompletionInput, complete(std.testing.allocator, .{ .input = "" }));
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

test {
    std.testing.refAllDecls(@This());
}
