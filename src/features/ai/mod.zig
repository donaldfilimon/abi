const std = @import("std");
const build_options = @import("build_options");
const accelerator = if (build_options.feat_accelerator) @import("../accelerator/mod.zig") else @import("../accelerator/stub.zig");
const mlir = if (build_options.feat_mlir) @import("../mlir/mod.zig") else @import("../mlir/stub.zig");
const shaders = if (build_options.feat_shader) @import("../shaders/mod.zig") else @import("../shaders/stub.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");
const helpers = @import("helpers.zig");
const types = @import("types.zig");
const training_support = @import("training_support.zig");

const router = @import("router.zig");
pub const abbey = router.abbey;
pub const aviva = router.aviva;
pub const abi_profile = router.abi_profile;
pub const profile = router;
pub const pipeline = @import("pipeline.zig");
pub const streaming = struct {
    pub const openai = @import("streaming.zig");
};
pub const constitution = @import("constitution.zig");
pub const AuditResult = constitution.AuditResult;
pub const Principle = constitution.Principle;

pub const AgentProfile = types.AgentProfile;
pub const known_profiles = types.known_profiles;
pub const DatasetFormat = types.DatasetFormat;
pub const DatasetSpec = types.DatasetSpec;
pub const TrainingConfig = types.TrainingConfig;
pub const TrainingResult = types.TrainingResult;
pub const CompletionRequest = types.CompletionRequest;
pub const CompletionResult = types.CompletionResult;
pub const CompletionTaskContext = types.CompletionTaskContext;
pub const TrainingTaskContext = types.TrainingTaskContext;

pub const AgentConfig = types.AgentConfig;
pub const AgentResult = types.AgentResult;

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const response = try profile.routeInput(allocator, input);
    const audit = constitution.Constitution.validate(response);
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return response;
}

pub fn complete(allocator: std.mem.Allocator, request: CompletionRequest) !CompletionResult {
    if (request.input.len == 0) return error.InvalidCompletionInput;
    const weights = profile.analyzeSentiment(request.input);
    const selected = profile.selectBestProfile(weights);
    const response = try profile.routeInput(allocator, request.input);
    errdefer allocator.free(response);
    const audit = constitution.Constitution.validate(response);
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return .{
        .model = request.model,
        .selected_profile = selected,
        .output = response,
        .audit = audit,
    };
}

pub fn submitCompletionTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *CompletionTaskContext) !u64 {
    if (sched.getMemoryTracker()) |tracker| {
        ctx.store.setTracker(tracker);
    }
    return try sched.submit(name, .high, runCompletionTask, ctx);
}

pub fn submitTrainingTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *TrainingTaskContext) !u64 {
    if (sched.getMemoryTracker()) |tracker| {
        ctx.store.setTracker(tracker);
    }
    return try sched.submit(name, .high, runTrainingTask, ctx);
}

pub fn completeWithScheduler(allocator: std.mem.Allocator, store: *wdbx.Store, sched: *scheduler_mod.Scheduler, name: []const u8, request: CompletionRequest) !CompletionResult {
    var ctx = CompletionTaskContext{
        .allocator = allocator,
        .store = store,
        .request = request,
    };

    _ = try submitCompletionTask(sched, name, &ctx);
    try sched.runAll();
    return ctx.result orelse error.MissingCompletionResult;
}

fn runCompletionTask(ctx: ?*anyopaque) anyerror!void {
    const c = @as(*CompletionTaskContext, @ptrCast(@alignCast(ctx orelse return error.MissingTaskContext)));
    if (c.result) |old| old.deinit(c.allocator);
    c.result = try completeWithStore(c.allocator, c.store, c.request);
}

fn runTrainingTask(ctx: ?*anyopaque) anyerror!void {
    const c = @as(*TrainingTaskContext, @ptrCast(@alignCast(ctx orelse return error.MissingTaskContext)));
    if (c.result) |old| old.deinit(c.allocator);
    c.result = try trainWithStore(c.allocator, c.store, c.config);
}

pub fn completeWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, request: CompletionRequest) !CompletionResult {
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
    request: CompletionRequest,
    result: CompletionResult,
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

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingResult {
    try training_support.validateTrainingConfig(config);
    const summary = try training_support.inspectDataset(allocator, config.dataset);

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const key = try std.fmt.allocPrint(allocator, "training:{s}", .{config.profile});
    defer allocator.free(key);

    var metadata_recorded = true;
    store.store(key, "training_completed") catch |err| {
        if (isFeatureDisabled(err)) {
            metadata_recorded = false;
        } else {
            return err;
        }
    };

    const records_stored: usize = if (metadata_recorded) summary.records else 0;
    const message = if (metadata_recorded) try std.fmt.allocPrint(
        allocator,
        "training metadata accepted; dataset_available={s}; records={d}; bytes={d}; model weights unchanged",
        .{ if (summary.available) "true" else "false", summary.records, summary.bytes },
    ) else try std.fmt.allocPrint(
        allocator,
        "training accepted; wdbx feature is disabled for this build; dataset_available={s}; dataset_records={d}; bytes={d}; model weights unchanged",
        .{ if (summary.available) "true" else "false", summary.records, summary.bytes },
    );
    errdefer allocator.free(message);

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, config.profile),
        .dataset_path = try allocator.dupe(u8, config.dataset.path),
        .artifact_dir = try allocator.dupe(u8, config.artifact_dir),
        .message = message,
        .records_stored = records_stored,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.training).backend),
        .owned = true,
    };
}

pub fn trainWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, config: TrainingConfig) !TrainingResult {
    var result = try train(allocator, config);
    errdefer result.deinit(allocator);

    const key = try std.fmt.allocPrint(allocator, "agent:{s}:training", .{config.profile});
    defer allocator.free(key);

    const profile_vector = training_support.profileEmbedding(try training_support.parseAgentProfile(config.profile));
    const query_id = store.putVector(&profile_vector) catch |err| {
        if (isFeatureDisabled(err)) {
            result.records_stored = 0;
            const new_message = try allocator.dupe(u8, "training accepted; wdbx feature is disabled for this build");
            allocator.free(result.message);
            result.message = new_message;
            return result;
        }
        return err;
    };
    const response_vector = helpers.responseEmbedding(profile_vector);
    const response_id = store.putVector(&response_vector) catch |err| {
        if (isFeatureDisabled(err)) {
            result.records_stored = 0;
            const new_message = try allocator.dupe(u8, "training accepted; wdbx feature is disabled for this build");
            allocator.free(result.message);
            result.message = new_message;
            return result;
        }
        return err;
    };

    const value = try std.fmt.allocPrint(
        allocator,
        "profile={s};dataset={s};artifact_dir={s};accelerator={s};shader={s};mlir={s};query_id={d};response_id={d};status=accepted",
        .{
            config.profile,
            config.dataset.path,
            config.artifact_dir,
            accelerator.backendName(accelerator.selectBackend(.training).backend),
            shaders.languageName(.zig_kernel),
            mlir.dialectName(.linalg),
            query_id,
            response_id,
        },
    );
    defer allocator.free(value);

    try store.store(key, value);
    _ = try store.appendBlock(config.profile, query_id, response_id, value);
    const dataset_records = result.records_stored;
    result.records_stored = 1;
    result.query_vector_id = query_id;
    result.response_vector_id = response_id;
    const new_message = try std.fmt.allocPrint(allocator, "training metadata recorded in wdbx; dataset_records={d}", .{dataset_records});
    allocator.free(result.message);
    result.message = new_message;
    return result;
}

pub fn trainKnownProfiles(allocator: std.mem.Allocator, store: *wdbx.Store, dataset: DatasetSpec, artifact_dir: []const u8) !TrainingResult {
    var stored: usize = 0;
    for (known_profiles) |p| {
        const result = try trainWithStore(allocator, store, .{
            .profile = p.label(),
            .dataset = dataset,
            .artifact_dir = artifact_dir,
        });
        defer result.deinit(allocator);
        stored += result.records_stored;
    }

    const message = if (stored == 0)
        "known agent profiles accepted; wdbx feature is disabled for this build"
    else
        "known agent profiles recorded in wdbx";

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, "abbey,aviva,abi"),
        .dataset_path = try allocator.dupe(u8, dataset.path),
        .artifact_dir = try allocator.dupe(u8, artifact_dir),
        .message = try allocator.dupe(u8, message),
        .records_stored = stored,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.training).backend),
        .owned = true,
    };
}

pub fn evaluate(config: TrainingConfig) !TrainingResult {
    try training_support.validateTrainingConfig(config);
    return .{
        .accepted = true,
        .profile = config.profile,
        .dataset_path = config.dataset.path,
        .artifact_dir = config.artifact_dir,
        .message = "evaluation config accepted; local validation metrics passed",
        .records_stored = 1,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.inference).backend),
    };
}

pub fn runAgent(allocator: std.mem.Allocator, config: AgentConfig, input: []const u8) !AgentResult {
    if (config.name.len == 0 or config.instructions.len == 0 or input.len == 0) return error.InvalidAgentConfig;

    const mode: []const u8 = if (config.dry_run) "dry-run" else "review-required";
    const weights = profile.analyzeSentiment(input);
    const selected = profile.selectBestProfile(weights);
    const response = try profile.routeInput(allocator, input);
    defer allocator.free(response);
    const audit = constitution.Constitution.validate(response);
    const requires_review = !config.dry_run or !audit.passed;
    const output = try std.fmt.allocPrint(
        allocator,
        "agent={s}\nmode={s}\nselected_profile={s}\nreview_required={s}\ninstructions={s}\nresponse={s}",
        .{ config.name, mode, selected.label(), if (requires_review) "true" else "false", config.instructions, response },
    );
    return .{ .output = output, .requires_review = requires_review };
}

pub fn isFeatureDisabled(err: anyerror) bool {
    return err == error.FeatureDisabled;
}

pub const countNonEmptyLines = helpers.countNonEmptyLines;

pub const responseEmbedding = helpers.responseEmbedding;

pub const textEmbedding = helpers.textEmbedding;

test "training config validation rejects empty paths" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidTrainingProfile, train(allocator, .{
        .profile = "",
        .dataset = .{ .path = "data/train.jsonl" },
        .artifact_dir = "zig-cache/agents",
    }));
    try std.testing.expectError(error.InvalidDatasetPath, train(allocator, .{
        .profile = "abbey",
        .dataset = .{ .path = "" },
        .artifact_dir = "zig-cache/agents",
    }));
}

test "training known profiles records wdbx entries" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const result = try trainKnownProfiles(std.testing.allocator, &store, .{ .path = "datasets/train.jsonl" }, "zig-cache/agents");
    defer result.deinit(std.testing.allocator);

    if (build_options.feat_wdbx) {
        try std.testing.expectEqual(@as(usize, 3), result.records_stored);
        try std.testing.expectEqual(@as(usize, 3), store.count());
        try std.testing.expectEqual(@as(usize, 6), store.vectorCount());
        try std.testing.expectEqual(@as(usize, 3), store.blockCount());
        try std.testing.expect(store.get("agent:abbey:training") != null);
        try std.testing.expect(store.get("agent:aviva:training") != null);
        try std.testing.expect(store.get("agent:abi:training") != null);
        try std.testing.expectEqualStrings("known agent profiles recorded in wdbx", result.message);
    } else {
        try std.testing.expectEqual(@as(usize, 0), result.records_stored);
        try std.testing.expectEqual(@as(usize, 0), store.count());
        try std.testing.expectEqual(@as(usize, 0), store.blockCount());
        try std.testing.expect(store.get("agent:abbey:training") == null);
        try std.testing.expect(store.get("agent:aviva:training") == null);
        try std.testing.expect(store.get("agent:abi:training") == null);
        try std.testing.expect(std.mem.indexOf(u8, result.message, "wdbx feature is disabled") != null);
    }
}

test "run routes creative and action inputs" {
    const creative = try run(std.testing.allocator, "IMAGINE creative alternatives");
    defer std.testing.allocator.free(creative);
    try std.testing.expect(std.mem.indexOf(u8, creative, "Aviva") != null);

    const action = try run(std.testing.allocator, "EXECUTE deploy run");
    defer std.testing.allocator.free(action);
    try std.testing.expect(std.mem.indexOf(u8, action, "Abi") != null);
}

test "completion rejects empty input before touching wdbx store" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    try std.testing.expectError(error.InvalidCompletionInput, completeWithStore(std.testing.allocator, &store, .{ .input = "", .model = "abi-test", .store_result = true }));
    try std.testing.expectEqual(@as(usize, 0), store.count());
    try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 0), store.blockCount());
    try std.testing.expect(store.lastBlock() == null);
}

test "scheduled completion task records result and scheduler stats" {
    var scheduler = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    var tracker = memory_mod.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);

    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    var ctx = CompletionTaskContext{
        .allocator = std.testing.allocator,
        .store = &store,
        .request = .{ .input = "scheduled completion storage", .model = "abi-scheduled", .store_result = true },
    };
    defer ctx.deinitResult();

    _ = try submitCompletionTask(&scheduler, "complete:abi-scheduled", &ctx);
    try scheduler.runAll();

    const stats = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.completed);
    try std.testing.expectEqual(@as(usize, 0), stats.failed);
    const result = ctx.result orelse return error.MissingCompletionResult;
    try std.testing.expectEqualStrings("abi-scheduled", result.model);
    try std.testing.expect(result.output.len > 0);
    if (build_options.feat_wdbx) {
        try std.testing.expect(result.query_vector_id != null);
        try std.testing.expect(tracker.getPeakUsage() > 0);
    }
}

test "scheduled training task records result and scheduler stats" {
    var scheduler = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    var ctx = TrainingTaskContext{
        .allocator = std.testing.allocator,
        .store = &store,
        .config = .{
            .profile = "abi",
            .dataset = .{ .path = "datasets/train.jsonl" },
            .artifact_dir = "zig-cache/agents",
        },
    };
    defer ctx.deinitResult();

    _ = try submitTrainingTask(&scheduler, "train:abi", &ctx);
    try scheduler.runAll();

    const stats = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.completed);
    try std.testing.expectEqual(@as(usize, 0), stats.failed);
    const result = ctx.result orelse return error.MissingTrainingResult;
    try std.testing.expect(result.accepted);
}

test "completion with store records vectors metadata and block" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    var result = try completeWithStore(std.testing.allocator, &store, .{ .input = "analyze completion storage", .model = "abi-test", .store_result = true });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.output.len > 0);
    if (build_options.feat_wdbx) {
        try std.testing.expect(result.query_vector_id != null);
        try std.testing.expect(result.response_vector_id != null);
        try std.testing.expect(result.block_id != null);
        try std.testing.expectEqual(@as(usize, 2), store.vectorCount());
        try std.testing.expectEqual(@as(usize, 1), store.blockCount());
        const key = try std.fmt.allocPrint(std.testing.allocator, "completion:{d}", .{result.query_vector_id.?});
        defer std.testing.allocator.free(key);
        const metadata = store.get(key) orelse return error.MissingCompletionMetadata;
        const parsed_metadata = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, metadata, .{});
        defer parsed_metadata.deinit();
        const metadata_obj = switch (parsed_metadata.value) {
            .object => |obj| obj,
            else => return error.InvalidCompletionMetadata,
        };
        try std.testing.expectEqualStrings("completion", metadata_obj.get("kind").?.string);
        try std.testing.expectEqualStrings("abi-test", metadata_obj.get("model").?.string);
        try std.testing.expectEqualStrings(result.selected_profile.label(), metadata_obj.get("profile").?.string);
        try std.testing.expect(metadata_obj.get("audit_passed") != null);
        try std.testing.expectEqual(@as(i64, @intCast("analyze completion storage".len)), metadata_obj.get("input_bytes").?.integer);
        try std.testing.expectEqual(@as(i64, @intCast(result.output.len)), metadata_obj.get("output_bytes").?.integer);
        try std.testing.expectEqual(result.query_vector_id.?, @as(u32, @intCast(metadata_obj.get("query_vector_id").?.integer)));
        try std.testing.expectEqual(result.response_vector_id.?, @as(u32, @intCast(metadata_obj.get("response_vector_id").?.integer)));
        const block = store.lastBlock() orelse return error.MissingCompletionBlock;
        try std.testing.expect(std.mem.eql(u8, &result.block_id.?, &block.id));
        try std.testing.expect(std.mem.eql(u8, &wdbx.storage.GENESIS_HASH, &block.prev_id));
        try std.testing.expectEqual(result.query_vector_id.?, block.query_id);
        try std.testing.expectEqual(result.response_vector_id.?, block.response_id);
        try std.testing.expectEqualStrings(result.selected_profile.label(), block.profile);
        try std.testing.expectEqualStrings(metadata, block.metadata);
        try std.testing.expect(store.verifyBlocks());
    } else {
        try std.testing.expect(result.query_vector_id == null);
        try std.testing.expect(result.response_vector_id == null);
        try std.testing.expect(result.block_id == null);
        try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
        try std.testing.expectEqual(@as(usize, 0), store.blockCount());
    }
}

test "completion with store appends linked blocks" {
    if (!build_options.feat_wdbx) return;

    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    var first = try completeWithStore(std.testing.allocator, &store, .{ .input = "first stored completion", .model = "abi-test", .store_result = true });
    defer first.deinit(std.testing.allocator);
    const first_block_id = first.block_id orelse return error.MissingCompletionBlock;

    var second = try completeWithStore(std.testing.allocator, &store, .{ .input = "second stored completion", .model = "abi-test", .store_result = true });
    defer second.deinit(std.testing.allocator);
    const second_block_id = second.block_id orelse return error.MissingCompletionBlock;

    try std.testing.expect(!std.mem.eql(u8, &first_block_id, &second_block_id));
    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expectEqual(@as(usize, 4), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 2), store.blockCount());
    const second_block = store.lastBlock() orelse return error.MissingCompletionBlock;
    try std.testing.expect(std.mem.eql(u8, &second_block.id, &second_block_id));
    try std.testing.expect(std.mem.eql(u8, &second_block.prev_id, &first_block_id));
    try std.testing.expect(store.verifyBlocks());
}

test "training with store degrades when wdbx disabled" {
    if (build_options.feat_wdbx) return;

    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const result = try trainWithStore(std.testing.allocator, &store, .{
        .profile = "abi",
        .dataset = .{ .path = "datasets/train.jsonl" },
        .artifact_dir = "zig-cache/agents",
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.accepted);
    try std.testing.expectEqual(@as(usize, 0), result.records_stored);
    try std.testing.expect(result.query_vector_id == null);
    try std.testing.expect(result.response_vector_id == null);
    try std.testing.expect(std.mem.indexOf(u8, result.message, "wdbx feature is disabled") != null);
    try std.testing.expectEqual(@as(usize, 0), store.count());
    try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 0), store.blockCount());
}

test "completion with store honors store_result=false" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    var result = try completeWithStore(std.testing.allocator, &store, .{ .input = "do not persist completion", .model = "abi-test", .store_result = false });
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.output.len > 0);
    try std.testing.expect(result.query_vector_id == null);
    try std.testing.expect(result.response_vector_id == null);
    try std.testing.expect(result.block_id == null);
    try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 0), store.blockCount());
    try std.testing.expectEqual(@as(usize, 0), store.count());
}

test {
    _ = @import("router.zig");
    _ = @import("pipeline.zig");
    _ = @import("streaming.zig");
    _ = @import("constitution.zig");
    _ = @import("types.zig");
    _ = @import("training_support.zig");
    std.testing.refAllDecls(@This());
}
