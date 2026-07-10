//! AI feature module — profiles (Abbey, Aviva, Abi), routing, constitution,
//! completion, training, and multi-agent orchestration. Enabled by `-Dfeat-ai`.
//! The router selects a persona based on sentiment/keyword analysis; completion
//! records query/response vectors + blocks to WDBX; training persists metadata
//! through the scheduler. All paths are fully local unless `--live` is passed.
const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const memory_mod = @import("../../core/memory.zig");
const helpers = @import("helpers.zig");
const types = @import("types.zig");
const completion = @import("completion.zig");
const training = @import("training.zig");

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
pub const models = @import("models.zig");
pub const iot_monitor = @import("iot_monitor.zig");
pub const multimodal_fusion = @import("multimodal_fusion.zig");
pub const CompletionRequest = types.CompletionRequest;
pub const CompletionResult = types.CompletionResult;
pub const CompletionTaskContext = types.CompletionTaskContext;
pub const TrainingTaskContext = types.TrainingTaskContext;
pub const AgentTaskContext = types.AgentTaskContext;

pub const AgentConfig = types.AgentConfig;
pub const AgentResult = types.AgentResult;

pub const AgentToolHint = types.AgentToolHint;
const orchestration = @import("orchestration.zig");
pub const AgentWorkerSpec = orchestration.AgentWorkerSpec;
pub const CustomMultiAgentResult = orchestration.CustomMultiAgentResult;
pub const BackgroundAgentBatch = orchestration.BackgroundAgentBatch;
pub const BrowserOrchestrationPlan = orchestration.BrowserOrchestrationPlan;
pub const parseWorkerSpecs = orchestration.parseWorkerSpecs;
pub const freeWorkerSpecs = orchestration.freeWorkerSpecs;
pub const planBrowserOrchestration = orchestration.planBrowserOrchestration;
pub const collectBackgroundBatch = orchestration.collectBackgroundBatch;

pub fn runCustomMultiAgentWithScheduler(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    base_name: []const u8,
    specs: []const AgentWorkerSpec,
    input: []const u8,
) !CustomMultiAgentResult {
    return orchestration.runCustomMultiAgentWithScheduler(allocator, sched, base_name, specs, input, submitAgentTask);
}

pub fn submitAgentsBackground(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    base_name: []const u8,
    specs: []const AgentWorkerSpec,
    input: []const u8,
) !BackgroundAgentBatch {
    return orchestration.submitAgentsBackground(allocator, sched, base_name, specs, input, submitAgentTask);
}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const response = try profile.routeInput(allocator, input);
    const audit = constitution.Constitution.validate(response);
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return response;
}

pub const complete = completion.complete;
pub const submitCompletionTask = completion.submitCompletionTask;
pub const submitTrainingTask = training.submitTrainingTask;

pub const completeWithScheduler = completion.completeWithScheduler;

pub const completeWithStore = completion.completeWithStore;
pub const completeWithStoreAdaptive = completion.completeWithStoreAdaptive;
pub const completion_kv_delta = completion.completion_kv_delta;
pub const completionMetadataKey = completion.completionMetadataKey;

pub const train = training.train;
pub const trainWithStore = training.trainWithStore;
pub const trainKnownProfiles = training.trainKnownProfiles;
pub const evaluate = training.evaluate;

pub fn runAgent(allocator: std.mem.Allocator, config: AgentConfig, input: []const u8) !AgentResult {
    if (config.name.len == 0 or config.instructions.len == 0 or input.len == 0) return error.InvalidAgentConfig;

    const mode: []const u8 = if (config.dry_run) "dry-run" else "review-required";
    const selected = config.profile_override orelse profile.selectBestProfile(profile.analyzeSentiment(input));
    const response = try profile.routeToProfile(allocator, selected, input);
    defer allocator.free(response);
    const audit = constitution.Constitution.validate(response);
    const requires_review = !config.dry_run or !audit.passed;
    const hints_text = try orchestration.formatToolHints(allocator, config.tool_hints);
    defer allocator.free(hints_text);
    const output = try std.fmt.allocPrint(
        allocator,
        "agent={s}\nmode={s}\nselected_profile={s}\nreview_required={s}\ntool_hints={s}\ninstructions={s}\nresponse={s}",
        .{ config.name, mode, selected.label(), if (requires_review) "true" else "false", hints_text, config.instructions, response },
    );
    return .{ .output = output, .requires_review = requires_review };
}

pub fn submitAgentTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *AgentTaskContext) !u64 {
    return try sched.submit(name, .normal, runAgentTask, ctx);
}

pub fn runAgentWithScheduler(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    name: []const u8,
    config: AgentConfig,
    input: []const u8,
) !AgentResult {
    var ctx = AgentTaskContext{
        .allocator = allocator,
        .config = config,
        .input = input,
    };
    defer ctx.deinitResult();

    _ = try submitAgentTask(sched, name, &ctx);
    try sched.runAll();
    const result = ctx.result orelse return error.MissingAgentResult;
    ctx.result = null;
    return result;
}

/// Multi-agent result holding outputs from all personas (Abbey, Aviva, Abi).
/// Enables true multi-agent workflows for TUI parity with Claude Code / OpenCode / Codex.
pub const MultiAgentResult = struct {
    abbey: AgentResult,
    aviva: AgentResult,
    abi: AgentResult,
    aggregated: []u8,

    pub fn deinit(self: *MultiAgentResult, allocator: std.mem.Allocator) void {
        self.abbey.deinit(allocator);
        self.aviva.deinit(allocator);
        self.abi.deinit(allocator);
        allocator.free(self.aggregated);
    }
};

/// Run all three agents (Abbey, Aviva, Abi) in parallel via scheduler for a given input.
/// Supports multi-agent orchestration, delegation, review handoff.
/// Each uses a persona-specific config for instructions.
pub fn runMultiAgentWithScheduler(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    base_name: []const u8,
    input: []const u8,
) !MultiAgentResult {
    var abbey_ctx = AgentTaskContext{
        .allocator = allocator,
        .config = .{ .name = "abbey", .instructions = "Analytical review and structured safety analysis.", .dry_run = true, .profile_override = .abbey },
        .input = input,
    };
    var aviva_ctx = AgentTaskContext{
        .allocator = allocator,
        .config = .{ .name = "aviva", .instructions = "Creative exploration and alternative perspectives.", .dry_run = true, .profile_override = .aviva },
        .input = input,
    };
    var abi_ctx = AgentTaskContext{
        .allocator = allocator,
        .config = .{ .name = "abi", .instructions = "Concise action-oriented execution plan.", .dry_run = true, .profile_override = .abi },
        .input = input,
    };
    defer abbey_ctx.deinitResult();
    defer aviva_ctx.deinitResult();
    defer abi_ctx.deinitResult();

    const abbey_name = try std.fmt.allocPrint(allocator, "{s}:abbey", .{base_name});
    defer allocator.free(abbey_name);
    const aviva_name = try std.fmt.allocPrint(allocator, "{s}:aviva", .{base_name});
    defer allocator.free(aviva_name);
    const abi_name = try std.fmt.allocPrint(allocator, "{s}:abi", .{base_name});
    defer allocator.free(abi_name);

    _ = try submitAgentTask(sched, abbey_name, &abbey_ctx);
    _ = try submitAgentTask(sched, aviva_name, &aviva_ctx);
    _ = try submitAgentTask(sched, abi_name, &abi_ctx);

    try sched.runAll();

    const abbey_res = abbey_ctx.result orelse return error.MissingAgentResult;
    const aviva_res = aviva_ctx.result orelse return error.MissingAgentResult;
    const abi_res = abi_ctx.result orelse return error.MissingAgentResult;

    // Aggregate for TUI display / further processing
    const aggregated = try std.fmt.allocPrint(allocator, "=== MULTI-AGENT RESULTS ===\n\n[ABBEY]\n{s}\n\n[AVIVA]\n{s}\n\n[ABI]\n{s}\n\n=== END ===", .{ abbey_res.output, aviva_res.output, abi_res.output });
    abbey_ctx.result = null;
    aviva_ctx.result = null;
    abi_ctx.result = null;

    return MultiAgentResult{
        .abbey = abbey_res,
        .aviva = aviva_res,
        .abi = abi_res,
        .aggregated = aggregated,
    };
}

fn runAgentTask(ctx: ?*anyopaque) anyerror!void {
    const c = @as(*AgentTaskContext, @ptrCast(@alignCast(ctx orelse return error.MissingTaskContext)));
    if (c.result) |old| old.deinit(c.allocator);
    c.result = try runAgent(c.allocator, c.config, c.input);
}

pub const isFeatureDisabled = training.isFeatureDisabled;

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

    var tracker = memory_mod.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);

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
    if (build_options.feat_wdbx) {
        try std.testing.expect(result.query_vector_id != null);
        try std.testing.expect(result.response_vector_id != null);
        try std.testing.expectEqual(@as(usize, 2), store.vectorCount());
        try std.testing.expectEqual(@as(usize, 1), store.blockCount());
        try std.testing.expect(tracker.getPeakUsage() > 0);
    }
}

test "scheduled agent task records result and scheduler stats" {
    var scheduler = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    var tracker = memory_mod.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();
    scheduler.setMemoryTracker(&tracker);

    var tracking_alloc = memory_mod.TrackingAllocator.init(std.testing.allocator, &tracker);

    var result = try runAgentWithScheduler(
        tracking_alloc.allocator(),
        &scheduler,
        "agent:plan",
        .{ .name = "test-agent", .instructions = "Plan only", .dry_run = true },
        "plan a safe refactor",
    );
    defer result.deinit(tracking_alloc.allocator());

    const stats = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.completed);
    try std.testing.expectEqual(@as(usize, 0), stats.failed);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "agent=test-agent") != null);
    try std.testing.expect(tracker.getPeakUsage() > 0);
}

test "custom multi-agent workers via orchestration" {
    var scheduler = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    const specs = [_]AgentWorkerSpec{
        .{ .name = "alpha", .instructions = "First worker", .profile_override = .abbey, .tool_hints = &.{.plan} },
        .{ .name = "beta", .instructions = "Second worker", .profile_override = .aviva, .tool_hints = &.{.explore} },
    };

    var result = try runCustomMultiAgentWithScheduler(std.testing.allocator, &scheduler, "agent:workers", &specs, "coordinate release");
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), result.results.len);
    try std.testing.expect(std.mem.indexOf(u8, result.results[0].result.output, "tool_hints=plan") != null);
}

test "multi-agent scheduler routes each named persona explicitly" {
    var scheduler = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    var result = try runMultiAgentWithScheduler(std.testing.allocator, &scheduler, "agent:multi", "neutral request");
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.abbey.output, "selected_profile=abbey") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.aviva.output, "selected_profile=aviva") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.abi.output, "selected_profile=abi") != null);
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
    _ = @import("completion.zig");
    _ = @import("training.zig");
    _ = @import("streaming.zig");
    _ = @import("constitution.zig");
    _ = @import("types.zig");
    _ = @import("training_support.zig");
    _ = @import("models.zig");
    _ = @import("iot_monitor.zig");
    _ = @import("multimodal_fusion.zig");
    _ = @import("orchestration.zig");
    std.testing.refAllDecls(@This());
}
