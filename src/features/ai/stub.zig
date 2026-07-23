//! AI feature stub — used when `-Dfeat-ai=false`. Maintains declaration-name
//! parity with `mod.zig` for all `pub const`/`pub fn` names (enforced by
//! `zig build check-parity`). Reuses the real `models.zig` (std-only),
//! `helpers.zig` (std-only), and `stub_types.zig`/`stub_profile.zig`/
//! `stub_constitution.zig` to keep dimensionality and type shapes identical.
//! Returns `"AI feature is disabled"` or `error.FeatureDisabled` at runtime.
const std = @import("std");
const build_options = @import("build_options");
const scheduler_mod = @import("../../core/scheduler.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const types = @import("stub_types.zig");
// `helpers` is dependency-free (std only), so the disabled-AI stub can reuse the
// real embedding to keep dimensionality identical across the mod/stub boundary.
const helpers = @import("helpers.zig");
pub const point_neural_net = @import("point_neural_net.zig");
pub const identity = @import("identity.zig");

pub fn routeInputWithSoul(
    allocator: std.mem.Allocator,
    net: ?*point_neural_net.PointNeuralNetwork,
    blend_alpha: f32,
    input: []const u8,
) ![]u8 {
    _ = allocator;
    _ = net;
    _ = blend_alpha;
    _ = input;
    return error.FeatureDisabled;
}

pub fn blendWeights(
    a: @import("router.zig").ProfileWeights,
    b: @import("router.zig").ProfileWeights,
    alpha: f32,
) @import("router.zig").ProfileWeights {
    _ = a;
    _ = b;
    _ = alpha;
    return .{
        .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
        .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
        .w_abi = identity.DEFAULT_ABI_WEIGHT,
    };
}

pub const soul_layout = struct {
    pub const SoulRecord = struct {
        label: []const u8,
        point: point_neural_net.Point,

        pub fn deinit(self: SoulRecord, allocator: std.mem.Allocator) void {
            {
                _ = self;
            }
            {
                _ = allocator;
            }
        }
    };

    pub const SoulLayout = struct {
        records: []SoulRecord,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *SoulLayout) void {
            {
                _ = self;
            }
        }

        pub fn fromJson(allocator: std.mem.Allocator, _: []const u8) !SoulLayout {
            {
                _ = allocator;
            }
            return error.FeatureDisabled;
        }

        pub fn bootstrap(self: *SoulLayout, _: *point_neural_net.PointNeuralNetwork) !void {
            {
                _ = self;
            }
            return error.FeatureDisabled;
        }
    };
};

pub const Principle = types.Principle;
pub const AuditResult = types.AuditResult;
pub const DatasetFormat = types.DatasetFormat;
pub const AgentProfile = types.AgentProfile;
pub const known_profiles = types.known_profiles;
pub const COMPLETION_KEY_FMT = types.COMPLETION_KEY_FMT;
pub const PROFILE_LABELS = types.PROFILE_LABELS;
pub const DatasetSpec = types.DatasetSpec;
pub const TrainingConfig = types.TrainingConfig;
pub const TrainingResult = types.TrainingResult;
pub const CompletionRequest = types.CompletionRequest;
pub const CompletionResult = types.CompletionResult;
pub const StreamChunk = types.StreamChunk;
pub const StreamCallback = types.StreamCallback;
pub const CompletionTaskContext = types.CompletionTaskContext;
pub const TrainingTaskContext = types.TrainingTaskContext;
pub const AgentTaskContext = types.AgentTaskContext;
pub const AgentConfig = types.AgentConfig;
pub const AgentResult = types.AgentResult;

pub const AgentToolHint = types.AgentToolHint;
pub const file_context = struct {
    pub const FileMention = struct { path: []const u8, start: usize, end: usize };
    pub const ContextBudget = struct {
        max_bytes: usize,
        used: usize = 0,
        pub fn init(max_bytes: usize) ContextBudget {
            return .{ .max_bytes = max_bytes };
        }
        pub fn remaining(self: ContextBudget) usize {
            return if (self.used >= self.max_bytes) 0 else self.max_bytes - self.used;
        }
        pub fn canFit(self: ContextBudget, bytes: usize) bool {
            return self.used + bytes <= self.max_bytes;
        }
        pub fn consume(self: *ContextBudget, bytes: usize) void {
            self.used += bytes;
            if (self.used > self.max_bytes) self.used = self.max_bytes;
        }
    };
    pub const TREE_MAX_DEPTH: usize = 3;
    pub const TREE_MAX_ENTRIES: usize = 64;
    pub const DEFAULT_BUDGET_BYTES: usize = 8192;
    pub const GIT_DIFF_DEFAULT_BUDGET_BYTES: usize = 2048;
    pub const TREE_DEFAULT_BUDGET_BYTES: usize = 2048;

    pub const AgentContextOptions = struct {
        include_tree: bool = true,
        include_git_diff: bool = true,
        git_stat_only: bool = true,
        open_path: []const u8 = "",
        open_content: []const u8 = "",
        tree_max_depth: usize = TREE_MAX_DEPTH,
        tree_max_entries: usize = TREE_MAX_ENTRIES,
    };

    pub fn buildAgentContext(io: std.Io, allocator: std.mem.Allocator, input: []const u8, root: []const u8, total_budget: usize, opts: AgentContextOptions) ![]u8 {
        _ = io;
        _ = root;
        _ = total_budget;
        _ = opts;
        return try allocator.dupe(u8, input);
    }
    pub fn parseFileMentions(allocator: std.mem.Allocator, input: []const u8) ![]FileMention {
        _ = input;
        return try allocator.alloc(FileMention, 0);
    }
    pub fn validateMentionPath(path: []const u8, root: []const u8) !void {
        _ = path;
        _ = root;
    }
    pub fn resolveAndInject(io: std.Io, allocator: std.mem.Allocator, input: []const u8, root: []const u8, budget: *ContextBudget) ![]u8 {
        _ = io;
        _ = root;
        _ = budget;
        return try allocator.dupe(u8, input);
    }
};
pub const AgentWorkerSpec = struct {
    name: []const u8,
    instructions: []const u8,
    dry_run: bool = true,
    profile_override: ?AgentProfile = null,
    tool_hints: []const AgentToolHint = &.{},
};

const NamedAgentResult = struct {
    name: []const u8,
    result: AgentResult,
};

pub const CustomMultiAgentResult = struct {
    results: []NamedAgentResult,
    aggregated: []u8,
    task_ids: []u64,

    pub fn deinit(self: *CustomMultiAgentResult, allocator: std.mem.Allocator) void {
        for (self.results) |entry| {
            entry.result.deinit(allocator);
            allocator.free(entry.name);
        }
        allocator.free(self.results);
        allocator.free(self.aggregated);
        allocator.free(self.task_ids);
    }
};

pub const BackgroundAgentBatch = struct {
    allocator: std.mem.Allocator,
    contexts: []*AgentTaskContext,
    task_ids: []u64,

    pub fn deinit(self: *BackgroundAgentBatch) void {
        for (self.contexts) |ctx| {
            ctx.deinitResult();
            self.allocator.destroy(ctx);
        }
        self.allocator.free(self.contexts);
        self.allocator.free(self.task_ids);
    }
};

pub const BrowserOrchestrationPlan = struct {
    output: []u8,
    requires_review: bool,
    execute_requested: bool,

    pub fn deinit(self: BrowserOrchestrationPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

const WorkerSpecError = error{ FeatureDisabled, InvalidWorkerSpec, InvalidAgentToolHint } || std.mem.Allocator.Error;

pub fn parseWorkerSpecs(_: std.mem.Allocator, _: []const u8) WorkerSpecError![]AgentWorkerSpec {
    return error.FeatureDisabled;
}

pub fn freeWorkerSpecs(allocator: std.mem.Allocator, specs: []AgentWorkerSpec) void {
    for (specs) |spec| {
        allocator.free(spec.name);
        allocator.free(spec.instructions);
        allocator.free(spec.tool_hints);
    }
    allocator.free(specs);
}

pub fn planBrowserOrchestration(_: std.mem.Allocator, _: []const u8, _: ?[]const u8, _: bool) !BrowserOrchestrationPlan {
    return error.FeatureDisabled;
}

pub fn runCustomMultiAgentWithScheduler(_: std.mem.Allocator, _: *scheduler_mod.Scheduler, _: []const u8, _: []const AgentWorkerSpec, _: []const u8) !CustomMultiAgentResult {
    return error.FeatureDisabled;
}

pub fn collectBackgroundBatch(_: std.mem.Allocator, _: *BackgroundAgentBatch, _: []const AgentWorkerSpec) !CustomMultiAgentResult {
    return error.FeatureDisabled;
}

pub fn submitAgentsBackground(_: std.mem.Allocator, _: *scheduler_mod.Scheduler, _: []const u8, _: []const AgentWorkerSpec, _: []const u8) !BackgroundAgentBatch {
    return error.FeatureDisabled;
}

pub const profile = @import("stub_profile.zig");
pub const abbey = profile.abbey;
pub const aviva = profile.aviva;
pub const abi_profile = profile.abi_profile;

pub const pipeline = struct {
    pub fn train(profile_name: []const u8) !void {
        _ = profile_name;
    }
};

pub const streaming = struct {
    pub const openai = struct {
        pub const OpenAIRequest = struct {
            model: []const u8,
            stream: bool = false,
            messages: []Message = &.{},
            max_tokens: ?u32 = null,

            pub const Message = struct {
                role: []const u8,
                content: []const u8,
            };
        };

        pub const OpenAIStreamChunk = struct {
            id: []const u8,
            object: []const u8,
            created: i64,
            model: []const u8,
            choices: []StreamChoice,

            pub const StreamChoice = struct {
                index: u32,
                delta: Delta,
                finish_reason: ?[]const u8 = null,

                pub const Delta = struct {
                    role: ?[]const u8 = null,
                    content: ?[]const u8 = null,
                };
            };
        };

        pub fn parseRequest(allocator: std.mem.Allocator, request_body: []const u8) !std.json.Parsed(OpenAIRequest) {
            return try std.json.parseFromSlice(OpenAIRequest, allocator, request_body, .{
                .ignore_unknown_fields = true,
            });
        }

        pub fn handleOpenAIChatCompletions(allocator: std.mem.Allocator, request: []const u8, writer: anytype) !void {
            _ = request;
            _ = allocator;
            try writer.writeAll("{\"error\":\"AI feature is disabled\"}");
        }
    };
};

pub const constitution = @import("stub_constitution.zig");

// Disabled-AI parity shims for the per-turn `PipelineTelemetry` snapshot
// (docs/spec/wdbx-rust-capability-extract.mdx §3). Names mirror `mod.zig`;
// bodies are inert (feature disabled).
pub const pipeline_telemetry = struct {
    pub const NeuralTelemetryView = struct {
        layer_count: usize = 0,
        total_weights: usize = 0,
        nonzero_weights: usize = 0,
        l1_norms: []const f32 = &.{},
    };

    pub const PipelineTelemetryImpl = struct {
        ethical_scores: [6]f32,
        escore: f32,
        vetoed: bool,
        neural: NeuralTelemetryView,
        provider: []u8,
        retrieval_summary: []u8,
        cluster_status: []u8,
        governance_version: []u8,
        prompt_version: []u8,
        guardrail_summary: []u8,
        p99_latency_ms: f64,
        total_errors: u64,
        responses_total: u64,

        pub fn deinit(self: *PipelineTelemetry, allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };

    pub const SnapshotOptions = struct {
        provider: []const u8 = "local",
        retrieval_summary: []const u8 = "",
        cluster_status: []const u8 = "single-node",
        governance_version: []const u8 = "constitution:6.0",
        prompt_version: []const u8 = "prompt:1.0",
        guardrail_summary: []const u8 = "ok",
        p99_latency_ms: f64 = 0,
    };

    pub const ObservabilityHubImpl = struct {
        allocator: std.mem.Allocator,
        responses_total: u64 = 0,
        total_errors: u64 = 0,
        constitution_blocks: u64 = 0,

        pub fn init(allocator: std.mem.Allocator) ObservabilityHub {
            return .{ .allocator = allocator };
        }
        pub fn recordResponse(self: *ObservabilityHub) void {
            _ = self;
        }
        pub fn recordError(self: *ObservabilityHub) void {
            _ = self;
        }
        pub fn recordConstitutionBlock(self: *ObservabilityHub) void {
            _ = self;
        }
        pub fn snapshot(
            self: *ObservabilityHub,
            allocator: std.mem.Allocator,
            audit: AuditResult,
            maybe_neural: ?point_neural_net.NeuralTelemetry,
            opts: SnapshotOptions,
        ) !PipelineTelemetry {
            _ = self;
            _ = maybe_neural;
            _ = opts;
            return .{
                .ethical_scores = audit.scores,
                .escore = audit.escore,
                .vetoed = audit.vetoed,
                .neural = .{},
                .provider = try allocator.dupe(u8, "disabled"),
                .retrieval_summary = try allocator.dupe(u8, ""),
                .cluster_status = try allocator.dupe(u8, "disabled"),
                .governance_version = try allocator.dupe(u8, ""),
                .prompt_version = try allocator.dupe(u8, ""),
                .guardrail_summary = try allocator.dupe(u8, ""),
                .p99_latency_ms = 0,
                .total_errors = 0,
                .responses_total = 0,
            };
        }
        pub fn finishTurn(
            self: *ObservabilityHub,
            allocator: std.mem.Allocator,
            audit: AuditResult,
            maybe_neural: ?point_neural_net.NeuralTelemetry,
            opts: SnapshotOptions,
        ) !PipelineTelemetry {
            return self.snapshot(allocator, audit, maybe_neural, opts);
        }
    };
};
pub const PipelineTelemetry = pipeline_telemetry.PipelineTelemetryImpl;
pub const ObservabilityHub = pipeline_telemetry.ObservabilityHubImpl;

// `models` is dependency-free (std only) plain data, so the disabled-AI stub
// reuses the real catalog to keep declaration parity across the mod/stub
// boundary (`zig build check-parity`).
pub const models = @import("models.zig");

// Stub shims for AI agent multi surfaces (names only; bodies disabled for feat-ai=false).
// Required for mod/stub parity check (top-level pub const/fn names).
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
pub fn runMultiAgentWithScheduler(allocator: std.mem.Allocator, _: *scheduler_mod.Scheduler, _: []const u8, _: []const u8) !MultiAgentResult {
    const abbey_output = try allocator.dupe(u8, "AI feature is disabled");
    errdefer allocator.free(abbey_output);
    const aviva_output = try allocator.dupe(u8, "AI feature is disabled");
    errdefer allocator.free(aviva_output);
    const abi_output = try allocator.dupe(u8, "AI feature is disabled");
    errdefer allocator.free(abi_output);
    const aggregated = try allocator.dupe(u8, "AI feature is disabled");
    return .{
        .abbey = .{ .output = abbey_output, .requires_review = true },
        .aviva = .{ .output = aviva_output, .requires_review = true },
        .abi = .{ .output = abi_output, .requires_review = true },
        .aggregated = aggregated,
    };
}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = input;
    return try allocator.dupe(u8, "AI feature is disabled");
}

pub fn complete(allocator: std.mem.Allocator, request: CompletionRequest) !CompletionResult {
    if (request.input.len == 0) return error.InvalidCompletionInput;
    return .{
        .model = request.model,
        .selected_profile = .abbey,
        .output = try allocator.dupe(u8, "AI feature is disabled"),
        .audit = constitution.Constitution.validate("AI feature is disabled"),
    };
}

pub fn submitCompletionTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *CompletionTaskContext) !u64 {
    _ = name;
    _ = ctx;
    _ = sched;
    return error.FeatureDisabled;
}

pub fn submitTrainingTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *TrainingTaskContext) !u64 {
    _ = name;
    _ = ctx;
    _ = sched;
    return error.FeatureDisabled;
}

pub fn submitAgentTask(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *AgentTaskContext) !u64 {
    _ = name;
    _ = ctx;
    _ = sched;
    return error.FeatureDisabled;
}

pub fn runAgentWithScheduler(allocator: std.mem.Allocator, sched: *scheduler_mod.Scheduler, name: []const u8, config: AgentConfig, input: []const u8) !AgentResult {
    _ = sched;
    _ = name;
    return runAgent(allocator, config, input);
}

pub fn completeWithScheduler(allocator: std.mem.Allocator, store: anytype, sched: *scheduler_mod.Scheduler, name: []const u8, request: CompletionRequest) !CompletionResult {
    _ = sched;
    _ = name;
    return completeWithStore(allocator, store, request);
}

pub fn completeStreaming(allocator: std.mem.Allocator, request: CompletionRequest, on_chunk: StreamCallback, callback_ctx: *anyopaque) !CompletionResult {
    const result = try complete(allocator, request);
    errdefer result.deinit(allocator);
    try on_chunk(callback_ctx, .{ .delta = result.output, .done = false });
    try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    return result;
}

pub fn completeWithSchedulerStreaming(allocator: std.mem.Allocator, store: anytype, sched: *scheduler_mod.Scheduler, name: []const u8, request: CompletionRequest, on_chunk: StreamCallback, callback_ctx: *anyopaque) !CompletionResult {
    _ = store;
    _ = sched;
    _ = name;
    return completeStreaming(allocator, request, on_chunk, callback_ctx);
}

pub fn completeWithStore(allocator: std.mem.Allocator, store: anytype, request: CompletionRequest) !CompletionResult {
    _ = store;
    return complete(allocator, request);
}

pub fn completeWithStoreAdaptive(allocator: std.mem.Allocator, store: anytype, request: CompletionRequest) !CompletionResult {
    return completeWithStore(allocator, store, request);
}

pub const completion_kv_delta = 0;

pub fn completionMetadataKey(allocator: std.mem.Allocator, query_id: u32) ![]const u8 {
    _ = query_id;
    return try allocator.dupe(u8, "");
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingResult {
    try validateTrainingConfig(config);
    return .{
        .accepted = false,
        .profile = try allocator.dupe(u8, config.profile),
        .dataset_path = try allocator.dupe(u8, config.dataset.path),
        .artifact_dir = try allocator.dupe(u8, config.artifact_dir),
        .message = try allocator.dupe(u8, "AI feature is disabled"),
        .owned = true,
    };
}

pub fn trainWithStore(allocator: std.mem.Allocator, store: anytype, config: TrainingConfig) !TrainingResult {
    _ = store;
    return train(allocator, config);
}

pub fn trainKnownProfiles(allocator: std.mem.Allocator, store: anytype, dataset: DatasetSpec, artifact_dir: []const u8) !TrainingResult {
    _ = store;
    return .{
        .accepted = false,
        .profile = try allocator.dupe(u8, "abbey,aviva,abi"),
        .dataset_path = try allocator.dupe(u8, dataset.path),
        .artifact_dir = try allocator.dupe(u8, artifact_dir),
        .message = try allocator.dupe(u8, "AI feature is disabled"),
        .records_stored = 0,
        .owned = true,
    };
}

pub fn evaluate(config: TrainingConfig) !TrainingResult {
    try validateTrainingConfig(config);
    return .{
        .accepted = false,
        .profile = config.profile,
        .dataset_path = config.dataset.path,
        .artifact_dir = config.artifact_dir,
        .message = "AI feature is disabled",
    };
}

pub fn runAgent(allocator: std.mem.Allocator, config: AgentConfig, input: []const u8) !AgentResult {
    if (config.name.len == 0 or config.instructions.len == 0 or input.len == 0) return error.InvalidAgentConfig;
    return .{ .output = try allocator.dupe(u8, "AI feature is disabled"), .requires_review = true };
}

pub const iot_monitor = struct {
    pub const IotMonitor = struct {
        allocator: std.mem.Allocator,
        history: std.ArrayListUnmanaged(f64) = .empty,
        z_threshold: f64,

        pub fn init(allocator: std.mem.Allocator) IotMonitor {
            return .{
                .allocator = allocator,
                .history = .empty,
                .z_threshold = 2.5,
            };
        }

        pub fn deinit(self: *IotMonitor) void {
            self.history.deinit(self.allocator);
        }

        pub fn feed(_: *IotMonitor, _: f64) !bool {
            return false;
        }

        pub fn count(self: IotMonitor) usize {
            return self.history.items.len;
        }

        pub fn reset(self: *IotMonitor) void {
            self.history.clearAndFree(self.allocator);
            self.z_threshold = 2.5;
        }
    };
};

pub const multimodal_fusion = struct {
    pub const VisionProcessor = struct {
        pub const EMBEDDING_LEN: usize = 64;

        pub fn encode(allocator: std.mem.Allocator, description: []const u8) ![]f32 {
            _ = description;
            return try allocator.alloc(f32, EMBEDDING_LEN);
        }
    };

    pub const AudioProcessor = struct {
        pub const EMBEDDING_LEN: usize = 32;

        pub fn encode(allocator: std.mem.Allocator, description: []const u8) ![]f32 {
            _ = description;
            return try allocator.alloc(f32, EMBEDDING_LEN);
        }
    };

    pub const IotProcessor = struct {
        pub const EMBEDDING_LEN: usize = 16;

        pub fn encode(allocator: std.mem.Allocator, mean_reading: f64) ![]f32 {
            _ = mean_reading;
            return try allocator.alloc(f32, EMBEDDING_LEN);
        }
    };

    pub fn fuse(allocator: std.mem.Allocator, vision: []const f32, audio: []const f32, iot: []const f32) ![]f32 {
        const total = vision.len + audio.len + iot.len;
        return try allocator.alloc(f32, total);
    }
};

pub fn isFeatureDisabled(err: anyerror) bool {
    return err == error.FeatureDisabled;
}

pub const countNonEmptyLines = helpers.countNonEmptyLines;
pub const textEmbedding = helpers.textEmbedding;
pub const responseEmbedding = helpers.responseEmbedding;

fn validateTrainingConfig(config: TrainingConfig) !void {
    if (config.profile.len == 0) return error.InvalidTrainingProfile;
    _ = parseAgentProfile(config.profile) catch return error.InvalidTrainingProfile;
    if (config.dataset.path.len == 0) return error.InvalidDatasetPath;
    if (config.artifact_dir.len == 0) return error.InvalidArtifactPath;
}

fn parseAgentProfile(name: []const u8) !AgentProfile {
    inline for (known_profiles) |p| {
        if (std.mem.eql(u8, name, p.label())) return p;
    }
    return error.UnknownAgentProfile;
}

test {
    std.testing.refAllDecls(@This());
}

test "ai stub preserves disabled feature contracts" {
    const allocator = std.testing.allocator;

    const run_response = try run(allocator, "hello");
    defer allocator.free(run_response);
    try std.testing.expectEqualStrings("AI feature is disabled", run_response);

    try std.testing.expectError(error.InvalidCompletionInput, complete(allocator, .{ .input = "", .model = "custom-model", .store_result = true }));

    var completion = try complete(allocator, .{ .input = "hello", .model = "custom-model", .store_result = true });
    defer completion.deinit(allocator);
    try std.testing.expectEqualStrings("custom-model", completion.model);
    try std.testing.expectEqualStrings("AI feature is disabled", completion.output);
    try std.testing.expect(completion.query_vector_id == null);
    try std.testing.expect(completion.response_vector_id == null);
    try std.testing.expect(completion.block_id == null);

    const training = try train(allocator, .{
        .profile = "abi",
        .dataset = .{ .path = "data.jsonl" },
        .artifact_dir = "zig-cache/agents",
    });
    defer training.deinit(allocator);
    try std.testing.expect(!training.accepted);
    try std.testing.expectEqualStrings("AI feature is disabled", training.message);

    const known = try trainKnownProfiles(allocator, {}, .{ .path = "data.jsonl" }, "zig-cache/agents");
    defer known.deinit(allocator);
    try std.testing.expect(!known.accepted);
    try std.testing.expectEqual(@as(usize, 0), known.records_stored);

    try std.testing.expectError(error.InvalidTrainingProfile, train(allocator, .{
        .profile = "unknown",
        .dataset = .{ .path = "data.jsonl" },
        .artifact_dir = "zig-cache/agents",
    }));
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    var scheduler = scheduler_mod.Scheduler.init(allocator);
    defer scheduler.deinit();
    var completion_ctx = CompletionTaskContext{
        .allocator = allocator,
        .store = &store,
        .request = .{ .input = "hello" },
    };
    try std.testing.expectError(error.FeatureDisabled, submitCompletionTask(&scheduler, "complete", &completion_ctx));

    var training_ctx = TrainingTaskContext{
        .allocator = allocator,
        .store = &store,
        .config = .{ .profile = "abi", .dataset = .{ .path = "data.jsonl" }, .artifact_dir = "zig-cache/agents" },
    };
    try std.testing.expectError(error.FeatureDisabled, submitTrainingTask(&scheduler, "train", &training_ctx));

    var agent_ctx = AgentTaskContext{
        .allocator = allocator,
        .config = .{ .name = "abi", .instructions = "be safe" },
        .input = "hello",
    };
    try std.testing.expectError(error.FeatureDisabled, submitAgentTask(&scheduler, "agent", &agent_ctx));

    try std.testing.expectError(error.InvalidAgentConfig, runAgent(allocator, .{ .name = "", .instructions = "be safe" }, "hello"));

    var agent_result = try runAgent(allocator, .{ .name = "abi", .instructions = "be safe" }, "hello");
    defer agent_result.deinit(allocator);
    try std.testing.expect(agent_result.requires_review);
    try std.testing.expectEqualStrings("AI feature is disabled", agent_result.output);

    var weights: profile.ProfileWeights = .{ .w_abbey = 2, .w_aviva = 1, .w_abi = 1 };
    weights.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights.w_abbey + weights.w_aviva + weights.w_abi, 0.001);
}
