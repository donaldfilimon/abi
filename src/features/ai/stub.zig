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

pub const Principle = types.Principle;
pub const AuditResult = types.AuditResult;
pub const DatasetFormat = types.DatasetFormat;
pub const AgentProfile = types.AgentProfile;
pub const known_profiles = types.known_profiles;
pub const DatasetSpec = types.DatasetSpec;
pub const TrainingConfig = types.TrainingConfig;
pub const TrainingResult = types.TrainingResult;
pub const CompletionRequest = types.CompletionRequest;
pub const CompletionResult = types.CompletionResult;
pub const CompletionTaskContext = types.CompletionTaskContext;
pub const TrainingTaskContext = types.TrainingTaskContext;
pub const AgentTaskContext = types.AgentTaskContext;
pub const AgentConfig = types.AgentConfig;
pub const AgentResult = types.AgentResult;

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

// `models` is dependency-free (std only) plain data, so the disabled-AI stub
// reuses the real catalog to keep declaration parity across the mod/stub
// boundary (`zig build check-parity`).
pub const models = @import("models.zig");

// Stub shims for AI agent multi surfaces (names only; bodies disabled for feat-ai=false).
// Required for mod/stub parity check (top-level pub const/fn names).
pub const MultiAgentResult = struct {
    pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
};
pub fn runMultiAgentWithScheduler(_: std.mem.Allocator, _: *scheduler_mod.Scheduler, _: []const u8, _: []const u8) !MultiAgentResult {
    return .{};
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
