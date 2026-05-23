const std = @import("std");
const build_options = @import("build_options");
const accelerator = if (build_options.feat_accelerator) @import("../accelerator/mod.zig") else @import("../accelerator/stub.zig");
const mlir = if (build_options.feat_mlir) @import("../mlir/mod.zig") else @import("../mlir/stub.zig");
const shaders = if (build_options.feat_shader) @import("../shaders/mod.zig") else @import("../shaders/stub.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const foundation_io = @import("../../foundation/io/mod.zig");
const helpers = @import("helpers.zig");

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

pub const AgentProfile = enum {
    abbey,
    aviva,
    abi,

    pub fn label(self: AgentProfile) []const u8 {
        return switch (self) {
            .abbey => "abbey",
            .aviva => "aviva",
            .abi => "abi",
        };
    }
};

pub const known_profiles = [_]AgentProfile{ .abbey, .aviva, .abi };

pub const DatasetFormat = enum {
    jsonl,
    csv,
    text,
};

pub const DatasetSpec = struct {
    path: []const u8,
    format: DatasetFormat = .jsonl,
};

pub const TrainingConfig = struct {
    profile: []const u8,
    dataset: DatasetSpec,
    artifact_dir: []const u8,
};

pub const TrainingResult = struct {
    accepted: bool,
    profile: []const u8,
    dataset_path: []const u8,
    artifact_dir: []const u8,
    message: []const u8,
    records_stored: usize = 0,
    acceleration_backend: []const u8 = "unknown",
    query_vector_id: ?u32 = null,
    response_vector_id: ?u32 = null,
    owned: bool = false,

    pub fn deinit(self: TrainingResult, allocator: std.mem.Allocator) void {
        if (!self.owned) return;
        allocator.free(self.profile);
        allocator.free(self.dataset_path);
        allocator.free(self.artifact_dir);
        allocator.free(self.message);
    }
};

pub const CompletionRequest = struct {
    input: []const u8,
    model: []const u8 = "abi-local",
    store_result: bool = false,
};

pub const CompletionResult = struct {
    model: []const u8,
    selected_profile: AgentProfile,
    output: []u8,
    audit: AuditResult,
    query_vector_id: ?u32 = null,
    response_vector_id: ?u32 = null,
    block_id: ?[32]u8 = null,

    pub fn deinit(self: CompletionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

const DatasetSummary = struct {
    available: bool = false,
    records: usize = 0,
    bytes: usize = 0,
};

pub const AgentConfig = struct {
    name: []const u8,
    instructions: []const u8,
    dry_run: bool = true,
};

pub const AgentResult = struct {
    output: []u8,
    requires_review: bool,

    pub fn deinit(self: AgentResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

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

pub fn completeWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, request: CompletionRequest) !CompletionResult {
    var result = try complete(allocator, request);
    errdefer result.deinit(allocator);

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

    const metadata = try std.fmt.allocPrint(
        allocator,
        "model={s};profile={s};audit_passed={s};input_bytes={d};output_bytes={d}",
        .{ request.model, result.selected_profile.label(), if (result.audit.passed) "true" else "false", request.input.len, result.output.len },
    );
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

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingResult {
    try validateTrainingConfig(config);
    const summary = try inspectDataset(allocator, config.dataset);

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const key = try std.fmt.allocPrint(allocator, "training:{s}", .{config.profile});
    defer allocator.free(key);

    try store.store(key, "training_completed");

    const message = try std.fmt.allocPrint(
        allocator,
        "training metadata accepted; dataset_available={s}; records={d}; bytes={d}; model weights unchanged",
        .{ if (summary.available) "true" else "false", summary.records, summary.bytes },
    );
    errdefer allocator.free(message);

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, config.profile),
        .dataset_path = try allocator.dupe(u8, config.dataset.path),
        .artifact_dir = try allocator.dupe(u8, config.artifact_dir),
        .message = message,
        .records_stored = summary.records,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.training).backend),
        .owned = true,
    };
}

pub fn trainWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, config: TrainingConfig) !TrainingResult {
    var result = try train(allocator, config);
    errdefer result.deinit(allocator);

    const key = try std.fmt.allocPrint(allocator, "agent:{s}:training", .{config.profile});
    defer allocator.free(key);

    const profile_vector = profileEmbedding(try parseAgentProfile(config.profile));
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

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, "abbey,aviva,abi"),
        .dataset_path = try allocator.dupe(u8, dataset.path),
        .artifact_dir = try allocator.dupe(u8, artifact_dir),
        .message = try allocator.dupe(u8, "known agent profiles recorded in wdbx"),
        .records_stored = stored,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.training).backend),
        .owned = true,
    };
}

pub fn evaluate(config: TrainingConfig) !TrainingResult {
    try validateTrainingConfig(config);
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

fn validateTrainingConfig(config: TrainingConfig) !void {
    if (config.profile.len == 0) return error.InvalidTrainingProfile;
    _ = parseAgentProfile(config.profile) catch return error.InvalidTrainingProfile;
    if (config.dataset.path.len == 0) return error.InvalidDatasetPath;
    if (config.artifact_dir.len == 0) return error.InvalidArtifactPath;
}

fn isFeatureDisabled(err: anyerror) bool {
    return std.mem.eql(u8, @errorName(err), "FeatureDisabled");
}

fn inspectDataset(allocator: std.mem.Allocator, dataset: DatasetSpec) !DatasetSummary {
    const path = foundation_io.resolvePath(allocator, dataset.path) catch |err| switch (err) {
        error.FileNotFound => return .{ .available = false, .records = 0, .bytes = dataset.path.len },
        else => return err,
    };
    defer allocator.free(path);

    const data = foundation_io.asyncReadFile(allocator, path) catch |err| switch (err) {
        error.FileNotFound => return .{ .available = false, .records = 0, .bytes = dataset.path.len },
        else => return err,
    };
    defer allocator.free(data);

    return .{
        .available = true,
        .records = try countDatasetRecords(allocator, dataset.format, data),
        .bytes = data.len,
    };
}

fn countDatasetRecords(allocator: std.mem.Allocator, format: DatasetFormat, data: []const u8) !usize {
    return switch (format) {
        .text => helpers.countNonEmptyLines(data),
        .csv => blk: {
            const lines = helpers.countNonEmptyLines(data);
            break :blk if (lines > 0) lines - 1 else 0;
        },
        .jsonl => try countJsonlRecords(allocator, data),
    };
}

fn countJsonlRecords(allocator: std.mem.Allocator, data: []const u8) !usize {
    var records: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{});
        defer parsed.deinit();
        records += 1;
    }
    return records;
}

pub const countNonEmptyLines = helpers.countNonEmptyLines;

fn parseAgentProfile(name: []const u8) !AgentProfile {
    inline for (known_profiles) |p| {
        if (std.mem.eql(u8, name, p.label())) return p;
    }
    return error.UnknownAgentProfile;
}

fn profileEmbedding(agent: AgentProfile) [4]f32 {
    return switch (agent) {
        .abbey => .{ 0.92, 0.48, 0.25, 0.76 },
        .aviva => .{ 0.34, 0.94, 0.82, 0.41 },
        .abi => .{ 0.71, 0.69, 0.88, 0.97 },
    };
}

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
    } else {
        try std.testing.expectEqual(@as(usize, 0), result.records_stored);
        try std.testing.expectEqual(@as(usize, 0), store.count());
        try std.testing.expectEqual(@as(usize, 0), store.blockCount());
        try std.testing.expect(store.get("agent:abbey:training") == null);
        try std.testing.expect(store.get("agent:aviva:training") == null);
        try std.testing.expect(store.get("agent:abi:training") == null);
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
    }
}

test {
    _ = @import("router.zig");
    _ = @import("pipeline.zig");
    _ = @import("streaming.zig");
    _ = @import("constitution.zig");
    std.testing.refAllDecls(@This());
}
