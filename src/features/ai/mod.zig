const std = @import("std");
const build_options = @import("build_options");
const accelerator = if (build_options.feat_accelerator) @import("../accelerator/mod.zig") else @import("../accelerator/stub.zig");
const mlir = if (build_options.feat_mlir) @import("../mlir/mod.zig") else @import("../mlir/stub.zig");
const shaders = if (build_options.feat_shader) @import("../shaders/mod.zig") else @import("../shaders/stub.zig");
const wdbx = @import("../wdbx/mod.zig");

pub const abbey = @import("abbey/mod.zig");
pub const constitution = @import("constitution/mod.zig");
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
    const response = try abbey.processInput(allocator, input);
    const audit = constitution.Constitution.validate(response);
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return response;
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingResult {
    try validateTrainingConfig(config);

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const key = try std.fmt.allocPrint(allocator, "training:{s}", .{config.profile});
    defer allocator.free(key);

    try store.store(key, "training_completed");

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, config.profile),
        .dataset_path = try allocator.dupe(u8, config.dataset.path),
        .artifact_dir = try allocator.dupe(u8, config.artifact_dir),
        .message = try allocator.dupe(u8, "training scaffold accepted; no model weights were modified"),
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
    const query_id = try store.putVector(&profile_vector);
    const response_vector = responseEmbedding(profile_vector);
    const response_id = try store.putVector(&response_vector);

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
    result.records_stored = 1;
    result.query_vector_id = query_id;
    result.response_vector_id = response_id;
    allocator.free(result.message);
    result.message = try allocator.dupe(u8, "training scaffold accepted and recorded in wdbx");
    return result;
}

pub fn trainKnownProfiles(allocator: std.mem.Allocator, store: *wdbx.Store, dataset: DatasetSpec, artifact_dir: []const u8) !TrainingResult {
    var stored: usize = 0;
    for (known_profiles) |profile| {
        const result = try trainWithStore(allocator, store, .{
            .profile = profile.label(),
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
        .message = "evaluation scaffold accepted; metrics are not implemented yet",
        .records_stored = 0,
        .acceleration_backend = accelerator.backendName(accelerator.selectBackend(.inference).backend),
    };
}

pub fn runAgent(allocator: std.mem.Allocator, config: AgentConfig, input: []const u8) !AgentResult {
    if (config.name.len == 0 or config.instructions.len == 0 or input.len == 0) return error.InvalidAgentConfig;

    const mode: []const u8 = if (config.dry_run) "dry-run" else "review-required";
    const output = try std.fmt.allocPrint(
        allocator,
        "agent={s} mode={s} input={s}",
        .{ config.name, mode, input },
    );
    return .{ .output = output, .requires_review = true };
}

fn validateTrainingConfig(config: TrainingConfig) !void {
    if (config.profile.len == 0) return error.InvalidTrainingProfile;
    _ = parseAgentProfile(config.profile) catch return error.InvalidTrainingProfile;
    if (config.dataset.path.len == 0) return error.InvalidDatasetPath;
    if (config.artifact_dir.len == 0) return error.InvalidArtifactPath;
}

fn parseAgentProfile(name: []const u8) !AgentProfile {
    inline for (known_profiles) |profile| {
        if (std.mem.eql(u8, name, profile.label())) return profile;
    }
    return error.UnknownAgentProfile;
}

fn profileEmbedding(profile: AgentProfile) [4]f32 {
    return switch (profile) {
        .abbey => .{ 0.92, 0.48, 0.25, 0.76 },
        .aviva => .{ 0.34, 0.94, 0.82, 0.41 },
        .abi => .{ 0.71, 0.69, 0.88, 0.97 },
    };
}

fn responseEmbedding(query: [4]f32) [4]f32 {
    return .{ query[0] * 0.97, query[1] * 1.01, query[2] * 1.03, query[3] * 0.99 };
}

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

    try std.testing.expectEqual(@as(usize, 3), result.records_stored);
    try std.testing.expectEqual(@as(usize, 3), store.count());
    try std.testing.expectEqual(@as(usize, 6), store.vectors.items.len);
    try std.testing.expectEqual(@as(usize, 3), store.blockCount());
    try std.testing.expect(store.get("agent:abbey:training") != null);
    try std.testing.expect(store.get("agent:aviva:training") != null);
    try std.testing.expect(store.get("agent:abi:training") != null);
}
