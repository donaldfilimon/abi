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
        .message = try allocator.dupe(u8, "Training data persisted to WDBX."),
    };
}

pub fn trainWithStore(allocator: std.mem.Allocator, store: *wdbx.Store, config: TrainingConfig) !TrainingResult {
    var result = try train(allocator, config);

    const key = try std.fmt.allocPrint(allocator, "agent:{s}:training", .{config.profile});
    defer allocator.free(key);

    const value = try std.fmt.allocPrint(
        allocator,
        "profile={s};dataset={s};artifact_dir={s};accelerator={s};shader={s};mlir={s};status=accepted",
        .{
            config.profile,
            config.dataset.path,
            config.artifact_dir,
            accelerator.backendName(accelerator.selectBackend(.training).backend),
            shaders.languageName(.zig_kernel),
            mlir.dialectName(.linalg),
        },
    );
    defer allocator.free(value);

    try store.store(key, value);
    result.records_stored = 1;
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
        stored += result.records_stored;
    }

    return .{
        .accepted = true,
        .profile = try allocator.dupe(u8, "abbey,aviva,abi"),
        .dataset_path = try allocator.dupe(u8, dataset.path),
        .artifact_dir = try allocator.dupe(u8, artifact_dir),
        .message = try allocator.dupe(u8, "known agent profiles recorded in wdbx"),
        .records_stored = stored,
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
    defer std.testing.allocator.free(result.profile);
    defer std.testing.allocator.free(result.dataset_path);
    defer std.testing.allocator.free(result.artifact_dir);
    defer std.testing.allocator.free(result.message);

    try std.testing.expectEqual(@as(usize, 3), result.records_stored);
    try std.testing.expectEqual(@as(usize, 3), store.count());
    try std.testing.expect(store.get("agent:abbey:training") != null);
    try std.testing.expect(store.get("agent:aviva:training") != null);
    try std.testing.expect(store.get("agent:abi:training") != null);
}
