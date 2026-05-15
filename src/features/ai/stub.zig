const std = @import("std");

pub const Principle = enum {
    truthfulness,
    safety,
    helpfulness,
    fairness,
    privacy,
    transparency,
};

pub const AuditResult = struct {
    passed: bool,
    violations: std.bit_set.IntegerBitSet(6),
};

pub const DatasetFormat = enum {
    jsonl,
    csv,
    text,
};

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

pub const abbey = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        _ = allocator;
        _ = input;
        return "";
    }
};

pub const constitution = struct {
    pub const Constitution = struct {
        pub fn validate(response: []const u8) AuditResult {
            _ = response;
            return .{
                .passed = true,
                .violations = std.bit_set.IntegerBitSet(6).initEmpty(),
            };
        }
    };
};

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return "";
}

pub fn train(config: TrainingConfig) !TrainingResult {
    return .{
        .accepted = false,
        .profile = config.profile,
        .dataset_path = config.dataset.path,
        .artifact_dir = config.artifact_dir,
        .message = "AI feature is disabled",
    };
}

pub fn trainWithStore(allocator: std.mem.Allocator, store: anytype, config: TrainingConfig) !TrainingResult {
    _ = allocator;
    _ = store;
    return train(config);
}

pub fn trainKnownProfiles(allocator: std.mem.Allocator, store: anytype, dataset: DatasetSpec, artifact_dir: []const u8) !TrainingResult {
    _ = allocator;
    _ = store;
    return .{
        .accepted = false,
        .profile = "abbey,aviva,abi",
        .dataset_path = dataset.path,
        .artifact_dir = artifact_dir,
        .message = "AI feature is disabled",
        .records_stored = 0,
    };
}

pub fn evaluate(config: TrainingConfig) !TrainingResult {
    return .{
        .accepted = false,
        .profile = config.profile,
        .dataset_path = config.dataset.path,
        .artifact_dir = config.artifact_dir,
        .message = "AI feature is disabled",
    };
}

pub fn runAgent(allocator: std.mem.Allocator, config: AgentConfig, input: []const u8) !AgentResult {
    _ = config;
    _ = input;
    return .{ .output = try allocator.dupe(u8, "AI feature is disabled"), .requires_review = true };
}
