const std = @import("std");
const build_options = @import("build_options");
const constitution = @import("constitution.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");

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
    audit: constitution.AuditResult,
    query_vector_id: ?u32 = null,
    response_vector_id: ?u32 = null,
    block_id: ?[32]u8 = null,

    pub fn deinit(self: CompletionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

pub const CompletionTaskContext = struct {
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    request: CompletionRequest,
    result: ?CompletionResult = null,

    pub fn deinitResult(self: *CompletionTaskContext) void {
        if (self.result) |res| {
            res.deinit(self.allocator);
            self.result = null;
        }
    }
};

pub const TrainingTaskContext = struct {
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    config: TrainingConfig,
    result: ?TrainingResult = null,

    pub fn deinitResult(self: *TrainingTaskContext) void {
        if (self.result) |res| {
            res.deinit(self.allocator);
            self.result = null;
        }
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

test "agent profile labels are stable" {
    try std.testing.expectEqualStrings("abbey", AgentProfile.abbey.label());
    try std.testing.expectEqualStrings("aviva", AgentProfile.aviva.label());
    try std.testing.expectEqualStrings("abi", AgentProfile.abi.label());
}

test {
    std.testing.refAllDecls(@This());
}
