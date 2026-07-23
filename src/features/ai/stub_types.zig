const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const core_memory = @import("../../core/memory.zig");

pub const COMPLETION_KEY_FMT = "completion:{d}";
pub const PROFILE_LABELS = [_][]const u8{ "abbey", "aviva", "abi" };

pub const Principle = enum {
    truthfulness,
    safety,
    helpfulness,
    fairness,
    privacy,
    transparency,

    pub fn label(self: Principle) []const u8 {
        return switch (self) {
            .truthfulness => "truthfulness",
            .safety => "safety",
            .helpfulness => "helpfulness",
            .fairness => "fairness",
            .privacy => "privacy",
            .transparency => "transparency",
        };
    }

    pub fn specAlias(self: Principle) []const u8 {
        return switch (self) {
            .truthfulness => "honesty",
            .helpfulness => "autonomy",
            else => self.label(),
        };
    }
};

/// Mirrors `constitution.PRINCIPLE_WEIGHTS` so the disabled-AI `AuditResult`
/// keeps an identical E-score shape (see `constitution.zig` for the pillar
/// mapping). Indexed by `@intFromEnum(Principle)`; sums to 1.0.
pub const PRINCIPLE_WEIGHTS = [6]f32{ 0.20, 0.175, 0.125, 0.20, 0.175, 0.125 };

/// Safety-class principles (Non-Maleficence pillar) — mirror of the real one.
pub const SAFETY_CLASS = [_]Principle{ .safety, .privacy };

/// Severity floor for the safety class — mirror of the real one.
pub const SAFETY_VETO_THRESHOLD: f32 = 0.5;

pub const AuditResult = struct {
    passed: bool,
    violations: std.bit_set.IntegerBitSet(6),
    scores: [6]f32 = .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    escore: f32 = 1.0,
    vetoed: bool = false,
    timestamp: i64 = 0,

    pub fn init() AuditResult {
        return .{
            .passed = true,
            .violations = std.bit_set.IntegerBitSet(6).empty,
        };
    }

    /// Compute the weighted E-score and apply the hard safety veto from the
    /// current `scores`. Mirror of `constitution.AuditResult.finalize`.
    pub fn finalize(self: *AuditResult) void {
        var e: f32 = 0;
        for (self.scores, PRINCIPLE_WEIGHTS) |s, w| e += s * w;
        self.escore = std.math.clamp(e, 0.0, 1.0);

        var vetoed = false;
        for (SAFETY_CLASS) |p| {
            if (self.scores[@intFromEnum(p)] < SAFETY_VETO_THRESHOLD) vetoed = true;
        }
        self.vetoed = vetoed;
        if (vetoed) self.passed = false;
    }
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
    /// Optional tracker (mirrors the real `types.TrainingConfig`). The disabled
    /// training path does no transient accounting, but the field must exist so
    /// callers constructing a `TrainingConfig` compile identically under both.
    tracker: ?*core_memory.MemoryTracker = null,
};

pub const TrainingResult = struct {
    accepted: bool,
    profile: []const u8,
    dataset_path: []const u8,
    artifact_dir: []const u8,
    message: []const u8,
    records_stored: usize = 0,
    acceleration_backend: []const u8 = "disabled",
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

pub const StreamChunk = struct {
    delta: []const u8,
    done: bool,
};

pub const StreamCallback = *const fn (ctx: *anyopaque, chunk: StreamChunk) anyerror!void;

pub const CompletionRequest = struct {
    input: []const u8,
    model: []const u8 = "claude-fable-5",
    /// Mirrors the real explicit-consent persistence switch. Disabled AI never
    /// persists a completion regardless of this value.
    store_result: bool = false,
    stream_callback: ?StreamCallback = null,
    stream_ctx: ?*anyopaque = null,
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

pub const AgentToolHint = enum {
    plan,
    explore,
    browser,

    pub fn label(self: AgentToolHint) []const u8 {
        return switch (self) {
            .plan => "plan",
            .explore => "explore",
            .browser => "browser",
        };
    }

    pub fn parse(name: []const u8) ?AgentToolHint {
        if (std.mem.eql(u8, name, "plan")) return .plan;
        if (std.mem.eql(u8, name, "explore")) return .explore;
        if (std.mem.eql(u8, name, "browser")) return .browser;
        return null;
    }
};

pub const AgentTaskContext = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    input: []const u8,
    result: ?AgentResult = null,

    pub fn deinitResult(self: *AgentTaskContext) void {
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
    profile_override: ?AgentProfile = null,
    tool_hints: []const AgentToolHint = &.{},
};

pub const AgentResult = struct {
    output: []u8,
    requires_review: bool,

    pub fn deinit(self: AgentResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

test {
    std.testing.refAllDecls(@This());
}
