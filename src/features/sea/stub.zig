//! Disabled SEA stub. Declaration-name parity with `mod.zig` is enforced by
//! `zig build check-parity`. With the feature off, the self-learning loop
//! degrades to a plain completion with no evidence recall and no router
//! adaptation.

const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const ai = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");

/// Types mirror — declaration-name parity with `mod.zig` for `MemoryKind`
/// and `Authority`. The arrays are empty and `score()` always returns 0.
pub const MemoryKind = enum(u8) {
    note,
    user_preference,
    project_decision,
    code_fact,
    tool_output,
    benchmark,
    constraint,
    contradiction,
    summary,

    pub fn parse(s: []const u8) ?MemoryKind {
        return std.meta.stringToEnum(MemoryKind, s);
    }

    pub fn text(self: MemoryKind) []const u8 {
        return @tagName(self);
    }
};

pub const Authority = enum(u8) {
    inferred,
    user_stated,
    tool_verified,
    file_verified,
    system_pinned,

    pub fn parse(s: []const u8) ?Authority {
        return std.meta.stringToEnum(Authority, s);
    }

    pub fn text(self: Authority) []const u8 {
        return @tagName(self);
    }

    pub fn score(_: Authority) f32 {
        return 0;
    }
};

pub const SeaSignals = struct {
    semantic: f32 = 0,
    keyword: f32 = 0,
    metadata: f32 = 0,
    recency: f32 = 0,
    authority: f32 = 0,
    graph: f32 = 0,
    contradiction: f32 = 0,
    task_fit: f32 = 0,
};

pub const SeaWeights = struct {
    semantic: f32 = 0.30,
    keyword: f32 = 0.15,
    metadata: f32 = 0.15,
    recency: f32 = 0.10,
    authority: f32 = 0.10,
    graph: f32 = 0.10,
    contradiction: f32 = 0.05,
    task_fit: f32 = 0.05,
};

pub const DEFAULT_SEA_WEIGHTS = SeaWeights{};

pub const SeaCandidate = struct {
    record_id: u32,
    cluster_id: u8,
    estimated_tokens: usize,
    signals: SeaSignals,
    final_score: f32,
};

pub const SeaOptions = struct {
    max_tokens: usize = 4096,
    max_records: usize = 16,
    per_cluster_limit: usize = 4,
    weights: SeaWeights = .{},
};

pub const SeaSelection = struct {
    selected_ids: []u32,
    rejected_ids: []u32,
    total_estimated_tokens: usize,
    reason: []const u8,
};

/// With the feature off the scorer always returns 0.
pub fn seaScore(_: SeaSignals, _: SeaWeights) f32 {
    return 0;
}

pub fn adjustWeightsForTask(base: SeaWeights, task: u8) SeaWeights {
    _ = task;
    return base;
}

pub fn selectSeaCandidates(allocator: std.mem.Allocator, candidates: []SeaCandidate, options: SeaOptions) !SeaSelection {
    _ = candidates;
    _ = options;
    return .{
        .selected_ids = try allocator.alloc(u32, 0),
        .rejected_ids = try allocator.alloc(u32, 0),
        .total_estimated_tokens = 0,
        .reason = "sea feature is disabled",
    };
}

pub fn contextPack(allocator: std.mem.Allocator, selected: *const SeaSelection, candidates: []const SeaCandidate, kind_texts: []const []const u8, snippets: []const []const u8) ![]u8 {
    _ = selected;
    _ = candidates;
    _ = kind_texts;
    _ = snippets;
    return allocator.dupe(u8, "[SEA evidence]\n(disabled)");
}

pub const EvidenceItem = struct {
    vector_id: u32 = 0,
    profile_label: []const u8 = "unknown",
    snippet: []u8 = &.{},
    score: f32 = 0,

    pub fn deinit(self: *EvidenceItem, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const EvidenceContext = struct {
    items: []EvidenceItem = &.{},
    allocator: std.mem.Allocator,

    pub fn deinit(self: *EvidenceContext) void {
        self.items = &.{};
    }

    pub fn isEmpty(self: *const EvidenceContext) bool {
        return self.items.len == 0;
    }
};

/// Mirror of the real `TaskType` so callers compile unchanged with `feat-sea`
/// off. Inference always degrades to `.general` (see `inferQueryPlan`).
pub const TaskType = enum(u8) {
    general,
    implementation_design,
    code_repair,
    legal_review,
    research_synthesis,
    project_recall,
    benchmark_review,

    pub fn text(self: TaskType) []const u8 {
        return @tagName(self);
    }
};

pub const QueryPlan = struct {
    task: TaskType = .general,
    query: []const u8,
    require_grounding: bool = true,
    exact_recall: bool = false,
    recency_bias: f32 = 0.40,
    risk: f32 = 0.50,
};

pub const LearnLoopConfig = struct {
    evidence_limit: usize = 5,
    persist: bool = true,
    adapt_router: bool = true,
    max_prompt_bytes: usize = 4096,
    stream_callback: ?ai.StreamCallback = null,
    stream_ctx: ?*anyopaque = null,
};

pub const LearnLoopResult = struct {
    completion: ai.CompletionResult,
    evidence_count: usize = 0,
    adapted: bool = false,
    query_task: TaskType = .general,

    pub fn deinit(self: *LearnLoopResult, allocator: std.mem.Allocator) void {
        self.completion.deinit(allocator);
    }
};

/// With the feature off there is no inference: every query degrades to a plain
/// `.general` plan with no task-aware tuning.
pub fn inferQueryPlan(query: []const u8) QueryPlan {
    return .{ .query = query };
}

pub fn gatherEvidence(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    limit: usize,
) !EvidenceContext {
    _ = store;
    _ = input;
    _ = limit;
    return .{ .items = &.{}, .allocator = allocator };
}

pub fn gatherEvidenceWithPlan(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    limit: usize,
    plan: QueryPlan,
) !EvidenceContext {
    _ = store;
    _ = input;
    _ = limit;
    _ = plan;
    return .{ .items = &.{}, .allocator = allocator };
}

pub fn augmentPrompt(allocator: std.mem.Allocator, input: []const u8, ctx: *const EvidenceContext) ![]u8 {
    _ = ctx;
    return allocator.dupe(u8, input);
}

pub fn runLearnLoop(
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    input: []const u8,
    model: []const u8,
    config: LearnLoopConfig,
) !LearnLoopResult {
    const completion = try ai.completeWithStore(allocator, store, .{
        .input = input,
        .model = model,
        .store_result = config.persist,
    });
    return .{ .completion = completion, .evidence_count = 0, .adapted = false };
}

test {
    std.testing.refAllDecls(@This());
}
