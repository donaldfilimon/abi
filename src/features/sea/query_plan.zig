//! QueryPlan / TaskType inference for SEA retrieval.
//!
//! A free-text query is turned into a small, structured plan of tuning knobs by
//! a *deterministic keyword heuristic* — no model call — mirroring the SEA
//! reference design (`docs/spec/sea-design-extract.md` §4). The plan's flags
//! (notably `exact_recall`) let `gatherEvidence` make retrieval task-aware.

const std = @import("std");

/// Recognized retrieval intents. Inferred from the query by `infer`.
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

/// Inferred, structured intent for a query. `query` is borrowed (never owned by
/// the plan). The scalars are reserved tuning knobs with sane defaults; SEA's
/// Phase-1 retrieval consumes `exact_recall` today (and `recency_bias` is
/// carried for future task-aware scoring).
pub const QueryPlan = struct {
    task: TaskType = .general,
    query: []const u8,
    /// Answers should be backed by selected evidence (design intent flag).
    require_grounding: bool = true,
    /// Trust provenance + exact wording over fuzzy semantic similarity. Set by
    /// `project_recall`; re-weights `gatherEvidence` toward lexical overlap.
    exact_recall: bool = false,
    /// Relative preference for fresher records, `[0,1]`. Reserved tuning knob.
    recency_bias: f32 = 0.40,
    /// Risk posture of the task, `[0,1]`. Reserved tuning knob.
    risk: f32 = 0.50,
};

/// One keyword group: any substring match maps the query to `task`.
const KeywordGroup = struct {
    task: TaskType,
    keywords: []const []const u8,
};

/// Keyword groups in ascending precedence — later groups overwrite earlier
/// matches (last-match-wins, per the reference design). `project_recall` is last
/// so an explicit recall intent dominates and enables `exact_recall`.
const KEYWORD_GROUPS = [_]KeywordGroup{
    .{ .task = .implementation_design, .keywords = &.{ "zig", "code", "build.zig", "implement", "design" } },
    .{ .task = .code_repair, .keywords = &.{ "bug", "patch", "compile", "fix", "error" } },
    .{ .task = .legal_review, .keywords = &.{ "contract", "legal", "license", "compliance" } },
    .{ .task = .research_synthesis, .keywords = &.{ "paper", "research", "eval", "study" } },
    .{ .task = .benchmark_review, .keywords = &.{ "benchmark", "throughput", "latency", "perf" } },
    .{ .task = .project_recall, .keywords = &.{ "remember", "prior", "decision", "recall", "earlier" } },
};

/// Deterministically infer a `QueryPlan` from `query` by case-insensitive
/// substring matching (no allocation, no model call). Last-match-wins across the
/// keyword groups; `project_recall` additionally sets `exact_recall`.
pub fn infer(query: []const u8) QueryPlan {
    var plan = QueryPlan{ .query = query };

    for (KEYWORD_GROUPS) |group| {
        for (group.keywords) |kw| {
            if (containsIgnoreCase(query, kw)) {
                plan.task = group.task;
                break;
            }
        }
    }

    switch (plan.task) {
        .project_recall => {
            plan.exact_recall = true;
            // Recall favors provenance/exact wording over freshness.
            plan.recency_bias = 0.20;
        },
        .code_repair => {
            // Repair leans on the freshest structural state.
            plan.recency_bias = 0.60;
        },
        else => {},
    }

    return plan;
}

/// Case-insensitive substring search (ASCII). Local to keep `query_plan` free of
/// cross-feature imports for stub parity.
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

test {
    std.testing.refAllDecls(@This());
}

test "infer defaults to general" {
    const plan = infer("what is the weather like today");
    try std.testing.expectEqual(TaskType.general, plan.task);
    try std.testing.expect(!plan.exact_recall);
}

test "infer detects implementation_design" {
    try std.testing.expectEqual(TaskType.implementation_design, infer("how do I write this in zig").task);
}

test "infer detects code_repair and raises recency bias" {
    const plan = infer("help me fix this compile bug");
    try std.testing.expectEqual(TaskType.code_repair, plan.task);
    try std.testing.expect(plan.recency_bias > 0.5);
}

test "infer detects legal_review" {
    try std.testing.expectEqual(TaskType.legal_review, infer("review this contract for compliance").task);
}

test "infer detects research_synthesis" {
    try std.testing.expectEqual(TaskType.research_synthesis, infer("summarize the research paper").task);
}

test "infer detects benchmark_review" {
    try std.testing.expectEqual(TaskType.benchmark_review, infer("what is the latency benchmark").task);
}

test "infer detects project_recall and enables exact_recall" {
    const plan = infer("remember the prior decision we made");
    try std.testing.expectEqual(TaskType.project_recall, plan.task);
    try std.testing.expect(plan.exact_recall);
    try std.testing.expect(plan.recency_bias < 0.40);
}

test "infer is case-insensitive" {
    try std.testing.expectEqual(TaskType.implementation_design, infer("Rewrite the CODE").task);
}

test "infer is last-match-wins: project_recall overrides earlier code match" {
    // contains both "code" (implementation_design) and "remember" (project_recall);
    // project_recall is later in precedence and wins.
    const plan = infer("remember the code decision");
    try std.testing.expectEqual(TaskType.project_recall, plan.task);
}
