//! SEA — Sparse Evidence Attention self-learning loop.
//!
//! Phase 1: evidence-augmented completion over a durable WDBX store. Recalls
//! prior records relevant to an input, prepends them as prompt context, runs the
//! existing AI completion, and adapts the persona-router weights. Reuses
//! `ai.completeWithStore` and `router.AdaptiveModulator`; introduces no new ML.

const std = @import("std");

const evidence = @import("evidence.zig");
const learn_loop = @import("learn_loop.zig");
const query_plan = @import("query_plan.zig");
const sea_types = @import("types.zig");
const scorer = @import("scorer.zig");

pub const EvidenceItem = evidence.EvidenceItem;
pub const EvidenceContext = evidence.EvidenceContext;
pub const gatherEvidence = evidence.gatherEvidence;
pub const gatherEvidenceWithPlan = evidence.gatherEvidenceWithPlan;
pub const augmentPrompt = evidence.augmentPrompt;

pub const QueryPlan = query_plan.QueryPlan;
pub const TaskType = query_plan.TaskType;
pub const inferQueryPlan = query_plan.infer;

pub const MemoryKind = sea_types.MemoryKind;
pub const Authority = sea_types.Authority;

pub const SeaSignals = scorer.SeaSignals;
pub const SeaWeights = scorer.SeaWeights;
pub const SeaCandidate = scorer.SeaCandidate;
pub const SeaOptions = scorer.SeaOptions;
pub const SeaSelection = scorer.SeaSelection;
pub const DEFAULT_SEA_WEIGHTS = scorer.DEFAULT_SEA_WEIGHTS;
pub const seaScore = scorer.seaScore;
pub const adjustWeightsForTask = scorer.adjustWeightsForTask;
pub const selectSeaCandidates = scorer.selectSeaCandidates;
pub const contextPack = scorer.contextPack;

pub const LearnLoopConfig = learn_loop.LearnLoopConfig;
pub const LearnLoopResult = learn_loop.LearnLoopResult;
pub const runLearnLoop = learn_loop.runLearnLoop;

test {
    _ = evidence;
    _ = learn_loop;
    _ = query_plan;
    _ = sea_types;
    _ = scorer;
    std.testing.refAllDecls(@This());
}
