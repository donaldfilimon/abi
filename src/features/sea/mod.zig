//! SEA — Sparse Evidence Attention self-learning loop.
//!
//! Phase 1: evidence-augmented completion over a durable WDBX store. Recalls
//! prior records relevant to an input, prepends them as prompt context, runs the
//! existing AI completion, and adapts the persona-router weights. Reuses
//! `ai.completeWithStore` and `router.AdaptiveModulator`; introduces no new ML.

const std = @import("std");

const evidence = @import("evidence.zig");
const learn_loop = @import("learn_loop.zig");

pub const EvidenceItem = evidence.EvidenceItem;
pub const EvidenceContext = evidence.EvidenceContext;
pub const gatherEvidence = evidence.gatherEvidence;
pub const augmentPrompt = evidence.augmentPrompt;

pub const LearnLoopConfig = learn_loop.LearnLoopConfig;
pub const LearnLoopResult = learn_loop.LearnLoopResult;
pub const runLearnLoop = learn_loop.runLearnLoop;

test {
    _ = evidence;
    _ = learn_loop;
    std.testing.refAllDecls(@This());
}
