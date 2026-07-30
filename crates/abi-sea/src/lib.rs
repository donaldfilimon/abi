//! SEA — Sparse Evidence Attention self-learning loop.
//!
//! Ported from `src/features/sea/`. Phase 1: evidence-augmented completion over
//! a durable WDBX store. Recalls prior records relevant to an input, prepends
//! them as prompt context, runs adaptive AI completion, and updates the
//! persona-router weights. No new ML is introduced.

pub mod evidence;
pub mod learn_loop;
pub mod query_plan;
pub mod scorer;
pub mod types;

pub use evidence::{
    EvidenceContext, EvidenceItem, MAX_PROMPT_BYTES, augment_prompt, gather_evidence,
    gather_evidence_with_plan,
};
pub use learn_loop::{LearnLoopConfig, LearnLoopResult, PersistedIds, run_learn_loop};
pub use query_plan::{QueryPlan, TaskType, infer as infer_query_plan};
pub use scorer::{
    DEFAULT_SEA_WEIGHTS, SeaCandidate, SeaOptions, SeaSelection, SeaSignals, SeaWeights,
    adjust_weights_for_task, sea_score, select_sea_candidates,
};
pub use types::{Authority, MemoryKind};
