//! Eight-signal SEA scoring and budgeted selection.
//!
//! Ported from `src/features/sea/scorer.zig`. The evidence-recall path builds
//! these candidates from durable WDBX records before applying this combiner.

use crate::query_plan::TaskType;

/// Eight orthogonal scoring signals.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SeaSignals {
    /// Semantic similarity.
    pub semantic: f32,
    /// Lexical overlap.
    pub keyword: f32,
    /// Structural / metadata relevance.
    pub metadata: f32,
    /// Freshness.
    pub recency: f32,
    /// Provenance trust.
    pub authority: f32,
    /// Graph connectivity.
    pub graph: f32,
    /// Explicit contradiction signal.
    pub contradiction: f32,
    /// Task alignment.
    pub task_fit: f32,
}

/// Signal-weight vector. Defaults sum to 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaWeights {
    /// Semantic weight.
    pub semantic: f32,
    /// Keyword weight.
    pub keyword: f32,
    /// Metadata weight.
    pub metadata: f32,
    /// Recency weight.
    pub recency: f32,
    /// Authority weight.
    pub authority: f32,
    /// Graph weight.
    pub graph: f32,
    /// Contradiction weight.
    pub contradiction: f32,
    /// Task-fit weight.
    pub task_fit: f32,
}

/// Default weights from the SEA design extract.
pub const DEFAULT_SEA_WEIGHTS: SeaWeights = SeaWeights {
    semantic: 0.30,
    keyword: 0.15,
    metadata: 0.15,
    recency: 0.10,
    authority: 0.10,
    graph: 0.10,
    contradiction: 0.05,
    task_fit: 0.05,
};

impl Default for SeaWeights {
    fn default() -> Self {
        DEFAULT_SEA_WEIGHTS
    }
}

/// One candidate for budgeted selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SeaCandidate {
    /// Stable WDBX vector id.
    pub record_id: u64,
    /// Diversity cluster id.
    pub cluster_id: u8,
    /// Estimated token cost of admitting this candidate.
    pub estimated_tokens: usize,
    /// Sub-scores.
    pub signals: SeaSignals,
    /// Combined final score.
    pub final_score: f32,
}

/// Budgets and weights for selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaOptions {
    /// Token budget.
    pub max_tokens: usize,
    /// Record-count budget.
    pub max_records: usize,
    /// Per-cluster diversity cap.
    pub per_cluster_limit: usize,
    /// Signal weights.
    pub weights: SeaWeights,
}

impl Default for SeaOptions {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            max_records: 16,
            per_cluster_limit: 4,
            weights: DEFAULT_SEA_WEIGHTS,
        }
    }
}

/// Result of budgeted selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SeaSelection {
    /// Admitted record ids.
    pub selected_ids: Vec<u64>,
    /// Rejected record ids.
    pub rejected_ids: Vec<u64>,
    /// Total estimated tokens of the admitted set.
    pub total_estimated_tokens: usize,
    /// Human-readable reason.
    pub reason: &'static str,
}

/// Weighted-sum combiner into `[0,1]`.
#[must_use]
pub fn sea_score(signals: SeaSignals, weights: SeaWeights) -> f32 {
    let raw = signals.semantic * weights.semantic
        + signals.keyword * weights.keyword
        + signals.metadata * weights.metadata
        + signals.recency * weights.recency
        + signals.authority * weights.authority
        + signals.graph * weights.graph
        + signals.contradiction * weights.contradiction
        + signals.task_fit * weights.task_fit;
    raw.clamp(0.0, 1.0)
}

/// Per-task additive weight adjustments.
#[must_use]
pub fn adjust_weights_for_task(base: SeaWeights, task: TaskType) -> SeaWeights {
    let mut w = base;
    match task {
        TaskType::CodeRepair => {
            w.metadata += 0.05;
            w.task_fit += 0.05;
            w.recency += 0.05;
            w.semantic -= 0.05;
        }
        TaskType::ProjectRecall => {
            w.authority += 0.10;
            w.keyword += 0.05;
            w.semantic -= 0.10;
        }
        TaskType::BenchmarkReview => {
            w.metadata += 0.05;
            w.task_fit += 0.10;
            w.semantic -= 0.05;
        }
        _ => {}
    }
    w
}

/// Budgeted greedy selection by `final_score` descending.
#[must_use]
pub fn select_sea_candidates(
    mut candidates: Vec<SeaCandidate>,
    options: SeaOptions,
) -> SeaSelection {
    if candidates.is_empty() {
        return SeaSelection {
            selected_ids: Vec::new(),
            rejected_ids: Vec::new(),
            total_estimated_tokens: 0,
            reason: "no candidates to select from",
        };
    }

    candidates.sort_by(|a, b| {
        b.final_score
            .total_cmp(&a.final_score)
            .then_with(|| a.record_id.cmp(&b.record_id))
    });

    let mut selected = Vec::new();
    let mut rejected = Vec::new();
    let mut used_tokens = 0_usize;
    let mut cluster_counts = [0_usize; 9];
    let mut seen = std::collections::BTreeSet::new();

    for c in &candidates {
        // A vector id is the durable record identity. Duplicate search hits or
        // callers repeating a candidate must not consume budget twice.
        if !seen.insert(c.record_id) {
            continue;
        }
        let cluster = usize::from(c.cluster_id.min(8));
        let count = cluster_counts[cluster];
        let too_many_cluster = count >= options.per_cluster_limit && c.final_score < 0.92;
        let too_many_records = selected.len() >= options.max_records;
        let too_many_tokens = used_tokens.saturating_add(c.estimated_tokens) > options.max_tokens;

        if too_many_cluster || too_many_records || too_many_tokens {
            rejected.push(c.record_id);
            continue;
        }

        selected.push(c.record_id);
        cluster_counts[cluster] += 1;
        used_tokens += c.estimated_tokens;
    }

    let reason = if rejected.is_empty() {
        "all candidates selected"
    } else {
        "budget-limited"
    };

    SeaSelection {
        selected_ids: selected,
        rejected_ids: rejected,
        total_estimated_tokens: used_tokens,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_sum_to_one() {
        let w = DEFAULT_SEA_WEIGHTS;
        let sum = w.semantic
            + w.keyword
            + w.metadata
            + w.recency
            + w.authority
            + w.graph
            + w.contradiction
            + w.task_fit;
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sea_score_is_clamped() {
        let signals = SeaSignals {
            semantic: 1.0,
            keyword: 1.0,
            metadata: 1.0,
            recency: 1.0,
            authority: 1.0,
            graph: 1.0,
            contradiction: 1.0,
            task_fit: 1.0,
        };
        assert!((sea_score(signals, DEFAULT_SEA_WEIGHTS) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn selection_respects_record_budget() {
        let candidates: Vec<_> = (0..10)
            .map(|i| SeaCandidate {
                record_id: i,
                cluster_id: u8::try_from(i).unwrap_or(8),
                estimated_tokens: 10,
                signals: SeaSignals::default(),
                final_score: 1.0 - f32::from(u16::try_from(i).unwrap_or(0)) * 0.01,
            })
            .collect();
        let selection = select_sea_candidates(
            candidates,
            SeaOptions {
                max_records: 3,
                ..SeaOptions::default()
            },
        );
        assert_eq!(selection.selected_ids.len(), 3);
        assert_eq!(selection.rejected_ids.len(), 7);
        assert_eq!(selection.reason, "budget-limited");
    }

    #[test]
    fn selection_is_deterministic_and_deduplicates_vector_ids() {
        let candidates = vec![
            SeaCandidate {
                record_id: 9,
                cluster_id: 0,
                estimated_tokens: 3,
                signals: SeaSignals::default(),
                final_score: 0.8,
            },
            SeaCandidate {
                record_id: 4,
                cluster_id: 1,
                estimated_tokens: 3,
                signals: SeaSignals::default(),
                final_score: 0.8,
            },
            SeaCandidate {
                record_id: 4,
                cluster_id: 2,
                estimated_tokens: 100,
                signals: SeaSignals::default(),
                final_score: 0.1,
            },
        ];
        let selection = select_sea_candidates(candidates, SeaOptions::default());
        assert_eq!(selection.selected_ids, vec![4, 9]);
        assert!(selection.rejected_ids.is_empty());
        assert_eq!(selection.total_estimated_tokens, 6);
    }

    #[test]
    fn selection_respects_token_and_cluster_budgets() {
        let candidates = vec![
            SeaCandidate {
                record_id: 1,
                cluster_id: 0,
                estimated_tokens: 6,
                signals: SeaSignals::default(),
                final_score: 0.90,
            },
            SeaCandidate {
                record_id: 2,
                cluster_id: 0,
                estimated_tokens: 2,
                signals: SeaSignals::default(),
                final_score: 0.80,
            },
            SeaCandidate {
                record_id: 3,
                cluster_id: 1,
                estimated_tokens: usize::MAX,
                signals: SeaSignals::default(),
                final_score: 0.70,
            },
        ];
        let selection = select_sea_candidates(
            candidates,
            SeaOptions {
                max_tokens: 8,
                max_records: 3,
                per_cluster_limit: 1,
                ..SeaOptions::default()
            },
        );
        assert_eq!(selection.selected_ids, vec![1]);
        assert_eq!(selection.rejected_ids, vec![2, 3]);
        assert_eq!(selection.total_estimated_tokens, 6);
    }
}
