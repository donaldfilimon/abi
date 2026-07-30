//! Temporal/causal graph reconstruction and hybrid vector ranking.
//!
//! This is the Rust counterpart of `temporal.zig` plus the default path in
//! `retrieval.zig`: HNSW supplies semantic candidates, then an in-memory graph
//! reconstructed from the durable snapshot re-ranks them by
//! `semantic * temporal * causal * persona`.

use crate::format::{TemporalKind, TemporalRecord};
use crate::{DurableError, DurableStore};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// One hybrid score split into its observable REST components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreComponents {
    /// Cosine similarity supplied by HNSW, clamped to `[0, 1]`.
    pub semantic: f32,
    /// Exponential recency weight.
    pub temporal: f32,
    /// Causal proximity weight.
    pub causal: f32,
    /// Caller-supplied persona affinity.
    pub persona: f32,
}

impl ScoreComponents {
    /// Multiplicative north-star ranking score.
    #[must_use]
    pub fn combined(self) -> f32 {
        self.semantic * self.temporal * self.causal * self.persona
    }
}

/// One vector after hybrid re-ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedNode {
    /// Durable vector id.
    pub id: u64,
    /// Combined ranking score.
    pub score: f32,
    /// Individual factors used to compute `score`.
    pub components: ScoreComponents,
}

/// Exponential recency weight, halving every `half_life_ms`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn temporal_weight(now_ms: i64, timestamp_ms: i64, half_life_ms: i64) -> f32 {
    if half_life_ms <= 0 {
        return 1.0;
    }
    let age_ms = now_ms.saturating_sub(timestamp_ms).max(0) as f64;
    let weight = 0.5_f64.powf(age_ms / half_life_ms as f64);
    (weight as f32).clamp(0.0, 1.0)
}

/// In-memory undirected reachability view of persisted causal records.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemporalCausalGraph {
    timestamps: BTreeMap<u64, i64>,
    adjacency: BTreeMap<u64, BTreeSet<u64>>,
}

impl TemporalCausalGraph {
    /// Reconstruct a graph from durable temporal records.
    ///
    /// Historical segment records used both `cause`/`effect` and `from`/`to`
    /// field names, so both spellings are accepted.
    #[must_use]
    pub fn from_records(records: &[TemporalRecord]) -> Self {
        let mut graph = Self::default();
        for record in records {
            match record.kind {
                TemporalKind::Node => {
                    let Some(id) = json_u64(record, "id") else {
                        continue;
                    };
                    let Some(timestamp_ms) = json_i64(record, "timestamp_ms") else {
                        continue;
                    };
                    graph.add_node(id, timestamp_ms);
                }
                TemporalKind::Edge => {
                    let cause = json_u64(record, "cause").or_else(|| json_u64(record, "from"));
                    let effect = json_u64(record, "effect").or_else(|| json_u64(record, "to"));
                    if let (Some(cause), Some(effect)) = (cause, effect) {
                        graph.add_causal_edge(cause, effect);
                    }
                }
            }
        }
        graph
    }

    /// Insert or replace a node timestamp.
    pub fn add_node(&mut self, id: u64, timestamp_ms: i64) {
        self.timestamps.insert(id, timestamp_ms);
    }

    /// Add an undirected reachability edge.
    ///
    /// Self-edges are ignored, matching the fact that they add no useful
    /// reachability information to a recovered graph.
    pub fn add_causal_edge(&mut self, cause: u64, effect: u64) {
        if cause == effect {
            return;
        }
        self.adjacency.entry(cause).or_default().insert(effect);
        self.adjacency.entry(effect).or_default().insert(cause);
    }

    /// Timestamp for a node, when one was persisted.
    #[must_use]
    pub fn timestamp_for(&self, id: u64) -> Option<i64> {
        self.timestamps.get(&id).copied()
    }

    /// Unique undirected edge count.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adjacency
            .iter()
            .map(|(from, neighbors)| neighbors.iter().filter(|to| from < *to).count())
            .sum()
    }

    /// Breadth-first hop distance, bounded by `max_hops`.
    #[must_use]
    pub fn hop_distance(&self, from: u64, to: u64, max_hops: u32) -> Option<u32> {
        if from == to {
            return Some(0);
        }

        let mut visited = BTreeSet::from([from]);
        let mut frontier = VecDeque::from([(from, 0_u32)]);
        while let Some((node, depth)) = frontier.pop_front() {
            if depth >= max_hops {
                continue;
            }
            let next_depth = depth + 1;
            for &neighbor in self.adjacency.get(&node).into_iter().flatten() {
                if neighbor == to {
                    return Some(next_depth);
                }
                if visited.insert(neighbor) {
                    frontier.push_back((neighbor, next_depth));
                }
            }
        }
        None
    }
}

/// Parameters for the multiplicative hybrid scorer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridScorer {
    /// Query-time Unix milliseconds.
    pub now_ms: i64,
    /// Recency half-life in milliseconds.
    pub half_life_ms: i64,
    /// Causal multiplier applied per graph hop.
    pub causal_decay: f32,
    /// Minimum causal weight for unrelated nodes.
    pub causal_floor: f32,
    /// Maximum BFS depth.
    pub max_hops: u32,
}

impl HybridScorer {
    /// Construct the default scorer used by the REST route.
    #[must_use]
    pub const fn new(now_ms: i64) -> Self {
        Self {
            now_ms,
            half_life_ms: 24 * 60 * 60 * 1000,
            causal_decay: 0.6,
            causal_floor: 0.25,
            max_hops: 4,
        }
    }

    /// Causal proximity for one candidate.
    #[must_use]
    pub fn causal_weight(self, graph: &TemporalCausalGraph, focus_id: u64, node_id: u64) -> f32 {
        let Some(hops) = graph.hop_distance(focus_id, node_id, self.max_hops) else {
            return self.causal_floor;
        };
        self.causal_floor.max(
            self.causal_decay
                .powi(i32::try_from(hops).unwrap_or(i32::MAX)),
        )
    }

    /// Score one semantic candidate.
    #[must_use]
    pub fn score(
        self,
        graph: &TemporalCausalGraph,
        focus_id: u64,
        node_id: u64,
        semantic: f32,
        persona: f32,
    ) -> ScoreComponents {
        let timestamp_ms = graph.timestamp_for(node_id).unwrap_or(self.now_ms);
        ScoreComponents {
            semantic: semantic.clamp(0.0, 1.0),
            temporal: temporal_weight(self.now_ms, timestamp_ms, self.half_life_ms),
            causal: self.causal_weight(graph, focus_id, node_id),
            persona: persona.clamp(0.0, 1.0),
        }
    }
}

/// Run HNSW search and apply the default REST hybrid model.
pub fn hybrid_search(
    store: &DurableStore,
    query: &[f32],
    limit: usize,
    now_ms: i64,
) -> Result<Vec<RankedNode>, DurableError> {
    let graph = TemporalCausalGraph::from_records(&store.snapshot().temporal);
    let focus_id = store.next_vector_id().saturating_sub(1).max(1);
    let scorer = HybridScorer::new(now_ms);
    let mut ranked = store
        .search(query, limit)?
        .into_iter()
        .map(|result| {
            let components = scorer.score(&graph, focus_id, result.id, result.score, 0.5);
            RankedNode {
                id: result.id,
                score: components.combined(),
                components,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(ranked)
}

fn json_u64(record: &TemporalRecord, field: &str) -> Option<u64> {
    record.fields.get(field).and_then(serde_json::Value::as_u64)
}

fn json_i64(record: &TemporalRecord, field: &str) -> Option<i64> {
    record.fields.get(field).and_then(serde_json::Value::as_i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(kind: TemporalKind, value: &serde_json::Value) -> TemporalRecord {
        TemporalRecord {
            kind,
            fields: value.as_object().expect("object").clone(),
        }
    }

    #[test]
    fn temporal_weight_halves_and_clamps_future_timestamps() {
        assert!((temporal_weight(1_000, 1_000, 1_000) - 1.0).abs() < 1e-6);
        assert!((temporal_weight(2_000, 1_000, 1_000) - 0.5).abs() < 1e-6);
        assert!((temporal_weight(3_000, 1_000, 1_000) - 0.25).abs() < 1e-6);
        assert!((temporal_weight(1_000, 5_000, 1_000) - 1.0).abs() < 1e-6);
        assert!((temporal_weight(5_000, 0, 0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn graph_accepts_both_historical_edge_spellings() {
        let records = vec![
            record(TemporalKind::Node, &json!({"id": 1, "timestamp_ms": 100})),
            record(TemporalKind::Edge, &json!({"cause": 1, "effect": 2})),
            record(TemporalKind::Edge, &json!({"from": 2, "to": 3})),
        ];
        let graph = TemporalCausalGraph::from_records(&records);
        assert_eq!(graph.timestamp_for(1), Some(100));
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.hop_distance(1, 3, 4), Some(2));
        assert_eq!(graph.hop_distance(1, 4, 4), None);
    }

    #[test]
    fn hybrid_score_matches_the_zig_multiplicative_model() {
        let mut graph = TemporalCausalGraph::default();
        graph.add_node(1, 1_000);
        graph.add_node(2, 0);
        graph.add_causal_edge(1, 2);
        let scorer = HybridScorer {
            now_ms: 1_000,
            half_life_ms: 1_000,
            causal_decay: 0.6,
            causal_floor: 0.25,
            max_hops: 4,
        };
        let components = scorer.score(&graph, 1, 2, 1.0, 0.5);
        assert!((components.temporal - 0.5).abs() < 1e-6);
        assert!((components.causal - 0.6).abs() < 1e-6);
        assert!((components.combined() - 0.15).abs() < 1e-6);
    }
}
