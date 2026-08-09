//! Persona-scoped and spatial hybrid retrieval over a durable WDBX store.

use crate::{
    DurableError, DurableStore, HybridScorer, Point3D, RankedNode, ScoreComponents,
    TemporalCausalGraph, cosine_similarity, euclidean_distance,
};

/// Hybrid-ranked vector with a zero-copy store view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedVector<'store> {
    /// Stable vector identifier.
    pub id: u64,
    /// Multiplicative combined score.
    pub score: f32,
    /// Observable score factors.
    pub components: ScoreComponents,
    /// Borrowed vector, valid until the store mutates.
    pub vector: &'store [f32],
}

/// Run semantic search and apply caller-provided persona affinity.
pub fn hybrid_search_with_persona<'store>(
    store: &'store DurableStore,
    query: &[f32],
    limit: usize,
    graph: &TemporalCausalGraph,
    scorer: HybridScorer,
    focus_id: u64,
    persona_for: impl Fn(u64) -> f32,
) -> Result<Vec<RankedVector<'store>>, DurableError> {
    let mut ranked = store
        .search(query, limit)?
        .into_iter()
        .map(|result| {
            let components = scorer.score(
                graph,
                focus_id,
                result.id,
                result.score,
                persona_for(result.id),
            );
            RankedVector {
                id: result.id,
                score: components.combined(),
                components,
                vector: result.vector,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(ranked)
}

/// Restrict semantic candidates to one persona before hybrid re-ranking.
pub fn hybrid_search_scoped<'store>(
    store: &'store DurableStore,
    query: &[f32],
    limit: usize,
    graph: &TemporalCausalGraph,
    scorer: HybridScorer,
    focus_id: u64,
    keep: impl Fn(u64) -> bool,
) -> Result<Vec<RankedVector<'store>>, DurableError> {
    let available = store.stats().vectors;
    let wanted = limit.saturating_mul(8);
    let pool = if available == 0 {
        limit
    } else {
        wanted.min(available)
    };
    let mut ranked = store
        .search(query, pool)?
        .into_iter()
        .filter(|result| keep(result.id))
        .map(|result| {
            let components = scorer.score(graph, focus_id, result.id, result.score, 1.0);
            RankedVector {
                id: result.id,
                score: components.combined(),
                components,
                vector: result.vector,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    ranked.truncate(limit);
    Ok(ranked)
}

/// Semantic plus 3-D spatial result with borrowed payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedSpatialResult<'store> {
    /// Shared vector/spatial identifier.
    pub id: u64,
    /// Stored point.
    pub point: Point3D,
    /// Borrowed spatial payload.
    pub payload: &'store str,
    /// Weighted combined score.
    pub score: f32,
    /// Cosine similarity component.
    pub semantic_score: f32,
    /// Normalized inverse-distance component.
    pub spatial_score: f32,
}

/// Blend semantic similarity with normalized Euclidean proximity.
pub fn hybrid_spatial_search<'store>(
    store: &'store DurableStore,
    query_vector: &[f32],
    center: Point3D,
    limit: usize,
    weight_semantic: f32,
    weight_spatial: f32,
) -> Vec<RankedSpatialResult<'store>> {
    let pool = limit.saturating_mul(8).min(store.snapshot().spatial.len());
    let mut spatial_hits = store
        .snapshot()
        .spatial
        .values()
        .map(|record| {
            let point = Point3D {
                x: record.x,
                y: record.y,
                z: record.z,
            };
            (record, point, euclidean_distance(center, point))
        })
        .collect::<Vec<_>>();
    spatial_hits.sort_by(|left, right| left.2.total_cmp(&right.2));
    spatial_hits.truncate(pool);
    let maximum_distance = spatial_hits
        .iter()
        .map(|(_, _, distance)| *distance)
        .fold(0.0, f32::max);

    let mut ranked = spatial_hits
        .into_iter()
        .filter_map(|(record, point, distance)| {
            let vector = store.get_vector(record.id)?;
            let semantic_score = cosine_similarity(query_vector, vector);
            let normalized_distance = if maximum_distance == 0.0 {
                0.0
            } else {
                distance / maximum_distance
            };
            let spatial_score = 1.0 - normalized_distance;
            Some(RankedSpatialResult {
                id: record.id,
                point,
                payload: &record.payload,
                score: weight_semantic * semantic_score + weight_spatial * spatial_score,
                semantic_score,
                spatial_score,
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    ranked.truncate(limit);
    ranked
}

/// Convert a borrowed ranked vector to the existing non-vector result shape.
#[must_use]
pub const fn without_vector(result: RankedVector<'_>) -> RankedNode {
    RankedNode {
        id: crate::RecordId::Legacy(result.id),
        score: result.score,
        components: result.components,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::{SpatialRecord, StorePaths};

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(label: &str) -> Self {
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("{label}_{}_{}", std::process::id(), sequence));
            std::fs::create_dir_all(&path).expect("fixture directory");
            Self(path)
        }

        fn store(&self) -> DurableStore {
            DurableStore::open(StorePaths::new(&self.0)).expect("store")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scorer() -> HybridScorer {
        HybridScorer {
            now_ms: 1_000,
            half_life_ms: 1_000,
            causal_decay: 0.6,
            causal_floor: 0.25,
            max_hops: 4,
        }
    }

    #[test]
    fn persona_scope_isolates_vectors_and_preserves_borrowing() {
        let fixture = Fixture::new("abi_retrieval_scope");
        let mut store = fixture.store();
        let first = store.put_vector(&[1.0, 0.0]).expect("first");
        let second = store.put_vector(&[0.0, 1.0]).expect("second");
        let third = store.put_vector(&[0.9, 0.1]).expect("third");
        let fourth = store.put_vector(&[0.0, 0.0]).expect("fourth");
        let mut graph = TemporalCausalGraph::default();
        for id in [first, second, third, fourth] {
            graph.add_node(id, 1_000);
        }
        let ranked = hybrid_search_scoped(&store, &[1.0, 0.0], 10, &graph, scorer(), first, |id| {
            id == first || id == third
        })
        .expect("scoped");
        assert_eq!(
            ranked.iter().map(|result| result.id).collect::<Vec<_>>(),
            [first, third]
        );
        assert_eq!(
            ranked[0].vector.as_ptr(),
            store.get_vector(first).expect("direct").as_ptr()
        );
    }

    #[test]
    fn recency_causality_and_persona_re_rank_equal_semantics() {
        let fixture = Fixture::new("abi_retrieval_persona");
        let mut store = fixture.store();
        let old = store.put_vector(&[1.0, 0.0]).expect("old");
        let recent = store.put_vector(&[1.0, 0.0]).expect("recent");
        let mut graph = TemporalCausalGraph::default();
        graph.add_node(old, 0);
        graph.add_node(recent, 1_000);
        let ranked =
            hybrid_search_with_persona(&store, &[1.0, 0.0], 10, &graph, scorer(), recent, |id| {
                if id == recent { 1.0 } else { 0.2 }
            })
            .expect("ranked");
        assert_eq!(ranked[0].id, recent);
        assert!(ranked[0].components.persona > ranked[1].components.persona);
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn spatial_blend_recovers_pure_semantic_and_pure_spatial_rankings() {
        let fixture = Fixture::new("abi_retrieval_spatial");
        let mut store = fixture.store();
        let semantic = store.put_vector(&[1.0, 0.0]).expect("semantic");
        store
            .put_spatial(SpatialRecord {
                id: semantic,
                x: 100.0,
                y: 100.0,
                z: 100.0,
                payload: "far".to_string(),
            })
            .expect("far");
        let spatial = store.put_vector(&[0.0, 1.0]).expect("spatial");
        store
            .put_spatial(SpatialRecord {
                id: spatial,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                payload: "near".to_string(),
            })
            .expect("near");
        store
            .put_spatial(SpatialRecord {
                id: 999,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                payload: "missing-vector".to_string(),
            })
            .expect("spatial only");

        let semantic_ranked =
            hybrid_spatial_search(&store, &[1.0, 0.0], Point3D::default(), 10, 1.0, 0.0);
        assert_eq!(semantic_ranked[0].id, semantic);
        let spatial_ranked =
            hybrid_spatial_search(&store, &[1.0, 0.0], Point3D::default(), 10, 0.0, 1.0);
        assert_eq!(spatial_ranked[0].id, spatial);
        assert!(spatial_ranked.iter().all(|result| result.id != 999));
        assert_eq!(spatial_ranked[0].payload, "near");
    }
}
