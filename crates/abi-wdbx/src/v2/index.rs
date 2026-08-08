//! Dense internal HNSW slots mapped to stable public v2 record identities.

use super::{MAX_V2_VECTOR_DIMENSIONS, RecordId, V2Snapshot};
use crate::hnsw::{HnswError, HnswIndex};
use abi_compute::Accelerator;
use std::collections::BTreeSet;
use std::sync::Arc;

/// One mutation-safe v2 vector match.
#[derive(Debug, Clone)]
pub struct V2SearchResult {
    /// Stable public identity, independent from the graph's dense slot.
    pub id: RecordId,
    /// Cosine similarity.
    pub score: f32,
    snapshot: Arc<V2Snapshot>,
}

impl V2SearchResult {
    /// Zero-copy vector view retained by the immutable snapshot.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        self.snapshot
            .preferred_vector(self.id)
            .expect("the retained index snapshot contains every returned id")
    }
}

/// HNSW graph whose internal contiguous slots never escape as public IDs.
#[derive(Debug, Clone)]
pub struct V2VectorIndex {
    graph: HnswIndex,
    slots: Vec<RecordId>,
    snapshot: Arc<V2Snapshot>,
}

impl V2VectorIndex {
    /// Rebuild a deterministic graph from the preferred current vector versions.
    pub fn from_snapshot(snapshot: Arc<V2Snapshot>) -> Result<Self, HnswError> {
        let ids: Vec<_> = snapshot.vector_ids().collect();
        let Some(first) = ids.first().copied() else {
            return Err(HnswError::EmptySnapshot);
        };
        let dimensions = snapshot
            .preferred_vector(first)
            .expect("vector_ids only returns present vectors")
            .len();
        let mut graph = HnswIndex::with_seed_and_limit(dimensions, 0, MAX_V2_VECTOR_DIMENSIONS)?;
        for (index, id) in ids.iter().copied().enumerate() {
            let values = snapshot
                .preferred_vector(id)
                .expect("vector_ids only returns present vectors");
            let slot = u64::try_from(index + 1).map_err(|_| HnswError::CorruptGraph {
                reason: "v2 dense slot space exceeds u64".into(),
            })?;
            graph.insert(slot, values)?;
            graph.commit_last_insert(slot);
        }
        Ok(Self {
            graph,
            slots: ids,
            snapshot,
        })
    }

    /// Fixed vector width.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.graph.dimensions()
    }

    /// Number of indexed public identities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Search and translate dense graph slots back to stable public IDs.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<V2SearchResult>, HnswError> {
        let mut results = self
            .graph
            .search(query, limit)?
            .into_iter()
            .map(|result| {
                let index = usize::try_from(result.id - 1).expect("dense slot fits usize");
                V2SearchResult {
                    id: self.slots[index],
                    score: result.score,
                    snapshot: Arc::clone(&self.snapshot),
                }
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(results)
    }

    /// Search the complete immutable vector batch through a selected accelerator.
    ///
    /// The adapter is responsible for deterministic CPU-oracle fallback. WDBX
    /// validates the returned dense positions before translating them back to
    /// stable public identities and retaining this index's immutable snapshot.
    pub fn search_with_accelerator(
        &self,
        query: &[f32],
        limit: usize,
        accelerator: &dyn Accelerator,
    ) -> Result<Vec<V2SearchResult>, HnswError> {
        if query.len() != self.dimensions() {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimensions(),
                found: query.len(),
            });
        }
        let candidates = self
            .slots
            .iter()
            .map(|id| {
                self.snapshot
                    .preferred_vector(*id)
                    .expect("the index snapshot contains every dense slot")
            })
            .collect::<Vec<_>>();
        let ranked = accelerator
            .top_k(query, &candidates, limit.min(candidates.len()))
            .map_err(|_| HnswError::DimensionMismatch {
                expected: self.dimensions(),
                found: query.len(),
            })?;
        let mut seen = BTreeSet::new();
        let mut results = Vec::with_capacity(ranked.len());
        for result in ranked {
            if result.index >= self.slots.len()
                || !result.score.is_finite()
                || !seen.insert(result.index)
            {
                return Err(HnswError::CorruptGraph {
                    reason: "accelerator returned an invalid dense ranking".into(),
                });
            }
            results.push(V2SearchResult {
                id: self.slots[result.index],
                score: result.score,
                snapshot: Arc::clone(&self.snapshot),
            });
        }
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{V2Mutation, V2Store};
    use abi_compute::{
        Backend, CapabilityEvidence, CapabilityState, ComputeError, CpuBackend, ScoredIndex,
    };

    #[derive(Debug, Default)]
    struct TestAccelerator {
        cpu: CpuBackend,
    }

    impl Accelerator for TestAccelerator {
        fn backend(&self) -> Backend {
            Backend::GpuMetal
        }

        fn capability(&self) -> CapabilityState {
            CapabilityState::new(
                Backend::GpuMetal,
                CapabilityEvidence::new()
                    .with_compiled(true)
                    .with_available(true)
                    .with_initialized(true),
                "test accelerator",
            )
            .unwrap()
        }

        fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
            self.cpu.dot(left, right)
        }
    }

    #[derive(Debug)]
    struct InvalidAccelerator;

    impl Accelerator for InvalidAccelerator {
        fn backend(&self) -> Backend {
            Backend::GpuMetal
        }

        fn capability(&self) -> CapabilityState {
            CapabilityState::new(
                Backend::GpuMetal,
                CapabilityEvidence::new().with_compiled(true),
                "invalid test adapter",
            )
            .unwrap()
        }

        fn dot(&self, _left: &[f32], _right: &[f32]) -> Result<f32, ComputeError> {
            Ok(0.0)
        }

        fn top_k(
            &self,
            _query: &[f32],
            _candidates: &[&[f32]],
            _limit: usize,
        ) -> Result<Vec<ScoredIndex>, ComputeError> {
            Ok(vec![ScoredIndex {
                index: usize::MAX,
                score: f32::NAN,
            }])
        }
    }

    #[test]
    fn dense_slots_preserve_public_ids_and_snapshot_lifetimes() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-index", "store");
        let mut store = V2Store::open(&root).unwrap();
        let legacy = RecordId::Legacy(99);
        let uuid = RecordId::new_v2();
        store
            .commit(vec![
                V2Mutation::PutVector {
                    id: uuid,
                    values: vec![1.0, 0.0],
                },
                V2Mutation::PutVector {
                    id: legacy,
                    values: vec![0.0, 1.0],
                },
            ])
            .unwrap();
        let index = V2VectorIndex::from_snapshot(store.snapshot()).unwrap();
        let result = index.search(&[1.0, 0.0], 1).unwrap().remove(0);
        assert_eq!(result.id, uuid);
        assert_eq!(result.vector(), [1.0, 0.0]);

        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![0.5, 0.5],
            }])
            .unwrap();
        assert_eq!(result.vector(), [1.0, 0.0]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn v2_hnsw_accepts_4096_dimensions_without_widening_v1() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-index-wide", "store");
        let mut store = V2Store::open(&root).unwrap();
        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![0.25; 4096],
            }])
            .unwrap();
        let index = V2VectorIndex::from_snapshot(store.snapshot()).unwrap();
        assert_eq!(index.dimensions(), 4096);
        assert_eq!(index.search(&vec![0.25; 4096], 1).unwrap().len(), 1);
        assert!(HnswIndex::new(4096).is_err(), "v1 remains capped at 128");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accelerator_batch_search_preserves_ids_ties_and_snapshot_lifetime() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-accelerated-index", "store");
        let mut store = V2Store::open(&root).unwrap();
        let first = RecordId::V2(uuid::Uuid::from_u128(1));
        let second = RecordId::V2(uuid::Uuid::from_u128(2));
        let unrelated = RecordId::V2(uuid::Uuid::from_u128(3));
        store
            .commit(vec![
                V2Mutation::PutVector {
                    id: second,
                    values: vec![2.0, 0.0],
                },
                V2Mutation::PutVector {
                    id: unrelated,
                    values: vec![0.0, 1.0],
                },
                V2Mutation::PutVector {
                    id: first,
                    values: vec![1.0, 0.0],
                },
            ])
            .unwrap();
        let index = V2VectorIndex::from_snapshot(store.snapshot()).unwrap();
        let accelerator = TestAccelerator::default();
        let results = index
            .search_with_accelerator(&[1.0, 0.0], 2, &accelerator)
            .unwrap();
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            [first, second]
        );
        assert_eq!(results[0].vector(), [1.0, 0.0]);

        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![0.5, 0.5],
            }])
            .unwrap();
        assert_eq!(results[0].vector(), [1.0, 0.0]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accelerator_batch_search_rejects_invalid_adapter_rankings() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-invalid-adapter", "store");
        let mut store = V2Store::open(&root).unwrap();
        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![1.0, 0.0],
            }])
            .unwrap();
        let index = V2VectorIndex::from_snapshot(store.snapshot()).unwrap();
        assert!(matches!(
            index.search_with_accelerator(&[1.0, 0.0], 1, &InvalidAccelerator),
            Err(HnswError::CorruptGraph { .. })
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
