//! Deterministic exact vector index for the Rust WDBX migration.
//!
//! This is the correctness backend for vector search while the Zig HNSW graph
//! is still being ported. It scans every vector, computes the same cosine score
//! shape as Zig (`1 - cosine_distance`), and returns stable score/id ordering.
//! It does not claim HNSW performance or candidate-selection parity.

use crate::Snapshot;
use crate::wal::MAX_VECTOR_DIMENSIONS;
use std::collections::BTreeMap;

/// A vector-index validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    /// Empty vectors and queries are invalid.
    InvalidVector,
    /// A vector or query did not match the index's active dimensions.
    DimensionMismatch {
        /// Dimensions required by the index.
        expected: usize,
        /// Dimensions supplied by the caller.
        found: usize,
    },
    /// The Zig-compatible 128-dimension ceiling was exceeded.
    TooManyDimensions {
        /// Dimensions supplied by the caller.
        found: usize,
    },
    /// An absolute replay id was not the next contiguous id.
    NonContiguousId {
        /// Next id required by the index.
        expected: u64,
        /// Id supplied by the caller.
        found: u64,
    },
    /// No further vector id can be represented.
    IdOverflow,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVector => formatter.write_str("vector must not be empty"),
            Self::DimensionMismatch { expected, found } => {
                write!(
                    formatter,
                    "vector has {found} dimensions; expected {expected}"
                )
            }
            Self::TooManyDimensions { found } => write!(
                formatter,
                "vector has {found} dimensions; maximum is {MAX_VECTOR_DIMENSIONS}"
            ),
            Self::NonContiguousId { expected, found } => {
                write!(formatter, "vector id {found} is not the next id {expected}")
            }
            Self::IdOverflow => formatter.write_str("vector id space is exhausted"),
        }
    }
}

impl std::error::Error for IndexError {}

/// One exact-search match.
///
/// The vector is borrowed from the index. Rust therefore prevents a mutating
/// insert while search results are alive, encoding Zig's documented
/// "valid-until-next-mutation" rule in the type system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult<'index> {
    /// Stable vector id.
    pub id: u64,
    /// Cosine similarity. Identical vectors score near `1`; orthogonal and
    /// zero-norm comparisons score `0`.
    pub score: f32,
    /// Zero-copy view of the stored vector.
    pub vector: &'index [f32],
}

/// An exact, deterministic in-memory vector index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExactIndex {
    vectors: BTreeMap<u64, Vec<f32>>,
    dimensions: Option<usize>,
    next_id: u64,
}

impl ExactIndex {
    /// Create an empty index whose first generated id is `1`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vectors: BTreeMap::new(),
            dimensions: None,
            next_id: 1,
        }
    }

    /// Build the exact index from a recovered checkpoint/WAL snapshot.
    pub fn from_snapshot(snapshot: &Snapshot) -> Result<Self, IndexError> {
        let mut index = Self::new();
        for (&id, values) in &snapshot.vectors {
            index.validate_dimensions(values)?;
            index.vectors.insert(id, values.clone());
        }
        index.next_id = match index.vectors.keys().next_back().copied() {
            Some(id) => id.checked_add(1).ok_or(IndexError::IdOverflow)?,
            None => 1,
        };
        Ok(index)
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether no vectors are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Active vector dimensions, established by the first insert.
    #[must_use]
    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    /// Id that the next successful generated insert will receive.
    #[must_use]
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Borrow one stored vector.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&[f32]> {
        self.vectors.get(&id).map(Vec::as_slice)
    }

    /// Insert a vector and return its contiguous generated id.
    pub fn put_vector(&mut self, values: &[f32]) -> Result<u64, IndexError> {
        let id = self.next_id;
        self.insert_absolute(id, values)?;
        Ok(id)
    }

    /// Insert a vector carrying an absolute WAL/recovery id.
    ///
    /// IDs must be contiguous, matching the WDBX WAL replay contract.
    pub fn insert_absolute(&mut self, id: u64, values: &[f32]) -> Result<(), IndexError> {
        self.validate_dimensions(values)?;
        if id != self.next_id {
            return Err(IndexError::NonContiguousId {
                expected: self.next_id,
                found: id,
            });
        }
        let next_id = id.checked_add(1).ok_or(IndexError::IdOverflow)?;
        self.vectors.insert(id, values.to_vec());
        self.next_id = next_id;
        Ok(())
    }

    /// Search every vector and return the best cosine-similarity matches.
    ///
    /// Ties are ordered by ascending id so repeated searches are byte-stable.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult<'_>>, IndexError> {
        self.validate_query(query)?;
        let mut results = self
            .vectors
            .iter()
            .map(|(&id, vector)| SearchResult {
                id,
                score: cosine_similarity(vector, query),
                vector,
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(limit);
        Ok(results)
    }

    fn validate_dimensions(&mut self, values: &[f32]) -> Result<(), IndexError> {
        validate_width(values)?;
        match self.dimensions {
            Some(expected) if expected != values.len() => Err(IndexError::DimensionMismatch {
                expected,
                found: values.len(),
            }),
            Some(_) => Ok(()),
            None => {
                self.dimensions = Some(values.len());
                Ok(())
            }
        }
    }

    fn validate_query(&self, query: &[f32]) -> Result<(), IndexError> {
        validate_width(query)?;
        if let Some(expected) = self.dimensions
            && expected != query.len()
        {
            return Err(IndexError::DimensionMismatch {
                expected,
                found: query.len(),
            });
        }
        Ok(())
    }
}

fn validate_width(values: &[f32]) -> Result<(), IndexError> {
    if values.is_empty() {
        return Err(IndexError::InvalidVector);
    }
    if values.len() > MAX_VECTOR_DIMENSIONS {
        return Err(IndexError::TooManyDimensions {
            found: values.len(),
        });
    }
    Ok(())
}

/// Zig-compatible scalar cosine similarity.
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_left = 0.0_f32;
    let mut norm_right = 0.0_f32;
    for (&left_value, &right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        norm_left += left_value * left_value;
        norm_right += right_value * right_value;
    }
    let denominator = norm_left.sqrt() * norm_right.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_start_at_one_and_are_contiguous() {
        let mut index = ExactIndex::new();
        assert_eq!(index.put_vector(&[1.0, 0.0]).expect("first"), 1);
        assert_eq!(index.put_vector(&[0.0, 1.0]).expect("second"), 2);
        assert_eq!(index.next_id(), 3);
    }

    #[test]
    fn dimensions_are_established_once() {
        let mut index = ExactIndex::new();
        index.put_vector(&[1.0, 0.0]).expect("first");
        assert_eq!(index.dimensions(), Some(2));
        assert_eq!(
            index.put_vector(&[1.0, 0.0, 0.0]),
            Err(IndexError::DimensionMismatch {
                expected: 2,
                found: 3,
            })
        );
    }

    #[test]
    fn empty_and_oversize_vectors_are_rejected() {
        let mut index = ExactIndex::new();
        assert_eq!(index.put_vector(&[]), Err(IndexError::InvalidVector));
        assert_eq!(
            index.put_vector(&vec![0.0; MAX_VECTOR_DIMENSIONS + 1]),
            Err(IndexError::TooManyDimensions {
                found: MAX_VECTOR_DIMENSIONS + 1,
            })
        );
    }

    #[test]
    fn exact_search_orders_by_cosine_similarity() {
        let mut index = ExactIndex::new();
        let identical = index.put_vector(&[1.0, 0.0]).expect("identical");
        let diagonal = index.put_vector(&[1.0, 1.0]).expect("diagonal");
        let orthogonal = index.put_vector(&[0.0, 1.0]).expect("orthogonal");
        let results = index.search(&[1.0, 0.0], 3).expect("search");
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            vec![identical, diagonal, orthogonal]
        );
        assert!((results[0].score - 1.0).abs() < 0.0001);
        assert!((results[1].score - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.0001);
        assert!(results[2].score.abs() < f32::EPSILON);
    }

    #[test]
    fn ties_are_stable_by_id() {
        let mut index = ExactIndex::new();
        index.put_vector(&[1.0, 0.0]).expect("first");
        index.put_vector(&[1.0, 0.0]).expect("second");
        let results = index.search(&[1.0, 0.0], 2).expect("search");
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn zero_norm_vectors_have_zero_similarity() {
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).abs() < f32::EPSILON);
        assert!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]).abs() < f32::EPSILON);
    }

    #[test]
    fn limit_zero_returns_no_matches_after_validation() {
        let mut index = ExactIndex::new();
        index.put_vector(&[1.0]).expect("insert");
        assert!(index.search(&[1.0], 0).expect("search").is_empty());
        assert_eq!(index.search(&[], 0), Err(IndexError::InvalidVector));
    }

    #[test]
    fn results_borrow_the_stored_vector_without_copying() {
        let mut index = ExactIndex::new();
        let id = index.put_vector(&[1.0, 2.0]).expect("insert");
        let stored = index.get(id).expect("stored");
        let results = index.search(&[1.0, 2.0], 1).expect("search");
        assert_eq!(results[0].vector.as_ptr(), stored.as_ptr());
    }

    #[test]
    fn absolute_ids_must_continue_the_sequence() {
        let mut index = ExactIndex::new();
        assert_eq!(
            index.insert_absolute(2, &[1.0]),
            Err(IndexError::NonContiguousId {
                expected: 1,
                found: 2,
            })
        );
        index.insert_absolute(1, &[1.0]).expect("first");
        index.insert_absolute(2, &[2.0]).expect("second");
    }

    #[test]
    fn snapshot_load_preserves_ids_and_selects_the_next_one() {
        let mut snapshot = Snapshot::new();
        snapshot.vectors.insert(4, vec![1.0, 0.0]);
        snapshot.vectors.insert(9, vec![0.0, 1.0]);
        let mut index = ExactIndex::from_snapshot(&snapshot).expect("load");
        assert_eq!(index.len(), 2);
        assert_eq!(index.next_id(), 10);
        assert_eq!(index.put_vector(&[1.0, 1.0]).expect("next"), 10);
    }

    #[test]
    fn snapshot_load_rejects_mixed_dimensions() {
        let mut snapshot = Snapshot::new();
        snapshot.vectors.insert(1, vec![1.0]);
        snapshot.vectors.insert(2, vec![1.0, 2.0]);
        assert_eq!(
            ExactIndex::from_snapshot(&snapshot),
            Err(IndexError::DimensionMismatch {
                expected: 1,
                found: 2,
            })
        );
    }
}
