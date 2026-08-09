//! Deterministic portable-SIMD CPU primitives.

use std::simd::{Simd, num::SimdFloat};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    Backend, CapabilityEvidence, CapabilityState, ScoredIndex, Selection, best_cpu_backend,
};

/// A compute operation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeError {
    /// Input vectors have different dimensions.
    DimensionMismatch,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("vector dimensions differ")
    }
}

impl std::error::Error for ComputeError {}

/// Stateful CPU backend whose capability record reflects successful execution.
#[derive(Debug)]
pub struct CpuBackend {
    backend: Backend,
    executed: AtomicBool,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// Construct the best CPU backend for the current host.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: best_cpu_backend(),
            executed: AtomicBool::new(false),
        }
    }

    /// Construct the portable scalar-compatible backend.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            backend: Backend::CpuScalar,
            executed: AtomicBool::new(false),
        }
    }

    /// Effective CPU backend.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Capability state, including whether this instance has executed.
    #[must_use]
    pub fn capability(&self) -> CapabilityState {
        let executed = self.executed.load(Ordering::Relaxed);
        CapabilityState::new(
            self.backend,
            CapabilityEvidence::new()
                .with_compiled(true)
                .with_available(true)
                .with_initialized(true)
                .with_executed(executed)
                .with_runtime_verified(executed),
            if executed {
                "deterministic CPU SIMD execution verified on this runtime"
            } else {
                "deterministic CPU SIMD implementation initialized"
            },
        )
        .expect("CPU capability state is valid")
    }

    /// Execute a dot product and record successful runtime execution.
    pub fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        let value = dot_for_backend(self.backend, left, right)?;
        self.executed.store(true, Ordering::Relaxed);
        Ok(value)
    }

    /// Execute a Euclidean norm and record successful runtime execution.
    pub fn norm(&self, values: &[f32]) -> f32 {
        let value = dot_for_backend(self.backend, values, values)
            .expect("a vector always has the same dimensions as itself")
            .sqrt();
        self.executed.store(true, Ordering::Relaxed);
        value
    }

    /// Execute cosine similarity and record successful runtime execution.
    pub fn cosine(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        if left.len() != right.len() {
            return Err(ComputeError::DimensionMismatch);
        }
        if left.is_empty() {
            self.executed.store(true, Ordering::Relaxed);
            return Ok(0.0);
        }
        let numerator = dot_for_backend(self.backend, left, right)?;
        let denominator = dot_for_backend(self.backend, left, left)?.sqrt()
            * dot_for_backend(self.backend, right, right)?.sqrt();
        self.executed.store(true, Ordering::Relaxed);
        Ok(if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        })
    }

    /// Rank a vector batch by cosine score with stable dense-index tie breaks.
    pub fn top_k(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        limit: usize,
    ) -> Result<Vec<ScoredIndex>, ComputeError> {
        let mut results = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                self.cosine(query, candidate)
                    .map(|score| ScoredIndex { index, score })
            })
            .collect::<Result<Vec<_>, _>>()?;
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.index.cmp(&right.index))
        });
        results.truncate(limit);
        Ok(results)
    }
}

/// Host-selected SIMD lane count for `f32`.
#[must_use]
pub fn simd_lanes() -> usize {
    lanes_for_backend(best_cpu_backend())
}

const fn lanes_for_backend(backend: Backend) -> usize {
    match backend {
        Backend::CpuAvx512 => 16,
        Backend::CpuAvx2 => 8,
        _ => 4,
    }
}

fn dot_lanes<const LANES: usize>(left: &[f32], right: &[f32]) -> f32 {
    let mut accumulator = Simd::<f32, LANES>::splat(0.0);
    let mut offset = 0;
    while offset + LANES <= left.len() {
        let left_lanes = Simd::from_slice(&left[offset..offset + LANES]);
        let right_lanes = Simd::from_slice(&right[offset..offset + LANES]);
        accumulator += left_lanes * right_lanes;
        offset += LANES;
    }
    let mut sum = accumulator.reduce_sum();
    for index in offset..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

fn dot_for_backend(backend: Backend, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
    if left.len() != right.len() {
        return Err(ComputeError::DimensionMismatch);
    }
    Ok(match lanes_for_backend(backend) {
        16 => dot_lanes::<16>(left, right),
        8 => dot_lanes::<8>(left, right),
        _ => dot_lanes::<4>(left, right),
    })
}

/// Execute a dot product on the selection's deterministic CPU path.
pub fn dot(selection: Selection, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
    dot_for_backend(selection.effective, left, right)
}

/// Compute a Euclidean norm on the deterministic CPU path.
#[must_use]
pub fn norm(values: &[f32]) -> f32 {
    dot_for_backend(best_cpu_backend(), values, values)
        .expect("a vector always has the same dimensions as itself")
        .sqrt()
}

/// Compute cosine similarity on the deterministic CPU path.
///
/// Empty and zero-norm vectors return `0.0`, matching WDBX's reference
/// behavior.
pub fn cosine(left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
    CpuBackend::new().cosine(left, right)
}

/// Rank a vector batch by cosine score with stable dense-index tie breaks.
pub fn top_k(
    query: &[f32],
    candidates: &[&[f32]],
    limit: usize,
) -> Result<Vec<ScoredIndex>, ComputeError> {
    CpuBackend::new().top_k(query, candidates, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_preserves_ragged_tail_and_dimension_behavior() {
        let left: Vec<f32> = (1_u8..=17).map(f32::from).collect();
        let right = vec![2.0; 17];
        let expected: f32 = left
            .iter()
            .zip(&right)
            .map(|(left, right)| left * right)
            .sum();
        assert!(
            (dot(crate::select(Backend::CpuScalar), &left, &right).expect("dot") - expected).abs()
                < 1e-4
        );
        assert_eq!(
            dot(crate::select(Backend::CpuScalar), &[1.0], &[1.0, 2.0]),
            Err(ComputeError::DimensionMismatch)
        );
    }

    #[test]
    fn accelerator_selection_preserves_cpu_fallback_parity() {
        let left = [1.0, 2.0, 3.0, 4.0, 5.0];
        let right = [5.0, 4.0, 3.0, 2.0, 1.0];
        let cpu_selection = crate::select(Backend::CpuScalar);
        let accelerator_selection = crate::select(Backend::GpuMetal);
        assert!(!accelerator_selection.native);
        assert!(accelerator_selection.effective.is_cpu());
        assert_eq!(
            dot(accelerator_selection, &left, &right),
            dot(cpu_selection, &left, &right)
        );
    }

    #[test]
    fn cosine_and_norm_match_reference_values() {
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < f32::EPSILON);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]).expect("cosine") - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), Ok(0.0));
        assert_eq!(cosine(&[], &[]), Ok(0.0));
    }

    #[test]
    fn top_k_is_score_sorted_with_stable_index_ties() {
        let query = [1.0, 0.0];
        let first = [1.0, 0.0];
        let second = [0.0, 1.0];
        let third = [2.0, 0.0];
        let results = top_k(&query, &[&first, &second, &third], 2).expect("top k");
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 2);
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!((results[1].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn execution_evidence_changes_only_after_success() {
        let cpu = CpuBackend::new();
        assert!(!cpu.capability().executed());
        assert_eq!(
            cpu.dot(&[1.0], &[1.0, 2.0]),
            Err(ComputeError::DimensionMismatch)
        );
        assert!(!cpu.capability().executed());
        assert_eq!(cpu.dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
        assert!(cpu.capability().executed());
        assert!(cpu.capability().runtime_verified());
    }
}
