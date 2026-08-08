//! Object-safe accelerator adapter contract.

use crate::{Backend, CapabilityState, ComputeError};

/// One dense batch position and its similarity score.
///
/// Storage crates remain responsible for mapping this index to their public
/// record identity. Keeping IDs out of this crate prevents a compute/storage
/// dependency cycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredIndex {
    /// Zero-based position in the supplied candidate batch.
    pub index: usize,
    /// Cosine similarity score.
    pub score: f32,
}

/// Native or remote accelerator implementation.
///
/// The trait is object-safe so callers can select adapters at runtime. An
/// implementation must update its [`CapabilityState`] only after real events:
/// compilation, availability detection, initialization, successful execution,
/// and runtime verification are separate states.
pub trait Accelerator: Send + Sync {
    /// Backend implemented by this adapter.
    fn backend(&self) -> Backend;

    /// Current capability and runtime-evidence state.
    fn capability(&self) -> CapabilityState;

    /// Execute a native or remote dot product.
    fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError>;

    /// Execute cosine similarity.
    fn cosine(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        if left.len() != right.len() {
            return Err(ComputeError::DimensionMismatch);
        }
        if left.is_empty() {
            return Ok(0.0);
        }
        let numerator = self.dot(left, right)?;
        let denominator = self.dot(left, left)?.sqrt() * self.dot(right, right)?.sqrt();
        Ok(if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        })
    }

    /// Execute a Euclidean norm.
    fn norm(&self, values: &[f32]) -> Result<f32, ComputeError> {
        Ok(self.dot(values, values)?.sqrt())
    }

    /// Rank a batch by cosine score.
    ///
    /// Adapters with native batch/top-k kernels should override this method.
    /// The default remains deterministic: descending score, then ascending
    /// dense batch index.
    fn top_k(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapabilityEvidence;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ReferenceAdapter {
        executed: AtomicBool,
    }

    impl Accelerator for ReferenceAdapter {
        fn backend(&self) -> Backend {
            Backend::GpuMetal
        }

        fn capability(&self) -> CapabilityState {
            let executed = self.executed.load(Ordering::Relaxed);
            CapabilityState::new(
                Backend::GpuMetal,
                CapabilityEvidence::new()
                    .with_compiled(true)
                    .with_available(true)
                    .with_initialized(true)
                    .with_executed(executed)
                    .with_runtime_verified(executed),
                "test adapter",
            )
            .expect("valid capability")
        }

        fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
            if left.len() != right.len() {
                return Err(ComputeError::DimensionMismatch);
            }
            let value = left.iter().zip(right).map(|(a, b)| a * b).sum();
            self.executed.store(true, Ordering::Relaxed);
            Ok(value)
        }
    }

    #[test]
    fn trait_is_object_safe_and_defaults_are_deterministic() {
        let adapter: Box<dyn Accelerator> = Box::new(ReferenceAdapter {
            executed: AtomicBool::new(false),
        });
        let query = [1.0, 0.0];
        let first = [1.0, 0.0];
        let second = [0.0, 1.0];
        let third = [2.0, 0.0];
        let results = adapter
            .top_k(&query, &[&first, &second, &third], 2)
            .expect("top k");
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 2);
        assert!(adapter.capability().executed());
    }
}
