//! True incremental character sampler.
//!
//! Advances model state **one character at a time** via [`SampleState::step`],
//! rather than only exposing a batch `sample(n)` that hides the loop. The
//! batch helper [`crate::train::sample`] is implemented on top of this path.

use crate::model::{Model, Scratch, forward_loss_no_target};

/// Mutable incremental sampling state for a trained (or loaded) model.
#[derive(Debug, Clone)]
pub struct SampleState {
    scratch: Scratch,
    remaining: usize,
    /// Characters already produced.
    produced: Vec<u8>,
}

impl SampleState {
    /// Begin greedy sampling of up to `n` characters starting from `seed_char`.
    ///
    /// Seed resolution matches the batch path: first corpus occurrence of
    /// `seed_char`, else corpus index 0.
    #[must_use]
    pub fn start(model: &Model, seed_char: u8, n: usize) -> Self {
        let mut scratch = Scratch::new(model);
        let start = model
            .corpus
            .iter()
            .position(|&b| b == seed_char)
            .unwrap_or(0);
        let _ = model.sample_at(start, &mut scratch.ctx);
        Self {
            scratch,
            remaining: n,
            produced: Vec::with_capacity(n),
        }
    }

    /// Produce the next character, advancing the context window.
    ///
    /// Returns `None` when the requested length is exhausted.
    pub fn step(&mut self, model: &Model) -> Option<u8> {
        if self.remaining == 0 {
            return None;
        }
        forward_loss_no_target(model, &mut self.scratch);
        let mut best = 0_usize;
        let mut best_p = self.scratch.probs[0];
        for (idx, &pv) in self.scratch.probs.iter().enumerate().skip(1) {
            if pv > best_p {
                best_p = pv;
                best = idx;
            }
        }
        let ch = model.id_to_byte[best];
        if model.seq_len > 1 {
            self.scratch.ctx.copy_within(1..model.seq_len, 0);
        }
        self.scratch.ctx[model.seq_len - 1] = best;
        self.remaining -= 1;
        self.produced.push(ch);
        Some(ch)
    }

    /// Characters produced so far.
    #[must_use]
    pub fn produced(&self) -> &[u8] {
        &self.produced
    }

    /// Steps still available.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.remaining
    }

    /// Whether sampling is finished.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.remaining == 0
    }

    /// Drain remaining steps into a buffer (true multi-step path).
    #[must_use]
    pub fn finish(mut self, model: &Model) -> Vec<u8> {
        while self.step(model).is_some() {}
        self.produced
    }
}

/// Greedy sample `n` characters using the incremental stepper.
#[must_use]
pub fn sample_incremental(model: &Model, seed_char: u8, n: usize) -> Vec<u8> {
    SampleState::start(model, seed_char, n).finish(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{sample, train_model};
    use crate::types::TrainConfig;

    fn tiny_model() -> Model {
        train_model(
            b"hello world ",
            TrainConfig {
                seq_len: 2,
                hidden: 16,
                embed_dim: 8,
                epochs: 200,
                lr: 0.5,
                seed: 11,
                ..TrainConfig::default()
            },
        )
        .expect("train")
    }

    #[test]
    fn stepwise_generation_advances_one_char_at_a_time() {
        let model = tiny_model();
        let mut state = SampleState::start(&model, b'h', 5);
        assert_eq!(state.remaining(), 5);
        assert!(!state.is_done());
        let mut chars = Vec::new();
        for expected_left in (0..5).rev() {
            let ch = state.step(&model).expect("step");
            chars.push(ch);
            assert_eq!(state.remaining(), expected_left);
            assert_eq!(state.produced(), chars.as_slice());
        }
        assert!(state.is_done());
        assert!(state.step(&model).is_none());
        assert_eq!(chars.len(), 5);
    }

    #[test]
    fn incremental_matches_batch_sample_byte_for_byte() {
        let model = tiny_model();
        let batch = sample(&model, b'h', 24);
        let inc = sample_incremental(&model, b'h', 24);
        assert_eq!(batch, inc);
    }

    #[test]
    fn finish_drains_all_steps_deterministically() {
        let model = tiny_model();
        let a = SampleState::start(&model, b'w', 12).finish(&model);
        let b = SampleState::start(&model, b'w', 12).finish(&model);
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        assert!(!a.is_empty());
    }
}
