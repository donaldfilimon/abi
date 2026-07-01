//! Shared types for the `nn` feature (pure-Zig character-level LM trainer).
//!
//! Both `mod.zig` (the real backprop trainer) and `stub.zig` (the disabled
//! path) import from here. Keep this module `std`-only so the stub compiles
//! cheaply and the two stay in declaration-name parity.

const std = @import("std");

/// Errors specific to the nn trainer.
pub const NnError = error{
    /// The feature is compiled out (stub path).
    FeatureDisabled,
    /// Training corpus was empty or too short for the configured context.
    EmptyCorpus,
    /// JSONL ingest extracted no usable text for the requested field.
    NoCorpusData,
    /// A `TrainConfig` field was zero/out of range.
    InvalidConfig,
    OutOfMemory,
};

pub const Error = NnError;

/// Hidden-layer activation.
pub const Activation = enum { tanh, relu };

/// Gradient-descent optimizer selection.
pub const Optimizer = enum { sgd, adam };

/// Hyper-parameters for a training run. Defaults are tuned for the tiny
/// in-memory corpora exercised by the inline tests; callers can override any
/// field. Determinism is governed entirely by `seed` (no wall-clock entropy).
pub const TrainConfig = struct {
    /// Context window: number of preceding characters used to predict the next.
    seq_len: usize = 2,
    /// Hidden-layer width.
    hidden: usize = 16,
    /// Per-character embedding dimensionality.
    embed_dim: usize = 8,
    /// Full-batch gradient-descent passes over the corpus.
    epochs: usize = 200,
    /// Learning rate.
    lr: f32 = 0.1,
    /// Deterministic PRNG seed for weight initialization.
    seed: u64 = 0x5EED,
    activation: Activation = .tanh,
    optimizer: Optimizer = .sgd,
};

/// Result of a training run. `improved` is the hard success signal used by the
/// loss-decrease gate: it is true iff the model's average cross-entropy strictly
/// decreased from initialization to the end of training.
pub const TrainReport = struct {
    initial_loss: f32,
    final_loss: f32,
    steps: usize,
    improved: bool,
};

test {
    std.testing.refAllDecls(@This());
}
