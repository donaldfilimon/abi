//! Shared types for the miniature char-LM demo trainer.
//!
//! Ported from `src/features/nn/types.zig`.

/// Errors specific to the nn trainer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NnError {
    /// Training corpus was empty or too short for the configured context.
    EmptyCorpus,
    /// JSONL ingest extracted no usable text for the requested field.
    NoCorpusData,
    /// A `TrainConfig` field was zero/out of range.
    InvalidConfig,
    /// I/O failure reading a dataset.
    Io(String),
}

impl std::fmt::Display for NnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCorpus => f.write_str("empty or too-short training corpus"),
            Self::NoCorpusData => f.write_str("no usable text field in JSONL dataset"),
            Self::InvalidConfig => f.write_str("invalid training configuration"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for NnError {}

/// Hidden-layer activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Hyperbolic tangent.
    #[default]
    Tanh,
    /// Rectified linear unit.
    Relu,
}

/// Gradient-descent optimizer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Optimizer {
    /// Plain SGD with mean-normalized gradients.
    #[default]
    Sgd,
    /// Adam with the usual β₁=0.9, β₂=0.999 defaults.
    Adam,
}

/// Hyper-parameters for a training run.
///
/// Defaults match Zig's `TrainConfig` and are tuned for the tiny in-memory
/// corpora exercised by the inline tests. Determinism is governed entirely by
/// `seed` (no wall-clock entropy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    /// Context window: number of preceding characters used to predict the next.
    pub seq_len: usize,
    /// Hidden-layer width.
    pub hidden: usize,
    /// Per-character embedding dimensionality.
    pub embed_dim: usize,
    /// Full-batch gradient-descent passes over the corpus.
    pub epochs: usize,
    /// Learning rate.
    pub lr: f32,
    /// Deterministic PRNG seed for weight initialization.
    pub seed: u64,
    /// Hidden activation.
    pub activation: Activation,
    /// Optimizer.
    pub optimizer: Optimizer,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seq_len: 2,
            hidden: 16,
            embed_dim: 8,
            epochs: 200,
            lr: 0.1,
            seed: 0x5EED,
            activation: Activation::Tanh,
            optimizer: Optimizer::Sgd,
        }
    }
}

/// Result of a training run.
///
/// `improved` is the hard success signal: true iff average cross-entropy
/// strictly decreased from initialization to the end of training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainReport {
    /// Cross-entropy before any gradient step.
    pub initial_loss: f32,
    /// Cross-entropy after training.
    pub final_loss: f32,
    /// Total parameter-update steps (`epochs * corpus_len`).
    pub steps: usize,
    /// Whether `final_loss < initial_loss`.
    pub improved: bool,
}
