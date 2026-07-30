//! Miniature character-level neural-net demo trainer.
//!
//! Ported from `src/features/nn/`. This is a **demonstration** trainer for the
//! local pipeline — not a production LLM, not distributed, and not network-
//! connected. It trains a tiny next-character model by hand-derived backprop
//! over a small in-memory corpus.

pub mod checkpoint;
pub mod gguf_demo;
pub mod model;
pub mod sample_inc;
pub mod train;
pub mod types;

pub use checkpoint::{load as load_checkpoint, save as save_checkpoint};
pub use gguf_demo::{
    GgufError, load_and_sample as load_gguf_and_sample, load_model_file, write_demo_gguf,
};
pub use model::{Grads, Matrix, Model, Scratch};
pub use sample_inc::{SampleState, sample_incremental};
pub use train::{extract_corpus_from_jsonl, sample, train_model, train_on_jsonl, train_on_text};
pub use types::{Activation, NnError, Optimizer, TrainConfig, TrainReport};

/// Format the standard `nn train:` report line printed by the CLI.
#[must_use]
pub fn format_report(report: &TrainReport) -> String {
    format!(
        "nn train: initial_loss={:.4} final_loss={:.4} improved={} steps={}",
        report.initial_loss,
        report.final_loss,
        if report.improved { "true" } else { "false" },
        report.steps,
    )
}
