//! Demo-grade GGUF-adjacent load path for the char-LM.
//!
//! Full ggml/llama.cpp is **not** vendored. This module accepts either:
//! 1. Our JSON checkpoint (via [`crate::checkpoint::load`]) when the path ends
//!    in `.json`, or
//! 2. A tiny **demo GGUF container** (`GGUF` magic + embedded JSON checkpoint
//!    payload) used only for CI/fixture proof that a model file can be loaded
//!    and sampled.
//!
//! Not a production LLM loader; not distributed; not network-connected.

use crate::checkpoint::{self, load as load_checkpoint, load_json};
use crate::model::Model;
use crate::types::NnError;
use std::path::Path;

const DEMO_MAGIC: &[u8; 4] = b"GGUF";
const DEMO_VERSION: u32 = 1;

/// Errors specific to demo GGUF loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// I/O failure.
    Io(String),
    /// Bad magic / truncated container.
    Format(String),
    /// Nested checkpoint parse failed.
    Checkpoint(NnError),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "gguf io: {m}"),
            Self::Format(m) => write!(f, "gguf format: {m}"),
            Self::Checkpoint(e) => write!(f, "gguf checkpoint: {e}"),
        }
    }
}

impl std::error::Error for GgufError {}

/// Write a demo GGUF container wrapping a JSON checkpoint payload.
pub fn write_demo_gguf(path: &str, model: &Model) -> Result<(), GgufError> {
    // Serialize via a private temp under std::env::temp_dir (never beside `path`).
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = std::env::temp_dir().join(format!("abi-nn-gguf-write-{ns}.json"));
    checkpoint::save(&tmp, model).map_err(GgufError::Checkpoint)?;
    let json = std::fs::read(&tmp).map_err(|e| GgufError::Io(e.to_string()))?;
    let _ = std::fs::remove_file(&tmp);
    let mut out = Vec::with_capacity(12 + json.len());
    out.extend_from_slice(DEMO_MAGIC);
    out.extend_from_slice(&DEMO_VERSION.to_le_bytes());
    let len =
        u32::try_from(json.len()).map_err(|_| GgufError::Format("payload too large".into()))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&json);
    std::fs::write(path, out).map_err(|e| GgufError::Io(e.to_string()))
}

/// Load a model from JSON checkpoint or demo GGUF container.
pub fn load_model_file(path: &str) -> Result<Model, GgufError> {
    if Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        return load_checkpoint(Path::new(path)).map_err(GgufError::Checkpoint);
    }
    let bytes = std::fs::read(path).map_err(|e| GgufError::Io(e.to_string()))?;
    if bytes.len() < 12 || &bytes[0..4] != DEMO_MAGIC {
        return Err(GgufError::Format(
            "expected demo GGUF magic 'GGUF' or a .json checkpoint".into(),
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != DEMO_VERSION {
        return Err(GgufError::Format(format!(
            "unsupported demo gguf version {version}"
        )));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() < 12 + len {
        return Err(GgufError::Format("truncated demo gguf payload".into()));
    }
    let json =
        std::str::from_utf8(&bytes[12..12 + len]).map_err(|e| GgufError::Format(e.to_string()))?;
    // Parse in-memory — never write a sidecar next to the model path.
    load_json(json).map_err(GgufError::Checkpoint)
}

/// Load a model file and run incremental sample for `n` characters.
pub fn load_and_sample(path: &str, seed_char: u8, n: usize) -> Result<Vec<u8>, GgufError> {
    let model = load_model_file(path)?;
    Ok(crate::sample_inc::sample_incremental(&model, seed_char, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::train_model;
    use crate::types::TrainConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(name: &str) -> String {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!(
            "{}/abi-nn-gguf-{}-{}.gguf",
            std::env::temp_dir().display(),
            name,
            ns
        )
    }

    #[test]
    fn demo_gguf_round_trip_load_and_sample() {
        let model = train_model(
            b"hello world ",
            TrainConfig {
                seq_len: 2,
                hidden: 12,
                embed_dim: 6,
                epochs: 80,
                lr: 0.5,
                seed: 3,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let path = scratch("rt");
        write_demo_gguf(&path, &model).unwrap();
        let out = load_and_sample(&path, b'h', 16).unwrap();
        assert_eq!(out.len(), 16);
        let _ = std::fs::remove_file(&path);
    }
}
