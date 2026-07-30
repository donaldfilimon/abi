//! JSON checkpoint for the demo char-LM (save / load).
//!
//! Demo durability only — not a production model format. Versioned so future
//! field changes can refuse incompatible files instead of silently corrupting.

use std::fs;
use std::path::Path;

use crate::model::{Matrix, Model};
use crate::types::{Activation, NnError, TrainReport};

const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CheckpointFile {
    version: u32,
    vocab: Vec<i16>,
    id_to_byte: Vec<u8>,
    vocab_size: usize,
    seq_len: usize,
    embed_dim: usize,
    hidden: usize,
    activation: String,
    embed: MatrixDto,
    w1: MatrixDto,
    b1: Vec<f32>,
    w2: MatrixDto,
    b2: Vec<f32>,
    corpus: Vec<u8>,
    report: ReportDto,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MatrixDto {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ReportDto {
    initial_loss: f32,
    final_loss: f32,
    steps: usize,
    improved: bool,
}

impl From<&Matrix> for MatrixDto {
    fn from(m: &Matrix) -> Self {
        Self {
            rows: m.rows,
            cols: m.cols,
            data: m.data.clone(),
        }
    }
}

impl TryFrom<MatrixDto> for Matrix {
    type Error = NnError;

    fn try_from(dto: MatrixDto) -> Result<Self, Self::Error> {
        if dto.data.len() != dto.rows.saturating_mul(dto.cols) {
            return Err(NnError::InvalidConfig);
        }
        Ok(Self {
            data: dto.data,
            rows: dto.rows,
            cols: dto.cols,
        })
    }
}

/// Write `model` as JSON to `path` (parent dirs must exist).
pub fn save(path: &Path, model: &Model) -> Result<(), NnError> {
    let file = CheckpointFile {
        version: CHECKPOINT_VERSION,
        vocab: model.vocab.to_vec(),
        id_to_byte: model.id_to_byte.clone(),
        vocab_size: model.vocab_size,
        seq_len: model.seq_len,
        embed_dim: model.embed_dim,
        hidden: model.hidden,
        activation: match model.activation {
            Activation::Tanh => "tanh".into(),
            Activation::Relu => "relu".into(),
        },
        embed: MatrixDto::from(&model.embed),
        w1: MatrixDto::from(&model.w1),
        b1: model.b1.clone(),
        w2: MatrixDto::from(&model.w2),
        b2: model.b2.clone(),
        corpus: model.corpus.clone(),
        report: ReportDto {
            initial_loss: model.report.initial_loss,
            final_loss: model.report.final_loss,
            steps: model.report.steps,
            improved: model.report.improved,
        },
    };
    let json = serde_json::to_string_pretty(&file).map_err(|e| NnError::Io(e.to_string()))?;
    fs::write(path, json).map_err(|e| NnError::Io(e.to_string()))
}

/// Load a demo checkpoint written by [`save`].
pub fn load(path: &Path) -> Result<Model, NnError> {
    let raw = fs::read_to_string(path).map_err(|e| NnError::Io(e.to_string()))?;
    load_json(&raw)
}

/// Parse a checkpoint from a JSON string (no filesystem).
pub fn load_json(raw: &str) -> Result<Model, NnError> {
    let file: CheckpointFile = serde_json::from_str(raw).map_err(|e| NnError::Io(e.to_string()))?;
    if file.version != CHECKPOINT_VERSION {
        return Err(NnError::Io(format!(
            "unsupported checkpoint version {} (want {CHECKPOINT_VERSION})",
            file.version
        )));
    }
    if file.vocab.len() != 256 {
        return Err(NnError::InvalidConfig);
    }
    let mut vocab = [-1_i16; 256];
    for (i, v) in file.vocab.iter().enumerate() {
        vocab[i] = *v;
    }
    let activation = match file.activation.as_str() {
        "tanh" => Activation::Tanh,
        "relu" => Activation::Relu,
        other => return Err(NnError::Io(format!("unknown activation {other:?}"))),
    };
    Ok(Model {
        vocab,
        id_to_byte: file.id_to_byte,
        vocab_size: file.vocab_size,
        seq_len: file.seq_len,
        embed_dim: file.embed_dim,
        hidden: file.hidden,
        activation,
        embed: Matrix::try_from(file.embed)?,
        w1: Matrix::try_from(file.w1)?,
        b1: file.b1,
        w2: Matrix::try_from(file.w2)?,
        b2: file.b2,
        corpus: file.corpus,
        report: TrainReport {
            initial_loss: file.report.initial_loss,
            final_loss: file.report.final_loss,
            steps: file.report.steps,
            improved: file.report.improved,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::train_model;
    use crate::types::TrainConfig;

    #[test]
    fn save_load_round_trip_preserves_sample() {
        let model =
            train_model(b"hello world hello world ", TrainConfig::default()).expect("train");
        let dir = std::env::temp_dir().join(format!("abi_nn_ckpt_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("model.json");
        save(&path, &model).expect("save");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.vocab_size, model.vocab_size);
        assert_eq!(loaded.embed.data, model.embed.data);
        assert_eq!(loaded.w1.data, model.w1.data);
        assert!((loaded.report.final_loss - model.report.final_loss).abs() < f32::EPSILON);
        let a = crate::train::sample(&model, b'h', 12);
        let b = crate::train::sample(&loaded, b'h', 12);
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
