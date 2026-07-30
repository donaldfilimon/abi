//! Training loop and corpus helpers.
//!
//! Ported from `src/features/nn/train.zig`. Weight initialization uses a
//! deterministic splitmix64 stream; values are not claimed to match Zig's
//! `DefaultPrng` bit-for-bit — only the loss-decrease and sampling properties
//! are contract.

use crate::model::{Grads, Matrix, Model, Scratch, backward, eval_loss, forward_loss};
use crate::types::{NnError, Optimizer, TrainConfig, TrainReport};

/// Splitmix64 — deterministic 64-bit mixer used only for weight init.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits → float in [0, 1).
        let bits = (self.next_u64() >> 40) as f32;
        bits / 16_777_216.0 // 2^24
    }
}

fn fill_uniform(rng: &mut SplitMix64, slice: &mut [f32], scale: f32) {
    for v in slice {
        *v = (rng.next_f32() * 2.0 - 1.0) * scale;
    }
}

fn init_model(text: &[u8], config: TrainConfig) -> Result<Model, NnError> {
    if text.is_empty() {
        return Err(NnError::EmptyCorpus);
    }
    if config.seq_len == 0 || config.hidden == 0 || config.embed_dim == 0 {
        return Err(NnError::InvalidConfig);
    }
    if text.len() <= config.seq_len {
        return Err(NnError::EmptyCorpus);
    }

    let mut present = [false; 256];
    for &b in text {
        present[b as usize] = true;
    }
    let vocab_size = present.iter().filter(|&&p| p).count();
    if vocab_size < 2 {
        return Err(NnError::InvalidConfig);
    }

    let mut id_to_byte = vec![0_u8; vocab_size];
    let mut vocab = [-1_i16; 256];
    let mut idx = 0_usize;
    for (b, &is_present) in present.iter().enumerate() {
        if is_present {
            // Demo vocab sizes are tiny (≪ i16::MAX / 256).
            vocab[b] = i16::try_from(idx).expect("vocab fits i16");
            id_to_byte[idx] = u8::try_from(b).expect("byte index fits u8");
            idx += 1;
        }
    }

    let in_dim = config.seq_len * config.embed_dim;
    let mut rng = SplitMix64::new(config.seed);

    let mut embed = Matrix::zeros(vocab_size, config.embed_dim);
    let mut w1 = Matrix::zeros(config.hidden, in_dim);
    let mut w2 = Matrix::zeros(vocab_size, config.hidden);
    fill_uniform(&mut rng, &mut embed.data, 0.1);
    #[allow(clippy::cast_precision_loss)]
    let w1_scale = 1.0 / (in_dim as f32).sqrt();
    #[allow(clippy::cast_precision_loss)]
    let w2_scale = 1.0 / (config.hidden as f32).sqrt();
    fill_uniform(&mut rng, &mut w1.data, w1_scale);
    fill_uniform(&mut rng, &mut w2.data, w2_scale);

    Ok(Model {
        vocab,
        id_to_byte,
        vocab_size,
        seq_len: config.seq_len,
        embed_dim: config.embed_dim,
        hidden: config.hidden,
        activation: config.activation,
        embed,
        w1,
        b1: vec![0.0; config.hidden],
        w2,
        b2: vec![0.0; vocab_size],
        corpus: text.to_vec(),
        report: TrainReport {
            initial_loss: 0.0,
            final_loss: 0.0,
            steps: 0,
            improved: false,
        },
    })
}

/// Train a model on `text` and return it with a filled [`TrainReport`].
///
/// # Errors
///
/// Empty/short corpus or invalid config.
pub fn train_model(text: &[u8], config: TrainConfig) -> Result<Model, NnError> {
    let mut model = init_model(text, config)?;
    let mut sc = Scratch::new(&model);
    let mut grads = Grads::new(&model);

    #[allow(clippy::cast_precision_loss)]
    let n = model.corpus.len() as f32;
    let initial = eval_loss(&model, &mut sc);

    let mut adam_m: Option<Vec<Vec<f32>>> = None;
    let mut adam_v: Option<Vec<Vec<f32>>> = None;
    if config.optimizer == Optimizer::Adam {
        adam_m = Some(vec![
            vec![0.0; model.embed.data.len()],
            vec![0.0; model.w1.data.len()],
            vec![0.0; model.w2.data.len()],
            vec![0.0; model.b1.len()],
            vec![0.0; model.b2.len()],
        ]);
        adam_v = Some(vec![
            vec![0.0; model.embed.data.len()],
            vec![0.0; model.w1.data.len()],
            vec![0.0; model.w2.data.len()],
            vec![0.0; model.b1.len()],
            vec![0.0; model.b2.len()],
        ]);
    }

    let mut t = 0.0_f32;
    for _ in 0..config.epochs {
        grads.zero();
        for p in 0..model.corpus.len() {
            let target = model.sample_at(p, &mut sc.ctx);
            let _ = forward_loss(&model, &mut sc, target);
            backward(&model, &mut grads, &mut sc, target);
        }

        match config.optimizer {
            Optimizer::Sgd => {
                apply_sgd(&mut model.embed.data, &grads.embed.data, config.lr, n);
                apply_sgd(&mut model.w1.data, &grads.w1.data, config.lr, n);
                apply_sgd(&mut model.w2.data, &grads.w2.data, config.lr, n);
                apply_sgd(&mut model.b1, &grads.b1, config.lr, n);
                apply_sgd(&mut model.b2, &grads.b2, config.lr, n);
            }
            Optimizer::Adam => {
                t += 1.0;
                let m = adam_m.as_mut().expect("adam_m");
                let v = adam_v.as_mut().expect("adam_v");
                apply_adam(
                    &mut model.embed.data,
                    &grads.embed.data,
                    &mut m[0],
                    &mut v[0],
                    config.lr,
                    n,
                    t,
                );
                apply_adam(
                    &mut model.w1.data,
                    &grads.w1.data,
                    &mut m[1],
                    &mut v[1],
                    config.lr,
                    n,
                    t,
                );
                apply_adam(
                    &mut model.w2.data,
                    &grads.w2.data,
                    &mut m[2],
                    &mut v[2],
                    config.lr,
                    n,
                    t,
                );
                apply_adam(
                    &mut model.b1,
                    &grads.b1,
                    &mut m[3],
                    &mut v[3],
                    config.lr,
                    n,
                    t,
                );
                apply_adam(
                    &mut model.b2,
                    &grads.b2,
                    &mut m[4],
                    &mut v[4],
                    config.lr,
                    n,
                    t,
                );
            }
        }
    }

    let final_loss = eval_loss(&model, &mut sc);
    model.report = TrainReport {
        initial_loss: initial,
        final_loss,
        steps: config.epochs * model.corpus.len(),
        improved: final_loss < initial,
    };
    Ok(model)
}

fn apply_sgd(params: &mut [f32], grads: &[f32], lr: f32, n: f32) {
    for (pv, gv) in params.iter_mut().zip(grads) {
        *pv -= lr * (gv / n);
    }
}

fn apply_adam(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    n: f32,
    t: f32,
) {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1e-8;
    let bc1 = 1.0 - BETA1.powf(t);
    let bc2 = 1.0 - BETA2.powf(t);
    for (((pv, gv_raw), mv), vv) in params
        .iter_mut()
        .zip(grads)
        .zip(m.iter_mut())
        .zip(v.iter_mut())
    {
        let gv = gv_raw / n;
        *mv = BETA1 * *mv + (1.0 - BETA1) * gv;
        *vv = BETA2 * *vv + (1.0 - BETA2) * gv * gv;
        let mhat = *mv / bc1;
        let vhat = *vv / bc2;
        *pv -= lr * mhat / (vhat.sqrt() + EPS);
    }
}

/// Train and return only the report.
///
/// # Errors
///
/// Propagates [`train_model`] errors.
pub fn train_on_text(text: &[u8], config: TrainConfig) -> Result<TrainReport, NnError> {
    Ok(train_model(text, config)?.report)
}

/// Extract and concatenate a string field across non-blank JSONL lines.
///
/// # Errors
///
/// [`NnError::NoCorpusData`] when nothing usable is found.
pub fn extract_corpus_from_jsonl(bytes: &str, field: &str) -> Result<Vec<u8>, NnError> {
    let mut corpus = Vec::new();
    for raw_line in bytes.split('\n') {
        let line = raw_line.trim_matches(|c| matches!(c, ' ' | '\t' | '\r' | '\n'));
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(obj) = value.as_object() else {
            continue;
        };
        let Some(text) = obj.get(field).and_then(|v| v.as_str()) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        if !corpus.is_empty() {
            corpus.push(b'\n');
        }
        corpus.extend_from_slice(text.as_bytes());
    }
    if corpus.is_empty() {
        return Err(NnError::NoCorpusData);
    }
    Ok(corpus)
}

/// Train on a JSONL file's `field` values.
///
/// # Errors
///
/// I/O, empty field, or training failures.
pub fn train_on_jsonl(
    path: &str,
    field: &str,
    config: TrainConfig,
) -> Result<TrainReport, NnError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| NnError::Io(e.to_string()))?;
    let corpus = extract_corpus_from_jsonl(&bytes, field)?;
    train_on_text(&corpus, config)
}

/// Greedy sample `n` characters starting from `seed_char`.
///
/// Implemented via the true incremental stepper in [`crate::sample_inc`].
#[must_use]
pub fn sample(model: &Model, seed_char: u8, n: usize) -> Vec<u8> {
    crate::sample_inc::sample_incremental(model, seed_char, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_strictly_decreases_cross_entropy() {
        let report = train_on_text(
            b"hello world ",
            TrainConfig {
                seq_len: 2,
                hidden: 16,
                embed_dim: 8,
                epochs: 300,
                lr: 0.5,
                seed: 42,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        assert!(report.steps > 0);
        assert!(report.improved);
        assert!(report.final_loss < report.initial_loss);
    }

    #[test]
    fn greedy_sampling_reproduces_the_learned_pattern() {
        let model = train_model(
            b"hello world ",
            TrainConfig {
                seq_len: 2,
                hidden: 16,
                embed_dim: 8,
                epochs: 400,
                lr: 0.5,
                seed: 7,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        assert!(model.report.improved);
        let out = sample(&model, b'h', 24);
        let text = String::from_utf8_lossy(&out);
        assert!(
            text.contains("hello world"),
            "sampled {text:?} should contain the trained phrase"
        );
    }

    #[test]
    fn extract_corpus_from_jsonl_concatenates_field() {
        let jsonl = r#"{"text":"hello","n":1}

{"text":"world"}
{"other":"ignored"}
"#;
        let corpus = extract_corpus_from_jsonl(jsonl, "text").unwrap();
        assert_eq!(corpus, b"hello\nworld");
    }

    #[test]
    fn extract_corpus_errors_when_field_absent() {
        let jsonl = r#"{"other":"x"}
{"other":"y"}
"#;
        assert_eq!(
            extract_corpus_from_jsonl(jsonl, "text"),
            Err(NnError::NoCorpusData)
        );
    }

    #[test]
    fn empty_and_degenerate_corpora_are_rejected() {
        assert_eq!(
            train_on_text(b"", TrainConfig::default()),
            Err(NnError::EmptyCorpus)
        );
        assert_eq!(
            train_on_text(
                b"ab",
                TrainConfig {
                    seq_len: 5,
                    ..TrainConfig::default()
                }
            ),
            Err(NnError::EmptyCorpus)
        );
        assert_eq!(
            train_on_text(
                b"aaaa",
                TrainConfig {
                    seq_len: 1,
                    ..TrainConfig::default()
                }
            ),
            Err(NnError::InvalidConfig)
        );
    }
}
