//! Gemma 3 text adapter: verified config and safetensors shards driven through
//! Candle's Gemma 3 decoder with deterministic greedy selection.
//!
//! The adapter adds no sampling, no chat-template file parsing, and no tool
//! protocol. Turns are rendered with the fixed Gemma control tokens, and every
//! control token id found inside caller-supplied content is dropped so content
//! cannot open or close a turn.

use crate::ModelRuntimeError;
use abi_agent_runtime::{ModelRequest, Role};
use abi_models::ModelManifest;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use tokenizers::Tokenizer;

/// Gemma 3 text-only decoder architecture loaded from safetensors shards.
pub const GEMMA3_TEXT_ARCHITECTURE: &str = "gemma3-text-v1";

/// Largest accepted architecture-config artifact: 1 MiB.
pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

const BOS: &str = "<bos>";
const EOS: &str = "<eos>";
const START_OF_TURN: &str = "<start_of_turn>";
const END_OF_TURN: &str = "<end_of_turn>";
const USER_TURN: &str = "user\n";
const MODEL_TURN: &str = "model\n";

/// Control-token identities resolved once from the verified tokenizer.
#[derive(Clone, Copy, Debug)]
struct ControlTokens {
    bos: u32,
    eos: u32,
    start_of_turn: u32,
    end_of_turn: u32,
}

impl ControlTokens {
    const fn contains(self, token: u32) -> bool {
        token == self.bos
            || token == self.eos
            || token == self.start_of_turn
            || token == self.end_of_turn
    }
}

/// Loaded Gemma 3 decoder plus the prompt pieces that never change.
pub(crate) struct GemmaEngine {
    model: Mutex<gemma3::Model>,
    device: Device,
    control: ControlTokens,
    user_turn: Vec<u32>,
    model_turn: Vec<u32>,
    newline: Vec<u32>,
}

/// Counts reported in the load evidence.
pub(crate) struct GemmaLoad {
    pub(crate) engine: GemmaEngine,
    pub(crate) tensor_count: usize,
    pub(crate) parameter_count: usize,
}

/// One exclusive generation over the decoder's key/value cache.
pub(crate) struct GemmaSession<'a> {
    model: MutexGuard<'a, gemma3::Model>,
    device: &'a Device,
    control: ControlTokens,
    position: usize,
}

impl GemmaEngine {
    /// Build the decoder from already hash-verified artifacts.
    pub(crate) fn load(
        manifest: &ModelManifest,
        config_path: &Path,
        weights_paths: &[PathBuf],
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> Result<GemmaLoad, ModelRuntimeError> {
        let config = read_config(manifest, config_path)?;
        let vocabulary = tokenizer.get_vocab_size(true);
        if vocabulary > config.vocab_size {
            return Err(config_error(
                manifest,
                format!(
                    "tokenizer vocabulary {vocabulary} exceeds config vocab_size {}",
                    config.vocab_size
                ),
            ));
        }
        let control = ControlTokens {
            bos: required_token(manifest, tokenizer, BOS)?,
            eos: required_token(manifest, tokenizer, EOS)?,
            start_of_turn: required_token(manifest, tokenizer, START_OF_TURN)?,
            end_of_turn: required_token(manifest, tokenizer, END_OF_TURN)?,
        };
        let (tensors, tensor_count, parameter_count) = read_shards(manifest, weights_paths)?;
        let builder = VarBuilder::from_tensors(tensors, DType::F32, device);
        let model = gemma3::Model::new(false, &config, builder)
            .map_err(|error| candle_error(manifest, &error))?;
        Ok(GemmaLoad {
            engine: Self {
                model: Mutex::new(model),
                device: device.clone(),
                control,
                user_turn: fixed_piece(manifest, tokenizer, USER_TURN)?,
                model_turn: fixed_piece(manifest, tokenizer, MODEL_TURN)?,
                newline: fixed_piece(manifest, tokenizer, "\n")?,
            },
            tensor_count,
            parameter_count,
        })
    }

    /// Render and tokenize the conversation as Gemma turns ending in an open
    /// model turn.
    pub(crate) fn encode_chat(
        &self,
        model_id: &str,
        tokenizer: &Tokenizer,
        request: &ModelRequest,
    ) -> Result<Vec<u32>, ModelRuntimeError> {
        let mut ids = vec![self.control.bos];
        let mut pending_system = request.system().map(str::to_owned);
        for message in request.messages() {
            let speaker = match message.role {
                Role::System => {
                    let joined = match pending_system.take() {
                        Some(existing) => format!("{existing}\n\n{}", message.content),
                        None => message.content.clone(),
                    };
                    pending_system = Some(joined);
                    continue;
                }
                Role::User => &self.user_turn,
                Role::Assistant => &self.model_turn,
                Role::Tool => {
                    return Err(ModelRuntimeError::UnsupportedRole {
                        model: model_id.to_owned(),
                        role: message.role.as_str(),
                    });
                }
            };
            let content = match (message.role, pending_system.take()) {
                (Role::User, Some(system)) => format!("{system}\n\n{}", message.content),
                (_, system) => {
                    pending_system = system;
                    message.content.clone()
                }
            };
            ids.push(self.control.start_of_turn);
            ids.extend_from_slice(speaker);
            ids.extend(self.encode_content(model_id, tokenizer, &content)?);
            ids.push(self.control.end_of_turn);
            ids.extend_from_slice(&self.newline);
        }
        if let Some(system) = pending_system {
            ids.push(self.control.start_of_turn);
            ids.extend_from_slice(&self.user_turn);
            ids.extend(self.encode_content(model_id, tokenizer, &system)?);
            ids.push(self.control.end_of_turn);
            ids.extend_from_slice(&self.newline);
        }
        ids.push(self.control.start_of_turn);
        ids.extend_from_slice(&self.model_turn);
        Ok(ids)
    }

    /// Take exclusive use of the decoder and reset its key/value cache.
    pub(crate) fn session(&self, model_id: &str) -> Result<GemmaSession<'_>, ModelRuntimeError> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| ModelRuntimeError::ReportStatePoisoned {
                model: model_id.to_owned(),
            })?;
        model.clear_kv_cache();
        Ok(GemmaSession {
            model,
            device: &self.device,
            control: self.control,
            position: 0,
        })
    }

    fn encode_content(
        &self,
        model_id: &str,
        tokenizer: &Tokenizer,
        content: &str,
    ) -> Result<Vec<u32>, ModelRuntimeError> {
        let encoding =
            tokenizer
                .encode(content, false)
                .map_err(|error| ModelRuntimeError::Tokenizer {
                    model: model_id.to_owned(),
                    detail: error.to_string(),
                })?;
        Ok(encoding
            .get_ids()
            .iter()
            .copied()
            .filter(|token| !self.control.contains(*token))
            .collect())
    }
}

impl GemmaSession<'_> {
    /// Native Candle operations performed by one [`Self::advance`] call:
    /// decoder forward, argmax, and scalar extraction.
    pub(crate) const OPERATIONS_PER_STEP: u64 = 3;

    /// Feed `tokens` at the current position and greedily select the next
    /// token. `None` means the model closed its turn.
    pub(crate) fn advance(&mut self, tokens: &[u32]) -> Result<Option<u32>, candle_core::Error> {
        let input = Tensor::new(tokens, self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.position)?;
        self.position = self.position.saturating_add(tokens.len());
        let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        if next == self.control.eos || next == self.control.end_of_turn {
            return Ok(None);
        }
        Ok(Some(next))
    }
}

fn read_config(manifest: &ModelManifest, path: &Path) -> Result<gemma3::Config, ModelRuntimeError> {
    let bytes = std::fs::read(path).map_err(|source| ModelRuntimeError::io(path, source))?;
    let document: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| config_error(manifest, error))?;
    // Multimodal publications nest the decoder settings under `text_config`.
    let text = match document.get("text_config") {
        Some(nested) => nested.clone(),
        None => document,
    };
    serde_json::from_value(text).map_err(|error| config_error(manifest, error))
}

fn read_shards(
    manifest: &ModelManifest,
    paths: &[PathBuf],
) -> Result<(HashMap<String, Tensor>, usize, usize), ModelRuntimeError> {
    let mut tensors = HashMap::new();
    let mut parameter_count = 0usize;
    for path in paths {
        let bytes = std::fs::read(path).map_err(|source| ModelRuntimeError::io(path, source))?;
        // Shards stay on the CPU here; the builder moves each tensor to the
        // selected device as the decoder requests it.
        let shard = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)
            .map_err(|error| candle_error(manifest, &error))?;
        for (name, tensor) in shard {
            parameter_count = parameter_count.saturating_add(tensor.elem_count());
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(ModelRuntimeError::Safetensors {
                    model: manifest.id.clone(),
                    detail: format!("tensor '{name}' appears in more than one shard"),
                });
            }
        }
    }
    let tensor_count = tensors.len();
    Ok((tensors, tensor_count, parameter_count))
}

fn required_token(
    manifest: &ModelManifest,
    tokenizer: &Tokenizer,
    token: &str,
) -> Result<u32, ModelRuntimeError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| ModelRuntimeError::Tokenizer {
            model: manifest.id.clone(),
            detail: format!("required {token} token is absent"),
        })
}

fn fixed_piece(
    manifest: &ModelManifest,
    tokenizer: &Tokenizer,
    piece: &str,
) -> Result<Vec<u32>, ModelRuntimeError> {
    tokenizer
        .encode(piece, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| ModelRuntimeError::Tokenizer {
            model: manifest.id.clone(),
            detail: error.to_string(),
        })
}

fn config_error(manifest: &ModelManifest, detail: impl std::fmt::Display) -> ModelRuntimeError {
    ModelRuntimeError::Config {
        model: manifest.id.clone(),
        detail: detail.to_string(),
    }
}

fn candle_error(manifest: &ModelManifest, error: &candle_core::Error) -> ModelRuntimeError {
    ModelRuntimeError::Candle {
        model: manifest.id.clone(),
        detail: error.to_string(),
    }
}
