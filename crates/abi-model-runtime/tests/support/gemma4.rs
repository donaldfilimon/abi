//! Scratch Gemma 4 fixture: a randomly initialized two-layer text decoder
//! stored under the checkpoint's language-model prefix, a word-level tokenizer
//! carrying the Gemma 4 control tokens, and a manifest
//! whose artifacts are hashed and stored like a real publication. Shared by the
//! runtime contract tests and the serving end-to-end test.

use abi_model_runtime::{
    DevicePreference, GEMMA4_TEXT_ARCHITECTURE, LoadConfig, LocalModelProvider, ModelRuntimeError,
};
use abi_models::{AcceptanceLedger, ModelManifest, ModelRegistry, StorageRoot, hash_file};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::gemma4::{Gemma4TextConfig, text::TextModel};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::{AddedToken, Tokenizer};

pub const MODEL_ID: &str = "scratch-gemma4";
pub const PRINCIPAL: &str = "operator@example.invalid";
const REVISION: &str = "456789abcdef0123456789abcdef0123456789ab";
pub const CONTEXT_TOKENS: u32 = 96;
pub const OUTPUT_TOKENS: u32 = 6;

/// Control tokens registered as special, in vocabulary order.
pub const CONTROL: [&str; 13] = [
    "<bos>",
    "<eos>",
    "<|turn>",
    "<turn|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
    "<|channel>",
    "<channel|>",
];

pub const WORDS: [&str; 8] = [
    "<unk>", "system", "user", "model", "hello", "world", "again", "terse",
];

pub fn vocabulary_len() -> usize {
    CONTROL.len() + WORDS.len()
}

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let root = abi_foundation::temp_path::temp_file_path("abi_model_runtime", "gemma");
        std::fs::create_dir_all(&root).expect("scratch directory");
        Self { root }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

pub struct Fixture {
    _scratch: Scratch,
    registry: ModelRegistry,
    ledger: AcceptanceLedger,
    storage: StorageRoot,
}

impl Fixture {
    pub fn load(&self) -> Result<LocalModelProvider, ModelRuntimeError> {
        LocalModelProvider::load(
            &self.registry,
            &self.ledger,
            &self.storage,
            MODEL_ID,
            PRINCIPAL,
            LoadConfig::new(DevicePreference::Cpu),
        )
    }
}

pub fn config() -> Value {
    json!({
        "head_dim": 8,
        "global_head_dim": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "vocab_size": vocabulary_len(),
        "sliding_window": 16,
        "final_logit_softcapping": null,
        "query_pre_attn_scalar": 8,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"]
    })
}

/// `tools` selects whether the tool-grammar tokens exist in the tokenizer.
fn write_tokenizer(path: &Path, tools: bool) {
    let control: Vec<&str> = CONTROL
        .iter()
        .copied()
        .filter(|token| tools || !token.contains("tool") && *token != "<|\"|>")
        .collect();
    let vocabulary = control
        .iter()
        .chain(WORDS.iter())
        .enumerate()
        .map(|(index, token)| {
            (
                (*token).to_owned(),
                u32::try_from(index).expect("small index"),
            )
        })
        .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("word-level model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    let special: Vec<AddedToken> = control
        .iter()
        .map(|token| AddedToken::from(*token, true))
        .collect();
    tokenizer.add_special_tokens(&special);
    tokenizer.save(path, true).expect("save tokenizer");
}

/// Write randomly initialized decoder weights split across `shards` files.
fn write_weights(directory: &Path, config: &Value, shards: usize) -> Vec<PathBuf> {
    let parsed: Gemma4TextConfig = serde_json::from_value(config.clone()).expect("scratch config");
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    TextModel::new(&parsed, builder.pp("model").pp("language_model")).expect("scratch decoder");
    let mut tensors: Vec<_> = variables
        .data()
        .lock()
        .expect("variable map")
        .iter()
        .map(|(name, variable)| (name.clone(), variable.as_tensor().clone()))
        .collect();
    tensors.sort_by(|left, right| left.0.cmp(&right.0));
    let per_shard = tensors.len().div_ceil(shards);
    tensors
        .chunks(per_shard)
        .enumerate()
        .map(|(index, chunk)| {
            let path = directory.join(format!("model-{index}.safetensors"));
            let shard = chunk.iter().cloned().collect();
            candle_core::safetensors::save(&shard, &path).expect("save shard");
            path
        })
        .collect()
}

fn artifact(path: &Path, kind: &str) -> Value {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("name");
    json!({
        "path": name,
        "kind": kind,
        "sha256": hash_file(path).expect("artifact hash").to_hex(),
        "size_bytes": std::fs::metadata(path).expect("artifact metadata").len(),
        "url": format!("https://models.example.invalid/{REVISION}/{name}")
    })
}

pub fn fixture_with(config: &Value, shards: usize, tools: bool) -> Fixture {
    let scratch = Scratch::new();
    let source = scratch.root.join("source");
    std::fs::create_dir_all(&source).expect("source directory");
    let tokenizer = source.join("tokenizer.json");
    write_tokenizer(&tokenizer, tools);
    let mut files = write_weights(&source, &self::config(), shards);
    let mut artifacts: Vec<Value> = files.iter().map(|path| artifact(path, "weights")).collect();
    artifacts.push(artifact(&tokenizer, "tokenizer"));
    files.push(tokenizer);
    let config_path = source.join("config.json");
    std::fs::write(&config_path, config.to_string()).expect("write config");
    artifacts.push(artifact(&config_path, "config"));
    files.push(config_path);
    let manifest = ModelManifest::from_json(
        "generated gemma4 fixture",
        &json!({
            "id": MODEL_ID,
            "repository": "abi-fixtures/scratch-gemma4",
            "revision": REVISION,
            "architecture": GEMMA4_TEXT_ARCHITECTURE,
            "license": "fixture-only-1.0",
            "license_sha256": abi_models::Sha256Digest::from_bytes([9; 32]).to_hex(),
            "modalities": ["text"],
            "tensor_format": "safetensors",
            "quantizations": ["f32"],
            "context": {
                "max_context_tokens": CONTEXT_TOKENS,
                "max_output_tokens": OUTPUT_TOKENS
            },
            "artifacts": artifacts
        })
        .to_string(),
    )
    .expect("generated manifest validates");

    let storage = StorageRoot::at(scratch.root.join("models"));
    for (source, declared) in files.iter().zip(&manifest.artifacts) {
        let target = storage.artifact_path(&manifest, declared);
        std::fs::create_dir_all(target.parent().expect("model directory")).expect("model dir");
        std::fs::copy(source, target).expect("copy artifact");
    }
    let mut registry = ModelRegistry::new();
    registry.insert(manifest.clone()).expect("insert manifest");
    let mut ledger = AcceptanceLedger::in_memory();
    ledger
        .accept(&manifest, PRINCIPAL)
        .expect("accept fixture license");
    Fixture {
        _scratch: scratch,
        registry,
        ledger,
        storage,
    }
}

pub fn fixture() -> Fixture {
    fixture_with(&config(), 2, true)
}
