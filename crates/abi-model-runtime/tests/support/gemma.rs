//! Scratch Gemma 3 fixture: a randomly initialized two-layer decoder, a
//! word-level tokenizer carrying the Gemma control tokens, and a manifest
//! whose artifacts are hashed and stored like a real publication. Shared by the
//! runtime contract tests and the serving end-to-end test.

use abi_model_runtime::{
    DevicePreference, GEMMA3_TEXT_ARCHITECTURE, LoadConfig, LocalModelProvider, ModelRuntimeError,
};
use abi_models::{AcceptanceLedger, ModelManifest, ModelRegistry, StorageRoot, hash_file};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::gemma3;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::{AddedToken, Tokenizer};

pub const MODEL_ID: &str = "scratch-gemma3";
pub const PRINCIPAL: &str = "operator@example.invalid";
const REVISION: &str = "89abcdef0123456789abcdef0123456789abcdef";
pub const CONTEXT_TOKENS: u32 = 48;
pub const OUTPUT_TOKENS: u32 = 6;

pub const VOCABULARY: [&str; 12] = [
    "<unk>",
    "<bos>",
    "<eos>",
    "<start_of_turn>",
    "<end_of_turn>",
    "user",
    "model",
    "hello",
    "world",
    "again",
    "be",
    "terse",
];

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
        "attention_bias": false,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "vocab_size": VOCABULARY.len(),
        "final_logit_softcapping": null,
        "attn_logit_softcapping": null,
        "query_pre_attn_scalar": 8,
        "sliding_window": 16,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 64
    })
}

fn write_tokenizer(path: &Path) {
    let vocabulary = VOCABULARY
        .iter()
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
    tokenizer.add_special_tokens(
        &["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"].map(|t| AddedToken::from(t, true)),
    );
    tokenizer.save(path, true).expect("save tokenizer");
}

/// Write randomly initialized decoder weights split across `shards` files.
fn write_weights(directory: &Path, config: &Value, shards: usize) -> Vec<PathBuf> {
    let parsed: gemma3::Config = serde_json::from_value(config.clone()).expect("scratch config");
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    gemma3::Model::new(false, &parsed, builder).expect("scratch decoder");
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

pub fn fixture_with(config: &Value, shards: usize, config_artifacts: usize) -> Fixture {
    let scratch = Scratch::new();
    let source = scratch.root.join("source");
    std::fs::create_dir_all(&source).expect("source directory");
    let tokenizer = source.join("tokenizer.json");
    write_tokenizer(&tokenizer);
    let mut files = write_weights(&source, &self::config(), shards);
    let mut artifacts: Vec<Value> = files.iter().map(|path| artifact(path, "weights")).collect();
    artifacts.push(artifact(&tokenizer, "tokenizer"));
    files.push(tokenizer);
    for index in 0..config_artifacts {
        let path = source.join(format!("config-{index}.json"));
        std::fs::write(&path, config.to_string()).expect("write config");
        artifacts.push(artifact(&path, "config"));
        files.push(path);
    }
    let manifest = ModelManifest::from_json(
        "generated gemma fixture",
        &json!({
            "id": MODEL_ID,
            "repository": "abi-fixtures/scratch-gemma3",
            "revision": REVISION,
            "architecture": GEMMA3_TEXT_ARCHITECTURE,
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
    fixture_with(&config(), 2, 1)
}
