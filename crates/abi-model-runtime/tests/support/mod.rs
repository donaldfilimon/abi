use abi_model_runtime::{DevicePreference, FIXTURE_BIGRAM_ARCHITECTURE, LoadConfig};
use abi_models::{AcceptanceLedger, ModelManifest, ModelRegistry, StorageRoot, hash_file};
use candle_core::{Device, Tensor};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

pub const MODEL_ID: &str = "fixture-bigram";
pub const PRINCIPAL: &str = "operator@example.invalid";
pub const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

pub struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let root = abi_foundation::temp_path::temp_file_path("abi_model_runtime", "scratch");
        std::fs::create_dir_all(&root).expect("scratch directory");
        Self { root }
    }

    fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.root.join(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

pub struct Fixture {
    _scratch: Scratch,
    pub registry: ModelRegistry,
    pub ledger: AcceptanceLedger,
    pub storage: StorageRoot,
    pub manifest: ModelManifest,
}

impl Fixture {
    pub fn load_cpu(&self) -> abi_model_runtime::LocalModelProvider {
        abi_model_runtime::LocalModelProvider::load(
            &self.registry,
            &self.ledger,
            &self.storage,
            MODEL_ID,
            PRINCIPAL,
            LoadConfig::new(DevicePreference::Cpu),
        )
        .expect("CPU fixture loads")
    }

    pub fn artifact_path(&self, name: &str) -> PathBuf {
        let artifact = self
            .manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.path == name)
            .expect("fixture artifact");
        self.storage.artifact_path(&self.manifest, artifact)
    }
}

pub fn fixture() -> Fixture {
    fixture_with_architecture(FIXTURE_BIGRAM_ARCHITECTURE)
}

pub fn fixture_with_architecture(architecture: &str) -> Fixture {
    let scratch = Scratch::new();
    let source_weights = scratch.join("source/model.safetensors");
    let source_tokenizer = scratch.join("source/tokenizer.json");
    std::fs::create_dir_all(source_weights.parent().expect("weights parent"))
        .expect("source directory");
    write_weights(&source_weights);
    write_tokenizer(&source_tokenizer);

    let weights_size = std::fs::metadata(&source_weights)
        .expect("weights metadata")
        .len();
    let tokenizer_size = std::fs::metadata(&source_tokenizer)
        .expect("tokenizer metadata")
        .len();
    let manifest = ModelManifest::from_json(
        "generated fixture",
        &json!({
            "id": MODEL_ID,
            "repository": "abi-fixtures/bigram",
            "revision": REVISION,
            "architecture": architecture,
            "license": "fixture-only-1.0",
            "license_sha256": abi_models::Sha256Digest::from_bytes([7; 32]).to_hex(),
            "modalities": ["text"],
            "tensor_format": "safetensors",
            "quantizations": ["f32"],
            "context": { "max_context_tokens": 8, "max_output_tokens": 4 },
            "artifacts": [
                {
                    "path": "model.safetensors",
                    "kind": "weights",
                    "sha256": hash_file(&source_weights).expect("weights hash").to_hex(),
                    "size_bytes": weights_size,
                    "url": format!("https://models.example.invalid/{REVISION}/model.safetensors")
                },
                {
                    "path": "tokenizer.json",
                    "kind": "tokenizer",
                    "sha256": hash_file(&source_tokenizer).expect("tokenizer hash").to_hex(),
                    "size_bytes": tokenizer_size,
                    "url": format!("https://models.example.invalid/{REVISION}/tokenizer.json")
                }
            ]
        })
        .to_string(),
    )
    .expect("generated manifest validates");

    let storage = StorageRoot::at(scratch.join("models"));
    let weights = storage.artifact_path(&manifest, &manifest.artifacts[0]);
    let tokenizer = storage.artifact_path(&manifest, &manifest.artifacts[1]);
    std::fs::create_dir_all(weights.parent().expect("model directory")).expect("model directory");
    std::fs::copy(source_weights, weights).expect("copy weights");
    std::fs::copy(source_tokenizer, tokenizer).expect("copy tokenizer");

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
        manifest,
    }
}

fn write_tokenizer(path: &Path) {
    let vocabulary = [
        ("<unk>".to_owned(), 0),
        ("<eos>".to_owned(), 1),
        ("hello".to_owned(), 2),
        ("world".to_owned(), 3),
        ("again".to_owned(), 4),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("word-level model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.save(path, true).expect("save tokenizer");
}

fn write_weights(path: &Path) {
    let vocabulary = 5usize;
    let mut logits = vec![-100.0f32; vocabulary * vocabulary];
    for token in 0..vocabulary {
        logits[token * vocabulary + 1] = 10.0;
    }
    logits[2 * vocabulary + 3] = 20.0;
    logits[3 * vocabulary + 4] = 20.0;
    logits[4 * vocabulary + 1] = 20.0;
    let tensor = Tensor::from_vec(logits, (vocabulary, vocabulary), &Device::Cpu)
        .expect("transition tensor");
    let tensors = HashMap::from([("transition".to_owned(), tensor)]);
    candle_core::safetensors::save(&tensors, path).expect("save safetensors");
}
