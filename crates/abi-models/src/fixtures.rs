//! Test-only fixtures: a scratch directory and a known-good manifest document.
//!
//! Tests build a `serde_json::Value` from [`manifest_value`] and mutate the one
//! field under test, so a negative case differs from the positive case by
//! exactly the thing it claims to be testing.

use crate::manifest::ModelManifest;
use crate::verify::hash_reader;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// A 40-character lowercase hex revision, i.e. an immutable git SHA-1.
pub const REVISION: &str = "0f1e2d3c4b5a6978879665544332211000ffeedd";

/// A second, distinct immutable revision.
pub const OTHER_REVISION: &str = "abcdef0123456789abcdef0123456789abcdef01";

/// Stand-in weight payload. Not tensor data — only bytes to hash.
pub const WEIGHTS_BYTES: &[u8] = b"pretend-safetensors-payload-0123456789";

/// Stand-in tokenizer payload.
pub const TOKENIZER_BYTES: &[u8] = b"pretend-tokenizer-payload";

/// Lowercase hex SHA-256 of `bytes`.
pub fn digest_hex(bytes: &[u8]) -> String {
    hash_reader(&mut &bytes[..])
        .expect("hashing an in-memory slice cannot fail")
        .to_hex()
}

/// A valid manifest document as a mutable JSON value.
pub fn manifest_value() -> Value {
    json!({
        "id": "example-2b",
        "repository": "example-org/example-2b",
        "revision": REVISION,
        "architecture": "example",
        "license": "example-community-license-1.0",
        "license_sha256": digest_hex(b"example license text v1"),
        "modalities": ["text"],
        "tensor_format": "safetensors",
        "quantizations": ["bf16", "q4_k_m"],
        "context": { "max_context_tokens": 8192, "max_output_tokens": 2048 },
        "artifacts": [
            {
                "path": "model.safetensors",
                "kind": "weights",
                "sha256": digest_hex(WEIGHTS_BYTES),
                "size_bytes": WEIGHTS_BYTES.len(),
                "url": format!("https://example.invalid/{REVISION}/model.safetensors")
            },
            {
                "path": "tokenizer.json",
                "kind": "tokenizer",
                "sha256": digest_hex(TOKENIZER_BYTES),
                "size_bytes": TOKENIZER_BYTES.len(),
                "url": format!("https://example.invalid/{REVISION}/tokenizer.json")
            }
        ]
    })
}

/// Change a fixture to another immutable revision, including pinned URLs.
pub fn set_revision(value: &mut Value, revision: &str) {
    value["revision"] = json!(revision);
    for artifact in value["artifacts"]
        .as_array_mut()
        .expect("fixture artifacts are an array")
    {
        let url = artifact["url"].as_str().expect("fixture URL");
        artifact["url"] = json!(url.replace(REVISION, revision));
    }
}

/// Parse a manifest document, expecting success.
pub fn parse(value: &Value) -> ModelManifest {
    ModelManifest::from_json("fixture", &value.to_string()).expect("fixture manifest must parse")
}

/// The known-good manifest.
pub fn manifest() -> ModelManifest {
    parse(&manifest_value())
}

/// A self-deleting scratch directory outside any repository.
#[derive(Debug)]
pub struct Scratch {
    /// Root of the scratch tree.
    path: PathBuf,
}

impl Scratch {
    /// Create a uniquely named scratch directory under the platform temp dir.
    pub fn new(tag: &str) -> Self {
        let path = abi_foundation::temp_path::temp_file_path(&format!("abi_models_{tag}"), "dir");
        std::fs::create_dir_all(&path).expect("scratch directory must be creatable");
        Self { path }
    }

    /// The scratch root.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the scratch root.
    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Write a file inside the scratch root and return its path.
    pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent directory must be creatable");
        }
        std::fs::write(&path, bytes).expect("scratch file must be writable");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
