//! End-to-end exercise of `abi-models` from outside the crate.
//!
//! The unit tests can reach crate internals; this one deliberately cannot. It
//! proves the *published* API is sufficient to go from a manifest document to a
//! verified, license-accepted model, and that the license gate cannot be
//! side-stepped by an external caller.

use abi_models::{
    AcceptanceLedger, Chunk, ChunkTransport, DownloadOutcome, ModelError, ModelRegistry,
    ResumableDownload, StorageRoot, verify::hash_reader, verify_artifact,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const REVISION: &str = "1122334455667788990011223344556677889900";
const WEIGHTS: &[u8] = b"external-client-weight-bytes";
const TOKENIZER: &[u8] = b"external-client-tokenizer-bytes";
const WEIGHTS_URL: &str = "https://example.invalid/model.safetensors";
const TOKENIZER_URL: &str = "https://example.invalid/tokenizer.json";

/// A self-deleting scratch directory.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let path = abi_foundation::temp_path::temp_file_path(&format!("abi_models_api_{tag}"), "d");
        std::fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Lowercase hex SHA-256 of `bytes`, via the crate's public hasher.
fn digest(bytes: &[u8]) -> String {
    hash_reader(&mut &bytes[..]).expect("hashes").to_hex()
}

/// A one-element registry document.
fn catalog() -> String {
    format!(
        r#"[{{
            "id": "external-2b",
            "repository": "example-org/external-2b",
            "revision": "{REVISION}",
            "architecture": "example",
            "license": "example-community-license-1.0",
            "modalities": ["text"],
            "tensor_format": "safetensors",
            "quantizations": ["bf16"],
            "context": {{ "max_context_tokens": 4096, "max_output_tokens": 1024 }},
            "artifacts": [
                {{
                    "path": "model.safetensors",
                    "kind": "weights",
                    "sha256": "{weights_hash}",
                    "size_bytes": {weights_len},
                    "url": "{WEIGHTS_URL}"
                }},
                {{
                    "path": "tokenizer.json",
                    "kind": "tokenizer",
                    "sha256": "{tokenizer_hash}",
                    "size_bytes": {tokenizer_len},
                    "url": "{TOKENIZER_URL}"
                }}
            ]
        }}]"#,
        weights_hash = digest(WEIGHTS),
        weights_len = WEIGHTS.len(),
        tokenizer_hash = digest(TOKENIZER),
        tokenizer_len = TOKENIZER.len(),
    )
}

/// Serves whole files from memory.
struct MemoryTransport(HashMap<String, Vec<u8>>);

impl ChunkTransport for MemoryTransport {
    fn fetch(&self, url: &str, offset: u64, max_len: u64) -> Result<Chunk, ModelError> {
        let data = self.0.get(url).ok_or_else(|| ModelError::Transport {
            url: url.to_owned(),
            detail: "no such object".to_owned(),
        })?;
        let total = u64::try_from(data.len()).expect("fits");
        let start = usize::try_from(offset).expect("fits");
        let want = usize::try_from(max_len).unwrap_or(usize::MAX);
        let end = (start + want).min(data.len());
        Ok(Chunk {
            bytes: data[start.min(data.len())..end].to_vec(),
            total_len: total,
        })
    }
}

#[test]
fn an_external_caller_cannot_obtain_a_usable_model_without_acceptance() {
    let registry = ModelRegistry::from_json_array("catalog", &catalog()).expect("loads");
    let ledger = AcceptanceLedger::in_memory();

    let error = registry
        .resolve("external-2b", &ledger)
        .expect_err("the license gate must hold for external callers too");
    assert!(
        matches!(error, ModelError::LicenseNotAccepted { .. }),
        "{error:?}"
    );
}

#[test]
fn the_published_api_covers_download_verify_accept_and_resolve() {
    let scratch = Scratch::new("full_flow");
    let registry = ModelRegistry::from_json_array("catalog", &catalog()).expect("loads");
    let manifest = registry.manifest("external-2b").expect("present");

    // Storage lands outside every repository.
    let root = StorageRoot::at(scratch.path().join("weights"));
    root.reject_inside(repo_root().as_path())
        .expect("scratch storage is outside the repository");

    let transport = MemoryTransport(HashMap::from([
        (WEIGHTS_URL.to_owned(), WEIGHTS.to_vec()),
        (TOKENIZER_URL.to_owned(), TOKENIZER.to_vec()),
    ]));

    for artifact in &manifest.artifacts {
        let destination = root.artifact_path(manifest, artifact);
        let download = ResumableDownload::new(artifact, &destination);
        let outcome = download.run(&transport).expect("downloads");
        assert!(matches!(outcome, DownloadOutcome::Downloaded { .. }));
        verify_artifact(&destination, artifact).expect("verifies against the manifest");
    }

    // Files landed under the revision-scoped directory, not the repository.
    let model_dir = root.model_dir(manifest);
    assert!(model_dir.ends_with(Path::new("example-org__external-2b").join(REVISION)));
    assert!(model_dir.join("model.safetensors").is_file());
    assert!(model_dir.join("tokenizer.json").is_file());

    // Downloaded and verified is still not usable until the license is accepted.
    let mut ledger = AcceptanceLedger::load(scratch.path().join("accepted.jsonl")).expect("loads");
    assert!(registry.resolve("external-2b", &ledger).is_err());

    ledger
        .accept(manifest, "operator@example.invalid")
        .expect("accepts");
    let usable = registry.resolve("external-2b", &ledger).expect("usable");
    assert_eq!(usable.manifest().id, "external-2b");
    assert_eq!(usable.manifest().revision.as_str(), REVISION);
}

#[test]
fn a_storage_root_inside_this_repository_is_refused_through_the_public_api() {
    let inside = repo_root().join("crates/abi-models/weights");
    let error = StorageRoot::at(inside)
        .reject_inside(repo_root().as_path())
        .expect_err("weights must never live inside the repository");
    assert!(
        matches!(error, ModelError::StorageInsideRepository { .. }),
        "{error:?}"
    );
}

/// The abi repository root, from this test's own source location.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("crate lives inside the repository")
}
