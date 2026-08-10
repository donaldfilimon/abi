//! Hash-verified model registry and download plumbing for ABI.
//!
//! This crate is the *infrastructure* half of local model support: it decides
//! which models exist, where their bytes may live, whether those bytes are the
//! bytes the publisher named, and whether the operator has accepted the license
//! that governs them. It performs no inference and links no tensor library.
//!
//! # What is implemented here
//!
//! - [`manifest`] — the manifest schema. A [`ModelManifest`] cannot be
//!   constructed with a floating revision or an unhashed artifact; those are
//!   parse-time errors, not warnings.
//! - [`registry`] — validated unsigned and publisher-authenticated manifest
//!   collections, keyed by model id.
//! - [`signed`] — Ed25519 signed manifest envelopes and an explicit publisher
//!   trust store. Signature verification never replaces artifact hashes.
//! - [`verify`] — streaming SHA-256 of a file, compared against the manifest.
//! - [`download`] — a bounded HTTPS byte-range transport plus the
//!   partial-file resume and atomic no-clobber publication state machine.
//! - [`license`] — a persisted acceptance ledger. Resolution of a usable model
//!   goes through it, so "usable" and "license accepted for this exact
//!   revision" are the same statement.
//! - [`storage`] — resolution of the weight-storage root, which is always
//!   outside any source repository.
//!
//! # Explicit boundary
//!
//! These are named here so no caller mistakes an absence for a capability:
//!
//! - **Loading, decoding or running weights.** No `safetensors`, `GGUF`, or
//!   tensor-library integration exists here, and no architecture is
//!   implemented. Manifest fields such as `tensor_format` and `architecture`
//!   are recorded metadata, not evidence that a loader exists.
//! - **Any throughput, memory or accuracy characterisation.** This crate makes
//!   no performance claim.
//! - **Repository-owned model data.** Weights, datasets, and generated adapters
//!   belong under an external [`StorageRoot`], never in this source tree.
//!
//! # Note on the environment variable
//!
//! `abi-foundation`'s `env` module is normally the single registry for `ABI_*`
//! variable names. [`storage::STORAGE_ROOT_VAR`] is declared locally instead,
//! to keep this crate's diff self-contained; it is still read through
//! `abi_foundation::env::get`, so the test overlay applies as usual.

pub mod download;
pub mod error;
#[cfg(test)]
mod fixtures;
pub mod license;
pub mod manifest;
pub mod registry;
pub mod signed;
pub mod storage;
pub mod verify;

pub use download::{Chunk, ChunkTransport, DownloadOutcome, HttpTransport, ResumableDownload};
pub use error::ModelError;
pub use license::{AcceptanceLedger, AcceptanceRecord, AcceptedArtifact};
pub use manifest::{
    Artifact, ArtifactKind, ContextLimits, Modality, ModelManifest, Quantization, Revision,
    Sha256Digest, TensorFormat,
};
pub use registry::{ModelRegistry, UsableModel};
pub use signed::{PublisherTrustStore, SIGNED_MANIFEST_VERSION, SignedManifestEnvelope};
pub use storage::StorageRoot;
pub use verify::{hash_file, verify_artifact};
