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
//! - [`registry`] — a validated set of manifests, keyed by model id.
//! - [`verify`] — streaming SHA-256 of a file, compared against the manifest.
//! - [`download`] — the byte-range/partial-file resume state machine, written
//!   against the [`ChunkTransport`] trait so it is exercised end to end without
//!   a network.
//! - [`license`] — a persisted acceptance ledger. Resolution of a usable model
//!   goes through it, so "usable" and "license accepted for this exact
//!   revision" are the same statement.
//! - [`storage`] — resolution of the weight-storage root, which is always
//!   outside any source repository.
//!
//! # Explicitly not implemented (Proposed)
//!
//! These are named here so no caller mistakes an absence for a capability:
//!
//! - **Signature verification.** Only content hashes are checked. Publisher
//!   signatures over manifests are Proposed; nothing in this crate verifies an
//!   identity, so a manifest is only as trustworthy as its own provenance.
//! - **A real network transport.** [`HttpTransport`] returns
//!   [`ModelError::TransportNotImplemented`] rather than panicking. No HTTP
//!   client is a dependency of this crate.
//! - **Loading, decoding or running weights.** No `safetensors`, `GGUF`, or
//!   tensor-library integration exists here, and no architecture is
//!   implemented. Manifest fields such as `tensor_format` and `architecture`
//!   are recorded metadata, not evidence that a loader exists.
//! - **Any throughput, memory or accuracy characterisation.** This crate makes
//!   no performance claim.
//!
//! # Note on the environment variable
//!
//! `abi-foundation`'s `env` module is normally the single registry for `ABI_*`
//! variable names. [`storage::STORAGE_ROOT_VAR`] is declared locally instead,
//! to keep this crate's diff self-contained; it is still read through
//! `abi_foundation::env::get`, so the test overlay applies as usual.

pub mod download;
pub mod error;
pub mod license;
pub mod manifest;
pub mod registry;
pub mod storage;
pub mod verify;

pub use download::{Chunk, ChunkTransport, DownloadOutcome, HttpTransport, ResumableDownload};
pub use error::ModelError;
pub use license::{AcceptanceLedger, AcceptanceRecord};
pub use manifest::{
    Artifact, ArtifactKind, ContextLimits, Modality, ModelManifest, Quantization, Revision,
    Sha256Digest, TensorFormat,
};
pub use registry::{ModelRegistry, UsableModel};
pub use storage::StorageRoot;
pub use verify::{hash_file, verify_artifact};
