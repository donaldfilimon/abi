//! Typed errors for manifest parsing, registry validation, hash verification,
//! download resumption, and license acceptance.
//!
//! Every failure mode in this crate is a distinct variant rather than a string,
//! because callers are expected to *act* on the distinction: a
//! [`ModelError::FloatingRevision`] is an authoring bug in a manifest, while a
//! [`ModelError::HashMismatch`] is a corrupt or tampered artifact.

use std::path::PathBuf;

/// Errors produced by the model registry and its download plumbing.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// A manifest omitted the `revision` field entirely.
    #[error("model '{model}': manifest has no revision; an immutable revision is required")]
    MissingRevision {
        /// Model identifier the manifest declared.
        model: String,
    },

    /// A manifest named a mutable ref (`main`, `HEAD`, a tag) instead of an
    /// immutable commit revision.
    #[error(
        "model '{model}': revision '{value}' is a floating ref; \
         an immutable 40- or 64-character lowercase hex revision is required"
    )]
    FloatingRevision {
        /// Model identifier the manifest declared.
        model: String,
        /// The rejected revision string.
        value: String,
    },

    /// An artifact entry omitted its `sha256` field.
    #[error(
        "model '{model}': artifact '{artifact}' has no sha256 hash; every artifact must be hashed"
    )]
    MissingHash {
        /// Model identifier the manifest declared.
        model: String,
        /// Relative path of the artifact that lacked a hash.
        artifact: String,
    },

    /// A hash was present but was not 64 lowercase hex characters.
    #[error("model '{model}': artifact '{artifact}' has malformed sha256 '{value}'")]
    MalformedHash {
        /// Model identifier the manifest declared.
        model: String,
        /// Relative path of the offending artifact.
        artifact: String,
        /// The rejected hash string.
        value: String,
    },

    /// A required manifest field was empty.
    #[error("model '{model}': field '{field}' must not be empty")]
    EmptyField {
        /// Model identifier the manifest declared (or `<unnamed>`).
        model: String,
        /// Name of the empty field.
        field: &'static str,
    },

    /// A manifest was structurally valid but semantically incoherent.
    #[error("model '{model}': {reason}")]
    InvalidManifest {
        /// Model identifier the manifest declared.
        model: String,
        /// Human-readable explanation of the inconsistency.
        reason: String,
    },

    /// An artifact path escaped the model directory or was absolute.
    #[error("model '{model}': artifact path '{path}' must be relative and must not contain '..'")]
    UnsafeArtifactPath {
        /// Model identifier the manifest declared.
        model: String,
        /// The rejected path.
        path: String,
    },

    /// Two manifests claimed the same model identifier.
    #[error("duplicate model id '{id}' in registry")]
    DuplicateModel {
        /// The repeated identifier.
        id: String,
    },

    /// A lookup named a model the registry does not hold.
    #[error("unknown model id '{id}'")]
    UnknownModel {
        /// The requested identifier.
        id: String,
    },

    /// A manifest document was not well-formed JSON, or did not match the schema.
    #[error("{context}: {source}")]
    Json {
        /// What was being parsed, for locating the offending document.
        context: String,
        /// Underlying `serde_json` failure.
        #[source]
        source: serde_json::Error,
    },

    /// A filesystem operation failed.
    #[error("io error at {path}: {source}")]
    Io {
        /// Path being read or written.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A file's streamed SHA-256 did not match the manifest.
    #[error("hash mismatch for {path}: expected {expected}, computed {actual}")]
    HashMismatch {
        /// File that was verified.
        path: PathBuf,
        /// Hex digest the manifest declared.
        expected: String,
        /// Hex digest actually computed.
        actual: String,
    },

    /// A downloaded artifact's byte count did not match the manifest.
    #[error("size mismatch for {path}: expected {expected} bytes, found {actual}")]
    SizeMismatch {
        /// File that was measured.
        path: PathBuf,
        /// Size the manifest declared.
        expected: u64,
        /// Size actually observed.
        actual: u64,
    },

    /// A partial file on disk was longer than the artifact it is resuming, so
    /// its contents cannot be a prefix of the expected bytes.
    #[error(
        "partial file {path} holds {have} bytes but the artifact is only {total}; refusing to resume"
    )]
    ResumeBeyondEnd {
        /// The `.part` file.
        path: PathBuf,
        /// Bytes already on disk.
        have: u64,
        /// Total expected artifact size.
        total: u64,
    },

    /// A transport returned no bytes while the download was still incomplete.
    #[error("transport stalled at offset {offset} of {total} for {url}")]
    TransportStalled {
        /// Source URL.
        url: String,
        /// Offset at which no progress was made.
        offset: u64,
        /// Total expected artifact size.
        total: u64,
    },

    /// A transport implementation failed.
    #[error("transport error for {url}: {detail}")]
    Transport {
        /// Source URL.
        url: String,
        /// Implementation-supplied detail.
        detail: String,
    },

    /// An artifact URL is not an immutable HTTPS source for its revision.
    #[error(
        "model '{model}': artifact URL '{url}' is not a pinned HTTPS URL for revision {revision}"
    )]
    UnpinnedArtifactUrl {
        /// Model identifier.
        model: String,
        /// Rejected URL.
        url: String,
        /// Immutable revision that must appear as a complete URL component.
        revision: String,
    },

    /// A declared or reported artifact is larger than the configured bound.
    #[error("artifact size {actual} bytes exceeds configured maximum {maximum} bytes for {url}")]
    ArtifactTooLarge {
        /// Artifact URL or destination identifying the rejected object.
        url: String,
        /// Size declared or reported.
        actual: u64,
        /// Maximum accepted size.
        maximum: u64,
    },

    /// A signed manifest named a publisher key absent from the trust store.
    #[error("manifest signature names untrusted publisher key '{key_id}'")]
    UntrustedPublisherKey {
        /// Publisher key identifier from the envelope.
        key_id: String,
    },

    /// A publisher key identifier was configured more than once.
    #[error("publisher key '{key_id}' is configured more than once")]
    DuplicatePublisherKey {
        /// Repeated publisher key identifier.
        key_id: String,
    },

    /// Publisher key or signature bytes were malformed.
    #[error("invalid {kind} hex for publisher key '{key_id}'")]
    MalformedSignatureMaterial {
        /// Whether the key or signature was malformed.
        kind: &'static str,
        /// Publisher key identifier.
        key_id: String,
    },

    /// Signature verification failed for a manifest envelope.
    #[error("manifest signature verification failed for publisher key '{key_id}'")]
    InvalidManifestSignature {
        /// Publisher key identifier.
        key_id: String,
    },

    /// The model's license has not been accepted for this exact revision.
    #[error(
        "model '{model}' at revision {revision} requires accepting license '{license}' before use"
    )]
    LicenseNotAccepted {
        /// Model identifier.
        model: String,
        /// Revision the caller tried to use.
        revision: String,
        /// License identifier that must be accepted.
        license: String,
    },

    /// An acceptance record named a license the manifest does not declare.
    #[error(
        "acceptance for model '{model}' names license '{recorded}' but the manifest declares '{declared}'"
    )]
    LicenseMismatch {
        /// Model identifier.
        model: String,
        /// License in the acceptance record.
        recorded: String,
        /// License in the manifest.
        declared: String,
    },

    /// A prior acceptance does not cover the current license digest or
    /// artifact hash set.
    #[error(
        "acceptance for model '{model}' does not cover the current license document and artifact hashes"
    )]
    AcceptanceBindingMismatch {
        /// Model identifier.
        model: String,
    },

    /// The requested principal has not accepted this exact model identity.
    #[error("principal '{principal}' has not accepted model '{model}' at revision {revision}")]
    PrincipalNotAccepted {
        /// Model identifier.
        model: String,
        /// Immutable model revision.
        revision: String,
        /// Principal required by the caller.
        principal: String,
    },

    /// A license acceptance principal was blank.
    #[error("license acceptance principal must not be blank")]
    InvalidAcceptancePrincipal,

    /// The configured weight-storage root lies inside a source repository.
    #[error(
        "model storage root {path} is inside repository {repository}; weights must live outside both repositories"
    )]
    StorageInsideRepository {
        /// The rejected storage root.
        path: PathBuf,
        /// The repository root it fell inside.
        repository: PathBuf,
    },

    /// The storage root could not be resolved because `HOME` is unset and no
    /// override was supplied.
    #[error("cannot resolve model storage root: neither {var} nor HOME is set")]
    StorageRootUnresolvable {
        /// Name of the override environment variable.
        var: &'static str,
    },
}

impl ModelError {
    /// Wrap an I/O failure with the path that produced it.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Wrap a `serde_json` failure with a description of what was being parsed.
    pub(crate) fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }
}
