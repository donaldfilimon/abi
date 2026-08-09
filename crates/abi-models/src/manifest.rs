//! The model manifest schema.
//!
//! A manifest is the complete, auditable description of one downloadable model:
//! where it comes from, at which *immutable* revision, which files it is made of
//! and what each of those files must hash to, what license governs it, and what
//! shapes of input the publisher accepts.
//!
//! ## Two-layer parsing
//!
//! Deserialization goes through a private [`RawManifest`] whose fields are all
//! plain strings and options. That raw document is then converted into
//! [`ModelManifest`], which is the validated type. The split exists so semantic
//! failures surface as typed [`ModelError`] variants — `FloatingRevision`,
//! `MissingHash` — instead of being flattened into a `serde_json` message.
//!
//! Because the only constructor is that fallible conversion, a `ModelManifest`
//! value cannot exist with a floating revision or an unhashed artifact.
//!
//! ## Document shape
//!
//! ```text
//! {
//!   "id": "example-2b",
//!   "repository": "example-org/example-2b",
//!   "revision": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
//!   "architecture": "example",
//!   "license": "apache-2.0",
//!   "modalities": ["text"],
//!   "tensor_format": "safetensors",
//!   "quantizations": ["bf16", "q4_k_m"],
//!   "context": { "max_context_tokens": 8192, "max_output_tokens": 2048 },
//!   "artifacts": [
//!     {
//!       "path": "model.safetensors",
//!       "kind": "weights",
//!       "sha256": "<64 lowercase hex characters>",
//!       "size_bytes": 12,
//!       "url": "https://example.invalid/model.safetensors"
//!     }
//!   ]
//! }
//! ```

use crate::error::ModelError;
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;

/// Number of hex characters in a git SHA-1 revision.
const SHA1_HEX_LEN: usize = 40;
/// Number of hex characters in a SHA-256 revision or digest.
const SHA256_HEX_LEN: usize = 64;

/// An immutable repository revision.
///
/// Only a full lowercase hex commit id is accepted — 40 characters (SHA-1) or
/// 64 (SHA-256). Branch names, `HEAD`, `latest` and *tags* are all rejected:
/// tags are as mutable as branches, so accepting them would defeat the point of
/// pinning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Revision(String);

impl Revision {
    /// Parse an immutable revision, rejecting any floating ref.
    pub fn parse(model: &str, value: &str) -> Result<Self, ModelError> {
        let len = value.len();
        let shaped = (len == SHA1_HEX_LEN || len == SHA256_HEX_LEN)
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'));
        if shaped {
            Ok(Self(value.to_owned()))
        } else {
            Err(ModelError::FloatingRevision {
                model: model.to_owned(),
                value: value.to_owned(),
            })
        }
    }

    /// The revision as a hex string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for Revision {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

/// A parsed SHA-256 digest.
///
/// Stored as bytes rather than text so comparisons cannot be confused by case
/// or whitespace, and rendered back to lowercase hex on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Parse 64 lowercase hex characters into a digest.
    pub fn parse(model: &str, artifact: &str, value: &str) -> Result<Self, ModelError> {
        let malformed = || ModelError::MalformedHash {
            model: model.to_owned(),
            artifact: artifact.to_owned(),
            value: value.to_owned(),
        };
        if value.len() != SHA256_HEX_LEN {
            return Err(malformed());
        }
        let mut bytes = [0u8; 32];
        for (index, slot) in bytes.iter_mut().enumerate() {
            let pair = value
                .get(index * 2..index * 2 + 2)
                .ok_or_else(malformed)?
                .as_bytes();
            let hi = hex_nibble(pair[0]).ok_or_else(malformed)?;
            let lo = hex_nibble(pair[1]).ok_or_else(malformed)?;
            *slot = (hi << 4) | lo;
        }
        Ok(Self(bytes))
    }

    /// Wrap raw digest bytes, as produced by a streaming hasher.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Render as 64 lowercase hex characters.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(SHA256_HEX_LEN);
        for byte in self.0 {
            out.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            out.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
        out
    }
}

/// Lowercase hex alphabet used by [`Sha256Digest::to_hex`].
const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";

/// Decode one lowercase hex character into its nibble value.
fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

/// An input or output modality the model handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Natural-language text.
    Text,
    /// Still images.
    Vision,
    /// Audio waveforms.
    Audio,
}

/// On-disk tensor container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorFormat {
    /// `safetensors` shards.
    Safetensors,
    /// `GGUF` single-file format.
    Gguf,
}

/// A weight quantization the publisher ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantization {
    /// 32-bit float.
    #[serde(rename = "f32")]
    F32,
    /// 16-bit float.
    #[serde(rename = "f16")]
    F16,
    /// bfloat16.
    #[serde(rename = "bf16")]
    Bf16,
    /// 8-bit integer, block scale.
    #[serde(rename = "q8_0")]
    Q8,
    /// 4-bit integer, k-quant medium.
    #[serde(rename = "q4_k_m")]
    Q4KM,
}

/// The role a file plays within a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Tensor data.
    Weights,
    /// Tokenizer vocabulary/merges.
    Tokenizer,
    /// Architecture or generation configuration.
    Config,
}

/// Accepted context limits, in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextLimits {
    /// Largest total context (prompt plus generation) the model accepts.
    pub max_context_tokens: u32,
    /// Largest generation length accepted within that context.
    pub max_output_tokens: u32,
}

/// One downloadable file belonging to a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Artifact {
    /// Path relative to the model's directory under the storage root.
    pub path: String,
    /// What this file is.
    pub kind: ArtifactKind,
    /// Expected SHA-256 of the complete file.
    pub sha256: Sha256Digest,
    /// Expected size in bytes.
    pub size_bytes: u64,
    /// Where the bytes are fetched from.
    pub url: String,
}

/// A validated model manifest.
///
/// Construct with [`ModelManifest::from_json`]; there is no way to build one
/// that skipped validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelManifest {
    /// Registry-unique local identifier.
    pub id: String,
    /// Upstream repository, e.g. `org/name`.
    pub repository: String,
    /// Immutable revision within that repository.
    pub revision: Revision,
    /// Architecture family name.
    pub architecture: String,
    /// License identifier that must be accepted before use.
    pub license: String,
    /// Modalities the model handles.
    pub modalities: Vec<Modality>,
    /// Container format of the weight files.
    pub tensor_format: TensorFormat,
    /// Quantizations the publisher ships.
    pub quantizations: Vec<Quantization>,
    /// Accepted context limits.
    pub context: ContextLimits,
    /// Files making up the model.
    pub artifacts: Vec<Artifact>,
}

impl ModelManifest {
    /// Parse and validate a manifest document.
    pub fn from_json(context: &str, document: &str) -> Result<Self, ModelError> {
        let raw: RawManifest =
            serde_json::from_str(document).map_err(|source| ModelError::json(context, source))?;
        Self::try_from(raw)
    }

    /// Serialize back to a pretty JSON document.
    pub fn to_json(&self) -> Result<String, ModelError> {
        serde_json::to_string_pretty(self).map_err(|source| ModelError::json(&self.id, source))
    }

    /// The single tokenizer artifact.
    #[must_use]
    pub fn tokenizer(&self) -> &Artifact {
        self.artifacts
            .iter()
            .find(|artifact| artifact.kind == ArtifactKind::Tokenizer)
            .expect("validated manifest always has a tokenizer artifact")
    }

    /// Every weight artifact, in manifest order.
    pub fn weights(&self) -> impl Iterator<Item = &Artifact> {
        self.artifacts
            .iter()
            .filter(|artifact| artifact.kind == ArtifactKind::Weights)
    }
}

/// The serde-facing manifest document, before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    id: Option<String>,
    repository: Option<String>,
    revision: Option<String>,
    architecture: Option<String>,
    license: Option<String>,
    modalities: Vec<Modality>,
    tensor_format: TensorFormat,
    quantizations: Vec<Quantization>,
    context: ContextLimits,
    artifacts: Vec<RawArtifact>,
}

/// The serde-facing artifact entry, before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    path: Option<String>,
    kind: ArtifactKind,
    sha256: Option<String>,
    size_bytes: u64,
    url: Option<String>,
}

impl TryFrom<RawManifest> for ModelManifest {
    type Error = ModelError;

    fn try_from(raw: RawManifest) -> Result<Self, Self::Error> {
        let id = required(raw.id.as_deref(), "<unnamed>", "id")?;
        let repository = required(raw.repository.as_deref(), &id, "repository")?;
        let architecture = required(raw.architecture.as_deref(), &id, "architecture")?;
        let license = required(raw.license.as_deref(), &id, "license")?;

        let Some(revision) = raw.revision.as_deref() else {
            return Err(ModelError::MissingRevision { model: id });
        };
        let revision = Revision::parse(&id, revision)?;

        if raw.modalities.is_empty() {
            return Err(ModelError::EmptyField {
                model: id,
                field: "modalities",
            });
        }
        if raw.quantizations.is_empty() {
            return Err(ModelError::EmptyField {
                model: id,
                field: "quantizations",
            });
        }
        validate_context(&id, raw.context)?;

        let mut artifacts = Vec::with_capacity(raw.artifacts.len());
        for entry in &raw.artifacts {
            artifacts.push(convert_artifact(&id, entry)?);
        }
        require_kinds(&id, &artifacts)?;

        Ok(Self {
            id,
            repository,
            revision,
            architecture,
            license,
            modalities: raw.modalities,
            tensor_format: raw.tensor_format,
            quantizations: raw.quantizations,
            context: raw.context,
            artifacts,
        })
    }
}

/// Require a present, non-blank string field.
fn required(value: Option<&str>, model: &str, field: &'static str) -> Result<String, ModelError> {
    match value {
        Some(text) if !text.trim().is_empty() => Ok(text.to_owned()),
        _ => Err(ModelError::EmptyField {
            model: model.to_owned(),
            field,
        }),
    }
}

/// Reject incoherent context limits.
fn validate_context(model: &str, context: ContextLimits) -> Result<(), ModelError> {
    if context.max_context_tokens == 0 {
        return Err(ModelError::InvalidManifest {
            model: model.to_owned(),
            reason: "max_context_tokens must be greater than zero".to_owned(),
        });
    }
    if context.max_output_tokens == 0 || context.max_output_tokens > context.max_context_tokens {
        return Err(ModelError::InvalidManifest {
            model: model.to_owned(),
            reason: format!(
                "max_output_tokens ({}) must be in 1..={}",
                context.max_output_tokens, context.max_context_tokens
            ),
        });
    }
    Ok(())
}

/// Validate one raw artifact entry.
fn convert_artifact(model: &str, raw: &RawArtifact) -> Result<Artifact, ModelError> {
    let path = required(raw.path.as_deref(), model, "artifacts[].path")?;
    if path.starts_with('/')
        || path.starts_with('\\')
        || path.split(['/', '\\']).any(|segment| segment == "..")
    {
        return Err(ModelError::UnsafeArtifactPath {
            model: model.to_owned(),
            path,
        });
    }
    let url = required(raw.url.as_deref(), model, "artifacts[].url")?;

    let Some(hash) = raw.sha256.as_deref() else {
        return Err(ModelError::MissingHash {
            model: model.to_owned(),
            artifact: path,
        });
    };
    let sha256 = Sha256Digest::parse(model, &path, hash)?;

    Ok(Artifact {
        path,
        kind: raw.kind,
        sha256,
        size_bytes: raw.size_bytes,
        url,
    })
}

/// A model needs at least one weight file and exactly one tokenizer.
fn require_kinds(model: &str, artifacts: &[Artifact]) -> Result<(), ModelError> {
    let weights = artifacts
        .iter()
        .filter(|entry| entry.kind == ArtifactKind::Weights)
        .count();
    let tokenizers = artifacts
        .iter()
        .filter(|entry| entry.kind == ArtifactKind::Tokenizer)
        .count();
    if weights == 0 {
        return Err(ModelError::InvalidManifest {
            model: model.to_owned(),
            reason: "manifest declares no weights artifact".to_owned(),
        });
    }
    if tokenizers != 1 {
        return Err(ModelError::InvalidManifest {
            model: model.to_owned(),
            reason: format!(
                "manifest declares {tokenizers} tokenizer artifacts, expected exactly 1"
            ),
        });
    }
    let mut seen: Vec<&str> = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        if seen.contains(&artifact.path.as_str()) {
            return Err(ModelError::InvalidManifest {
                model: model.to_owned(),
                reason: format!("duplicate artifact path '{}'", artifact.path),
            });
        }
        seen.push(&artifact.path);
    }
    Ok(())
}
