//! The model manifest schema.
//!
//! A manifest is the complete, auditable description of one downloadable model:
//! where it comes from, at which *immutable* revision, which files it is made of
//! and what each of those files must hash to, what license governs it, and what
//! shapes of input the publisher accepts.
//!
//! ## Two-layer parsing
//!
//! Deserialization goes through a private `RawManifest` whose fields are all
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
//!   "license_sha256": "<64 lowercase hex characters>",
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
//!       "url": "https://example.invalid/<revision>/model.safetensors"
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
    /// SHA-256 of the exact license document accepted by the operator.
    pub license_sha256: Sha256Digest,
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
    license_sha256: Option<String>,
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
        let Some(license_hash) = raw.license_sha256.as_deref() else {
            return Err(ModelError::MissingHash {
                model: id,
                artifact: "<license>".to_owned(),
            });
        };
        let license_sha256 = Sha256Digest::parse(&id, "<license>", license_hash)?;

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
            artifacts.push(convert_artifact(&id, revision.as_str(), entry)?);
        }
        require_kinds(&id, &artifacts)?;

        Ok(Self {
            id,
            repository,
            revision,
            architecture,
            license,
            license_sha256,
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
fn convert_artifact(
    model: &str,
    revision: &str,
    raw: &RawArtifact,
) -> Result<Artifact, ModelError> {
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
    validate_pinned_https_url(model, revision, &url)?;
    if raw.size_bytes == 0 {
        return Err(ModelError::InvalidManifest {
            model: model.to_owned(),
            reason: format!("artifact '{path}' must declare a non-zero size"),
        });
    }

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

/// Require HTTPS and the immutable revision as one complete URL component.
fn validate_pinned_https_url(model: &str, revision: &str, url: &str) -> Result<(), ModelError> {
    let parsed: ureq::http::Uri = url.parse().map_err(|_| ModelError::UnpinnedArtifactUrl {
        model: model.to_owned(),
        url: url.to_owned(),
        revision: revision.to_owned(),
    })?;
    let pinned = parsed.scheme_str() == Some("https")
        && parsed
            .authority()
            .is_some_and(|authority| !authority.as_str().contains('@'))
        && !url.contains('#')
        && url
            .split(['/', '?', '&', '='])
            .any(|component| component == revision);
    if !pinned {
        return Err(ModelError::UnpinnedArtifactUrl {
            model: model.to_owned(),
            url: url.to_owned(),
            revision: revision.to_owned(),
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{OTHER_REVISION, REVISION, manifest_value, parse, set_revision};
    use serde_json::json;

    /// Parse a mutated fixture, expecting failure.
    fn reject(mutate: impl FnOnce(&mut serde_json::Value)) -> ModelError {
        let mut value = manifest_value();
        mutate(&mut value);
        ModelManifest::from_json("fixture", &value.to_string())
            .expect_err("mutated fixture must be rejected")
    }

    #[test]
    fn valid_manifest_parses_with_every_schema_field() {
        let manifest = parse(&manifest_value());
        assert_eq!(manifest.id, "example-2b");
        assert_eq!(manifest.repository, "example-org/example-2b");
        assert_eq!(manifest.revision.as_str(), REVISION);
        assert_eq!(manifest.architecture, "example");
        assert_eq!(manifest.license, "example-community-license-1.0");
        assert_eq!(
            manifest.license_sha256.to_hex(),
            crate::fixtures::digest_hex(b"example license text v1")
        );
        assert_eq!(manifest.modalities, vec![Modality::Text]);
        assert_eq!(manifest.tensor_format, TensorFormat::Safetensors);
        assert_eq!(
            manifest.quantizations,
            vec![Quantization::Bf16, Quantization::Q4KM]
        );
        assert_eq!(manifest.context.max_context_tokens, 8192);
        assert_eq!(manifest.context.max_output_tokens, 2048);
        assert_eq!(manifest.artifacts.len(), 2);
        assert_eq!(manifest.tokenizer().path, "tokenizer.json");
        assert_eq!(manifest.weights().count(), 1);
    }

    #[test]
    fn artifact_urls_require_https_and_the_exact_revision_component() {
        for url in [
            format!("http://example.invalid/{REVISION}/model.safetensors"),
            "https://example.invalid/model.safetensors".to_owned(),
            format!("https://example.invalid/{REVISION}extra/model.safetensors"),
        ] {
            let error = reject(|value| value["artifacts"][0]["url"] = json!(url));
            assert!(matches!(error, ModelError::UnpinnedArtifactUrl { .. }));
        }
    }

    #[test]
    fn artifact_sizes_must_be_nonzero() {
        let error = reject(|value| value["artifacts"][0]["size_bytes"] = json!(0));
        assert!(matches!(error, ModelError::InvalidManifest { .. }));
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = parse(&manifest_value());
        let document = manifest.to_json().expect("serializable");
        let reparsed = ModelManifest::from_json("round-trip", &document).expect("reparses");
        assert_eq!(manifest, reparsed);
    }

    #[test]
    fn a_floating_ref_is_a_hard_error() {
        // Tags are as mutable as branches, so `v1.0.0` must be refused too.
        for floating in [
            "main",
            "HEAD",
            "latest",
            "v1.0.0",
            "refs/heads/main",
            "",
            "0f1e2d3c",                                 // truncated
            "0F1E2D3C4B5A6978879665544332211000FFEEDD", // uppercase
            "0f1e2d3c4b5a6978879665544332211000ffeedg", // non-hex
            "0f1e2d3c4b5a6978879665544332211000ffeedd0f1e2d3c4b5a69788796655443", // 66 chars
        ] {
            let error = reject(|value| value["revision"] = json!(floating));
            assert!(
                matches!(error, ModelError::FloatingRevision { ref value, .. } if value == floating),
                "revision {floating:?} produced {error:?}"
            );
        }
    }

    #[test]
    fn a_sha256_revision_is_accepted() {
        let long = "a".repeat(64);
        let mut value = manifest_value();
        set_revision(&mut value, &long);
        assert_eq!(parse(&value).revision.as_str(), long);
    }

    #[test]
    fn an_absent_revision_is_a_hard_error() {
        let error = reject(|value| {
            value
                .as_object_mut()
                .expect("object")
                .remove("revision")
                .expect("fixture has a revision");
        });
        assert!(
            matches!(error, ModelError::MissingRevision { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn an_artifact_without_a_hash_is_a_hard_error() {
        let error = reject(|value| {
            value["artifacts"][0]
                .as_object_mut()
                .expect("object")
                .remove("sha256")
                .expect("fixture artifact has a hash");
        });
        assert!(
            matches!(error, ModelError::MissingHash { ref artifact, .. }
                if artifact == "model.safetensors"),
            "{error:?}"
        );
    }

    #[test]
    fn a_malformed_hash_is_a_hard_error() {
        for bad in [
            "",
            "deadbeef",
            &"a".repeat(63),
            &"a".repeat(65),
            &"A".repeat(64),
            &"z".repeat(64),
        ] {
            let error = reject(|value| value["artifacts"][0]["sha256"] = json!(bad));
            assert!(
                matches!(error, ModelError::MalformedHash { .. }),
                "{bad:?} gave {error:?}"
            );
        }
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let error = reject(|value| value["surprise"] = json!(true));
        assert!(matches!(error, ModelError::Json { .. }), "{error:?}");
    }

    #[test]
    fn a_manifest_needs_weights_and_exactly_one_tokenizer() {
        let only_tokenizer = reject(|value| {
            value["artifacts"] = json!([manifest_value()["artifacts"][1]]);
        });
        assert!(matches!(only_tokenizer, ModelError::InvalidManifest { .. }));

        let no_tokenizer = reject(|value| {
            value["artifacts"] = json!([manifest_value()["artifacts"][0]]);
        });
        assert!(matches!(no_tokenizer, ModelError::InvalidManifest { .. }));

        let two_tokenizers = reject(|value| {
            let mut extra = manifest_value()["artifacts"][1].clone();
            extra["path"] = json!("tokenizer.model");
            value["artifacts"] = json!([
                manifest_value()["artifacts"][0],
                manifest_value()["artifacts"][1],
                extra
            ]);
        });
        assert!(matches!(two_tokenizers, ModelError::InvalidManifest { .. }));
    }

    #[test]
    fn duplicate_artifact_paths_are_rejected() {
        let error = reject(|value| value["artifacts"][1]["path"] = json!("model.safetensors"));
        assert!(
            matches!(error, ModelError::InvalidManifest { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn an_escaping_artifact_path_is_rejected() {
        for bad in ["../outside.bin", "/etc/passwd", "nested/../../escape.bin"] {
            let error = reject(|value| value["artifacts"][0]["path"] = json!(bad));
            assert!(
                matches!(error, ModelError::UnsafeArtifactPath { .. }),
                "{bad:?} gave {error:?}"
            );
        }
    }

    #[test]
    fn incoherent_context_limits_are_rejected() {
        let zero_context = reject(|value| value["context"]["max_context_tokens"] = json!(0));
        assert!(matches!(zero_context, ModelError::InvalidManifest { .. }));

        let zero_output = reject(|value| value["context"]["max_output_tokens"] = json!(0));
        assert!(matches!(zero_output, ModelError::InvalidManifest { .. }));

        let too_long = reject(|value| value["context"]["max_output_tokens"] = json!(9000));
        assert!(matches!(too_long, ModelError::InvalidManifest { .. }));
    }

    #[test]
    fn blank_required_fields_are_rejected() {
        for field in ["id", "repository", "architecture", "license"] {
            let error = reject(|value| value[field] = json!("   "));
            assert!(
                matches!(error, ModelError::EmptyField { field: got, .. } if got == field),
                "{field} gave {error:?}"
            );
        }
        let no_modalities = reject(|value| value["modalities"] = json!([]));
        assert!(matches!(no_modalities, ModelError::EmptyField { .. }));
        let no_quantizations = reject(|value| value["quantizations"] = json!([]));
        assert!(matches!(no_quantizations, ModelError::EmptyField { .. }));
    }

    #[test]
    fn digests_round_trip_through_hex() {
        let hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let digest = Sha256Digest::parse("m", "a", hex).expect("valid digest");
        assert_eq!(digest.to_hex(), hex);
        assert_eq!(digest.as_bytes()[0], 0xba);
        assert_eq!(Sha256Digest::from_bytes(*digest.as_bytes()), digest);
    }

    #[test]
    fn revisions_are_distinguishable() {
        let first = Revision::parse("m", REVISION).expect("valid");
        let second = Revision::parse("m", OTHER_REVISION).expect("valid");
        assert_ne!(first, second);
        assert_eq!(first.to_string(), REVISION);
    }
}
