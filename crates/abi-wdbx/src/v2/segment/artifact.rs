//! Independently authenticated, content-addressed codec artifacts.

use super::codec::{CodecArtifactPayload, SegmentCodecKind};
use crate::codecs::CodecMetrics;
use crate::v2::{ObjectKind, ObjectSecurity, V2Error, open_object, seal_object};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};

const ARTIFACT_CONTAINER_VERSION: u8 = 1;
const MAX_ARTIFACT_OBJECT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ArtifactReference {
    artifact_format_version: u8,
    pub(super) object_id: String,
    pub(super) plaintext_sha256: String,
    encrypted: bool,
    signed: bool,
    pub(super) metrics: CodecMetrics,
}

impl ArtifactReference {
    pub(super) fn validate(&self, kind: SegmentCodecKind) -> Result<(), V2Error> {
        let expected_id = artifact_object_id(kind, &self.plaintext_sha256);
        if self.artifact_format_version != ARTIFACT_CONTAINER_VERSION
            || self.object_id != expected_id
            || !valid_digest(&self.plaintext_sha256)
            || !metrics_valid(&self.metrics)
        {
            return Err(V2Error::InvalidMutation(
                "invalid segment codec artifact reference".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ArtifactContainer {
    format_version: u8,
    payload: CodecArtifactPayload,
}

pub(super) fn write_artifact(
    root: &Path,
    payload: &CodecArtifactPayload,
    security: &ObjectSecurity,
) -> Result<ArtifactReference, V2Error> {
    payload
        .validate()
        .map_err(|error| invalid_artifact(root, error))?;
    let container = ArtifactContainer {
        format_version: ARTIFACT_CONTAINER_VERSION,
        payload: payload.clone(),
    };
    let plaintext =
        serde_json::to_vec(&container).map_err(|error| invalid_artifact(root, error))?;
    let digest = digest(&plaintext);
    let object_id = artifact_object_id(payload.kind(), &digest);
    let directory = root.join("artifacts");
    std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, &error))?;
    let path = directory.join(format!("{object_id}.object"));
    // Seal a candidate even when this content-addressed identity already
    // exists. Its opened flags describe the policy requested by this write.
    // Reusing a plaintext artifact for a newly encrypted segment (or the
    // reverse) would otherwise publish a segment that can never reopen.
    let candidate = seal_object(ObjectKind::Artifact, &object_id, &plaintext, security)
        .map_err(|error| security_error(&path, error))?;
    let desired =
        open_object(&candidate, security, false).map_err(|error| security_error(&path, error))?;
    if !path.exists() {
        super::super::atomic_write(&path, &candidate)?;
    }
    let opened = open_artifact_object(&path, security, false)?;
    if opened.object_id != object_id
        || opened.plaintext != plaintext
        || opened.kind != ObjectKind::Artifact
        || opened.encrypted != desired.encrypted
        || opened.signed != desired.signed
    {
        return Err(invalid_artifact(
            &path,
            "published codec artifact verification mismatch",
        ));
    }
    let reference = ArtifactReference {
        artifact_format_version: ARTIFACT_CONTAINER_VERSION,
        object_id,
        plaintext_sha256: digest,
        encrypted: opened.encrypted,
        signed: opened.signed,
        metrics: payload.metrics().clone(),
    };
    reference.validate(payload.kind())?;
    Ok(reference)
}

pub(super) fn read_artifact(
    segment_path: &Path,
    reference: &ArtifactReference,
    security: &ObjectSecurity,
    require_signature: bool,
    segment_encrypted: bool,
    segment_signed: bool,
) -> Result<CodecArtifactPayload, V2Error> {
    let kind = kind_from_object_id(&reference.object_id)
        .ok_or_else(|| invalid_artifact(segment_path, "unknown codec artifact identity"))?;
    reference.validate(kind)?;
    let root = segment_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid_artifact(segment_path, "segment has no generation root"))?;
    let path = root
        .join("artifacts")
        .join(format!("{}.object", reference.object_id));
    let opened = open_artifact_object(&path, security, require_signature)?;
    if opened.kind != ObjectKind::Artifact
        || opened.object_id != reference.object_id
        || opened.encrypted != reference.encrypted
        || opened.signed != reference.signed
        || opened.encrypted != segment_encrypted
        || opened.signed != segment_signed
        || digest(&opened.plaintext) != reference.plaintext_sha256
    {
        return Err(invalid_artifact(
            &path,
            "codec artifact identity, digest, or security policy mismatch",
        ));
    }
    let container: ArtifactContainer = serde_json::from_slice(&opened.plaintext)
        .map_err(|error| invalid_artifact(&path, error))?;
    if container.format_version != ARTIFACT_CONTAINER_VERSION
        || container.payload.kind() != kind
        || container.payload.metrics() != &reference.metrics
    {
        return Err(invalid_artifact(
            &path,
            "codec artifact version, kind, or metrics mismatch",
        ));
    }
    container
        .payload
        .validate()
        .map_err(|error| invalid_artifact(&path, error))?;
    Ok(container.payload)
}

fn open_artifact_object(
    path: &Path,
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<crate::v2::OpenedObject, V2Error> {
    let encoded = std::fs::read(path).map_err(|error| io_error(path, &error))?;
    if encoded.len() > MAX_ARTIFACT_OBJECT_BYTES {
        return Err(invalid_artifact(path, "codec artifact exceeds size bound"));
    }
    open_object(&encoded, security, require_signature).map_err(|error| security_error(path, error))
}

fn artifact_object_id(kind: SegmentCodecKind, digest: &str) -> String {
    format!("artifact-{}-v1-{digest}", kind.label())
}

fn kind_from_object_id(object_id: &str) -> Option<SegmentCodecKind> {
    if object_id.starts_with("artifact-pq-v1-") {
        Some(SegmentCodecKind::Pq)
    } else if object_id.starts_with("artifact-autoencoder-v1-") {
        Some(SegmentCodecKind::Autoencoder)
    } else {
        None
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn metrics_valid(metrics: &CodecMetrics) -> bool {
    metrics.reconstruction_mse.is_finite()
        && metrics.reconstruction_mse >= 0.0
        && metrics.compression_ratio.is_finite()
        && metrics.compression_ratio > 0.0
        && metrics.recall_at_k.is_finite()
        && (0.0..=1.0).contains(&metrics.recall_at_k)
        && metrics.recall_k > 0
}

fn invalid_artifact(path: &Path, error: impl std::fmt::Display) -> V2Error {
    V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line: 0,
        reason: format!("invalid codec artifact: {error}"),
    }
}

fn security_error(path: &Path, error: impl std::fmt::Display) -> V2Error {
    V2Error::Security {
        path: path.to_path_buf(),
        reason: error.to_string(),
    }
}

fn io_error(path: &Path, error: &std::io::Error) -> V2Error {
    V2Error::Io {
        path: PathBuf::from(path),
        message: error.to_string(),
    }
}
