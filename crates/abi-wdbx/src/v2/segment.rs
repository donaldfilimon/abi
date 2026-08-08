//! Immutable authenticated v2 compaction segments.
//!
//! Format 1 is the original exact `V2Snapshot` JSON image and remains readable
//! byte-for-byte. Format 2 separates vector values from their causal metadata
//! and may reference one independently authenticated learned-codec artifact.

mod artifact;
mod codec;

use self::artifact::{read_artifact, write_artifact};
use self::codec::{EncodedSnapshotV2, SegmentCodecDescriptor, decode_snapshot, prepare_snapshot};
pub use self::codec::{SegmentCodecKind, SegmentCodecPolicy};
use super::{
    CausalHeads, ObjectKind, ObjectSecurity, V2Error, V2Snapshot, atomic_write, open_object,
    seal_object,
};
use crate::codecs::CodecMetrics;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const LEGACY_SEGMENT_FORMAT_VERSION: u8 = 1;
const CODEC_SEGMENT_FORMAT_VERSION: u8 = 2;

/// Result of publishing one immutable compaction segment.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionReport {
    /// Published immutable object path.
    pub path: PathBuf,
    /// Authenticated object identity.
    pub object_id: String,
    /// Causal writer frontier covered by the snapshot.
    pub covered_heads: CausalHeads,
    /// Whether the segment payload is encrypted.
    pub encrypted: bool,
    /// Whether the segment header is signed.
    pub signed: bool,
    /// Vector representation declared by the segment.
    pub codec: SegmentCodecKind,
    /// Content-addressed learned artifact, when the codec requires one.
    pub artifact_id: Option<String>,
    /// Training quality evidence stored with the artifact.
    pub codec_metrics: Option<CodecMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SegmentImageV1 {
    format_version: u8,
    covered_heads: CausalHeads,
    snapshot: V2Snapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SegmentImageV2 {
    format_version: u8,
    covered_heads: CausalHeads,
    codec: SegmentCodecDescriptor,
    snapshot: EncodedSnapshotV2,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct SegmentVersionProbe {
    format_version: u8,
}

/// Preserve the original exact format-1 writer used by ordinary compaction.
pub(super) fn write_segment(
    root: &Path,
    snapshot: &V2Snapshot,
    security: &ObjectSecurity,
) -> Result<CompactionReport, V2Error> {
    let image = SegmentImageV1 {
        format_version: LEGACY_SEGMENT_FORMAT_VERSION,
        covered_heads: snapshot.heads.clone(),
        snapshot: snapshot.clone(),
    };
    let plaintext =
        serde_json::to_vec(&image).map_err(|error| corrupt(&root.join("segments"), error))?;
    let verified = publish_segment(root, &plaintext, security)?;
    if verified.snapshot != *snapshot
        || verified.covered_heads != image.covered_heads
        || verified.codec.kind != SegmentCodecKind::None
    {
        remove_failed_publication(&verified.path);
        return Err(corrupt(
            &verified.path,
            "published exact segment verification mismatch",
        ));
    }
    Ok(report_from_verified(verified))
}

/// Publish a format-2 segment using an explicit representation policy.
///
/// `None` is exact. PQ and autoencoder policies are intentionally lossy and
/// must be selected explicitly by a caller; ordinary compaction never invokes
/// this function.
pub(super) fn write_segment_with_codec(
    root: &Path,
    snapshot: &V2Snapshot,
    security: &ObjectSecurity,
    policy: SegmentCodecPolicy,
) -> Result<CompactionReport, V2Error> {
    let prepared = prepare_snapshot(snapshot, policy)
        .map_err(|error| codec_write_error("prepare segment", error))?;
    let artifact = prepared
        .artifact
        .as_ref()
        .map(|payload| write_artifact(root, payload, security))
        .transpose()?;
    let descriptor = SegmentCodecDescriptor::new(prepared.kind, artifact)?;
    let expected_metrics = descriptor.metrics().cloned();
    let image = SegmentImageV2 {
        format_version: CODEC_SEGMENT_FORMAT_VERSION,
        covered_heads: snapshot.heads.clone(),
        codec: descriptor,
        snapshot: prepared.snapshot,
    };
    let plaintext =
        serde_json::to_vec(&image).map_err(|error| corrupt(&root.join("segments"), error))?;
    let verified = publish_segment(root, &plaintext, security)?;
    if verified.covered_heads != image.covered_heads
        || verified.codec.kind != image.codec.kind
        || verified.codec_metrics != expected_metrics
    {
        remove_failed_publication(&verified.path);
        return Err(corrupt(
            &verified.path,
            "published codec segment verification mismatch",
        ));
    }
    Ok(report_from_verified(verified))
}

pub(super) fn load_segment(
    root: &Path,
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<Option<V2Snapshot>, V2Error> {
    let directory = root.join("segments");
    if !directory.is_dir() {
        return Ok(None);
    }
    let mut paths: Vec<_> = std::fs::read_dir(&directory)
        .map_err(|error| io_error(&directory, &error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "object")
        })
        .collect();
    paths.sort();
    let mut candidates = Vec::with_capacity(paths.len());
    for path in paths {
        candidates.push(read_segment(&path, security, require_signature)?);
    }
    let Some(selected) = candidates
        .iter()
        .filter(|candidate| {
            candidates
                .iter()
                .all(|other| heads_cover(&candidate.covered_heads, &other.covered_heads))
        })
        .max_by(|left, right| left.object_id.cmp(&right.object_id))
    else {
        // Incomparable concurrent compactions are safe but neither alone proves
        // coverage of the other. Replaying all journals is conservative.
        return Ok(None);
    };
    Ok(Some(selected.snapshot.clone()))
}

struct DecodedSegment {
    path: PathBuf,
    object_id: String,
    covered_heads: CausalHeads,
    snapshot: V2Snapshot,
    codec: SegmentCodecDescriptor,
    codec_metrics: Option<CodecMetrics>,
    encrypted: bool,
    signed: bool,
}

fn publish_segment(
    root: &Path,
    plaintext: &[u8],
    security: &ObjectSecurity,
) -> Result<DecodedSegment, V2Error> {
    let directory = root.join("segments");
    std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, &error))?;
    let object_id = format!("segment-{}", Uuid::new_v4());
    let path = directory.join(format!("{object_id}.object"));
    let encoded = seal_object(ObjectKind::Segment, &object_id, plaintext, security)
        .map_err(|error| security_error(&path, error))?;
    atomic_write(&path, &encoded)?;
    let verified = match read_segment(&path, security, false) {
        Ok(verified) => verified,
        Err(error) => {
            remove_failed_publication(&path);
            return Err(error);
        }
    };
    if verified.object_id != object_id {
        remove_failed_publication(&path);
        return Err(corrupt(&path, "published segment identity mismatch"));
    }
    Ok(verified)
}

fn read_segment(
    path: &Path,
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<DecodedSegment, V2Error> {
    let encoded = std::fs::read(path).map_err(|error| io_error(path, &error))?;
    let opened = open_object(&encoded, security, require_signature)
        .map_err(|error| security_error(path, error))?;
    if opened.kind != ObjectKind::Segment {
        return Err(corrupt(path, "immutable object kind is not segment"));
    }
    let expected_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| corrupt(path, "segment filename is not valid UTF-8"))?;
    if opened.object_id != expected_id {
        return Err(corrupt(
            path,
            "segment identity does not match its filename",
        ));
    }
    let probe: SegmentVersionProbe =
        serde_json::from_slice(&opened.plaintext).map_err(|error| corrupt(path, error))?;
    match probe.format_version {
        LEGACY_SEGMENT_FORMAT_VERSION => {
            let image: SegmentImageV1 =
                serde_json::from_slice(&opened.plaintext).map_err(|error| corrupt(path, error))?;
            if image.covered_heads != image.snapshot.heads {
                return Err(corrupt(path, "segment causal coverage mismatch"));
            }
            Ok(DecodedSegment {
                path: path.to_path_buf(),
                object_id: opened.object_id,
                covered_heads: image.covered_heads,
                snapshot: image.snapshot,
                codec: SegmentCodecDescriptor::none(),
                codec_metrics: None,
                encrypted: opened.encrypted,
                signed: opened.signed,
            })
        }
        CODEC_SEGMENT_FORMAT_VERSION => {
            let image: SegmentImageV2 =
                serde_json::from_slice(&opened.plaintext).map_err(|error| corrupt(path, error))?;
            image.codec.validate()?;
            let payload = image
                .codec
                .artifact
                .as_ref()
                .map(|reference| {
                    read_artifact(
                        path,
                        reference,
                        security,
                        require_signature,
                        opened.encrypted,
                        opened.signed,
                    )
                })
                .transpose()?;
            let snapshot = decode_snapshot(image.snapshot, &image.codec, payload.as_ref())
                .map_err(|error| corrupt(path, error))?;
            if image.covered_heads != snapshot.heads {
                return Err(corrupt(path, "segment causal coverage mismatch"));
            }
            let metrics = image.codec.metrics().cloned();
            Ok(DecodedSegment {
                path: path.to_path_buf(),
                object_id: opened.object_id,
                covered_heads: image.covered_heads,
                snapshot,
                codec: image.codec,
                codec_metrics: metrics,
                encrypted: opened.encrypted,
                signed: opened.signed,
            })
        }
        version => Err(corrupt(
            path,
            format!("unsupported immutable segment format version {version}"),
        )),
    }
}

fn report_from_verified(verified: DecodedSegment) -> CompactionReport {
    CompactionReport {
        path: verified.path,
        object_id: verified.object_id,
        covered_heads: verified.covered_heads,
        encrypted: verified.encrypted,
        signed: verified.signed,
        codec: verified.codec.kind,
        artifact_id: verified
            .codec
            .artifact
            .as_ref()
            .map(|artifact| artifact.object_id.clone()),
        codec_metrics: verified.codec_metrics,
    }
}

fn heads_cover(candidate: &CausalHeads, other: &CausalHeads) -> bool {
    other.iter().all(|(writer, sequence)| {
        candidate
            .get(writer)
            .is_some_and(|candidate_sequence| candidate_sequence >= sequence)
    })
}

fn codec_write_error(action: &str, error: impl std::fmt::Display) -> V2Error {
    V2Error::InvalidMutation(format!("{action}: {error}"))
}

fn remove_failed_publication(path: &Path) {
    // The path contains a fresh random segment UUID allocated by this call, so
    // bounded cleanup cannot remove a pre-existing object. Artifact objects
    // remain immutable and may be conservatively retained as harmless orphans.
    let _ = std::fs::remove_file(path);
}

fn corrupt(path: &Path, error: impl std::fmt::Display) -> V2Error {
    V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line: 0,
        reason: error.to_string(),
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
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
#[path = "segment/tests.rs"]
mod tests;
