//! Immutable authenticated v2 compaction segments.

use super::{
    CausalHeads, ObjectKind, ObjectSecurity, V2Error, V2Snapshot, atomic_write, open_object,
    seal_object,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const SEGMENT_FORMAT_VERSION: u8 = 1;

/// Result of publishing one immutable compaction segment.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SegmentImage {
    format_version: u8,
    covered_heads: CausalHeads,
    snapshot: V2Snapshot,
}

pub(super) fn write_segment(
    root: &Path,
    snapshot: &V2Snapshot,
    security: &ObjectSecurity,
) -> Result<CompactionReport, V2Error> {
    let directory = root.join("segments");
    std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, &error))?;
    let object_id = format!("segment-{}", Uuid::new_v4());
    let path = directory.join(format!("{object_id}.object"));
    let image = SegmentImage {
        format_version: SEGMENT_FORMAT_VERSION,
        covered_heads: snapshot.heads.clone(),
        snapshot: snapshot.clone(),
    };
    let plaintext = serde_json::to_vec(&image).map_err(|error| corrupt(&path, error))?;
    let encoded = seal_object(ObjectKind::Segment, &object_id, &plaintext, security)
        .map_err(|error| security_error(&path, error))?;
    atomic_write(&path, &encoded)?;

    // Publication succeeds only after reopening the exact bytes through the
    // authentication path and comparing the canonical decoded image.
    let verified = read_segment(&path, security, false)?;
    if verified.object_id != object_id || verified.image != image {
        return Err(V2Error::CorruptJournal {
            path,
            line: 0,
            reason: "published segment verification mismatch".into(),
        });
    }
    Ok(CompactionReport {
        path,
        object_id,
        covered_heads: image.covered_heads,
        encrypted: verified.encrypted,
        signed: verified.signed,
    })
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
            candidates.iter().all(|other| {
                heads_cover(&candidate.image.covered_heads, &other.image.covered_heads)
            })
        })
        .max_by(|left, right| left.object_id.cmp(&right.object_id))
    else {
        // Incomparable concurrent compactions are safe but neither alone proves
        // coverage of the other. Replaying all journals is conservative.
        return Ok(None);
    };
    Ok(Some(selected.image.snapshot.clone()))
}

struct DecodedSegment {
    object_id: String,
    image: SegmentImage,
    encrypted: bool,
    signed: bool,
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
        return Err(V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
            reason: "immutable object kind is not segment".into(),
        });
    }
    let expected_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
            reason: "segment filename is not valid UTF-8".into(),
        })?;
    if opened.object_id != expected_id {
        return Err(V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
            reason: "segment identity does not match its filename".into(),
        });
    }
    let image: SegmentImage =
        serde_json::from_slice(&opened.plaintext).map_err(|error| corrupt(path, error))?;
    if image.format_version != SEGMENT_FORMAT_VERSION || image.covered_heads != image.snapshot.heads
    {
        return Err(V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
            reason: "segment version or causal coverage mismatch".into(),
        });
    }
    Ok(DecodedSegment {
        object_id: opened.object_id,
        image,
        encrypted: opened.encrypted,
        signed: opened.signed,
    })
}

fn heads_cover(candidate: &CausalHeads, other: &CausalHeads) -> bool {
    other.iter().all(|(writer, sequence)| {
        candidate
            .get(writer)
            .is_some_and(|candidate_sequence| candidate_sequence >= sequence)
    })
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
mod tests {
    use super::*;

    #[test]
    fn incomparable_segments_fall_back_until_one_frontier_covers_both() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-segments", "store");
        let security = ObjectSecurity::plaintext();
        let left_writer = Uuid::new_v4();
        let right_writer = Uuid::new_v4();
        let mut left = V2Snapshot::default();
        left.heads.insert(left_writer, 1);
        let mut right = V2Snapshot::default();
        right.heads.insert(right_writer, 1);
        write_segment(&root, &left, &security).unwrap();
        write_segment(&root, &right, &security).unwrap();
        assert!(load_segment(&root, &security, false).unwrap().is_none());

        let mut covering = V2Snapshot::default();
        covering.heads.insert(left_writer, 1);
        covering.heads.insert(right_writer, 1);
        write_segment(&root, &covering, &security).unwrap();
        assert_eq!(
            load_segment(&root, &security, false).unwrap().unwrap(),
            covering
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
