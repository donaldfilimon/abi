//! Writing WDBX v1 checkpoints and atomically publishing their manifests.
//!
//! Every segment is a full snapshot. A flush writes and syncs the immutable
//! segment first, then atomically replaces the manifest that makes the epoch
//! visible. Readers therefore observe either the previous checkpoint set or the
//! complete new one, never a manifest pointing at a partially written segment.

use crate::format::{
    BlockRecord, CHECKSUM_PREFIX, FormatError, Hash, Result, SEGMENT_HEADER, SpatialRecord,
    StorePaths, TemporalKind, TemporalRecord,
};
use crate::store::Snapshot;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Serialize)]
struct KvLine<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    key: &'a str,
    value: &'a str,
}

#[derive(Serialize)]
struct VectorLine<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    id: u64,
    values: &'a [f32],
}

#[derive(Serialize)]
struct BlockLine<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    hash: Hash,
    prev_hash: Hash,
    timestamp_ms: i64,
    sequence: u64,
    profile: &'a str,
    query_id: u64,
    response_id: u64,
    metadata: &'a str,
}

#[derive(Serialize)]
struct SpatialLine<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    id: u64,
    x: f32,
    y: f32,
    z: f32,
    payload: &'a str,
}

/// Serialize a complete snapshot as a checksummed WDBX v1 segment.
pub fn serialize_snapshot(snapshot: &Snapshot) -> Result<Vec<u8>> {
    validate_snapshot(snapshot)?;

    let mut body = String::new();
    for (key, value) in &snapshot.kv {
        push_json(
            &mut body,
            &KvLine {
                kind: "kv",
                key,
                value,
            },
            "kv",
        )?;
    }
    for (&id, values) in &snapshot.vectors {
        push_json(
            &mut body,
            &VectorLine {
                kind: "vector",
                id,
                values,
            },
            "vector",
        )?;
    }
    for block in &snapshot.blocks {
        push_block(&mut body, block)?;
    }
    for spatial in snapshot.spatial.values() {
        push_spatial(&mut body, spatial)?;
    }
    for temporal in snapshot
        .temporal
        .iter()
        .filter(|record| record.kind == TemporalKind::Node)
        .chain(
            snapshot
                .temporal
                .iter()
                .filter(|record| record.kind == TemporalKind::Edge),
        )
    {
        push_temporal(&mut body, temporal)?;
    }

    let digest = Sha256::digest(body.as_bytes());
    let mut output = String::with_capacity(
        SEGMENT_HEADER.len() + body.len() + CHECKSUM_PREFIX.len() + digest.len() * 2 + 3,
    );
    output.push_str(SEGMENT_HEADER);
    output.push('\n');
    output.push_str(&body);
    output.push_str(CHECKSUM_PREFIX);
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    output.push('\n');
    Ok(output.into_bytes())
}

/// Atomically write one snapshot file without changing a segment manifest.
pub fn write_snapshot(path: impl AsRef<Path>, snapshot: &Snapshot) -> Result<()> {
    let path = path.as_ref();
    let bytes = serialize_snapshot(snapshot)?;
    abi_foundation::io::write_file_atomic(path, bytes).map_err(|error| FormatError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

/// Flush a full snapshot into the next epoch and atomically publish it.
///
/// Returns the new epoch. If publishing the manifest fails, the segment remains
/// an unreferenced orphan and normal readers ignore it.
pub fn flush(paths: &StorePaths, snapshot: &Snapshot) -> Result<u64> {
    let mut manifest = paths.read_manifest()?;
    let epoch = manifest.next_epoch;
    let next_epoch = epoch
        .checked_add(1)
        .ok_or_else(|| FormatError::InvalidManifest {
            reason: "next_epoch overflow".to_string(),
        })?;

    write_snapshot(paths.segment(epoch), snapshot)?;

    manifest.active.push(epoch);
    manifest.active.sort_unstable();
    manifest.active.dedup();
    manifest.next_epoch = next_epoch;
    let manifest_path = paths.manifest();
    abi_foundation::io::write_file_atomic(&manifest_path, manifest.render()).map_err(|error| {
        FormatError::Io {
            path: manifest_path,
            message: error.to_string(),
        }
    })?;
    Ok(epoch)
}

fn validate_snapshot(snapshot: &Snapshot) -> Result<()> {
    snapshot
        .verify_chain()
        .map_err(|error| FormatError::InvalidField {
            record: "block",
            field: "prev_hash",
            reason: error.to_string(),
        })?;

    let mut dimensions = None;
    for values in snapshot.vectors.values() {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FormatError::InvalidField {
                record: "vector",
                field: "values",
                reason: "components must be finite".to_string(),
            });
        }
        match dimensions {
            None => dimensions = Some(values.len()),
            Some(expected) if expected == values.len() => {}
            Some(expected) => {
                return Err(FormatError::InvalidField {
                    record: "vector",
                    field: "values",
                    reason: format!(
                        "dimension mismatch: expected {expected}, found {}",
                        values.len()
                    ),
                });
            }
        }
    }

    for record in snapshot.spatial.values() {
        if !record.x.is_finite() || !record.y.is_finite() || !record.z.is_finite() {
            return Err(FormatError::InvalidField {
                record: "spatial",
                field: "coordinates",
                reason: "coordinates must be finite".to_string(),
            });
        }
    }
    Ok(())
}

fn push_json<T: Serialize>(body: &mut String, value: &T, record: &'static str) -> Result<()> {
    let line = serde_json::to_string(value).map_err(|error| FormatError::InvalidField {
        record,
        field: "<serialize>",
        reason: error.to_string(),
    })?;
    body.push_str(&line);
    body.push('\n');
    Ok(())
}

fn push_block(body: &mut String, block: &BlockRecord) -> Result<()> {
    push_json(
        body,
        &BlockLine {
            kind: "block",
            hash: block.hash,
            prev_hash: block.prev_hash,
            timestamp_ms: block.timestamp_ms,
            sequence: block.sequence,
            profile: &block.profile,
            query_id: block.query_id,
            response_id: block.response_id,
            metadata: &block.metadata,
        },
        "block",
    )
}

fn push_spatial(body: &mut String, record: &SpatialRecord) -> Result<()> {
    push_json(
        body,
        &SpatialLine {
            kind: "spatial",
            id: record.id,
            x: record.x,
            y: record.y,
            z: record.z,
            payload: &record.payload,
        },
        "spatial",
    )
}

fn push_temporal(body: &mut String, record: &TemporalRecord) -> Result<()> {
    let mut fields = record.fields.clone();
    fields.insert(
        "type".to_string(),
        serde_json::Value::String(
            match record.kind {
                TemporalKind::Node => "temporal_node",
                TemporalKind::Edge => "temporal_edge",
            }
            .to_string(),
        ),
    );
    push_json(body, &fields, record_type(record))
}

const fn record_type(record: &TemporalRecord) -> &'static str {
    match record.kind {
        TemporalKind::Node => "temporal_node",
        TemporalKind::Edge => "temporal_edge",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Record, read_segment};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = abi_foundation::temp_path::temp_file_path(name, "store");
            std::fs::create_dir_all(&dir).expect("create store directory");
            Self { dir }
        }

        fn paths(&self) -> StorePaths {
            StorePaths::new(&self.dir)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    fn complete_snapshot() -> Snapshot {
        let mut temporal_node = serde_json::Map::new();
        temporal_node.insert("id".to_string(), serde_json::json!(3));
        temporal_node.insert("timestamp_ms".to_string(), serde_json::json!(99));
        let mut temporal_edge = serde_json::Map::new();
        temporal_edge.insert("cause".to_string(), serde_json::json!(3));
        temporal_edge.insert("effect".to_string(), serde_json::json!(4));

        let timestamp_ms = 1_753_000_000_000;
        let profile = "abi".to_string();
        let metadata = "roundtrip".to_string();
        let block = BlockRecord {
            hash: crate::wal::compute_block_hash(
                Hash::GENESIS,
                timestamp_ms,
                0,
                &profile,
                &metadata,
            ),
            prev_hash: Hash::GENESIS,
            timestamp_ms,
            sequence: 0,
            profile,
            query_id: 1,
            response_id: 2,
            metadata,
        };
        let mut snapshot = Snapshot {
            kv: BTreeMap::from([("key".to_string(), "opaque value".to_string())]),
            vectors: BTreeMap::from([(7, vec![0.25, 0.5, 0.75])]),
            blocks: vec![block],
            spatial: BTreeMap::from([(
                5,
                SpatialRecord {
                    id: 5,
                    x: 1.0,
                    y: -2.0,
                    z: 3.5,
                    payload: "point".to_string(),
                },
            )]),
            temporal: vec![
                TemporalRecord {
                    kind: TemporalKind::Edge,
                    fields: temporal_edge,
                },
                TemporalRecord {
                    kind: TemporalKind::Node,
                    fields: temporal_node,
                },
            ],
            stats: crate::store::SnapshotStats::default(),
        };
        snapshot.recount();
        snapshot
    }

    #[test]
    fn snapshot_round_trips_all_record_types() {
        let fixture = Fixture::new("abi_wdbx_write_roundtrip");
        let original = complete_snapshot();

        assert_eq!(flush(&fixture.paths(), &original).expect("flush"), 0);
        let loaded = crate::store::load(&fixture.paths()).expect("load");
        assert_eq!(loaded.kv, original.kv);
        assert_eq!(loaded.vectors, original.vectors);
        assert_eq!(loaded.blocks, original.blocks);
        assert_eq!(loaded.spatial, original.spatial);
        assert_eq!(loaded.temporal.len(), original.temporal.len());
        assert_eq!(loaded.stats.epochs_loaded, 1);
        loaded.verify_chain().expect("chain remains valid");
    }

    #[test]
    fn a_second_flush_publishes_a_new_full_checkpoint() {
        let fixture = Fixture::new("abi_wdbx_write_epochs");
        let first = complete_snapshot();
        flush(&fixture.paths(), &first).expect("first");

        let mut second = Snapshot::new();
        second
            .kv
            .insert("replacement".to_string(), "yes".to_string());
        second.recount();
        assert_eq!(flush(&fixture.paths(), &second).expect("second"), 1);

        let manifest = fixture.paths().read_manifest().expect("manifest");
        assert_eq!(manifest.next_epoch, 2);
        assert_eq!(manifest.active, [0, 1]);
        let loaded = crate::store::load(&fixture.paths()).expect("load latest");
        assert_eq!(loaded.kv, second.kv);
        assert!(!loaded.kv.contains_key("key"));
    }

    #[test]
    fn checksum_detects_body_tampering() {
        let fixture = Fixture::new("abi_wdbx_checksum");
        let snapshot = complete_snapshot();
        write_snapshot(fixture.paths().segment(0), &snapshot).expect("write");

        let path = fixture.paths().segment(0);
        let content = std::fs::read_to_string(&path).expect("read");
        let tampered = content.replacen("opaque value", "altered value", 1);
        std::fs::write(&path, tampered).expect("tamper");

        let error = read_segment(0, &path).expect_err("checksum must fail");
        assert!(matches!(error, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn writer_rejects_mixed_dimensions_before_creating_files() {
        let fixture = Fixture::new("abi_wdbx_bad_dimensions");
        let mut snapshot = Snapshot::new();
        snapshot.vectors.insert(1, vec![1.0, 2.0]);
        snapshot.vectors.insert(2, vec![1.0, 2.0, 3.0]);

        let error = flush(&fixture.paths(), &snapshot).expect_err("must reject");
        assert!(matches!(
            error,
            FormatError::InvalidField {
                record: "vector",
                field: "values",
                ..
            }
        ));
        assert!(!fixture.paths().manifest().exists());
        assert!(!fixture.paths().segment(0).exists());
    }

    #[test]
    fn serialized_records_reparse_individually() {
        let bytes = serialize_snapshot(&complete_snapshot()).expect("serialize");
        let text = std::str::from_utf8(&bytes).expect("utf8");
        for line in text.lines().skip(1) {
            if line.starts_with('#') {
                continue;
            }
            Record::parse(line).expect("writer only emits parseable records");
        }
    }

    #[test]
    fn writer_rejects_an_invalid_audit_chain() {
        let fixture = Fixture::new("abi_wdbx_bad_chain");
        let mut snapshot = Snapshot::new();
        snapshot.blocks.push(BlockRecord {
            hash: Hash([1; 32]),
            prev_hash: Hash([9; 32]),
            timestamp_ms: 0,
            sequence: 0,
            profile: "abi".to_string(),
            query_id: 0,
            response_id: 0,
            metadata: String::new(),
        });

        let error = flush(&fixture.paths(), &snapshot).expect_err("must reject");
        assert!(matches!(
            error,
            FormatError::InvalidField {
                record: "block",
                field: "prev_hash",
                ..
            }
        ));
        assert!(!fixture.paths().manifest().exists());
    }

    #[test]
    fn writer_rejects_non_finite_spatial_coordinates() {
        let fixture = Fixture::new("abi_wdbx_bad_spatial");
        let mut snapshot = Snapshot::new();
        snapshot.spatial.insert(
            1,
            SpatialRecord {
                id: 1,
                x: f32::NAN,
                y: 0.0,
                z: 0.0,
                payload: String::new(),
            },
        );

        let error = flush(&fixture.paths(), &snapshot).expect_err("must reject");
        assert!(matches!(
            error,
            FormatError::InvalidField {
                record: "spatial",
                field: "coordinates",
                ..
            }
        ));
    }
}
