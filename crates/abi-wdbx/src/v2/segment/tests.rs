//! Segment compatibility, codec, and artifact tests.

use super::*;
use crate::codecs::{AutoencoderConfigV1, PqConfigV1};
use crate::v2::{
    RecordId, V2AuditBlock, V2SpatialRecord, V2TemporalKind, V2TemporalRecord, Version,
    generate_key_material,
};
use serde_json::json;
use std::collections::BTreeMap;

fn scratch(label: &str) -> PathBuf {
    abi_foundation::temp_path::temp_file_path(label, "store")
}

#[allow(clippy::cast_precision_loss)]
fn pq_values() -> Vec<Vec<f32>> {
    (0..32)
        .map(|index| {
            let group = (index / 4) as f32;
            let offset = (index % 4) as f32 * 0.015;
            vec![
                group * 0.3 + offset,
                group.sin() + offset,
                group.cos() - offset,
                group * -0.12,
                (group * 0.7).sin(),
                (group * 0.4).cos(),
                offset - group * 0.05,
            ]
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn autoencoder_values() -> Vec<Vec<f32>> {
    (0..24)
        .map(|index| {
            let first = (index as f32 - 12.0) / 12.0;
            let second = ((index * 7) % 23) as f32 / 11.5 - 1.0;
            vec![
                first,
                second,
                first * 0.7 + second * 0.2,
                first * -0.3 + second * 0.8,
                first * first * 0.2,
                second * second * -0.15,
            ]
        })
        .collect()
}

fn pq_config(dimensions: usize) -> PqConfigV1 {
    PqConfigV1 {
        dimensions,
        subspaces: 3,
        centroids: 8,
        iterations: 20,
        seed: 0xD15C_A11E,
        recall_k: 3,
    }
}

fn autoencoder_config() -> AutoencoderConfigV1 {
    AutoencoderConfigV1 {
        input_dimensions: 6,
        latent_dimensions: 3,
        seed: 0xA070_EC0D,
        epochs: 600,
        learning_rate: 0.08,
        recall_k: 3,
    }
}

fn snapshot_with_vectors(values: &[Vec<f32>], head_sequence: u64) -> V2Snapshot {
    let writer = Uuid::from_u128(0xAB1);
    let mut snapshot = V2Snapshot::default();
    snapshot.heads.insert(writer, head_sequence);
    for (index, value) in values.iter().enumerate() {
        let id = RecordId::Legacy(u64::try_from(index + 1).unwrap());
        snapshot.vectors.insert(
            id,
            vec![Version {
                version_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                writer_id: writer,
                sequence: u64::try_from(index + 1).unwrap(),
                observed_heads: CausalHeads::new(),
                value: value.clone(),
            }],
        );
    }
    snapshot.kv.insert(
        "codec:metadata".into(),
        vec![Version {
            version_id: Uuid::from_u128(0xF001),
            writer_id: writer,
            sequence: 1,
            observed_heads: CausalHeads::new(),
            value: "preserved".into(),
        }],
    );
    let spatial = V2SpatialRecord {
        id: RecordId::Legacy(9001),
        x: 1.0,
        y: 2.0,
        z: 3.0,
        payload: "spatial".into(),
    };
    snapshot.spatial.insert(
        spatial.id,
        vec![Version {
            version_id: Uuid::from_u128(0xF002),
            writer_id: writer,
            sequence: 1,
            observed_heads: CausalHeads::new(),
            value: spatial,
        }],
    );
    let mut fields = serde_json::Map::new();
    fields.insert("label".into(), json!("temporal"));
    snapshot.temporal.insert(
        "node:codec".into(),
        vec![Version {
            version_id: Uuid::from_u128(0xF003),
            writer_id: writer,
            sequence: 1,
            observed_heads: CausalHeads::new(),
            value: V2TemporalRecord {
                kind: V2TemporalKind::Node,
                fields,
            },
        }],
    );
    let hash = "a".repeat(64);
    snapshot.audit.insert(
        hash.clone(),
        V2AuditBlock {
            hash: hash.clone(),
            parents: Vec::new(),
            timestamp_ms: 1,
            sequence: 1,
            profile: "abbey".into(),
            query_id: RecordId::Legacy(1),
            response_id: RecordId::Legacy(2),
            metadata: "audit".into(),
        },
    );
    snapshot.audit_order.push(hash);
    snapshot.committed_transactions = values.len();
    snapshot
}

fn artifact_path(root: &Path, report: &CompactionReport) -> PathBuf {
    root.join("artifacts").join(format!(
        "{}.object",
        report.artifact_id.as_deref().expect("codec artifact")
    ))
}

fn assert_non_vector_state_eq(left: &V2Snapshot, right: &V2Snapshot) {
    assert_eq!(left.heads, right.heads);
    assert_eq!(left.kv, right.kv);
    assert_eq!(left.spatial, right.spatial);
    assert_eq!(left.temporal, right.temporal);
    assert_eq!(left.audit, right.audit);
    assert_eq!(left.audit_order, right.audit_order);
    assert_eq!(left.committed_transactions, right.committed_transactions);
}

fn assert_vector_metadata_eq(left: &V2Snapshot, right: &V2Snapshot) {
    assert_eq!(
        left.vectors.keys().collect::<Vec<_>>(),
        right.vectors.keys().collect::<Vec<_>>()
    );
    for (id, left_versions) in &left.vectors {
        let right_versions = &right.vectors[id];
        assert_eq!(left_versions.len(), right_versions.len());
        for (left_version, right_version) in left_versions.iter().zip(right_versions) {
            assert_eq!(left_version.version_id, right_version.version_id);
            assert_eq!(left_version.writer_id, right_version.writer_id);
            assert_eq!(left_version.sequence, right_version.sequence);
            assert_eq!(left_version.observed_heads, right_version.observed_heads);
        }
    }
}

#[test]
fn legacy_exact_segment_remains_readable_without_format_drift() {
    let root = scratch("wdbx-v2-segment-legacy");
    let snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let report = write_segment(&root, &snapshot, &ObjectSecurity::plaintext()).unwrap();
    assert_eq!(report.codec, SegmentCodecKind::None);
    assert!(report.artifact_id.is_none());
    let encoded = std::fs::read(&report.path).unwrap();
    let opened = open_object(&encoded, &ObjectSecurity::plaintext(), false).unwrap();
    let image: SegmentImageV1 = serde_json::from_slice(&opened.plaintext).unwrap();
    assert_eq!(image.format_version, LEGACY_SEGMENT_FORMAT_VERSION);
    assert_eq!(image.snapshot, snapshot);
    assert_eq!(
        load_segment(&root, &ObjectSecurity::plaintext(), false).unwrap(),
        Some(snapshot)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn format_two_none_is_exact_and_declares_no_artifact() {
    let root = scratch("wdbx-v2-segment-none");
    let snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let report = write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::None,
    )
    .unwrap();
    assert_eq!(report.codec, SegmentCodecKind::None);
    assert!(report.artifact_id.is_none());
    assert!(report.codec_metrics.is_none());
    assert_eq!(
        load_segment(&root, &ObjectSecurity::plaintext(), false).unwrap(),
        Some(snapshot)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn pq_segment_uses_fixture_quality_and_preserves_all_metadata() {
    let root = scratch("wdbx-v2-segment-pq");
    let snapshot = snapshot_with_vectors(&pq_values(), 1);
    let report = write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(7)),
    )
    .unwrap();
    assert_eq!(report.codec, SegmentCodecKind::Pq);
    let metrics = report.codec_metrics.as_ref().unwrap();
    assert!(metrics.reconstruction_mse <= 0.007);
    assert!(metrics.recall_at_k >= 0.99);
    assert!(metrics.compression_ratio > 1.0);
    assert!(artifact_path(&root, &report).is_file());
    let recovered = load_segment(&root, &ObjectSecurity::plaintext(), false)
        .unwrap()
        .unwrap();
    assert_non_vector_state_eq(&snapshot, &recovered);
    assert_vector_metadata_eq(&snapshot, &recovered);
    assert_eq!(snapshot.vectors.len(), recovered.vectors.len());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn autoencoder_segment_uses_fixture_quality_and_reopens() {
    let root = scratch("wdbx-v2-segment-autoencoder");
    let snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let report = write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Autoencoder(autoencoder_config()),
    )
    .unwrap();
    assert_eq!(report.codec, SegmentCodecKind::Autoencoder);
    let metrics = report.codec_metrics.as_ref().unwrap();
    assert!(metrics.reconstruction_mse <= 0.0011);
    assert!(metrics.recall_at_k >= 0.87);
    let recovered = read_segment(&report.path, &ObjectSecurity::plaintext(), false)
        .unwrap()
        .snapshot;
    assert_non_vector_state_eq(&snapshot, &recovered);
    assert_vector_metadata_eq(&snapshot, &recovered);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sequential_mixed_generations_select_the_covering_codec_segment() {
    let root = scratch("wdbx-v2-segment-mixed");
    let mut snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    write_segment(&root, &snapshot, &ObjectSecurity::plaintext()).unwrap();
    let writer = *snapshot.heads.keys().next().unwrap();
    snapshot.heads.insert(writer, 2);
    write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    snapshot.heads.insert(writer, 3);
    let latest = write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Autoencoder(autoencoder_config()),
    )
    .unwrap();
    let expected = read_segment(&latest.path, &ObjectSecurity::plaintext(), false)
        .unwrap()
        .snapshot;
    let recovered = load_segment(&root, &ObjectSecurity::plaintext(), false)
        .unwrap()
        .unwrap();
    assert_eq!(recovered, expected);
    assert_eq!(recovered.heads.get(&writer), Some(&3));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_vector_versions_are_all_encoded_without_collapsing_conflicts() {
    let root = scratch("wdbx-v2-segment-conflicts");
    let mut snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let concurrent_writer = Uuid::from_u128(0xC011_F11C7);
    snapshot
        .vectors
        .get_mut(&RecordId::Legacy(1))
        .unwrap()
        .push(Version {
            version_id: Uuid::from_u128(0xC0FF_11C7),
            writer_id: concurrent_writer,
            sequence: 1,
            observed_heads: BTreeMap::new(),
            value: vec![0.2; 6],
        });
    snapshot.heads.insert(concurrent_writer, 1);
    let report = write_segment_with_codec(
        &root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    let recovered = read_segment(&report.path, &ObjectSecurity::plaintext(), false)
        .unwrap()
        .snapshot;
    assert_vector_metadata_eq(&snapshot, &recovered);
    assert_eq!(recovered.vectors[&RecordId::Legacy(1)].len(), 2);
    assert_eq!(
        recovered
            .get_vector(RecordId::Legacy(1))
            .unwrap()
            .conflicts
            .len(),
        1
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn missing_wrong_and_digest_mismatched_artifacts_fail_closed() {
    let values = autoencoder_values();
    let snapshot = snapshot_with_vectors(&values, 1);

    let missing_root = scratch("wdbx-v2-segment-artifact-missing");
    let missing = write_segment_with_codec(
        &missing_root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    std::fs::remove_file(artifact_path(&missing_root, &missing)).unwrap();
    assert!(load_segment(&missing_root, &ObjectSecurity::plaintext(), false).is_err());
    std::fs::remove_dir_all(missing_root).unwrap();

    let wrong_root = scratch("wdbx-v2-segment-artifact-kind");
    let wrong = write_segment_with_codec(
        &wrong_root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    let wrong_id = wrong.artifact_id.as_deref().unwrap();
    let wrong_bytes = seal_object(
        ObjectKind::Segment,
        wrong_id,
        b"{}",
        &ObjectSecurity::plaintext(),
    )
    .unwrap();
    std::fs::write(artifact_path(&wrong_root, &wrong), wrong_bytes).unwrap();
    assert!(load_segment(&wrong_root, &ObjectSecurity::plaintext(), false).is_err());
    std::fs::remove_dir_all(wrong_root).unwrap();

    let digest_root = scratch("wdbx-v2-segment-artifact-digest");
    let digest_report = write_segment_with_codec(
        &digest_root,
        &snapshot,
        &ObjectSecurity::plaintext(),
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    let encoded = std::fs::read(&digest_report.path).unwrap();
    let opened = open_object(&encoded, &ObjectSecurity::plaintext(), false).unwrap();
    let mut image: SegmentImageV2 = serde_json::from_slice(&opened.plaintext).unwrap();
    image.codec.artifact.as_mut().unwrap().plaintext_sha256 = "0".repeat(64);
    let tampered_plaintext = serde_json::to_vec(&image).unwrap();
    let tampered = seal_object(
        ObjectKind::Segment,
        &digest_report.object_id,
        &tampered_plaintext,
        &ObjectSecurity::plaintext(),
    )
    .unwrap();
    std::fs::write(&digest_report.path, tampered).unwrap();
    assert!(load_segment(&digest_root, &ObjectSecurity::plaintext(), false).is_err());
    std::fs::remove_dir_all(digest_root).unwrap();
}

#[test]
fn encrypted_signed_artifact_authenticates_before_parse_and_tampering_fails() {
    let root = scratch("wdbx-v2-segment-artifact-secure");
    let (encryption, signing, verifying) = generate_key_material();
    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let report = write_segment_with_codec(
        &root,
        &snapshot,
        &security,
        SegmentCodecPolicy::Pq(pq_config(6)),
    )
    .unwrap();
    assert!(report.encrypted);
    assert!(report.signed);
    let artifact = artifact_path(&root, &report);
    let mut bytes = std::fs::read(&artifact).unwrap();
    assert!(
        !bytes
            .windows(b"codebooks".len())
            .any(|window| window == b"codebooks")
    );
    assert!(load_segment(&root, &security, true).unwrap().is_some());
    let midpoint = bytes.len() / 2;
    bytes[midpoint] ^= 1;
    std::fs::write(&artifact, bytes).unwrap();
    let error = load_segment(&root, &security, true).unwrap_err();
    assert!(error.to_string().contains("authentication failed"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_security_policy_mismatch_fails_before_publishing_a_segment() {
    let root = scratch("wdbx-v2-segment-artifact-policy");
    let snapshot = snapshot_with_vectors(&autoencoder_values(), 1);
    let policy = SegmentCodecPolicy::Pq(pq_config(6));
    write_segment_with_codec(&root, &snapshot, &ObjectSecurity::plaintext(), policy).unwrap();
    let segments_before = std::fs::read_dir(root.join("segments")).unwrap().count();
    let (encryption, signing, verifying) = generate_key_material();
    let secure =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let error = write_segment_with_codec(&root, &snapshot, &secure, policy).unwrap_err();
    assert!(error.to_string().contains("verification mismatch"));
    assert_eq!(
        std::fs::read_dir(root.join("segments")).unwrap().count(),
        segments_before
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn incomparable_segments_still_fall_back_conservatively() {
    let root = scratch("wdbx-v2-segments-incomparable");
    let security = ObjectSecurity::plaintext();
    let left_writer = Uuid::new_v4();
    let right_writer = Uuid::new_v4();
    let mut left = V2Snapshot::default();
    left.heads.insert(left_writer, 1);
    let mut right = V2Snapshot::default();
    right.heads.insert(right_writer, 1);
    write_segment(&root, &left, &security).unwrap();
    write_segment_with_codec(&root, &right, &security, SegmentCodecPolicy::None).unwrap();
    assert!(load_segment(&root, &security, false).unwrap().is_none());

    let mut covering = V2Snapshot::default();
    covering.heads.insert(left_writer, 1);
    covering.heads.insert(right_writer, 1);
    write_segment(&root, &covering, &security).unwrap();
    assert_eq!(
        load_segment(&root, &security, false).unwrap(),
        Some(covering)
    );
    std::fs::remove_dir_all(root).unwrap();
}
