//! Versioned segment representations and learned-codec transforms.

use super::artifact::ArtifactReference;
use crate::codecs::{
    AutoencoderArtifactV1, AutoencoderConfigV1, CodecError, CodecMetrics, CodecRecord,
    PersistedAutoencoderV1, PqArtifactV1, PqConfigV1, ProductQuantizerV1, train_autoencoder_v1,
    train_product_quantizer_v1,
};
use crate::v2::{
    CausalHeads, RecordId, V2AuditBlock, V2Snapshot, V2SpatialRecord, V2TemporalRecord, Version,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const CODEC_DESCRIPTOR_VERSION: u8 = 1;

/// Vector representation declared by one immutable segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentCodecKind {
    /// Exact vectors stored as ordinary finite f32 arrays.
    None,
    /// Product-quantized vectors backed by a v1 codebook artifact.
    Pq,
    /// Latent vectors backed by a v1 persisted autoencoder artifact.
    Autoencoder,
}

impl SegmentCodecKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Pq => "pq",
            Self::Autoencoder => "autoencoder",
        }
    }
}

/// Explicit policy for writing one format-2 segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentCodecPolicy {
    /// Preserve vector bits exactly without an artifact.
    None,
    /// Train and persist deterministic product quantization.
    Pq(PqConfigV1),
    /// Train and persist a deterministic reference autoencoder.
    Autoencoder(AutoencoderConfigV1),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct SegmentCodecDescriptor {
    descriptor_version: u8,
    pub(super) kind: SegmentCodecKind,
    pub(super) artifact: Option<ArtifactReference>,
}

impl SegmentCodecDescriptor {
    pub(super) const fn none() -> Self {
        Self {
            descriptor_version: CODEC_DESCRIPTOR_VERSION,
            kind: SegmentCodecKind::None,
            artifact: None,
        }
    }

    pub(super) fn new(
        kind: SegmentCodecKind,
        artifact: Option<ArtifactReference>,
    ) -> Result<Self, crate::v2::V2Error> {
        let descriptor = Self {
            descriptor_version: CODEC_DESCRIPTOR_VERSION,
            kind,
            artifact,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub(super) fn validate(&self) -> Result<(), crate::v2::V2Error> {
        if self.descriptor_version != CODEC_DESCRIPTOR_VERSION
            || matches!(self.kind, SegmentCodecKind::None) != self.artifact.is_none()
        {
            return Err(crate::v2::V2Error::InvalidMutation(
                "invalid segment codec descriptor".into(),
            ));
        }
        if let Some(artifact) = &self.artifact {
            artifact.validate(self.kind)?;
        }
        Ok(())
    }

    pub(super) fn metrics(&self) -> Option<&CodecMetrics> {
        self.artifact.as_ref().map(|artifact| &artifact.metrics)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "encoding", content = "values", rename_all = "snake_case")]
enum EncodedVector {
    Raw(Vec<f32>),
    Pq(Vec<u8>),
    Autoencoder(Vec<f32>),
}

/// Snapshot metadata with only vector payloads replaced by codec values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct EncodedSnapshotV2 {
    heads: CausalHeads,
    kv: BTreeMap<String, Vec<Version<String>>>,
    vectors: BTreeMap<RecordId, Vec<Version<EncodedVector>>>,
    spatial: BTreeMap<RecordId, Vec<Version<V2SpatialRecord>>>,
    temporal: BTreeMap<String, Vec<Version<V2TemporalRecord>>>,
    audit: BTreeMap<String, V2AuditBlock>,
    audit_order: Vec<String>,
    committed_transactions: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "codec", content = "artifact", rename_all = "snake_case")]
pub(super) enum CodecArtifactPayload {
    Pq(PqArtifactV1),
    Autoencoder(AutoencoderArtifactV1),
}

impl CodecArtifactPayload {
    pub(super) const fn kind(&self) -> SegmentCodecKind {
        match self {
            Self::Pq(_) => SegmentCodecKind::Pq,
            Self::Autoencoder(_) => SegmentCodecKind::Autoencoder,
        }
    }

    pub(super) fn metrics(&self) -> &CodecMetrics {
        match self {
            Self::Pq(artifact) => &artifact.metrics,
            Self::Autoencoder(artifact) => &artifact.metrics,
        }
    }

    pub(super) fn validate(&self) -> Result<(), CodecError> {
        match self {
            Self::Pq(artifact) => artifact.validate(),
            Self::Autoencoder(artifact) => artifact.validate(),
        }
    }
}

pub(super) struct PreparedSnapshot {
    pub(super) kind: SegmentCodecKind,
    pub(super) snapshot: EncodedSnapshotV2,
    pub(super) artifact: Option<CodecArtifactPayload>,
}

pub(super) fn prepare_snapshot(
    snapshot: &V2Snapshot,
    policy: SegmentCodecPolicy,
) -> Result<PreparedSnapshot, CodecError> {
    match policy {
        SegmentCodecPolicy::None => Ok(PreparedSnapshot {
            kind: SegmentCodecKind::None,
            snapshot: encode_snapshot(snapshot, |values| Ok(EncodedVector::Raw(values.to_vec())))?,
            artifact: None,
        }),
        SegmentCodecPolicy::Pq(config) => {
            let records = codec_records(snapshot);
            let quantizer = train_product_quantizer_v1(&records, config)?;
            let encoded = encode_snapshot(snapshot, |values| {
                quantizer.encode(values).map(EncodedVector::Pq)
            })?;
            Ok(PreparedSnapshot {
                kind: SegmentCodecKind::Pq,
                snapshot: encoded,
                artifact: Some(CodecArtifactPayload::Pq(quantizer.artifact().clone())),
            })
        }
        SegmentCodecPolicy::Autoencoder(config) => {
            let records = codec_records(snapshot);
            let autoencoder = train_autoencoder_v1(&records, config)?;
            let encoded = encode_snapshot(snapshot, |values| {
                autoencoder.encode(values).map(EncodedVector::Autoencoder)
            })?;
            Ok(PreparedSnapshot {
                kind: SegmentCodecKind::Autoencoder,
                snapshot: encoded,
                artifact: Some(CodecArtifactPayload::Autoencoder(
                    autoencoder.artifact().clone(),
                )),
            })
        }
    }
}

pub(super) fn decode_snapshot(
    encoded: EncodedSnapshotV2,
    descriptor: &SegmentCodecDescriptor,
    artifact: Option<&CodecArtifactPayload>,
) -> Result<V2Snapshot, CodecError> {
    descriptor_payload_match(descriptor, artifact)?;
    let decoder = match artifact {
        None => Decoder::None,
        Some(CodecArtifactPayload::Pq(artifact)) => Decoder::Pq(Box::new(
            ProductQuantizerV1::from_artifact(artifact.clone())?,
        )),
        Some(CodecArtifactPayload::Autoencoder(artifact)) => Decoder::Autoencoder(Box::new(
            PersistedAutoencoderV1::from_artifact(artifact.clone())?,
        )),
    };
    let artifact_records = artifact.map_or(0, |payload| match payload {
        CodecArtifactPayload::Pq(artifact) => artifact.training_record_count,
        CodecArtifactPayload::Autoencoder(artifact) => artifact.training_record_count,
    });
    let encoded_records = encoded.vectors.values().map(Vec::len).sum::<usize>();
    if artifact.is_some() && artifact_records != encoded_records {
        return Err(CodecError::InvalidArtifact);
    }
    let vectors = encoded
        .vectors
        .into_iter()
        .map(|(id, versions)| {
            versions
                .into_iter()
                .map(|version| {
                    let value = decoder.decode(version.value)?;
                    Ok(Version {
                        version_id: version.version_id,
                        writer_id: version.writer_id,
                        sequence: version.sequence,
                        observed_heads: version.observed_heads,
                        value,
                    })
                })
                .collect::<Result<Vec<_>, CodecError>>()
                .map(|versions| (id, versions))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(V2Snapshot {
        heads: encoded.heads,
        kv: encoded.kv,
        vectors,
        spatial: encoded.spatial,
        temporal: encoded.temporal,
        audit: encoded.audit,
        audit_order: encoded.audit_order,
        committed_transactions: encoded.committed_transactions,
    })
}

enum Decoder {
    None,
    Pq(Box<ProductQuantizerV1>),
    Autoencoder(Box<PersistedAutoencoderV1>),
}

impl Decoder {
    fn decode(&self, encoded: EncodedVector) -> Result<Vec<f32>, CodecError> {
        match (self, encoded) {
            (Self::None, EncodedVector::Raw(values)) => validate_raw(values),
            (Self::Pq(quantizer), EncodedVector::Pq(encoded_codes)) => {
                quantizer.decode(&encoded_codes)
            }
            (Self::Autoencoder(model), EncodedVector::Autoencoder(latent)) => model.decode(&latent),
            _ => Err(CodecError::InvalidCodes),
        }
    }
}

fn descriptor_payload_match(
    descriptor: &SegmentCodecDescriptor,
    artifact: Option<&CodecArtifactPayload>,
) -> Result<(), CodecError> {
    match (descriptor.kind, artifact) {
        (SegmentCodecKind::None, None) => Ok(()),
        (expected, Some(payload)) if expected == payload.kind() => {
            payload.validate()?;
            let reference = descriptor
                .artifact
                .as_ref()
                .ok_or(CodecError::InvalidArtifact)?;
            if reference.metrics != *payload.metrics() {
                return Err(CodecError::InvalidArtifact);
            }
            Ok(())
        }
        _ => Err(CodecError::InvalidArtifact),
    }
}

fn codec_records(snapshot: &V2Snapshot) -> Vec<CodecRecord<'_>> {
    snapshot
        .vectors
        .values()
        .flatten()
        .map(|version| CodecRecord {
            // One public vector identity may retain concurrent versions. The
            // immutable version identity is globally stable and lets training
            // include every value without collapsing its conflict set.
            id: RecordId::V2(version.version_id),
            values: &version.value,
        })
        .collect()
}

fn encode_snapshot(
    snapshot: &V2Snapshot,
    mut encode: impl FnMut(&[f32]) -> Result<EncodedVector, CodecError>,
) -> Result<EncodedSnapshotV2, CodecError> {
    let vectors = snapshot
        .vectors
        .iter()
        .map(|(id, versions)| {
            versions
                .iter()
                .map(|version| {
                    Ok(Version {
                        version_id: version.version_id,
                        writer_id: version.writer_id,
                        sequence: version.sequence,
                        observed_heads: version.observed_heads.clone(),
                        value: encode(&version.value)?,
                    })
                })
                .collect::<Result<Vec<_>, CodecError>>()
                .map(|versions| (*id, versions))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(EncodedSnapshotV2 {
        heads: snapshot.heads.clone(),
        kv: snapshot.kv.clone(),
        vectors,
        spatial: snapshot.spatial.clone(),
        temporal: snapshot.temporal.clone(),
        audit: snapshot.audit.clone(),
        audit_order: snapshot.audit_order.clone(),
        committed_transactions: snapshot.committed_transactions,
    })
}

fn validate_raw(values: Vec<f32>) -> Result<Vec<f32>, CodecError> {
    if values.is_empty() || values.len() > crate::v2::MAX_V2_VECTOR_DIMENSIONS {
        return Err(CodecError::InvalidDimensions);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CodecError::NonFinite);
    }
    Ok(values)
}
