//! Deterministic learned-codec artifacts and shared quality evidence.
//!
//! These reference codecs provide reproducible fixtures and persisted model
//! metadata. They are not presented as state of the art or independently
//! security audited.

mod autoencoder;
mod pq;

use crate::RecordId;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub use autoencoder::{
    AutoencoderArchitectureV1, AutoencoderArtifactV1, AutoencoderConfigV1, NormalizationParamsV1,
    PersistedAutoencoderV1, train_autoencoder_v1,
};
pub use pq::{PqArtifactV1, PqConfigV1, ProductQuantizerV1, train_product_quantizer_v1};

/// One identity-associated vector used for deterministic codec training.
#[derive(Clone, Copy, Debug)]
pub struct CodecRecord<'a> {
    /// Stable public record identity.
    pub id: RecordId,
    /// Vector values.
    pub values: &'a [f32],
}

/// Versioned quality evidence stored with a trained artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodecMetrics {
    /// Mean squared reconstruction error over every scalar in the fixture.
    pub reconstruction_mse: f32,
    /// Total raw source-vector bytes divided by the complete learned
    /// representation: artifact metadata/model plus every encoded source row.
    pub compression_ratio: f32,
    /// Mean overlap between exact and reconstructed/approximate top-k sets.
    pub recall_at_k: f32,
    /// The `k` used to calculate recall.
    pub recall_k: usize,
}

/// Invalid training data, configuration, encoded values, or artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// At least one training vector is required.
    EmptyDataset,
    /// Vector dimensions or artifact shapes are invalid.
    InvalidDimensions,
    /// Records in one vector space have different dimensions.
    MixedDimensions,
    /// A vector, parameter, normalization value, or metric is non-finite.
    NonFinite,
    /// Record identities must be unique within one training fixture.
    DuplicateRecordId,
    /// A version, bound, iteration count, or other codec setting is invalid.
    InvalidConfig,
    /// Encoded PQ codes or distance tables do not match the artifact.
    InvalidCodes,
    /// Serialized artifact input is malformed.
    InvalidArtifact,
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::EmptyDataset => "codec training dataset is empty",
            Self::InvalidDimensions => "codec dimensions are invalid",
            Self::MixedDimensions => "codec vectors have mixed dimensions",
            Self::NonFinite => "codec input contains a non-finite value",
            Self::DuplicateRecordId => "codec training record IDs must be unique",
            Self::InvalidConfig => "codec configuration is invalid",
            Self::InvalidCodes => "codec codes or distance table are invalid",
            Self::InvalidArtifact => "codec artifact is malformed",
        })
    }
}

impl std::error::Error for CodecError {}

pub(super) const BINARY_U32_BYTES: usize = 4;
pub(super) const BINARY_U64_BYTES: usize = 8;
pub(super) const BINARY_F32_BYTES: usize = 4;
pub(super) const BINARY_SHA256_BYTES: usize = 32;
pub(super) const BINARY_METRICS_BYTES: usize = 3 * BINARY_F32_BYTES + BINARY_U64_BYTES;

pub(super) fn source_vector_bytes(
    record_count: usize,
    dimensions: usize,
) -> Result<usize, CodecError> {
    record_count
        .checked_mul(dimensions)
        .and_then(|scalars| scalars.checked_mul(BINARY_F32_BYTES))
        .ok_or(CodecError::InvalidArtifact)
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn representation_ratio(source: usize, persisted: usize) -> f32 {
    source as f32 / persisted as f32
}

pub(super) fn sorted_records<'a>(
    records: &'a [CodecRecord<'a>],
) -> Result<Vec<CodecRecord<'a>>, CodecError> {
    if records.is_empty() {
        return Err(CodecError::EmptyDataset);
    }
    let dimensions = records[0].values.len();
    if dimensions == 0 || dimensions > crate::v2::MAX_V2_VECTOR_DIMENSIONS {
        return Err(CodecError::InvalidDimensions);
    }
    if records
        .iter()
        .any(|record| record.values.len() != dimensions)
    {
        return Err(CodecError::MixedDimensions);
    }
    if records
        .iter()
        .flat_map(|record| record.values)
        .any(|value| !value.is_finite())
    {
        return Err(CodecError::NonFinite);
    }
    let mut sorted = records.to_vec();
    sorted.sort_unstable_by_key(|record| record.id);
    let unique = sorted
        .iter()
        .map(|record| record.id)
        .collect::<BTreeSet<_>>();
    if unique.len() != sorted.len() {
        return Err(CodecError::DuplicateRecordId);
    }
    Ok(sorted)
}

/// Compute the canonical SHA-256 source digest used by learned artifacts.
///
/// Records are ordered by `RecordId`; identities are domain-separated by
/// their v1/v2 variant, and vector values use their exact little-endian f32
/// representation. This makes the digest independent of insertion order.
pub fn codec_source_digest(records: &[CodecRecord<'_>]) -> Result<String, CodecError> {
    let sorted = sorted_records(records)?;
    let mut digest = Sha256::new();
    digest.update(b"abi-wdbx-codec-source-v1\0");
    for record in sorted {
        match record.id {
            RecordId::Legacy(id) => {
                digest.update([0]);
                digest.update(id.to_le_bytes());
            }
            RecordId::V2(id) => {
                digest.update([1]);
                digest.update(id.as_bytes());
            }
        }
        for value in record.values {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum()
}

pub(super) fn recall_at_k(
    records: &[CodecRecord<'_>],
    approximations: &[Vec<f32>],
    k: usize,
) -> Result<f32, CodecError> {
    if k == 0 || k >= records.len() || approximations.len() != records.len() {
        return Err(CodecError::InvalidConfig);
    }
    let mut total = 0.0_f32;
    for query in 0..records.len() {
        let exact = ranked_neighbors(records, query, |candidate| {
            squared_distance(records[query].values, records[candidate].values)
        });
        let approximate = ranked_neighbors(records, query, |candidate| {
            squared_distance(&approximations[query], &approximations[candidate])
        });
        let expected = exact.into_iter().take(k).collect::<BTreeSet<_>>();
        let found = approximate.into_iter().take(k).collect::<BTreeSet<_>>();
        #[allow(clippy::cast_precision_loss)]
        let overlap = expected.intersection(&found).count() as f32 / k as f32;
        total += overlap;
    }
    #[allow(clippy::cast_precision_loss)]
    let count = records.len() as f32;
    Ok(total / count)
}

fn ranked_neighbors(
    records: &[CodecRecord<'_>],
    query: usize,
    distance: impl Fn(usize) -> f32,
) -> Vec<RecordId> {
    let mut ranked = records
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != query)
        .map(|(index, record)| (distance(index), record.id))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked.into_iter().map(|(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn source_digest_is_record_id_sorted_and_bit_exact() {
        let uuid = RecordId::V2(Uuid::from_u128(7));
        let first = [1.0_f32, -0.0];
        let second = [2.0_f32, 3.0];
        let forward = [
            CodecRecord {
                id: uuid,
                values: &second,
            },
            CodecRecord {
                id: RecordId::Legacy(4),
                values: &first,
            },
        ];
        let reverse = [forward[1], forward[0]];
        assert_eq!(
            codec_source_digest(&forward).unwrap(),
            codec_source_digest(&reverse).unwrap()
        );
        let changed = [1.0_f32, 0.0];
        assert_ne!(
            codec_source_digest(&forward).unwrap(),
            codec_source_digest(&[
                CodecRecord {
                    id: uuid,
                    values: &second,
                },
                CodecRecord {
                    id: RecordId::Legacy(4),
                    values: &changed,
                },
            ])
            .unwrap()
        );
    }

    #[test]
    fn source_validation_rejects_empty_mixed_nonfinite_and_duplicates() {
        assert_eq!(codec_source_digest(&[]), Err(CodecError::EmptyDataset));
        let one = [1.0];
        let two = [1.0, 2.0];
        assert_eq!(
            codec_source_digest(&[
                CodecRecord {
                    id: 1_u64.into(),
                    values: &one,
                },
                CodecRecord {
                    id: 2_u64.into(),
                    values: &two,
                },
            ]),
            Err(CodecError::MixedDimensions)
        );
        let bad = [f32::INFINITY];
        assert_eq!(
            codec_source_digest(&[CodecRecord {
                id: 1_u64.into(),
                values: &bad,
            }]),
            Err(CodecError::NonFinite)
        );
        assert_eq!(
            codec_source_digest(&[
                CodecRecord {
                    id: 1_u64.into(),
                    values: &one,
                },
                CodecRecord {
                    id: 1_u64.into(),
                    values: &one,
                },
            ]),
            Err(CodecError::DuplicateRecordId)
        );
    }
}
