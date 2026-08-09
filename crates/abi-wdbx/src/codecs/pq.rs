//! Deterministic product-quantization artifact version 1.

use super::{
    BINARY_F32_BYTES, BINARY_METRICS_BYTES, BINARY_SHA256_BYTES, BINARY_U32_BYTES,
    BINARY_U64_BYTES, CodecError, CodecMetrics, CodecRecord, codec_source_digest, recall_at_k,
    representation_ratio, sorted_records, source_vector_bytes, squared_distance,
};
use serde::{Deserialize, Serialize};

/// Artifact format version for [`ProductQuantizerV1`].
pub const PQ_ARTIFACT_VERSION: u32 = 1;

const MAX_ITERATIONS: usize = 4096;
const PQ_BINARY_MAGIC: &[u8] = b"ABI-WDBX-PQ-V1\0";

/// Reproducible product-quantizer training configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PqConfigV1 {
    /// Full vector width.
    pub dimensions: usize,
    /// Number of contiguous, possibly uneven subspaces.
    pub subspaces: usize,
    /// Centroids per subspace; bounded by the u8 code space.
    pub centroids: usize,
    /// Fixed k-means iteration count.
    pub iterations: usize,
    /// Seed used for deterministic centroid initialization.
    pub seed: u64,
    /// Neighborhood size used by artifact quality evidence.
    pub recall_k: usize,
}

/// Serializable, versioned PQ training result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PqArtifactV1 {
    /// Artifact schema version.
    pub version: u32,
    /// Number of source vectors represented by the artifact metrics and codes.
    pub training_record_count: usize,
    /// Exact training configuration.
    pub config: PqConfigV1,
    /// Inclusive-start offsets plus the terminal full dimension.
    pub subspace_offsets: Vec<usize>,
    /// `[subspace][centroid][component]` codebook values.
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Canonical SHA-256 of ID-sorted source vectors.
    pub source_digest: String,
    /// Deterministic reconstruction and retrieval evidence.
    pub metrics: CodecMetrics,
}

impl PqArtifactV1 {
    /// Serialize with stable struct-field and array ordering.
    pub fn to_json(&self) -> Result<String, CodecError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|_| CodecError::InvalidArtifact)
    }

    /// Parse and fully validate a persisted artifact.
    pub fn from_json(json: &str) -> Result<Self, CodecError> {
        let artifact: Self = serde_json::from_str(json).map_err(|_| CodecError::InvalidArtifact)?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validate schema, shapes, bounds, hashes, and quality evidence.
    pub fn validate(&self) -> Result<(), CodecError> {
        validate_config(&self.config, Some(self.training_record_count))?;
        if self.version != PQ_ARTIFACT_VERSION
            || self.source_digest.len() != 64
            || !self
                .source_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.subspace_offsets != offsets(&self.config)
            || self.codebooks.len() != self.config.subspaces
        {
            return Err(CodecError::InvalidArtifact);
        }
        for (subspace, codebook) in self.codebooks.iter().enumerate() {
            let width = self.subspace_offsets[subspace + 1] - self.subspace_offsets[subspace];
            if codebook.len() != self.config.centroids
                || codebook.iter().any(|centroid| centroid.len() != width)
            {
                return Err(CodecError::InvalidArtifact);
            }
        }
        if self
            .codebooks
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
            || !metrics_valid(&self.metrics, self.config.recall_k)
        {
            return Err(CodecError::NonFinite);
        }
        let expected_ratio = representation_ratio(
            source_vector_bytes(self.training_record_count, self.config.dimensions)?,
            self.accounted_persisted_bytes_unchecked()?,
        );
        if self.metrics.compression_ratio.to_bits() != expected_ratio.to_bits() {
            return Err(CodecError::InvalidArtifact);
        }
        Ok(())
    }

    /// Size of the complete learned representation under the PQ-v1 binary
    /// accounting convention.
    ///
    /// This is deliberately independent of JSON formatting. It includes a
    /// fixed magic/version/config descriptor, the training count, u64 offsets
    /// and array lengths, every f32 codebook component, the raw 32-byte source
    /// digest, the fixed-width metric record, and one u8 code per subspace for
    /// every training row.
    pub fn accounted_persisted_bytes(&self) -> Result<usize, CodecError> {
        self.validate()?;
        self.accounted_persisted_bytes_unchecked()
    }

    fn accounted_persisted_bytes_unchecked(&self) -> Result<usize, CodecError> {
        let config_bytes = 5 * BINARY_U64_BYTES + BINARY_U64_BYTES;
        let offsets_bytes = BINARY_U64_BYTES
            .checked_add(
                self.subspace_offsets
                    .len()
                    .checked_mul(BINARY_U64_BYTES)
                    .ok_or(CodecError::InvalidArtifact)?,
            )
            .ok_or(CodecError::InvalidArtifact)?;
        let mut codebook_bytes = BINARY_U64_BYTES;
        for codebook in &self.codebooks {
            codebook_bytes = codebook_bytes
                .checked_add(BINARY_U64_BYTES)
                .ok_or(CodecError::InvalidArtifact)?;
            for centroid in codebook {
                let centroid_bytes = BINARY_U64_BYTES
                    .checked_add(
                        centroid
                            .len()
                            .checked_mul(BINARY_F32_BYTES)
                            .ok_or(CodecError::InvalidArtifact)?,
                    )
                    .ok_or(CodecError::InvalidArtifact)?;
                codebook_bytes = codebook_bytes
                    .checked_add(centroid_bytes)
                    .ok_or(CodecError::InvalidArtifact)?;
            }
        }
        let codes_bytes = BINARY_U64_BYTES
            .checked_add(
                self.training_record_count
                    .checked_mul(self.config.subspaces)
                    .ok_or(CodecError::InvalidArtifact)?,
            )
            .ok_or(CodecError::InvalidArtifact)?;
        [
            PQ_BINARY_MAGIC.len(),
            BINARY_U32_BYTES,
            BINARY_U64_BYTES,
            config_bytes,
            offsets_bytes,
            codebook_bytes,
            BINARY_SHA256_BYTES,
            BINARY_METRICS_BYTES,
            codes_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total.checked_add(bytes).ok_or(CodecError::InvalidArtifact)
        })
    }
}

/// Runtime view of a validated PQ artifact.
#[derive(Clone, Debug)]
pub struct ProductQuantizerV1 {
    artifact: PqArtifactV1,
}

impl ProductQuantizerV1 {
    /// Load a validated artifact.
    pub fn from_artifact(artifact: PqArtifactV1) -> Result<Self, CodecError> {
        artifact.validate()?;
        Ok(Self { artifact })
    }

    /// Borrow the persisted representation.
    #[must_use]
    pub const fn artifact(&self) -> &PqArtifactV1 {
        &self.artifact
    }

    /// Quantize one vector into exactly one u8 code per subspace.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, CodecError> {
        validate_vector(vector, self.artifact.config.dimensions)?;
        let mut codes = Vec::with_capacity(self.artifact.config.subspaces);
        for (subspace, codebook) in self.artifact.codebooks.iter().enumerate() {
            let start = self.artifact.subspace_offsets[subspace];
            let end = self.artifact.subspace_offsets[subspace + 1];
            let code = nearest_centroid(&vector[start..end], codebook);
            codes.push(u8::try_from(code).map_err(|_| CodecError::InvalidConfig)?);
        }
        Ok(codes)
    }

    /// Reconstruct a vector from its subspace codes.
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>, CodecError> {
        validate_codes(codes, &self.artifact.config)?;
        let mut decoded = Vec::with_capacity(self.artifact.config.dimensions);
        for (subspace, &code) in codes.iter().enumerate() {
            decoded.extend_from_slice(&self.artifact.codebooks[subspace][usize::from(code)]);
        }
        Ok(decoded)
    }

    /// Precompute squared query-to-centroid distances for every subspace.
    pub fn distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, CodecError> {
        validate_vector(query, self.artifact.config.dimensions)?;
        Ok(self
            .artifact
            .codebooks
            .iter()
            .enumerate()
            .map(|(subspace, codebook)| {
                let start = self.artifact.subspace_offsets[subspace];
                let end = self.artifact.subspace_offsets[subspace + 1];
                codebook
                    .iter()
                    .map(|centroid| squared_distance(&query[start..end], centroid))
                    .collect()
            })
            .collect())
    }

    /// Sum a code's lookup-table entries to obtain approximate squared distance.
    pub fn distance_from_table(&self, codes: &[u8], table: &[Vec<f32>]) -> Result<f32, CodecError> {
        validate_codes(codes, &self.artifact.config)?;
        if table.len() != self.artifact.config.subspaces
            || table.iter().any(|row| {
                row.len() != self.artifact.config.centroids
                    || row.iter().any(|value| !value.is_finite())
            })
        {
            return Err(CodecError::InvalidCodes);
        }
        Ok(codes
            .iter()
            .enumerate()
            .map(|(subspace, code)| table[subspace][usize::from(*code)])
            .sum())
    }
}

/// Train deterministic fixed-iteration k-means in every contiguous subspace.
pub fn train_product_quantizer_v1(
    records: &[CodecRecord<'_>],
    config: PqConfigV1,
) -> Result<ProductQuantizerV1, CodecError> {
    let sorted = sorted_records(records)?;
    validate_config(&config, Some(sorted.len()))?;
    if sorted[0].values.len() != config.dimensions {
        return Err(CodecError::InvalidDimensions);
    }
    let subspace_offsets = offsets(&config);
    let mut codebooks = Vec::with_capacity(config.subspaces);
    for subspace in 0..config.subspaces {
        let start = subspace_offsets[subspace];
        let end = subspace_offsets[subspace + 1];
        codebooks.push(train_subspace(
            &sorted,
            start,
            end,
            config.centroids,
            config.iterations,
            config.seed ^ (subspace as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        ));
    }
    let provisional = PqArtifactV1 {
        version: PQ_ARTIFACT_VERSION,
        training_record_count: sorted.len(),
        config,
        subspace_offsets,
        codebooks,
        source_digest: codec_source_digest(records)?,
        metrics: CodecMetrics {
            reconstruction_mse: 0.0,
            compression_ratio: 0.0,
            recall_at_k: 0.0,
            recall_k: config.recall_k,
        },
    };
    let mut codec = ProductQuantizerV1 {
        artifact: provisional,
    };
    let reconstructions = sorted
        .iter()
        .map(|record| {
            codec
                .encode(record.values)
                .and_then(|codes| codec.decode(&codes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scalar_count = sorted.len() * config.dimensions;
    let squared_error = sorted
        .iter()
        .zip(&reconstructions)
        .map(|(record, reconstruction)| squared_distance(record.values, reconstruction))
        .sum::<f32>();
    #[allow(clippy::cast_precision_loss)]
    let reconstruction_mse = squared_error / scalar_count as f32;
    let compression_ratio = representation_ratio(
        source_vector_bytes(sorted.len(), config.dimensions)?,
        codec.artifact.accounted_persisted_bytes_unchecked()?,
    );
    codec.artifact.metrics = CodecMetrics {
        reconstruction_mse,
        compression_ratio,
        recall_at_k: recall_at_k(&sorted, &reconstructions, config.recall_k)?,
        recall_k: config.recall_k,
    };
    codec.artifact.validate()?;
    Ok(codec)
}

fn validate_config(config: &PqConfigV1, record_count: Option<usize>) -> Result<(), CodecError> {
    if config.dimensions == 0
        || config.dimensions > crate::v2::MAX_V2_VECTOR_DIMENSIONS
        || config.subspaces == 0
        || config.subspaces > config.dimensions
        || config.centroids == 0
        || config.centroids > usize::from(u8::MAX) + 1
        || config.iterations == 0
        || config.iterations > MAX_ITERATIONS
        || config.recall_k == 0
    {
        return Err(CodecError::InvalidConfig);
    }
    if let Some(count) = record_count
        && (config.centroids > count || config.recall_k >= count)
    {
        return Err(CodecError::InvalidConfig);
    }
    Ok(())
}

fn validate_vector(vector: &[f32], dimensions: usize) -> Result<(), CodecError> {
    if vector.len() != dimensions {
        return Err(CodecError::InvalidDimensions);
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(CodecError::NonFinite);
    }
    Ok(())
}

fn validate_codes(codes: &[u8], config: &PqConfigV1) -> Result<(), CodecError> {
    if codes.len() != config.subspaces
        || codes
            .iter()
            .any(|code| usize::from(*code) >= config.centroids)
    {
        return Err(CodecError::InvalidCodes);
    }
    Ok(())
}

fn offsets(config: &PqConfigV1) -> Vec<usize> {
    (0..=config.subspaces)
        .map(|subspace| subspace * config.dimensions / config.subspaces)
        .collect()
}

fn train_subspace(
    records: &[CodecRecord<'_>],
    start: usize,
    end: usize,
    centroid_count: usize,
    iterations: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let width = end - start;
    let mut indices = (0..records.len()).collect::<Vec<_>>();
    let mut random = StableRng::new(seed);
    for index in (1..indices.len()).rev() {
        let swap = random.index(index + 1);
        indices.swap(index, swap);
    }
    let mut centroids = indices
        .into_iter()
        .take(centroid_count)
        .map(|index| records[index].values[start..end].to_vec())
        .collect::<Vec<_>>();
    let mut assignments = vec![0; records.len()];
    for _ in 0..iterations {
        for (index, record) in records.iter().enumerate() {
            assignments[index] = nearest_centroid(&record.values[start..end], &centroids);
        }
        let mut sums = vec![vec![0.0_f32; width]; centroid_count];
        let mut counts = vec![0_usize; centroid_count];
        for (record, &assignment) in records.iter().zip(&assignments) {
            counts[assignment] += 1;
            for (sum, value) in sums[assignment].iter_mut().zip(&record.values[start..end]) {
                *sum += value;
            }
        }
        for centroid in 0..centroid_count {
            if counts[centroid] == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let denominator = counts[centroid] as f32;
            for (value, sum) in centroids[centroid].iter_mut().zip(&sums[centroid]) {
                *value = *sum / denominator;
            }
        }
    }
    centroids
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(index, centroid)| (squared_distance(vector, centroid), index))
        .min_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        })
        .map_or(0, |(_, index)| index)
}

fn metrics_valid(metrics: &CodecMetrics, recall_k: usize) -> bool {
    metrics.reconstruction_mse.is_finite()
        && metrics.reconstruction_mse >= 0.0
        && metrics.compression_ratio.is_finite()
        && metrics.compression_ratio > 0.0
        && metrics.recall_at_k.is_finite()
        && (0.0..=1.0).contains(&metrics.recall_at_k)
        && metrics.recall_k == recall_k
}

#[derive(Clone, Copy)]
struct StableRng(u64);

impl StableRng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        usize::try_from(self.next() % upper as u64).expect("remainder is smaller than usize upper")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RecordId;

    const V1_FIXTURE_MAX_MSE: f32 = 0.007;
    const V1_FIXTURE_MIN_RECALL_AT_3: f32 = 0.99;
    const V1_FIXTURE_SOURCE_BYTES: usize = 896;
    const V1_FIXTURE_PERSISTED_BYTES: usize = 719;

    #[allow(clippy::cast_precision_loss)]
    fn fixture() -> (Vec<[f32; 7]>, Vec<RecordId>) {
        let vectors = (0..32)
            .map(|index| {
                let group = (index / 4) as f32;
                let offset = (index % 4) as f32 * 0.015;
                [
                    group * 0.3 + offset,
                    group.sin() + offset,
                    group.cos() - offset,
                    group * -0.12,
                    (group * 0.7).sin(),
                    (group * 0.4).cos(),
                    offset - group * 0.05,
                ]
            })
            .collect();
        let ids = (0..32).map(|id| RecordId::Legacy(id + 1)).collect();
        (vectors, ids)
    }

    fn config() -> PqConfigV1 {
        PqConfigV1 {
            dimensions: 7,
            subspaces: 3,
            centroids: 8,
            iterations: 20,
            seed: 0xD15C_A11E,
            recall_k: 3,
        }
    }

    #[test]
    fn deterministic_quality_fixture_and_reload() {
        let (vectors, ids) = fixture();
        let records = ids
            .iter()
            .zip(&vectors)
            .map(|(id, values)| CodecRecord { id: *id, values })
            .collect::<Vec<_>>();
        let first = train_product_quantizer_v1(&records, config()).unwrap();
        let second = train_product_quantizer_v1(&records, config()).unwrap();
        assert_eq!(first.artifact(), second.artifact());
        assert_eq!(first.artifact().subspace_offsets, [0, 2, 4, 7]);
        assert!(first.artifact().metrics.reconstruction_mse <= V1_FIXTURE_MAX_MSE);
        assert_eq!(
            first.artifact().accounted_persisted_bytes().unwrap(),
            V1_FIXTURE_PERSISTED_BYTES
        );
        assert_eq!(
            first.artifact().metrics.compression_ratio.to_bits(),
            representation_ratio(V1_FIXTURE_SOURCE_BYTES, V1_FIXTURE_PERSISTED_BYTES).to_bits()
        );
        assert!(first.artifact().metrics.recall_at_k >= V1_FIXTURE_MIN_RECALL_AT_3);
        let json = first.artifact().to_json().unwrap();
        let parsed = PqArtifactV1::from_json(&json).unwrap();
        let reloaded = ProductQuantizerV1::from_artifact(parsed).unwrap();
        let codes = reloaded.encode(&vectors[5]).unwrap();
        assert_eq!(codes.len(), 3);
        let table = reloaded.distance_table(&vectors[4]).unwrap();
        let distance = reloaded.distance_from_table(&codes, &table).unwrap();
        assert!(distance.is_finite());
        assert_eq!(
            reloaded.decode(&codes).unwrap(),
            first.decode(&codes).unwrap()
        );
    }

    #[test]
    fn ties_choose_the_lowest_centroid_and_inputs_are_strict() {
        let code = nearest_centroid(&[0.0], &[vec![-1.0], vec![1.0]]);
        assert_eq!(code, 0);
        let values = [0.0, 1.0];
        let records = [CodecRecord {
            id: 1_u64.into(),
            values: &values,
        }];
        assert_eq!(
            train_product_quantizer_v1(
                &records,
                PqConfigV1 {
                    dimensions: 2,
                    subspaces: 1,
                    centroids: 2,
                    iterations: 1,
                    seed: 1,
                    recall_k: 1,
                }
            )
            .unwrap_err(),
            CodecError::InvalidConfig
        );
        let bad = [f32::NAN, 1.0];
        assert_eq!(
            train_product_quantizer_v1(
                &[CodecRecord {
                    id: 1_u64.into(),
                    values: &bad,
                }],
                PqConfigV1 {
                    dimensions: 2,
                    subspaces: 1,
                    centroids: 1,
                    iterations: 1,
                    seed: 1,
                    recall_k: 1,
                }
            )
            .unwrap_err(),
            CodecError::NonFinite
        );
    }

    #[test]
    fn tampered_artifacts_codes_and_tables_fail_closed() {
        let (vectors, ids) = fixture();
        let records = ids
            .iter()
            .zip(&vectors)
            .map(|(id, values)| CodecRecord { id: *id, values })
            .collect::<Vec<_>>();
        let codec = train_product_quantizer_v1(&records, config()).unwrap();
        assert_eq!(codec.decode(&[0, 0]), Err(CodecError::InvalidCodes));
        let encoded = codec.encode(&vectors[0]).unwrap();
        assert_eq!(
            codec.distance_from_table(&encoded, &[vec![0.0]]),
            Err(CodecError::InvalidCodes)
        );
        let mut artifact = codec.artifact().clone();
        artifact.codebooks[0][0].push(0.0);
        assert_eq!(artifact.validate(), Err(CodecError::InvalidArtifact));
        let mut artifact = codec.artifact().clone();
        artifact.metrics.recall_at_k = f32::NAN;
        assert_eq!(artifact.validate(), Err(CodecError::NonFinite));
        let mut artifact = codec.artifact().clone();
        artifact.metrics.compression_ratio = 9.333_333;
        assert_eq!(artifact.validate(), Err(CodecError::InvalidArtifact));
    }
}
