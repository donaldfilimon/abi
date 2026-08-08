//! Persisted deterministic autoencoder artifact version 1.

use super::{
    BINARY_F32_BYTES, BINARY_METRICS_BYTES, BINARY_SHA256_BYTES, BINARY_U32_BYTES,
    BINARY_U64_BYTES, CodecError, CodecMetrics, CodecRecord, codec_source_digest, recall_at_k,
    representation_ratio, sorted_records, source_vector_bytes, squared_distance,
};
use crate::neural_compress::{Autoencoder, AutoencoderError, AutoencoderParameters};
use serde::{Deserialize, Serialize};

/// Artifact schema version.
pub const AUTOENCODER_ARTIFACT_VERSION: u32 = 1;

const MAX_EPOCHS: usize = 100_000;
const AUTOENCODER_BINARY_MAGIC: &[u8] = b"ABI-WDBX-AUTOENCODER-V1\0";

/// Architecture metadata that fixes activation and decoder semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoencoderArchitectureV1 {
    /// One tanh encoder layer followed by one linear decoder layer.
    TanhEncoderLinearDecoder,
}

/// Reproducible autoencoder training configuration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoencoderConfigV1 {
    /// Full vector width.
    pub input_dimensions: usize,
    /// Encoded float width.
    pub latent_dimensions: usize,
    /// Weight initialization seed.
    pub seed: u64,
    /// Fixed number of insertion-order epochs.
    pub epochs: usize,
    /// Fixed SGD learning rate.
    pub learning_rate: f32,
    /// Neighborhood size used by artifact quality evidence.
    pub recall_k: usize,
}

/// Explicit per-dimension affine normalization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizationParamsV1 {
    /// Training-set mean for each input dimension.
    pub mean: Vec<f32>,
    /// Positive maximum-centered-magnitude scale for each dimension.
    pub scale: Vec<f32>,
}

/// Serializable autoencoder architecture, weights, and training evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoencoderArtifactV1 {
    /// Artifact schema version.
    pub version: u32,
    /// Number of source vectors represented by the artifact metrics and codes.
    pub training_record_count: usize,
    /// Explicit activation/layer architecture.
    pub architecture: AutoencoderArchitectureV1,
    /// Exact training configuration.
    pub config: AutoencoderConfigV1,
    /// Affine input normalization learned from the source fixture.
    pub normalization: NormalizationParamsV1,
    /// Row-major `[latent][input]` encoder weights.
    pub encoder_weights: Vec<f32>,
    /// Encoder bias values.
    pub encoder_bias: Vec<f32>,
    /// Row-major `[input][latent]` decoder weights.
    pub decoder_weights: Vec<f32>,
    /// Decoder bias values.
    pub decoder_bias: Vec<f32>,
    /// Canonical SHA-256 of ID-sorted source vectors.
    pub source_digest: String,
    /// Deterministic reconstruction and retrieval evidence.
    pub metrics: CodecMetrics,
}

impl AutoencoderArtifactV1 {
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

    /// Validate versions, shapes, bounds, normalization, and finite values.
    pub fn validate(&self) -> Result<(), CodecError> {
        validate_config(&self.config, Some(self.training_record_count))?;
        let input = self.config.input_dimensions;
        let latent = self.config.latent_dimensions;
        if self.version != AUTOENCODER_ARTIFACT_VERSION
            || self.architecture != AutoencoderArchitectureV1::TanhEncoderLinearDecoder
            || self.normalization.mean.len() != input
            || self.normalization.scale.len() != input
            || self.encoder_weights.len() != input * latent
            || self.encoder_bias.len() != latent
            || self.decoder_weights.len() != input * latent
            || self.decoder_bias.len() != input
            || self.source_digest.len() != 64
            || !self
                .source_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CodecError::InvalidArtifact);
        }
        if self
            .normalization
            .mean
            .iter()
            .chain(&self.normalization.scale)
            .chain(&self.encoder_weights)
            .chain(&self.encoder_bias)
            .chain(&self.decoder_weights)
            .chain(&self.decoder_bias)
            .any(|value| !value.is_finite())
            || !self.config.learning_rate.is_finite()
            || !metrics_valid(&self.metrics, self.config.recall_k)
        {
            return Err(CodecError::NonFinite);
        }
        if self.normalization.scale.iter().any(|scale| *scale <= 0.0) {
            return Err(CodecError::InvalidArtifact);
        }
        let expected_ratio = representation_ratio(
            source_vector_bytes(self.training_record_count, self.config.input_dimensions)?,
            self.accounted_persisted_bytes_unchecked()?,
        );
        if self.metrics.compression_ratio.to_bits() != expected_ratio.to_bits() {
            return Err(CodecError::InvalidArtifact);
        }
        Ok(())
    }

    /// Size of the complete learned representation under the autoencoder-v1
    /// binary accounting convention.
    ///
    /// This is deliberately independent of JSON float formatting. It includes
    /// a fixed magic/version/architecture/config descriptor, the training
    /// count, u64 array lengths, normalization, every f32 weight and bias, the
    /// raw 32-byte source digest, the fixed-width metric record, and every f32
    /// latent code for the training rows.
    pub fn accounted_persisted_bytes(&self) -> Result<usize, CodecError> {
        self.validate()?;
        self.accounted_persisted_bytes_unchecked()
    }

    fn accounted_persisted_bytes_unchecked(&self) -> Result<usize, CodecError> {
        let config_bytes = 5 * BINARY_U64_BYTES + BINARY_F32_BYTES;
        let normalization_bytes = vector_bytes(&self.normalization.mean)?
            .checked_add(vector_bytes(&self.normalization.scale)?)
            .ok_or(CodecError::InvalidArtifact)?;
        let parameter_bytes = [
            &self.encoder_weights,
            &self.encoder_bias,
            &self.decoder_weights,
            &self.decoder_bias,
        ]
        .into_iter()
        .try_fold(0_usize, |total, values| {
            total
                .checked_add(vector_bytes(values)?)
                .ok_or(CodecError::InvalidArtifact)
        })?;
        let latent_bytes = BINARY_U64_BYTES
            .checked_add(
                self.training_record_count
                    .checked_mul(self.config.latent_dimensions)
                    .and_then(|values| values.checked_mul(BINARY_F32_BYTES))
                    .ok_or(CodecError::InvalidArtifact)?,
            )
            .ok_or(CodecError::InvalidArtifact)?;
        [
            AUTOENCODER_BINARY_MAGIC.len(),
            BINARY_U32_BYTES,
            BINARY_U32_BYTES,
            BINARY_U64_BYTES,
            config_bytes,
            normalization_bytes,
            parameter_bytes,
            BINARY_SHA256_BYTES,
            BINARY_METRICS_BYTES,
            latent_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total.checked_add(bytes).ok_or(CodecError::InvalidArtifact)
        })
    }
}

fn vector_bytes(values: &[f32]) -> Result<usize, CodecError> {
    BINARY_U64_BYTES
        .checked_add(
            values
                .len()
                .checked_mul(BINARY_F32_BYTES)
                .ok_or(CodecError::InvalidArtifact)?,
        )
        .ok_or(CodecError::InvalidArtifact)
}

/// Validated runtime autoencoder with original-space normalization helpers.
#[derive(Clone, Debug)]
pub struct PersistedAutoencoderV1 {
    artifact: AutoencoderArtifactV1,
    model: Autoencoder,
}

impl PersistedAutoencoderV1 {
    /// Load a persisted artifact only after validating every parameter.
    pub fn from_artifact(artifact: AutoencoderArtifactV1) -> Result<Self, CodecError> {
        artifact.validate()?;
        let parameters = AutoencoderParameters {
            encoder_weights: artifact.encoder_weights.clone(),
            encoder_bias: artifact.encoder_bias.clone(),
            decoder_weights: artifact.decoder_weights.clone(),
            decoder_bias: artifact.decoder_bias.clone(),
        };
        let model = Autoencoder::from_parameters(
            artifact.config.input_dimensions,
            artifact.config.latent_dimensions,
            parameters,
        )
        .map_err(map_autoencoder_error)?;
        Ok(Self { artifact, model })
    }

    /// Borrow the validated persisted representation.
    #[must_use]
    pub const fn artifact(&self) -> &AutoencoderArtifactV1 {
        &self.artifact
    }

    /// Encode an original-space vector into the artifact's latent space.
    pub fn encode(&self, input: &[f32]) -> Result<Vec<f32>, CodecError> {
        let normalized = self.normalize(input)?;
        let mut latent = vec![0.0; self.artifact.config.latent_dimensions];
        self.model
            .encode_into(&normalized, &mut latent)
            .map_err(map_autoencoder_error)?;
        Ok(latent)
    }

    /// Decode latent values into the original, de-normalized vector space.
    pub fn decode(&self, latent: &[f32]) -> Result<Vec<f32>, CodecError> {
        let mut normalized = vec![0.0; self.artifact.config.input_dimensions];
        self.model
            .decode_into(latent, &mut normalized)
            .map_err(map_autoencoder_error)?;
        Ok(normalized
            .iter()
            .zip(&self.artifact.normalization.mean)
            .zip(&self.artifact.normalization.scale)
            .map(|((value, mean), scale)| value * scale + mean)
            .collect())
    }

    /// Encode and decode one original-space vector.
    pub fn reconstruct(&self, input: &[f32]) -> Result<Vec<f32>, CodecError> {
        self.encode(input).and_then(|latent| self.decode(&latent))
    }

    fn normalize(&self, input: &[f32]) -> Result<Vec<f32>, CodecError> {
        if input.len() != self.artifact.config.input_dimensions {
            return Err(CodecError::InvalidDimensions);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(CodecError::NonFinite);
        }
        Ok(input
            .iter()
            .zip(&self.artifact.normalization.mean)
            .zip(&self.artifact.normalization.scale)
            .map(|((value, mean), scale)| (value - mean) / scale)
            .collect())
    }
}

/// Train a normalized deterministic autoencoder and attach quality evidence.
pub fn train_autoencoder_v1(
    records: &[CodecRecord<'_>],
    config: AutoencoderConfigV1,
) -> Result<PersistedAutoencoderV1, CodecError> {
    let sorted = sorted_records(records)?;
    validate_config(&config, Some(sorted.len()))?;
    if sorted[0].values.len() != config.input_dimensions {
        return Err(CodecError::InvalidDimensions);
    }
    let normalization = normalization(&sorted, config.input_dimensions);
    let normalized = sorted
        .iter()
        .map(|record| {
            record
                .values
                .iter()
                .zip(&normalization.mean)
                .zip(&normalization.scale)
                .map(|((value, mean), scale)| (value - mean) / scale)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let normalized_refs = normalized.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut model = Autoencoder::new(
        config.input_dimensions,
        config.latent_dimensions,
        config.seed,
    )
    .map_err(map_autoencoder_error)?;
    for _ in 0..config.epochs {
        model
            .train_epoch(&normalized_refs, config.learning_rate)
            .map_err(map_autoencoder_error)?;
    }
    let parameters = model.parameters();
    let provisional = AutoencoderArtifactV1 {
        version: AUTOENCODER_ARTIFACT_VERSION,
        training_record_count: sorted.len(),
        architecture: AutoencoderArchitectureV1::TanhEncoderLinearDecoder,
        config,
        normalization,
        encoder_weights: parameters.encoder_weights,
        encoder_bias: parameters.encoder_bias,
        decoder_weights: parameters.decoder_weights,
        decoder_bias: parameters.decoder_bias,
        source_digest: codec_source_digest(records)?,
        metrics: CodecMetrics {
            reconstruction_mse: 0.0,
            compression_ratio: 0.0,
            recall_at_k: 0.0,
            recall_k: config.recall_k,
        },
    };
    let mut persisted = PersistedAutoencoderV1::from_artifact_with_pending_metrics(provisional)?;
    let reconstructions = sorted
        .iter()
        .map(|record| persisted.reconstruct(record.values))
        .collect::<Result<Vec<_>, _>>()?;
    let error = sorted
        .iter()
        .zip(&reconstructions)
        .map(|(record, reconstruction)| squared_distance(record.values, reconstruction))
        .sum::<f32>();
    #[allow(clippy::cast_precision_loss)]
    let reconstruction_mse = error / (sorted.len() * config.input_dimensions) as f32;
    let compression_ratio = representation_ratio(
        source_vector_bytes(sorted.len(), config.input_dimensions)?,
        persisted.artifact.accounted_persisted_bytes_unchecked()?,
    );
    persisted.artifact.metrics = CodecMetrics {
        reconstruction_mse,
        compression_ratio,
        recall_at_k: recall_at_k(&sorted, &reconstructions, config.recall_k)?,
        recall_k: config.recall_k,
    };
    persisted.artifact.validate()?;
    Ok(persisted)
}

impl PersistedAutoencoderV1 {
    fn from_artifact_with_pending_metrics(
        artifact: AutoencoderArtifactV1,
    ) -> Result<Self, CodecError> {
        validate_config(&artifact.config, None)?;
        let parameters = AutoencoderParameters {
            encoder_weights: artifact.encoder_weights.clone(),
            encoder_bias: artifact.encoder_bias.clone(),
            decoder_weights: artifact.decoder_weights.clone(),
            decoder_bias: artifact.decoder_bias.clone(),
        };
        let model = Autoencoder::from_parameters(
            artifact.config.input_dimensions,
            artifact.config.latent_dimensions,
            parameters,
        )
        .map_err(map_autoencoder_error)?;
        Ok(Self { artifact, model })
    }
}

fn normalization(records: &[CodecRecord<'_>], dimensions: usize) -> NormalizationParamsV1 {
    let mut mean = vec![0.0_f32; dimensions];
    for record in records {
        for (mean, value) in mean.iter_mut().zip(record.values) {
            *mean += value;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = records.len() as f32;
    for value in &mut mean {
        *value /= count;
    }
    let mut scale = vec![0.0_f32; dimensions];
    for record in records {
        for ((scale, value), mean) in scale.iter_mut().zip(record.values).zip(&mean) {
            *scale = scale.max((value - mean).abs());
        }
    }
    for value in &mut scale {
        if *value <= f32::EPSILON {
            *value = 1.0;
        }
    }
    NormalizationParamsV1 { mean, scale }
}

fn validate_config(
    config: &AutoencoderConfigV1,
    record_count: Option<usize>,
) -> Result<(), CodecError> {
    if config.input_dimensions == 0
        || config.input_dimensions > crate::v2::MAX_V2_VECTOR_DIMENSIONS
        || config.latent_dimensions == 0
        || config.latent_dimensions >= config.input_dimensions
        || config.epochs == 0
        || config.epochs > MAX_EPOCHS
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
        || config.recall_k == 0
    {
        return Err(CodecError::InvalidConfig);
    }
    if let Some(count) = record_count
        && config.recall_k >= count
    {
        return Err(CodecError::InvalidConfig);
    }
    Ok(())
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

const fn map_autoencoder_error(error: AutoencoderError) -> CodecError {
    match error {
        AutoencoderError::InvalidDimensions | AutoencoderError::DimensionMismatch => {
            CodecError::InvalidDimensions
        }
        AutoencoderError::NonFinite => CodecError::NonFinite,
        AutoencoderError::InvalidParameters => CodecError::InvalidArtifact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RecordId;

    const V1_FIXTURE_MAX_MSE: f32 = 0.0011;
    const V1_FIXTURE_MIN_RECALL_AT_3: f32 = 0.87;
    const V1_FIXTURE_SOURCE_BYTES: usize = 576;
    const V1_FIXTURE_PERSISTED_BYTES: usize = 708;

    #[allow(clippy::cast_precision_loss)]
    fn fixture() -> (Vec<[f32; 6]>, Vec<RecordId>) {
        let vectors = (0..24)
            .map(|index| {
                let first = (index as f32 - 12.0) / 12.0;
                let second = ((index * 7) % 23) as f32 / 11.5 - 1.0;
                [
                    first,
                    second,
                    first * 0.7 + second * 0.2,
                    first * -0.3 + second * 0.8,
                    first * first * 0.2,
                    second * second * -0.15,
                ]
            })
            .collect();
        let ids = (0..24).map(|id| RecordId::Legacy(id + 1)).collect();
        (vectors, ids)
    }

    fn config() -> AutoencoderConfigV1 {
        AutoencoderConfigV1 {
            input_dimensions: 6,
            latent_dimensions: 3,
            seed: 0xA070_EC0D,
            epochs: 600,
            learning_rate: 0.08,
            recall_k: 3,
        }
    }

    #[test]
    fn trained_artifact_round_trips_deterministically() {
        let (vectors, ids) = fixture();
        let records = ids
            .iter()
            .zip(&vectors)
            .map(|(id, values)| CodecRecord { id: *id, values })
            .collect::<Vec<_>>();
        let trained = train_autoencoder_v1(&records, config()).unwrap();
        let again = train_autoencoder_v1(&records, config()).unwrap();
        assert_eq!(trained.artifact(), again.artifact());
        assert!(trained.artifact().metrics.reconstruction_mse <= V1_FIXTURE_MAX_MSE);
        assert_eq!(
            trained.artifact().accounted_persisted_bytes().unwrap(),
            V1_FIXTURE_PERSISTED_BYTES
        );
        assert_eq!(
            trained.artifact().metrics.compression_ratio.to_bits(),
            representation_ratio(V1_FIXTURE_SOURCE_BYTES, V1_FIXTURE_PERSISTED_BYTES).to_bits()
        );
        assert!(trained.artifact().metrics.recall_at_k >= V1_FIXTURE_MIN_RECALL_AT_3);
        let before = trained.reconstruct(&vectors[7]).unwrap();
        let json = trained.artifact().to_json().unwrap();
        let parsed = AutoencoderArtifactV1::from_json(&json).unwrap();
        assert_eq!(json, parsed.to_json().unwrap());
        let loaded = PersistedAutoencoderV1::from_artifact(parsed).unwrap();
        let after = loaded.reconstruct(&vectors[7]).unwrap();
        assert_eq!(
            before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            after
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn artifact_shape_bounds_and_finite_checks_fail_closed() {
        let (vectors, ids) = fixture();
        let records = ids
            .iter()
            .zip(&vectors)
            .map(|(id, values)| CodecRecord { id: *id, values })
            .collect::<Vec<_>>();
        let trained = train_autoencoder_v1(&records, config()).unwrap();
        let mut bad = trained.artifact().clone();
        bad.encoder_weights.pop();
        assert_eq!(bad.validate(), Err(CodecError::InvalidArtifact));
        let mut bad = trained.artifact().clone();
        bad.normalization.scale[0] = 0.0;
        assert_eq!(bad.validate(), Err(CodecError::InvalidArtifact));
        let mut bad = trained.artifact().clone();
        bad.decoder_bias[0] = f32::INFINITY;
        assert_eq!(bad.validate(), Err(CodecError::NonFinite));
        let mut bad = trained.artifact().clone();
        bad.metrics.compression_ratio = 2.0;
        assert_eq!(bad.validate(), Err(CodecError::InvalidArtifact));
        assert_eq!(
            trained.reconstruct(&[0.0; 5]),
            Err(CodecError::InvalidDimensions)
        );
    }
}
