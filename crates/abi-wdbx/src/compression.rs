//! Reference int8-style affine quantization for embedding vectors.

/// Quantized vector with a per-vector affine range.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    /// Minimum source value.
    pub min: f32,
    /// Quantization step, or zero for a constant vector.
    pub scale: f32,
    /// One code per source component.
    pub codes: Vec<u8>,
}

impl Quantized {
    /// Raw `f32` bytes divided by codes plus the two-`f32` header.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        let raw = (self.codes.len() * size_of::<f32>()) as f32;
        let packed = (self.codes.len() + 2 * size_of::<f32>()) as f32;
        raw / packed
    }
}

/// Quantization failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionError {
    /// An affine range cannot be derived from an empty vector.
    EmptyVector,
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("cannot quantize an empty vector")
    }
}

impl std::error::Error for CompressionError {}

/// Quantize one vector into 256 affine levels.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn quantize(vector: &[f32]) -> Result<Quantized, CompressionError> {
    let Some(&first) = vector.first() else {
        return Err(CompressionError::EmptyVector);
    };
    let mut min = first;
    let mut max = first;
    for &value in vector {
        min = min.min(value);
        max = max.max(value);
    }
    let span = max - min;
    let scale = if span == 0.0 { 0.0 } else { span / 255.0 };
    let codes = vector
        .iter()
        .map(|&value| {
            if scale == 0.0 {
                0
            } else {
                ((value - min) / scale).round().clamp(0.0, 255.0) as u8
            }
        })
        .collect();
    Ok(Quantized { min, scale, codes })
}

/// Reconstruct a quantized vector.
#[must_use]
pub fn dequantize(quantized: &Quantized) -> Vec<f32> {
    quantized
        .codes
        .iter()
        .map(|&code| quantized.min + f32::from(code) * quantized.scale)
        .collect()
}

/// Maximum absolute reconstruction error over corresponding components.
#[must_use]
pub fn max_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    original
        .iter()
        .zip(reconstructed)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realistic_vector_round_trip_stays_within_one_step() {
        let vector: Vec<_> = (0_u16..128)
            .map(|index| (f32::from(index) * 0.1).sin())
            .collect();
        let quantized = quantize(&vector).expect("quantize");
        let reconstructed = dequantize(&quantized);
        assert!(max_error(&vector, &reconstructed) <= quantized.scale + 1e-6);
        assert!(quantized.compression_ratio() > 3.0);
    }

    #[test]
    fn constant_vector_is_lossless_and_empty_is_rejected() {
        let vector = [0.5; 4];
        let quantized = quantize(&vector).expect("quantize");
        assert!(quantized.scale.abs() < f32::EPSILON);
        assert!(max_error(&vector, &dequantize(&quantized)) < f32::EPSILON);
        assert_eq!(quantize(&[]), Err(CompressionError::EmptyVector));
    }

    #[test]
    fn quantization_is_deterministic() {
        let vector = [-2.0, -0.5, 0.0, 1.25, 9.0];
        assert_eq!(quantize(&vector), quantize(&vector));
    }
}
