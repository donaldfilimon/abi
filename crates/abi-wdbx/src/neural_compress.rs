//! Small deterministic in-process autoencoder for reference vector compression.

/// Invalid autoencoder shape or buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutoencoderError {
    /// The latent dimension must be nonzero and smaller than the input.
    InvalidDimensions,
    /// A caller-provided input or output buffer has the wrong length.
    DimensionMismatch,
}

impl std::fmt::Display for AutoencoderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => formatter.write_str("invalid autoencoder dimensions"),
            Self::DimensionMismatch => formatter.write_str("autoencoder buffer dimensions differ"),
        }
    }
}

impl std::error::Error for AutoencoderError {}

#[derive(Clone, Copy)]
struct ReferenceRng {
    state: [u64; 4],
}

impl ReferenceRng {
    fn new(seed: u64) -> Self {
        let mut split_mix_state = seed;
        let mut state = [0; 4];
        for value in &mut state {
            split_mix_state = split_mix_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = split_mix_state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *value = mixed ^ (mixed >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let temporary = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_f32(&mut self) -> f32 {
        let random = self.next_u64();
        let mut leading_zeros = random.leading_zeros();
        if leading_zeros >= 41 {
            leading_zeros = 41 + self.next_u64().leading_zeros();
            if leading_zeros == 105 {
                leading_zeros += (u32::from_le_bytes(
                    self.next_u64().to_le_bytes()[..4]
                        .try_into()
                        .expect("four bytes"),
                ) | 0x7FF)
                    .leading_zeros();
            }
        }
        let mantissa =
            u32::from_le_bytes(random.to_le_bytes()[..4].try_into().expect("four bytes"))
                & 0x7F_FFFF;
        let exponent = (126 - leading_zeros) << 23;
        f32::from_bits(exponent | mantissa)
    }

    fn centered_weight(&mut self) -> f32 {
        (self.next_f32() - 0.5) * 0.2
    }
}

/// One-hidden-layer autoencoder with tanh encoder and linear decoder.
#[derive(Clone, Debug)]
pub struct Autoencoder {
    input_dim: usize,
    latent_dim: usize,
    encoder_weights: Vec<f32>,
    encoder_bias: Vec<f32>,
    decoder_weights: Vec<f32>,
    decoder_bias: Vec<f32>,
    hidden: Vec<f32>,
    reconstruction: Vec<f32>,
    reconstruction_gradient: Vec<f32>,
    hidden_pre_gradient: Vec<f32>,
}

impl Autoencoder {
    /// Create a seeded codec with small weights and zero biases.
    pub fn new(input_dim: usize, latent_dim: usize, seed: u64) -> Result<Self, AutoencoderError> {
        if latent_dim == 0 || latent_dim >= input_dim {
            return Err(AutoencoderError::InvalidDimensions);
        }
        let mut random = ReferenceRng::new(seed);
        let encoder_weights = (0..latent_dim * input_dim)
            .map(|_| random.centered_weight())
            .collect();
        let decoder_weights = (0..input_dim * latent_dim)
            .map(|_| random.centered_weight())
            .collect();
        Ok(Self {
            input_dim,
            latent_dim,
            encoder_weights,
            encoder_bias: vec![0.0; latent_dim],
            decoder_weights,
            decoder_bias: vec![0.0; input_dim],
            hidden: vec![0.0; latent_dim],
            reconstruction: vec![0.0; input_dim],
            reconstruction_gradient: vec![0.0; input_dim],
            hidden_pre_gradient: vec![0.0; latent_dim],
        })
    }

    /// Input vector dimension.
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Latent vector dimension.
    #[must_use]
    pub const fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Input floats divided by latent floats.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        self.input_dim as f32 / self.latent_dim as f32
    }

    /// Encode into a caller-provided latent buffer without allocating.
    pub fn encode_into(&self, input: &[f32], latent: &mut [f32]) -> Result<(), AutoencoderError> {
        if input.len() != self.input_dim || latent.len() != self.latent_dim {
            return Err(AutoencoderError::DimensionMismatch);
        }
        encode_raw(
            self.input_dim,
            self.latent_dim,
            &self.encoder_weights,
            &self.encoder_bias,
            input,
            latent,
        );
        Ok(())
    }

    /// Decode into a caller-provided output buffer without allocating.
    pub fn decode_into(&self, latent: &[f32], output: &mut [f32]) -> Result<(), AutoencoderError> {
        if latent.len() != self.latent_dim || output.len() != self.input_dim {
            return Err(AutoencoderError::DimensionMismatch);
        }
        decode_raw(
            self.input_dim,
            self.latent_dim,
            &self.decoder_weights,
            &self.decoder_bias,
            latent,
            output,
        );
        Ok(())
    }

    /// Mean-squared reconstruction error without updating weights.
    pub fn reconstruction_mse(&mut self, input: &[f32]) -> Result<f32, AutoencoderError> {
        if input.len() != self.input_dim {
            return Err(AutoencoderError::DimensionMismatch);
        }
        encode_raw(
            self.input_dim,
            self.latent_dim,
            &self.encoder_weights,
            &self.encoder_bias,
            input,
            &mut self.hidden,
        );
        decode_raw(
            self.input_dim,
            self.latent_dim,
            &self.decoder_weights,
            &self.decoder_bias,
            &self.hidden,
            &mut self.reconstruction,
        );
        Ok(mean_squared_error(&self.reconstruction, input))
    }

    /// Run one allocation-free SGD step, returning the pre-update MSE.
    pub fn train_step(
        &mut self,
        input: &[f32],
        learning_rate: f32,
    ) -> Result<f32, AutoencoderError> {
        if input.len() != self.input_dim {
            return Err(AutoencoderError::DimensionMismatch);
        }
        encode_raw(
            self.input_dim,
            self.latent_dim,
            &self.encoder_weights,
            &self.encoder_bias,
            input,
            &mut self.hidden,
        );
        decode_raw(
            self.input_dim,
            self.latent_dim,
            &self.decoder_weights,
            &self.decoder_bias,
            &self.hidden,
            &mut self.reconstruction,
        );

        #[allow(clippy::cast_precision_loss)]
        let dimension = self.input_dim as f32;
        let mut loss = 0.0;
        for (index, &target) in input.iter().enumerate() {
            let difference = self.reconstruction[index] - target;
            loss += difference * difference;
            self.reconstruction_gradient[index] = 2.0 * difference / dimension;
        }
        loss /= dimension;

        for latent in 0..self.latent_dim {
            let mut gradient = 0.0;
            for input_index in 0..self.input_dim {
                gradient += self.decoder_weights[input_index * self.latent_dim + latent]
                    * self.reconstruction_gradient[input_index];
            }
            self.hidden_pre_gradient[latent] =
                gradient * (1.0 - self.hidden[latent] * self.hidden[latent]);
        }

        for input_index in 0..self.input_dim {
            self.decoder_bias[input_index] -=
                learning_rate * self.reconstruction_gradient[input_index];
            for latent in 0..self.latent_dim {
                self.decoder_weights[input_index * self.latent_dim + latent] -=
                    learning_rate * self.reconstruction_gradient[input_index] * self.hidden[latent];
            }
        }
        for latent in 0..self.latent_dim {
            self.encoder_bias[latent] -= learning_rate * self.hidden_pre_gradient[latent];
            for (input_index, &value) in input.iter().enumerate() {
                self.encoder_weights[latent * self.input_dim + input_index] -=
                    learning_rate * self.hidden_pre_gradient[latent] * value;
            }
        }
        Ok(loss)
    }

    /// Train one insertion-order epoch and return average pre-update MSE.
    pub fn train_epoch(
        &mut self,
        vectors: &[&[f32]],
        learning_rate: f32,
    ) -> Result<f32, AutoencoderError> {
        if vectors.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0;
        for vector in vectors {
            total += self.train_step(vector, learning_rate)?;
        }
        #[allow(clippy::cast_precision_loss)]
        let count = vectors.len() as f32;
        Ok(total / count)
    }
}

fn encode_raw(
    input_dim: usize,
    latent_dim: usize,
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    latent: &mut [f32],
) {
    for hidden in 0..latent_dim {
        let mut sum = bias[hidden];
        for input_index in 0..input_dim {
            sum += weights[hidden * input_dim + input_index] * input[input_index];
        }
        latent[hidden] = sum.tanh();
    }
}

fn decode_raw(
    input_dim: usize,
    latent_dim: usize,
    weights: &[f32],
    bias: &[f32],
    latent: &[f32],
    output: &mut [f32],
) {
    for input_index in 0..input_dim {
        let mut sum = bias[input_index];
        for hidden in 0..latent_dim {
            sum += weights[input_index * latent_dim + hidden] * latent[hidden];
        }
        output[input_index] = sum;
    }
}

#[allow(clippy::cast_precision_loss)]
fn mean_squared_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f32>()
        / left.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn low_rank_dataset<const COUNT: usize, const DIMENSION: usize>(
        mut random: ReferenceRng,
        basis: &[[f32; DIMENSION]],
    ) -> [[f32; DIMENSION]; COUNT] {
        let mut storage = [[0.0; DIMENSION]; COUNT];
        for vector in &mut storage {
            for basis_vector in basis {
                let coefficient = random.centered_weight() * 5.0;
                for index in 0..DIMENSION {
                    vector[index] += coefficient * basis_vector[index];
                }
            }
        }
        storage
    }

    #[test]
    fn learns_to_reconstruct_low_rank_vectors() {
        let basis = [
            [0.4, -0.3, 0.1, 0.2, -0.1, 0.3, -0.2, 0.1],
            [-0.2, 0.3, 0.25, -0.15, 0.2, -0.1, 0.3, -0.25],
            [0.1, 0.1, -0.3, 0.3, 0.15, -0.2, 0.1, 0.2],
        ];
        let storage = low_rank_dataset::<24, 8>(ReferenceRng::new(0xABCD_EF01), &basis);
        let dataset: Vec<&[f32]> = storage.iter().map(<[f32; 8]>::as_slice).collect();
        let mut autoencoder = Autoencoder::new(8, 3, 0x00C0_FFEE).expect("codec");
        assert!((autoencoder.compression_ratio() - 8.0 / 3.0).abs() < 1e-6);
        let before = dataset
            .iter()
            .map(|vector| autoencoder.reconstruction_mse(vector).expect("mse"))
            .sum::<f32>()
            / 24.0;
        for _ in 0..400 {
            autoencoder.train_epoch(&dataset, 0.1).expect("epoch");
        }
        let after = dataset
            .iter()
            .map(|vector| autoencoder.reconstruction_mse(vector).expect("mse"))
            .sum::<f32>()
            / 24.0;
        assert!(after < before * 0.25);
        assert!(after < 0.01);
    }

    #[test]
    fn caller_buffers_and_dimension_validation_are_explicit() {
        let autoencoder = Autoencoder::new(6, 2, 0x1234).expect("codec");
        let input = [0.1, -0.2, 0.05, 0.0, -0.1, 0.15];
        let mut latent = [0.0; 2];
        let mut output = [0.0; 6];
        autoencoder
            .encode_into(&input, &mut latent)
            .expect("encode");
        autoencoder
            .decode_into(&latent, &mut output)
            .expect("decode");
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(
            autoencoder.encode_into(&input[..5], &mut latent),
            Err(AutoencoderError::DimensionMismatch)
        );
        assert!(matches!(
            Autoencoder::new(6, 6, 1),
            Err(AutoencoderError::InvalidDimensions)
        ));
    }

    #[test]
    fn higher_learning_rate_remains_stable() {
        let basis = [
            [0.3, -0.2, 0.1, 0.2, -0.1, 0.15],
            [-0.1, 0.25, 0.2, -0.15, 0.2, -0.1],
        ];
        let storage = low_rank_dataset::<16, 6>(ReferenceRng::new(0x5151_ABCD), &basis);
        let dataset: Vec<&[f32]> = storage.iter().map(<[f32; 6]>::as_slice).collect();
        let mut autoencoder = Autoencoder::new(6, 2, 0x2727_BEEF).expect("codec");
        let before = dataset
            .iter()
            .map(|vector| autoencoder.reconstruction_mse(vector).expect("mse"))
            .sum::<f32>();
        for _ in 0..200 {
            autoencoder.train_epoch(&dataset, 0.4).expect("epoch");
        }
        let after = dataset
            .iter()
            .map(|vector| autoencoder.reconstruction_mse(vector).expect("mse"))
            .sum::<f32>();
        assert!(after.is_finite());
        assert!(after < before);
    }

    #[test]
    fn identical_seed_and_training_order_are_deterministic() {
        let input = [0.1, -0.2, 0.05, 0.0, -0.1, 0.15, 0.2, -0.05];
        let mut left = Autoencoder::new(8, 3, 0xBEEF_1234).expect("left");
        let mut right = Autoencoder::new(8, 3, 0xBEEF_1234).expect("right");
        assert_eq!(left.encoder_weights, right.encoder_weights);
        assert_eq!(left.decoder_weights, right.decoder_weights);
        left.train_step(&input, 0.1).expect("left train");
        right.train_step(&input, 0.1).expect("right train");
        assert_eq!(left.encoder_weights, right.encoder_weights);
        assert_eq!(left.decoder_weights, right.decoder_weights);
        let mut left_latent = [0.0; 3];
        let mut right_latent = [0.0; 3];
        left.encode_into(&input, &mut left_latent).expect("encode");
        right
            .encode_into(&input, &mut right_latent)
            .expect("encode");
        assert_eq!(
            left_latent.map(f32::to_bits),
            right_latent.map(f32::to_bits)
        );
    }

    #[test]
    fn reference_generator_matches_zig_xoshiro256_sequence() {
        let mut random = ReferenceRng::new(0);
        assert_eq!(random.next_u64(), 0x5317_5D61_490B_23DF);
        assert_eq!(random.next_u64(), 0x61DA_6F3D_C380_D507);
        assert_eq!(random.next_u64(), 0x5C0F_DF91_EC9A_7BFC);
    }
}
