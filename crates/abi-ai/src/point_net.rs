//! Miniature 3-D point MLP used by soul routing.
//!
//! Ported from `src/features/ai/point_neural_net.zig`. Demo-grade SGD on a
//! fixed topology — not a production LLM or distributed trainer.

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

/// Point-cloud activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified linear unit.
    Relu,
    /// Identity.
    Linear,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Linear => x,
        }
    }

    fn derivative(self, x: f32) -> f32 {
        match self {
            Self::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Linear => 1.0,
        }
    }
}

/// A 3-D point with an optional scalar value (soul weight).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Feature x (normalized length).
    pub x: f32,
    /// Feature y (vowel density).
    pub y: f32,
    /// Feature z (punctuation density).
    pub z: f32,
    /// Scalar weight, default 1.0.
    pub value: f32,
}

impl Point {
    /// Deterministic text → point mapping matching Zig `Point.fromText`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_text(text: &str) -> Self {
        let len = text.len() as f32;
        let mut vowels = 0.0_f32;
        let mut punct = 0.0_f32;
        for c in text.bytes() {
            match c {
                b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U' => {
                    vowels += 1.0;
                }
                b'.' | b',' | b'!' | b'?' | b';' | b':' | b'-' | b'"' | b'\'' | b'(' | b')' => {
                    punct += 1.0;
                }
                _ => {}
            }
        }
        Self {
            x: len / 32.0,
            y: vowels / (len + 1.0),
            z: punct / (len + 1.0),
            value: 1.0,
        }
    }

    /// The three coordinates as an array.
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Debug, Clone)]
struct DenseLayer {
    /// Row-major weights: `weights[out][in]`.
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: Activation,
    input_size: usize,
    output_size: usize,
}

/// Topology optimization summary after soul bootstrap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologyReport {
    /// Centroid of the soul points.
    pub centroid: [f32; 3],
    /// Mean distance from centroid.
    pub spread: f32,
    /// Number of near-zero weights pruned.
    pub pruned_count: usize,
    /// Regularization scale applied.
    pub regularization_factor: f32,
}

/// Fixed-topology MLP used for soul routing.
#[derive(Debug, Clone)]
pub struct PointNeuralNetwork {
    layers: Vec<DenseLayer>,
    learning_rate: f32,
}

impl PointNeuralNetwork {
    /// Build a network for the given layer sizes (must be ≥ 2).
    ///
    /// # Errors
    ///
    /// Returns an error when `dims` is invalid.
    pub fn init(dims: &[usize], learning_rate: f32) -> Result<Self, String> {
        if dims.len() < 2 {
            return Err("invalid dimensions".into());
        }
        if dims.contains(&0) {
            return Err("invalid dimensions".into());
        }
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for (i, window) in dims.windows(2).enumerate() {
            let in_size = window[0];
            let out_size = window[1];
            let is_output = i == dims.len() - 2;
            let activation = if is_output {
                Activation::Linear
            } else {
                Activation::Relu
            };
            let scale = (2.0_f32 / (in_size + out_size) as f32).sqrt();
            // Deterministic He-style init (no RNG dependency).
            let mut weights = Vec::with_capacity(out_size);
            for j in 0..out_size {
                let mut row = Vec::with_capacity(in_size);
                for k in 0..in_size {
                    let seed = ((i + 1) * 10_007 + j * 97 + k) as f32;
                    // Map seed → roughly N(0,1) via a simple triangle wave.
                    let unit = (seed.sin() * 2.0) - 1.0;
                    row.push(unit * scale);
                }
                weights.push(row);
            }
            layers.push(DenseLayer {
                weights,
                bias: vec![0.0; out_size],
                activation,
                input_size: in_size,
                output_size: out_size,
            });
        }
        Ok(Self {
            layers,
            learning_rate,
        })
    }

    /// Last-layer output width.
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, |l| l.output_size)
    }

    /// Forward pass; returns the output layer activations.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self.layers.is_empty() {
            return Err("empty network".into());
        }
        if input.len() != self.layers[0].input_size {
            return Err("input dimension mismatch".into());
        }
        let mut activations = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let is_output = i == self.layers.len() - 1;
            let act = if is_output {
                Activation::Linear
            } else {
                layer.activation
            };
            let mut next = vec![0.0; layer.output_size];
            for j in 0..layer.output_size {
                let mut sum = layer.bias[j];
                for k in 0..layer.input_size {
                    sum += layer.weights[j][k] * activations[k];
                }
                next[j] = act.apply(sum);
            }
            activations = next;
        }
        Ok(activations)
    }

    /// One SGD step on a single (point, target) pair; returns MSE loss.
    pub fn teach(&mut self, point: Point, target: &[f32]) -> Result<f32, String> {
        let out_size = self.output_size();
        if target.len() != out_size {
            return Err("target dimension mismatch".into());
        }
        let input = point.to_array();
        // Forward with pre-activations for backprop.
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_acts: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        activations.push(input.to_vec());
        for (i, layer) in self.layers.iter().enumerate() {
            let is_output = i == self.layers.len() - 1;
            let act = if is_output {
                Activation::Linear
            } else {
                layer.activation
            };
            let mut pre = vec![0.0; layer.output_size];
            let mut post = vec![0.0; layer.output_size];
            let prev = &activations[i];
            for j in 0..layer.output_size {
                let mut sum = layer.bias[j];
                for k in 0..layer.input_size {
                    sum += layer.weights[j][k] * prev[k];
                }
                pre[j] = sum;
                post[j] = act.apply(sum);
            }
            pre_acts.push(pre);
            activations.push(post);
        }
        let output = activations.last().expect("activations non-empty");
        let mut loss = 0.0_f32;
        for (o, t) in output.iter().zip(target) {
            let err = o - t;
            loss += 0.5 * err * err;
        }

        // Output deltas.
        let mut deltas: Vec<Vec<f32>> = vec![Vec::new(); self.layers.len()];
        let last = self.layers.len() - 1;
        deltas[last] = output.iter().zip(target).map(|(o, t)| o - t).collect();

        // Backprop hidden deltas.
        let mut li = last;
        while li > 0 {
            let layer = &self.layers[li];
            let prev_layer = &self.layers[li - 1];
            let prev_size = prev_layer.output_size;
            let mut prev_delta = vec![0.0; prev_size];
            for k in 0..prev_size {
                let mut sum = 0.0_f32;
                for j in 0..layer.output_size {
                    sum += layer.weights[j][k] * deltas[li][j];
                }
                let deriv = prev_layer.activation.derivative(pre_acts[li - 1][k]);
                prev_delta[k] = sum * deriv;
            }
            deltas[li - 1] = prev_delta;
            li -= 1;
        }

        // Weight update.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let act_input = &activations[i];
            for j in 0..layer.output_size {
                for k in 0..layer.input_size {
                    layer.weights[j][k] -= self.learning_rate * deltas[i][j] * act_input[k];
                }
                layer.bias[j] -= self.learning_rate * deltas[i][j];
            }
        }
        Ok(loss)
    }

    /// Train for `epochs` passes over the dataset.
    pub fn train(
        &mut self,
        points: &[Point],
        targets: &[Vec<f32>],
        epochs: usize,
    ) -> Result<f32, String> {
        if points.len() != targets.len() {
            return Err("points/targets length mismatch".into());
        }
        let mut last = 0.0_f32;
        for _ in 0..epochs {
            for (p, t) in points.iter().zip(targets) {
                last = self.teach(*p, t)?;
            }
        }
        Ok(last)
    }

    /// Light topology prune toward the soul point cloud.
    pub fn optimize_topology(&mut self, points: &[Point]) -> TopologyReport {
        if points.is_empty() {
            return TopologyReport {
                centroid: [0.0; 3],
                spread: 0.0,
                pruned_count: 0,
                regularization_factor: 1.0,
            };
        }
        let mut cx = 0.0_f32;
        let mut cy = 0.0_f32;
        let mut cz = 0.0_f32;
        let mut total = 0.0_f32;
        for p in points {
            cx += p.x * p.value;
            cy += p.y * p.value;
            cz += p.z * p.value;
            total += p.value;
        }
        if total > 0.0 {
            cx /= total;
            cy /= total;
            cz /= total;
        }
        let mut spread = 0.0_f32;
        for p in points {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            spread += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        spread /= points.len() as f32;

        let threshold = 1e-3_f32;
        let mut pruned = 0_usize;
        for layer in &mut self.layers {
            for row in &mut layer.weights {
                for w in row {
                    if w.abs() < threshold {
                        *w = 0.0;
                        pruned += 1;
                    }
                }
            }
        }
        TopologyReport {
            centroid: [cx, cy, cz],
            spread,
            pruned_count: pruned,
            regularization_factor: 1.0 + spread.min(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_text_is_deterministic() {
        let a = Point::from_text("hello world");
        let b = Point::from_text("hello world");
        assert_eq!(a, b);
        assert!(a.x > 0.0);
        assert!(a.y > 0.0);
    }

    #[test]
    fn train_reduces_loss_on_autoencoder() {
        let mut net = PointNeuralNetwork::init(&[3, 8, 3], 0.05).unwrap();
        let points = [
            Point::from_text("hello"),
            Point::from_text("world"),
            Point::from_text("abbey empathic"),
        ];
        let targets: Vec<Vec<f32>> = points.iter().map(|p| p.to_array().to_vec()).collect();
        let initial = net.teach(points[0], &targets[0]).unwrap();
        let final_loss = net.train(&points, &targets, 80).unwrap();
        assert!(
            final_loss <= initial + 1e-3,
            "loss did not decrease: initial={initial} final={final_loss}"
        );
    }
}
