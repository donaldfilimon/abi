//! Configuration for trainable models.

const std = @import("std");

/// Gradient checkpointing strategy.
pub const CheckpointingStrategy = enum {
    /// No checkpointing - store all activations
    none,
    /// Checkpoint every N layers (trades memory for compute)
    every_n_layers,
    /// Only checkpoint attention (most memory-heavy)
    attention_only,
    /// Full checkpointing - recompute everything
    full,
};

/// Configuration for a trainable model.
pub const TrainableModelConfig = struct {
    /// Model dimension (hidden size)
    hidden_dim: u32,
    /// Number of transformer layers
    num_layers: u32,
    /// Number of attention heads
    num_heads: u32,
    /// Number of key-value heads (for GQA)
    num_kv_heads: u32,
    /// Intermediate dimension for FFN
    intermediate_dim: u32,
    /// Vocabulary size
    vocab_size: u32,
    /// Maximum sequence length
    max_seq_len: u32 = 2048,
    /// RoPE theta base
    rope_theta: f32 = 10000.0,
    /// RMSNorm epsilon
    norm_eps: f32 = 1e-5,
    /// Whether to use tied embeddings
    tie_embeddings: bool = true,
    /// Gradient checkpointing strategy
    checkpointing: CheckpointingStrategy = .none,
    /// Checkpoint interval (for every_n_layers)
    checkpoint_interval: u32 = 4,

    /// Compute head dimension.
    pub fn headDim(self: TrainableModelConfig) u32 {
        return self.hidden_dim / self.num_heads;
    }

    /// Compute total number of parameters.
    pub fn numParams(self: TrainableModelConfig) usize {
        const head_dim = self.headDim();
        const kv_dim = self.num_kv_heads * head_dim;

        var total: usize = 0;

        // Token embedding
        total += self.vocab_size * self.hidden_dim;

        // Per-layer weights
        const per_layer: usize =
            // Attention: Q, K, V projections
            self.hidden_dim * self.hidden_dim + // W_q
            self.hidden_dim * kv_dim + // W_k
            self.hidden_dim * kv_dim + // W_v
            self.hidden_dim * self.hidden_dim + // W_o
            // Attention norm
            self.hidden_dim +
            // FFN: gate, up, down
            self.hidden_dim * self.intermediate_dim + // gate
            self.hidden_dim * self.intermediate_dim + // up
            self.intermediate_dim * self.hidden_dim + // down
            // FFN norm
            self.hidden_dim;

        total += per_layer * self.num_layers;

        // Final norm
        total += self.hidden_dim;

        // Output projection (if not tied)
        if (!self.tie_embeddings) {
            total += self.hidden_dim * self.vocab_size;
        }

        return total;
    }
};
