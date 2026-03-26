//! Shared types for the transformer module.
//!
//! Used by both mod.zig and stub.zig to prevent type drift between
//! enabled and disabled paths.

const std = @import("std");

/// Errors common to both real and stub transformer implementations.
/// Stubs extend this with `FeatureDisabled` via a separate error set union.
pub const TransformerError = error{
    InvalidConfiguration,
    EmptyInput,
    ContextTooLarge,
    OutOfMemory,
};

pub const TransformerConfig = struct {
    layers: u16 = 4,
    hidden_size: u16 = 256,
    intermediate_size: u16 = 1024,
    vocab_size: u32 = 8192,
    max_tokens: u32 = 128,
    num_heads: u16 = 8,
    seed: u64 = 0x2a9d_7d3c_b1e5_4f03,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,

    pub fn validate(self: TransformerConfig) TransformerError!void {
        if (self.layers == 0) return TransformerError.InvalidConfiguration;
        if (self.hidden_size == 0) return TransformerError.InvalidConfiguration;
        if (self.intermediate_size == 0) return TransformerError.InvalidConfiguration;
        if (self.vocab_size < 2) return TransformerError.InvalidConfiguration;
        if (self.max_tokens == 0) return TransformerError.InvalidConfiguration;
        if (self.num_heads == 0) return TransformerError.InvalidConfiguration;
        if (self.hidden_size % self.num_heads != 0) {
            return TransformerError.InvalidConfiguration;
        }
        if (self.temperature < 0 or self.temperature > 2.0) {
            return TransformerError.InvalidConfiguration;
        }
        if (self.top_p < 0 or self.top_p > 1.0) return TransformerError.InvalidConfiguration;
    }
};


test {
    std.testing.refAllDecls(@This());
}
