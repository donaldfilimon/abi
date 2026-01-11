//! Tokenizer module for text-to-token conversion.
//!
//! Implements Byte-Pair Encoding (BPE) tokenization compatible with
//! LLaMA and other modern language models.

const std = @import("std");

pub const bpe = @import("bpe.zig");
pub const vocab = @import("vocab.zig");
pub const special_tokens = @import("special_tokens.zig");

// Re-exports
pub const BpeTokenizer = bpe.BpeTokenizer;
pub const Vocab = vocab.Vocab;
pub const SpecialTokens = special_tokens.SpecialTokens;
pub const TokenizerError = bpe.TokenizerError;

/// Quick encode helper.
pub fn encode(allocator: std.mem.Allocator, tokenizer: *BpeTokenizer, text: []const u8) ![]u32 {
    return tokenizer.encode(allocator, text);
}

/// Quick decode helper.
pub fn decode(allocator: std.mem.Allocator, tokenizer: *BpeTokenizer, tokens: []const u32) ![]u8 {
    return tokenizer.decode(allocator, tokens);
}

test "tokenizer module imports" {
    _ = bpe;
    _ = vocab;
    _ = special_tokens;
}
