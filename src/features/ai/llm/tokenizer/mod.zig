//! Tokenizer module for text-to-token conversion.
//!
//! Implements tokenization algorithms compatible with LLaMA and other
//! modern language models:
//! - BPE (Byte-Pair Encoding) - GPT-style tokenization
//! - SentencePiece (Unigram) - LLaMA-style tokenization

const std = @import("std");

pub const bpe = @import("bpe.zig");
pub const sentencepiece = @import("sentencepiece.zig");
pub const vocab = @import("vocab.zig");
pub const special_tokens = @import("special_tokens.zig");

// Re-exports
pub const BpeTokenizer = bpe.BpeTokenizer;
pub const CharTokenizer = bpe.CharTokenizer;
pub const SentencePieceTokenizer = sentencepiece.SentencePieceTokenizer;
pub const Vocab = vocab.Vocab;
pub const SpecialTokens = special_tokens.SpecialTokens;
pub const TokenizerError = bpe.TokenizerError;
pub const SentencePieceError = sentencepiece.SentencePieceError;
pub const TokenType = sentencepiece.TokenType;

/// Tokenizer type enumeration for auto-detection.
pub const TokenizerKind = enum {
    bpe,
    sentencepiece,
    unknown,

    /// Detect tokenizer type from GGUF model metadata.
    /// Checks the "tokenizer.ggml.model" key.
    pub fn fromGgufModel(model_type: ?[]const u8) TokenizerKind {
        const model = model_type orelse return .unknown;

        // GPT-2 style BPE
        if (std.mem.eql(u8, model, "gpt2")) return .bpe;
        if (std.mem.eql(u8, model, "llama-bpe")) return .bpe;

        // SentencePiece/Unigram style
        if (std.mem.eql(u8, model, "llama")) return .sentencepiece;
        if (std.mem.eql(u8, model, "mistral")) return .sentencepiece;

        return .unknown;
    }
};

/// Unified tokenizer that wraps BPE or SentencePiece implementations.
/// Provides a common interface for encoding/decoding text.
pub const Tokenizer = union(TokenizerKind) {
    bpe: BpeTokenizer,
    sentencepiece: SentencePieceTokenizer,
    unknown: void,

    /// Initialize a tokenizer of the specified kind.
    pub fn init(allocator: std.mem.Allocator, kind: TokenizerKind) Tokenizer {
        return switch (kind) {
            .bpe => .{ .bpe = BpeTokenizer.init(allocator) },
            .sentencepiece => .{ .sentencepiece = SentencePieceTokenizer.init(allocator) },
            .unknown => .{ .unknown = {} },
        };
    }

    /// Deinitialize the tokenizer.
    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            .bpe => |*t| t.deinit(),
            .sentencepiece => |*t| t.deinit(),
            .unknown => {},
        }
    }

    /// Encode text to token IDs.
    pub fn encode(self: *Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.*) {
            .bpe => |*t| t.encode(allocator, text),
            .sentencepiece => |*t| t.encode(allocator, text),
            .unknown => error.VocabNotLoaded,
        };
    }

    /// Decode token IDs to text.
    pub fn decode(self: *Tokenizer, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        return switch (self.*) {
            .bpe => |*t| t.decode(allocator, tokens),
            .sentencepiece => |*t| t.decode(allocator, tokens),
            .unknown => error.VocabNotLoaded,
        };
    }

    /// Get vocabulary size.
    pub fn vocabSize(self: *const Tokenizer) u32 {
        return switch (self.*) {
            .bpe => |*t| t.vocabSize(),
            .sentencepiece => |*t| t.vocabSize(),
            .unknown => 0,
        };
    }

    /// Check if a token ID is valid.
    pub fn isValidToken(self: *const Tokenizer, token_id: u32) bool {
        return switch (self.*) {
            .bpe => |*t| t.isValidToken(token_id),
            .sentencepiece => |*t| t.isValidToken(token_id),
            .unknown => false,
        };
    }

    /// Get string representation of a token.
    pub fn tokenToString(self: *const Tokenizer, token_id: u32) ?[]const u8 {
        return switch (self.*) {
            .bpe => |*t| t.tokenToString(token_id),
            .sentencepiece => |*t| t.tokenToString(token_id),
            .unknown => null,
        };
    }

    /// Get the tokenizer kind.
    pub fn getKind(self: *const Tokenizer) TokenizerKind {
        return @as(TokenizerKind, self.*);
    }

    /// Set whether to add BOS token.
    pub fn setAddBos(self: *Tokenizer, add_bos: bool) void {
        switch (self.*) {
            .bpe => |*t| t.add_bos = add_bos,
            .sentencepiece => |*t| t.add_bos = add_bos,
            .unknown => {},
        }
    }

    /// Set whether to add EOS token.
    pub fn setAddEos(self: *Tokenizer, add_eos: bool) void {
        switch (self.*) {
            .bpe => |*t| t.add_eos = add_eos,
            .sentencepiece => |*t| t.add_eos = add_eos,
            .unknown => {},
        }
    }
};

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
    _ = sentencepiece;
    _ = vocab;
    _ = special_tokens;
}

test "tokenizer kind detection" {
    try std.testing.expectEqual(TokenizerKind.bpe, TokenizerKind.fromGgufModel("gpt2"));
    try std.testing.expectEqual(TokenizerKind.bpe, TokenizerKind.fromGgufModel("llama-bpe"));
    try std.testing.expectEqual(TokenizerKind.sentencepiece, TokenizerKind.fromGgufModel("llama"));
    try std.testing.expectEqual(TokenizerKind.sentencepiece, TokenizerKind.fromGgufModel("mistral"));
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel(null));
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel("other"));
}

test "unified tokenizer init" {
    const allocator = std.testing.allocator;

    // Test BPE variant
    var bpe_tok = Tokenizer.init(allocator, .bpe);
    defer bpe_tok.deinit();
    try std.testing.expectEqual(TokenizerKind.bpe, bpe_tok.getKind());

    // Test SentencePiece variant
    var sp_tok = Tokenizer.init(allocator, .sentencepiece);
    defer sp_tok.deinit();
    try std.testing.expectEqual(TokenizerKind.sentencepiece, sp_tok.getKind());

    // Test unknown variant
    var unk_tok = Tokenizer.init(allocator, .unknown);
    defer unk_tok.deinit();
    try std.testing.expectEqual(TokenizerKind.unknown, unk_tok.getKind());
    try std.testing.expectEqual(@as(u32, 0), unk_tok.vocabSize());
}
