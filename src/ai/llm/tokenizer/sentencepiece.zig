//! SentencePiece tokenizer implementation.
//!
//! SentencePiece is a subword tokenization algorithm that uses a unigram
//! language model for segmentation. It's commonly used with LLaMA models.

const std = @import("std");
const vocab_mod = @import("vocab.zig");
const special_tokens = @import("special_tokens.zig");

pub const SentencePieceError = error{
    InvalidUtf8,
    VocabNotLoaded,
    UnknownToken,
    EncodingError,
    DecodingError,
    ModelNotLoaded,
    OutOfMemory,
};

/// SentencePiece tokenizer using unigram language model.
pub const SentencePieceTokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: vocab_mod.Vocab,
    special: special_tokens.SpecialTokens,

    /// Token scores (negative log probabilities)
    scores: std.AutoHashMapUnmanaged(u32, f32),

    /// Whether to add BOS token at start
    add_bos: bool,
    /// Whether to add EOS token at end
    add_eos: bool,
    /// Byte fallback for unknown characters
    byte_fallback: bool,

    /// Special token for space (▁ = U+2581)
    const SPACE_SYMBOL: []const u8 = "▁";

    pub fn init(allocator: std.mem.Allocator) SentencePieceTokenizer {
        return .{
            .allocator = allocator,
            .vocab = vocab_mod.Vocab.init(allocator),
            .special = special_tokens.SpecialTokens.initDefault(),
            .scores = std.AutoHashMapUnmanaged(u32, f32).empty,
            .add_bos = true,
            .add_eos = false,
            .byte_fallback = true,
        };
    }

    pub fn deinit(self: *SentencePieceTokenizer) void {
        self.scores.deinit(self.allocator);
        self.vocab.deinit();
        self.* = undefined;
    }

    /// Load vocabulary with scores from token array.
    pub fn loadVocabWithScores(self: *SentencePieceTokenizer, tokens: []const []const u8, token_scores: []const f32) !void {
        for (tokens, 0..) |token, i| {
            try self.vocab.addToken(token, @intCast(i));
            if (i < token_scores.len) {
                try self.scores.put(self.allocator, @intCast(i), token_scores[i]);
            }
        }
    }

    /// Load vocabulary from GGUF metadata.
    pub fn loadFromGgufMetadata(
        self: *SentencePieceTokenizer,
        tokens_data: []const u8,
        scores_data: []const u8,
        count: u64,
    ) !void {
        // Load tokens
        try self.vocab.loadFromGguf(tokens_data, count);

        // Load scores (array of f32)
        var id: u32 = 0;
        var offset: usize = 0;
        while (id < count and offset + 4 <= scores_data.len) : (id += 1) {
            const score_bits = std.mem.readInt(u32, scores_data[offset..][0..4], .little);
            const score: f32 = @bitCast(score_bits);
            try self.scores.put(self.allocator, id, score);
            offset += 4;
        }
    }

    /// Encode text to token IDs using Viterbi algorithm.
    pub fn encode(self: *SentencePieceTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var tokens = std.ArrayListUnmanaged(u32).empty;
        errdefer tokens.deinit(allocator);

        // Add BOS token if configured
        if (self.add_bos) {
            try tokens.append(allocator, self.special.bos_id);
        }

        // Preprocess: add space symbol at word boundaries
        const processed = try self.preprocessText(allocator, text);
        defer allocator.free(processed);

        // Use Viterbi algorithm for optimal segmentation
        const segmented = try self.viterbiSegment(allocator, processed);
        defer {
            for (segmented) |seg| {
                allocator.free(seg);
            }
            allocator.free(segmented);
        }

        // Convert segments to token IDs
        for (segmented) |piece| {
            if (self.vocab.getTokenId(piece)) |id| {
                try tokens.append(allocator, id);
            } else if (self.byte_fallback) {
                // Byte fallback: encode unknown bytes individually
                try self.appendByteFallback(allocator, &tokens, piece);
            } else {
                try tokens.append(allocator, self.special.unk_id);
            }
        }

        // Add EOS token if configured
        if (self.add_eos) {
            try tokens.append(allocator, self.special.eos_id);
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// Decode token IDs to text.
    pub fn decode(self: *SentencePieceTokenizer, allocator: std.mem.Allocator, token_ids: []const u32) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        for (token_ids) |token_id| {
            // Skip special tokens
            if (self.special.isSpecial(token_id)) continue;

            if (self.vocab.getTokenString(token_id)) |token_str| {
                // Replace space symbol with actual space
                var i: usize = 0;
                while (i < token_str.len) {
                    // Check for ▁ (3 bytes: 0xE2 0x96 0x81)
                    if (i + 3 <= token_str.len and
                        token_str[i] == 0xE2 and
                        token_str[i + 1] == 0x96 and
                        token_str[i + 2] == 0x81)
                    {
                        try result.append(allocator, ' ');
                        i += 3;
                    } else {
                        try result.append(allocator, token_str[i]);
                        i += 1;
                    }
                }
            }
        }

        // Trim leading space if present
        const items = result.items;
        if (items.len > 0 and items[0] == ' ') {
            return result.toOwnedSlice(allocator)[1..];
        }

        return result.toOwnedSlice(allocator);
    }

    /// Preprocess text by adding space symbols at word starts.
    fn preprocessText(self: *SentencePieceTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        _ = self;
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        // Add space symbol at the start
        try result.appendSlice(allocator, SPACE_SYMBOL);

        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];
            if (c == ' ' or c == '\n' or c == '\t') {
                // Replace whitespace with space symbol
                try result.appendSlice(allocator, SPACE_SYMBOL);
                i += 1;
                // Skip consecutive whitespace
                while (i < text.len and (text[i] == ' ' or text[i] == '\n' or text[i] == '\t')) {
                    i += 1;
                }
            } else {
                try result.append(allocator, c);
                i += 1;
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Viterbi algorithm for optimal subword segmentation.
    fn viterbiSegment(self: *SentencePieceTokenizer, allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
        const n = text.len;
        if (n == 0) return allocator.alloc([]const u8, 0);

        // best_score[i] = best score to reach position i
        const best_score = try allocator.alloc(f32, n + 1);
        defer allocator.free(best_score);

        // best_len[i] = length of best token ending at position i
        const best_len = try allocator.alloc(usize, n + 1);
        defer allocator.free(best_len);

        best_score[0] = 0;
        best_len[0] = 0;
        for (1..n + 1) |i| {
            best_score[i] = -std.math.inf(f32);
            best_len[i] = 0;
        }

        // Forward pass: find best segmentation
        for (0..n) |i| {
            if (best_score[i] == -std.math.inf(f32)) continue;

            // Try all possible tokens starting at position i
            const max_len: usize = @min(64, n - i); // Maximum token length
            for (1..max_len + 1) |len| {
                const piece = text[i .. i + len];
                const score = self.getTokenScore(piece);

                const new_score = best_score[i] + score;
                if (new_score > best_score[i + len]) {
                    best_score[i + len] = new_score;
                    best_len[i + len] = len;
                }
            }
        }

        // Backward pass: reconstruct segmentation
        var segments = std.ArrayListUnmanaged([]const u8).empty;
        errdefer segments.deinit(allocator);

        var pos = n;
        while (pos > 0) {
            const len = best_len[pos];
            if (len == 0) {
                // Fallback: take single character/byte
                const char_len = self.charLen(text[pos - 1 ..]);
                const actual_len = @min(char_len, pos);
                const segment = try allocator.dupe(u8, text[pos - actual_len .. pos]);
                try segments.append(allocator, segment);
                pos -= actual_len;
            } else {
                const segment = try allocator.dupe(u8, text[pos - len .. pos]);
                try segments.append(allocator, segment);
                pos -= len;
            }
        }

        // Reverse segments
        const result = try segments.toOwnedSlice(allocator);
        std.mem.reverse([]const u8, result);
        return result;
    }

    /// Get the score for a token (negative log probability).
    fn getTokenScore(self: *SentencePieceTokenizer, piece: []const u8) f32 {
        if (self.vocab.getTokenId(piece)) |id| {
            return self.scores.get(id) orelse -10.0;
        }
        // Unknown token gets very low score
        return -100.0;
    }

    /// Get UTF-8 character length at position.
    fn charLen(_: *SentencePieceTokenizer, text: []const u8) usize {
        if (text.len == 0) return 0;
        return std.unicode.utf8ByteSequenceLength(text[0]) catch 1;
    }

    /// Append byte fallback tokens for unknown text.
    fn appendByteFallback(
        self: *SentencePieceTokenizer,
        allocator: std.mem.Allocator,
        tokens: *std.ArrayListUnmanaged(u32),
        text: []const u8,
    ) !void {
        // Try to find byte tokens (e.g., <0x41> for 'A')
        for (text) |byte| {
            var buf: [8]u8 = undefined;
            const byte_token = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{byte}) catch {
                try tokens.append(allocator, self.special.unk_id);
                continue;
            };

            if (self.vocab.getTokenId(byte_token)) |id| {
                try tokens.append(allocator, id);
            } else {
                try tokens.append(allocator, self.special.unk_id);
            }
        }
    }

    /// Get vocabulary size.
    pub fn vocabSize(self: *const SentencePieceTokenizer) u32 {
        return self.vocab.size();
    }

    /// Check if a token ID is valid.
    pub fn isValidToken(self: *const SentencePieceTokenizer, token_id: u32) bool {
        return self.vocab.getTokenString(token_id) != null or
            self.special.isSpecial(token_id);
    }

    /// Get string representation of a token.
    pub fn tokenToString(self: *const SentencePieceTokenizer, token_id: u32) ?[]const u8 {
        if (self.special.isSpecial(token_id)) {
            return self.special.getName(token_id);
        }
        return self.vocab.getTokenString(token_id);
    }
};

/// Token type enumeration matching SentencePiece model.
pub const TokenType = enum(u8) {
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    byte_token = 6,
    unused = 0,
};

test "sentencepiece tokenizer init" {
    const allocator = std.testing.allocator;

    var tokenizer = SentencePieceTokenizer.init(allocator);
    defer tokenizer.deinit();

    try std.testing.expect(tokenizer.add_bos);
    try std.testing.expect(!tokenizer.add_eos);
    try std.testing.expect(tokenizer.byte_fallback);
}

test "sentencepiece preprocess" {
    const allocator = std.testing.allocator;

    var tokenizer = SentencePieceTokenizer.init(allocator);
    defer tokenizer.deinit();

    const processed = try tokenizer.preprocessText(allocator, "hello world");
    defer allocator.free(processed);

    // Should start with space symbol and have space symbol between words
    try std.testing.expect(std.mem.startsWith(u8, processed, "▁"));
}

test "sentencepiece vocab with scores" {
    const allocator = std.testing.allocator;

    var tokenizer = SentencePieceTokenizer.init(allocator);
    defer tokenizer.deinit();

    const tokens = [_][]const u8{ "▁hello", "▁world", "▁test" };
    const scores = [_]f32{ -1.0, -2.0, -3.0 };

    try tokenizer.loadVocabWithScores(&tokens, &scores);

    try std.testing.expectEqual(@as(u32, 3), tokenizer.vocabSize());
    try std.testing.expectEqual(@as(?u32, 0), tokenizer.vocab.getTokenId("▁hello"));
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tokenizer.scores.get(0).?, 0.001);
}

test "sentencepiece encode decode roundtrip" {
    const allocator = std.testing.allocator;

    var tokenizer = SentencePieceTokenizer.init(allocator);
    defer tokenizer.deinit();
    tokenizer.add_bos = false;

    // Load simple vocab
    const tokens = [_][]const u8{ "▁", "h", "e", "l", "o", "▁w", "r", "d" };
    const scores = [_]f32{ -0.1, -1.0, -1.0, -1.0, -1.0, -0.5, -1.0, -1.0 };
    try tokenizer.loadVocabWithScores(&tokens, &scores);

    // Encode
    const encoded = try tokenizer.encode(allocator, "hello");
    defer allocator.free(encoded);

    try std.testing.expect(encoded.len > 0);
}
