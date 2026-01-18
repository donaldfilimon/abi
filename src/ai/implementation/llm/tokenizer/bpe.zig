//! Byte-Pair Encoding (BPE) tokenizer implementation.
//!
//! BPE is a subword tokenization algorithm that iteratively merges
//! the most frequent pairs of tokens until a vocabulary size is reached.

const std = @import("std");
const vocab_mod = @import("vocab.zig");
const special_tokens = @import("special_tokens.zig");

pub const TokenizerError = error{
    InvalidUtf8,
    VocabNotLoaded,
    UnknownToken,
    EncodingError,
    DecodingError,
    OutOfMemory,
};

/// BPE tokenizer for encoding text to tokens and decoding tokens to text.
pub const BpeTokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: vocab_mod.Vocab,
    special: special_tokens.SpecialTokens,

    /// Merge priority scores (lower = higher priority)
    merge_ranks: std.StringHashMapUnmanaged(u32),

    /// Whether to add BOS token at start
    add_bos: bool,
    /// Whether to add EOS token at end
    add_eos: bool,

    pub fn init(allocator: std.mem.Allocator) BpeTokenizer {
        return .{
            .allocator = allocator,
            .vocab = vocab_mod.Vocab.init(allocator),
            .special = special_tokens.SpecialTokens.initDefault(),
            .merge_ranks = std.StringHashMapUnmanaged(u32).empty,
            .add_bos = true,
            .add_eos = false,
        };
    }

    pub fn deinit(self: *BpeTokenizer) void {
        // Free merge rank keys
        var iter = self.merge_ranks.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.merge_ranks.deinit(self.allocator);
        self.vocab.deinit();
        self.* = undefined;
    }

    /// Load vocabulary from a list of token strings.
    pub fn loadVocab(self: *BpeTokenizer, tokens: []const []const u8) !void {
        for (tokens, 0..) |token, i| {
            try self.vocab.addToken(token, @intCast(i));
        }
    }

    /// Load merge rules from pairs with priorities.
    pub fn loadMerges(self: *BpeTokenizer, merges: []const [2][]const u8) !void {
        for (merges, 0..) |merge, rank| {
            // Create merged key "a b"
            const key = try std.fmt.allocPrint(self.allocator, "{s} {s}", .{ merge[0], merge[1] });
            try self.merge_ranks.put(self.allocator, key, @intCast(rank));
        }
    }

    /// Encode text to token IDs.
    pub fn encode(self: *BpeTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var tokens = std.ArrayListUnmanaged(u32).empty;
        errdefer tokens.deinit(allocator);

        // Add BOS token if configured
        if (self.add_bos) {
            try tokens.append(allocator, self.special.bos_id);
        }

        // Split text into initial character tokens
        var chars = std.ArrayListUnmanaged([]const u8).empty;
        defer chars.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            const char_len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            const char_slice = text[i..@min(i + char_len, text.len)];
            try chars.append(allocator, char_slice);
            i += char_len;
        }

        // Apply BPE merges
        var working = try allocator.alloc([]const u8, chars.items.len);
        defer allocator.free(working);
        @memcpy(working, chars.items);

        var working_len = chars.items.len;

        while (working_len > 1) {
            // Find best merge
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;

            var j: usize = 0;
            while (j < working_len - 1) : (j += 1) {
                const key = std.fmt.allocPrint(allocator, "{s} {s}", .{ working[j], working[j + 1] }) catch continue;
                defer allocator.free(key);

                if (self.merge_ranks.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = j;
                    }
                }
            }

            // No more merges possible
            if (best_idx == null) break;

            // Apply merge at best_idx
            const idx = best_idx.?;
            const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{ working[idx], working[idx + 1] });

            // Shift remaining items
            working[idx] = merged;
            for (idx + 1..working_len - 1) |k| {
                working[k] = working[k + 1];
            }
            working_len -= 1;
        }

        // Convert to token IDs
        for (working[0..working_len]) |piece| {
            if (self.vocab.getTokenId(piece)) |id| {
                try tokens.append(allocator, id);
            } else {
                // Unknown token - try byte fallback or use UNK
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
    pub fn decode(self: *BpeTokenizer, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        for (tokens) |token_id| {
            // Skip special tokens in output
            if (self.special.isSpecial(token_id)) continue;

            if (self.vocab.getTokenString(token_id)) |token_str| {
                // Handle sentencepiece-style space encoding
                for (token_str) |c| {
                    if (c == '\xe2') {
                        // Check for "▁" (U+2581) - sentencepiece space marker
                        continue;
                    }
                    try result.append(allocator, c);
                }
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get vocabulary size.
    pub fn vocabSize(self: *const BpeTokenizer) u32 {
        return self.vocab.size();
    }

    /// Check if a token ID is valid.
    pub fn isValidToken(self: *const BpeTokenizer, token_id: u32) bool {
        return self.vocab.getTokenString(token_id) != null or
            self.special.isSpecial(token_id);
    }

    /// Get string representation of a token.
    pub fn tokenToString(self: *const BpeTokenizer, token_id: u32) ?[]const u8 {
        if (self.special.isSpecial(token_id)) {
            return self.special.getName(token_id);
        }
        return self.vocab.getTokenString(token_id);
    }
};

/// Simple character-level tokenizer for fallback.
pub const CharTokenizer = struct {
    allocator: std.mem.Allocator,
    char_to_id: std.AutoHashMapUnmanaged(u21, u32),
    id_to_char: std.AutoHashMapUnmanaged(u32, u21),
    next_id: u32,

    pub fn init(allocator: std.mem.Allocator) CharTokenizer {
        return .{
            .allocator = allocator,
            .char_to_id = std.AutoHashMapUnmanaged(u21, u32).empty,
            .id_to_char = std.AutoHashMapUnmanaged(u32, u21).empty,
            .next_id = 0,
        };
    }

    pub fn deinit(self: *CharTokenizer) void {
        self.char_to_id.deinit(self.allocator);
        self.id_to_char.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn encode(self: *CharTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var tokens = std.ArrayListUnmanaged(u32).empty;
        errdefer tokens.deinit(allocator);

        var iter = std.unicode.Utf8View.initUnchecked(text).iterator();
        while (iter.nextCodepoint()) |cp| {
            const id = self.getOrCreateId(cp);
            try tokens.append(allocator, id);
        }

        return tokens.toOwnedSlice(allocator);
    }

    pub fn decode(self: *CharTokenizer, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        for (tokens) |id| {
            if (self.id_to_char.get(id)) |cp| {
                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(cp, &buf) catch continue;
                try result.appendSlice(allocator, buf[0..len]);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    fn getOrCreateId(self: *CharTokenizer, codepoint: u21) u32 {
        if (self.char_to_id.get(codepoint)) |id| {
            return id;
        }

        const id = self.next_id;
        self.next_id += 1;

        self.char_to_id.put(self.allocator, codepoint, id) catch return 0;
        self.id_to_char.put(self.allocator, id, codepoint) catch return 0;

        return id;
    }
};

test "char tokenizer roundtrip" {
    const allocator = std.testing.allocator;

    var tokenizer = CharTokenizer.init(allocator);
    defer tokenizer.deinit();

    const text = "Hello, world!";
    const tokens = try tokenizer.encode(allocator, text);
    defer allocator.free(tokens);

    const decoded = try tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(text, decoded);
}

test "bpe tokenizer init" {
    const allocator = std.testing.allocator;

    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try std.testing.expect(tokenizer.add_bos);
    try std.testing.expect(!tokenizer.add_eos);
}
