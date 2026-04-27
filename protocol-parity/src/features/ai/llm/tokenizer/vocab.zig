//! Vocabulary management for tokenizers.
//!
//! Handles bidirectional mapping between token strings and IDs.

const std = @import("std");

/// Vocabulary for token-to-ID and ID-to-token mapping.
pub const Vocab = struct {
    allocator: std.mem.Allocator,

    /// Token string -> ID mapping
    token_to_id: std.StringHashMapUnmanaged(u32),

    /// ID -> Token string mapping (uses same strings as token_to_id)
    id_to_token: std.AutoHashMapUnmanaged(u32, []const u8),

    /// Total vocabulary size
    vocab_size: u32,

    pub fn init(allocator: std.mem.Allocator) Vocab {
        return .{
            .allocator = allocator,
            .token_to_id = std.StringHashMapUnmanaged(u32).empty,
            .id_to_token = std.AutoHashMapUnmanaged(u32, []const u8).empty,
            .vocab_size = 0,
        };
    }

    pub fn deinit(self: *Vocab) void {
        // Free owned token strings
        var iter = self.id_to_token.valueIterator();
        while (iter.next()) |token| {
            self.allocator.free(token.*);
        }
        self.token_to_id.deinit(self.allocator);
        self.id_to_token.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a token to the vocabulary.
    pub fn addToken(self: *Vocab, token: []const u8, id: u32) !void {
        // Duplicate the token string for ownership
        const owned = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(owned);

        try self.token_to_id.put(self.allocator, owned, id);
        try self.id_to_token.put(self.allocator, id, owned);

        if (id >= self.vocab_size) {
            self.vocab_size = id + 1;
        }
    }

    /// Get token ID for a string.
    pub fn getTokenId(self: *const Vocab, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    /// Get token string for an ID.
    pub fn getTokenString(self: *const Vocab, id: u32) ?[]const u8 {
        return self.id_to_token.get(id);
    }

    /// Get vocabulary size.
    pub fn size(self: *const Vocab) u32 {
        return self.vocab_size;
    }

    /// Check if a token exists.
    pub fn hasToken(self: *const Vocab, token: []const u8) bool {
        return self.token_to_id.contains(token);
    }

    /// Check if an ID is valid.
    pub fn hasId(self: *const Vocab, id: u32) bool {
        return self.id_to_token.contains(id);
    }

    /// Load vocabulary from GGUF metadata array.
    pub fn loadFromGguf(self: *Vocab, tokens_data: []const u8, count: u64) !void {
        // GGUF stores tokens as length-prefixed strings in the array
        var offset: usize = 0;
        var id: u32 = 0;

        while (id < count and offset < tokens_data.len) : (id += 1) {
            // Read string length (u64)
            if (offset + 8 > tokens_data.len) break;
            const len = std.mem.readInt(u64, tokens_data[offset..][0..8], .little);
            offset += 8;

            // Read string data
            if (offset + len > tokens_data.len) break;
            const token = tokens_data[offset .. offset + @as(usize, @intCast(len))];
            offset += @intCast(len);

            try self.addToken(token, id);
        }
    }

    /// Create a simple byte-level vocabulary (256 tokens for raw bytes).
    pub fn initByteLevelVocab(allocator: std.mem.Allocator) !Vocab {
        var self = Vocab.init(allocator);
        errdefer self.deinit();

        // Add single-byte tokens
        for (0..256) |byte| {
            var buf: [1]u8 = .{@intCast(byte)};
            try self.addToken(&buf, @intCast(byte));
        }

        return self;
    }
};

/// Token score information (for sorting by frequency).
pub const TokenScore = struct {
    token: []const u8,
    id: u32,
    score: f32,
};

/// Compare tokens by score (descending).
pub fn compareByScore(_: void, a: TokenScore, b: TokenScore) bool {
    return a.score > b.score;
}

test "vocab basic operations" {
    const allocator = std.testing.allocator;

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    try vocab.addToken("hello", 0);
    try vocab.addToken("world", 1);

    try std.testing.expectEqual(@as(?u32, 0), vocab.getTokenId("hello"));
    try std.testing.expectEqual(@as(?u32, 1), vocab.getTokenId("world"));
    try std.testing.expectEqual(@as(?u32, null), vocab.getTokenId("foo"));

    try std.testing.expectEqualStrings("hello", vocab.getTokenString(0).?);
    try std.testing.expectEqualStrings("world", vocab.getTokenString(1).?);

    try std.testing.expectEqual(@as(u32, 2), vocab.size());
}

test "vocab byte level" {
    const allocator = std.testing.allocator;

    var vocab = try Vocab.initByteLevelVocab(allocator);
    defer vocab.deinit();

    try std.testing.expectEqual(@as(u32, 256), vocab.size());
    try std.testing.expect(vocab.hasId(0));
    try std.testing.expect(vocab.hasId(255));
}

test {
    std.testing.refAllDecls(@This());
}
