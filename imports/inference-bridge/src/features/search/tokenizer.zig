//! Tokenizer
//!
//! Text tokenization with lowercase normalization and stop word filtering.

const std = @import("std");

pub const stop_words = [_][]const u8{
    "a",     "an",    "the", "and",  "or",   "but",  "in",   "on",    "at",    "to",     "for",
    "of",    "with",  "by",  "from", "is",   "are",  "was",  "were",  "be",    "been",   "being",
    "have",  "has",   "had", "do",   "does", "did",  "will", "would", "could", "should", "may",
    "might", "shall", "can", "it",   "its",  "this", "that", "these", "those", "i",      "we",
    "you",   "he",    "she", "they", "not",
};

const stop_word_set = std.StaticStringMap(void).initComptime(.{
    .{ "a", {} },     .{ "an", {} },    .{ "the", {} },   .{ "and", {} },
    .{ "or", {} },    .{ "but", {} },   .{ "in", {} },    .{ "on", {} },
    .{ "at", {} },    .{ "to", {} },    .{ "for", {} },   .{ "of", {} },
    .{ "with", {} },  .{ "by", {} },    .{ "from", {} },  .{ "is", {} },
    .{ "are", {} },   .{ "was", {} },   .{ "were", {} },  .{ "be", {} },
    .{ "been", {} },  .{ "being", {} }, .{ "have", {} },  .{ "has", {} },
    .{ "had", {} },   .{ "do", {} },    .{ "does", {} },  .{ "did", {} },
    .{ "will", {} },  .{ "would", {} }, .{ "could", {} }, .{ "should", {} },
    .{ "may", {} },   .{ "might", {} }, .{ "shall", {} }, .{ "can", {} },
    .{ "it", {} },    .{ "its", {} },   .{ "this", {} },  .{ "that", {} },
    .{ "these", {} }, .{ "those", {} }, .{ "i", {} },     .{ "we", {} },
    .{ "you", {} },   .{ "he", {} },    .{ "she", {} },   .{ "they", {} },
    .{ "not", {} },
});

pub fn isStopWord(word: []const u8) bool {
    return stop_word_set.has(word);
}

pub fn tokenize(
    allocator: std.mem.Allocator,
    text: []const u8,
    filter_stops: bool,
) !std.ArrayListUnmanaged([]u8) {
    var tokens: std.ArrayListUnmanaged([]u8) = .empty;
    errdefer {
        for (tokens.items) |t| allocator.free(t);
        tokens.deinit(allocator);
    }

    var i: usize = 0;
    while (i < text.len) {
        // Skip non-alpha
        while (i < text.len and !std.ascii.isAlphabetic(text[i]) and !std.ascii.isDigit(text[i])) : (i += 1) {}
        if (i >= text.len) break;

        const start = i;
        while (i < text.len and (std.ascii.isAlphabetic(text[i]) or std.ascii.isDigit(text[i]))) : (i += 1) {}

        const word = text[start..i];
        if (word.len == 0 or word.len > 100) continue;

        // Lowercase into stack buffer first to check stop words without allocation
        var lower_buf: [100]u8 = undefined;
        for (word, 0..) |c, j| {
            lower_buf[j] = std.ascii.toLower(c);
        }
        const lower_word = lower_buf[0..word.len];

        if (filter_stops and isStopWord(lower_word)) continue;

        // Only allocate for non-stop words
        const lower = try allocator.dupe(u8, lower_word);
        try tokens.append(allocator, lower);
    }

    return tokens;
}
