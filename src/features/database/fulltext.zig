//! Full-text search with BM25 ranking.
//!
//! Provides inverted index-based full-text search with BM25 scoring
//! for relevance ranking.

const std = @import("std");

/// BM25 configuration parameters.
pub const Bm25Config = struct {
    /// Term frequency saturation parameter.
    k1: f32 = 1.2,
    /// Document length normalization parameter.
    b: f32 = 0.75,
    /// Term frequency boost for title field.
    title_boost: f32 = 2.0,
};

/// Text tokenizer configuration.
pub const TokenizerConfig = struct {
    /// Convert to lowercase.
    lowercase: bool = true,
    /// Remove punctuation.
    remove_punctuation: bool = true,
    /// Minimum token length.
    min_token_length: usize = 2,
    /// Maximum token length.
    max_token_length: usize = 64,
    /// Enable stemming (basic suffix removal).
    enable_stemming: bool = true,
};

/// Token information.
pub const TokenInfo = struct {
    term: []const u8,
    position: u32,
    field: []const u8,
};

/// Document in the index.
pub const IndexedDocument = struct {
    id: u64,
    field_lengths: std.StringHashMapUnmanaged(u32),
    term_frequencies: std.StringHashMapUnmanaged(u32),
};

/// Posting list entry.
pub const Posting = struct {
    doc_id: u64,
    frequency: u32,
    positions: []const u32,
};

/// Search result.
pub const TextSearchResult = struct {
    doc_id: u64,
    score: f32,
    matched_terms: []const []const u8,

    pub fn lessThan(_: void, a: TextSearchResult, b: TextSearchResult) bool {
        return a.score > b.score; // Higher score first
    }
};

/// Inverted index for full-text search.
pub const InvertedIndex = struct {
    allocator: std.mem.Allocator,
    bm25_config: Bm25Config,
    tokenizer_config: TokenizerConfig,
    postings: std.StringHashMapUnmanaged(PostingList),
    documents: std.AutoHashMapUnmanaged(u64, DocumentMeta),
    total_docs: u64,
    avg_doc_length: f32,
    total_field_lengths: u64,

    const PostingList = struct {
        doc_ids: std.ArrayListUnmanaged(u64),
        frequencies: std.ArrayListUnmanaged(u32),
    };

    const DocumentMeta = struct {
        field_length: u32,
        term_count: u32,
    };

    /// Initialize inverted index.
    pub fn init(allocator: std.mem.Allocator, bm25_config: Bm25Config, tokenizer_config: TokenizerConfig) InvertedIndex {
        return .{
            .allocator = allocator,
            .bm25_config = bm25_config,
            .tokenizer_config = tokenizer_config,
            .postings = .{},
            .documents = .{},
            .total_docs = 0,
            .avg_doc_length = 0,
            .total_field_lengths = 0,
        };
    }

    /// Deinitialize index.
    pub fn deinit(self: *InvertedIndex) void {
        var posting_iter = self.postings.iterator();
        while (posting_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.doc_ids.deinit(self.allocator);
            entry.value_ptr.frequencies.deinit(self.allocator);
        }
        self.postings.deinit(self.allocator);
        self.documents.deinit(self.allocator);
        self.* = undefined;
    }

    /// Index a document.
    pub fn indexDocument(self: *InvertedIndex, doc_id: u64, text: []const u8) !void {
        var tokens = try self.tokenize(text);
        defer {
            for (tokens.items) |token| {
                self.allocator.free(token);
            }
            tokens.deinit(self.allocator);
        }

        // Count term frequencies
        var term_freqs = std.StringHashMapUnmanaged(u32){};
        defer term_freqs.deinit(self.allocator);

        for (tokens.items) |token| {
            const entry = try term_freqs.getOrPut(self.allocator, token);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }

        // Add to postings
        var term_iter = term_freqs.iterator();
        while (term_iter.next()) |entry| {
            const term = entry.key_ptr.*;
            const freq = entry.value_ptr.*;

            // Get or create posting list
            const posting_entry = try self.postings.getOrPut(self.allocator, term);
            if (!posting_entry.found_existing) {
                const owned_term = try self.allocator.dupe(u8, term);
                posting_entry.key_ptr.* = owned_term;
                posting_entry.value_ptr.* = .{
                    .doc_ids = .{},
                    .frequencies = .{},
                };
            }

            try posting_entry.value_ptr.doc_ids.append(self.allocator, doc_id);
            try posting_entry.value_ptr.frequencies.append(self.allocator, freq);
        }

        // Store document metadata
        try self.documents.put(self.allocator, doc_id, .{
            .field_length = @intCast(tokens.items.len),
            .term_count = @intCast(term_freqs.count()),
        });

        // Update statistics
        self.total_docs += 1;
        self.total_field_lengths += tokens.items.len;
        self.avg_doc_length = @as(f32, @floatFromInt(self.total_field_lengths)) /
            @as(f32, @floatFromInt(self.total_docs));
    }

    /// Remove a document from the index.
    pub fn removeDocument(self: *InvertedIndex, doc_id: u64) !void {
        // Get document metadata
        const doc_meta = self.documents.get(doc_id) orelse return;

        // Update statistics
        self.total_docs -|= 1;
        self.total_field_lengths -|= doc_meta.field_length;
        if (self.total_docs > 0) {
            self.avg_doc_length = @as(f32, @floatFromInt(self.total_field_lengths)) /
                @as(f32, @floatFromInt(self.total_docs));
        } else {
            self.avg_doc_length = 0;
        }

        // Remove from postings (expensive operation)
        var posting_iter = self.postings.iterator();
        while (posting_iter.next()) |entry| {
            const posting_list = entry.value_ptr;

            // Find and remove document from posting list
            var i: usize = 0;
            while (i < posting_list.doc_ids.items.len) {
                if (posting_list.doc_ids.items[i] == doc_id) {
                    _ = posting_list.doc_ids.orderedRemove(i);
                    _ = posting_list.frequencies.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Remove document metadata
        _ = self.documents.remove(doc_id);
    }

    /// Search the index.
    pub fn search(self: *InvertedIndex, query: []const u8, top_k: usize) ![]TextSearchResult {
        var query_tokens = try self.tokenize(query);
        defer {
            for (query_tokens.items) |token| {
                self.allocator.free(token);
            }
            query_tokens.deinit(self.allocator);
        }

        if (query_tokens.items.len == 0) {
            return try self.allocator.alloc(TextSearchResult, 0);
        }

        // Collect candidate documents and their scores
        var scores = std.AutoHashMapUnmanaged(u64, ScoreAccum){};
        defer scores.deinit(self.allocator);

        for (query_tokens.items) |term| {
            if (self.postings.get(term)) |posting_list| {
                // Calculate IDF
                const df = posting_list.doc_ids.items.len;
                const idf = self.calculateIdf(df);

                // Score each document
                for (posting_list.doc_ids.items, posting_list.frequencies.items) |doc_id, freq| {
                    const doc_meta = self.documents.get(doc_id) orelse continue;
                    const tf_score = self.calculateTf(freq, doc_meta.field_length);
                    const term_score = tf_score * idf;

                    const score_entry = try scores.getOrPut(self.allocator, doc_id);
                    if (!score_entry.found_existing) {
                        score_entry.value_ptr.* = .{ .score = 0, .term_count = 0 };
                    }
                    score_entry.value_ptr.score += term_score;
                    score_entry.value_ptr.term_count += 1;
                }
            }
        }

        // Convert to results and sort
        var results = std.ArrayListUnmanaged(TextSearchResult){};
        defer results.deinit(self.allocator);

        var score_iter = scores.iterator();
        while (score_iter.next()) |entry| {
            try results.append(self.allocator, .{
                .doc_id = entry.key_ptr.*,
                .score = entry.value_ptr.score,
                .matched_terms = &.{}, // Would need to track matched terms
            });
        }

        // Sort by score descending
        std.mem.sort(TextSearchResult, results.items, {}, TextSearchResult.lessThan);

        // Return top_k results
        const result_count = @min(top_k, results.items.len);
        const final_results = try self.allocator.alloc(TextSearchResult, result_count);
        @memcpy(final_results, results.items[0..result_count]);

        return final_results;
    }

    /// Get document frequency for a term.
    pub fn getDocumentFrequency(self: *const InvertedIndex, term: []const u8) usize {
        if (self.postings.get(term)) |posting_list| {
            return posting_list.doc_ids.items.len;
        }
        return 0;
    }

    /// Get total document count.
    pub fn documentCount(self: *const InvertedIndex) u64 {
        return self.total_docs;
    }

    /// Get term count.
    pub fn termCount(self: *const InvertedIndex) usize {
        return self.postings.count();
    }

    const ScoreAccum = struct {
        score: f32,
        term_count: u32,
    };

    // BM25 calculations
    fn calculateIdf(self: *const InvertedIndex, df: usize) f32 {
        if (df == 0 or self.total_docs == 0) return 0;

        const n = @as(f32, @floatFromInt(self.total_docs));
        const df_f = @as(f32, @floatFromInt(df));

        // IDF = ln((N - df + 0.5) / (df + 0.5) + 1)
        return @log((n - df_f + 0.5) / (df_f + 0.5) + 1.0);
    }

    fn calculateTf(self: *const InvertedIndex, freq: u32, doc_length: u32) f32 {
        const k1 = self.bm25_config.k1;
        const b = self.bm25_config.b;

        const f = @as(f32, @floatFromInt(freq));
        const dl = @as(f32, @floatFromInt(doc_length));
        const avgdl = self.avg_doc_length;

        if (avgdl == 0) return 0;

        // TF = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        return (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl));
    }

    // Tokenization
    fn tokenize(self: *InvertedIndex, text: []const u8) !std.ArrayListUnmanaged([]u8) {
        var tokens = std.ArrayListUnmanaged([]u8){};
        errdefer {
            for (tokens.items) |token| {
                self.allocator.free(token);
            }
            tokens.deinit(self.allocator);
        }

        var start: usize = 0;
        var i: usize = 0;

        while (i < text.len) : (i += 1) {
            const c = text[i];
            const is_boundary = !std.ascii.isAlphanumeric(c);

            if (is_boundary) {
                if (i > start) {
                    const token = try self.processToken(text[start..i]);
                    if (token) |t| {
                        try tokens.append(self.allocator, t);
                    }
                }
                start = i + 1;
            }
        }

        // Handle last token
        if (text.len > start) {
            const token = try self.processToken(text[start..]);
            if (token) |t| {
                try tokens.append(self.allocator, t);
            }
        }

        return tokens;
    }

    fn processToken(self: *InvertedIndex, token: []const u8) !?[]u8 {
        if (token.len < self.tokenizer_config.min_token_length) return null;
        if (token.len > self.tokenizer_config.max_token_length) return null;

        var processed = try self.allocator.alloc(u8, token.len);
        errdefer self.allocator.free(processed);

        // Lowercase
        if (self.tokenizer_config.lowercase) {
            for (token, 0..) |c, j| {
                processed[j] = std.ascii.toLower(c);
            }
        } else {
            @memcpy(processed, token);
        }

        // Basic stemming (remove common suffixes)
        if (self.tokenizer_config.enable_stemming) {
            var len = processed.len;

            // Remove common English suffixes
            if (len > 4) {
                if (std.mem.endsWith(u8, processed[0..len], "ing")) {
                    len -= 3;
                } else if (std.mem.endsWith(u8, processed[0..len], "ed")) {
                    len -= 2;
                } else if (std.mem.endsWith(u8, processed[0..len], "ly")) {
                    len -= 2;
                } else if (std.mem.endsWith(u8, processed[0..len], "ness")) {
                    len -= 4;
                } else if (std.mem.endsWith(u8, processed[0..len], "ment")) {
                    len -= 4;
                }
            }

            if (len < processed.len) {
                const stemmed = try self.allocator.realloc(processed, len);
                return stemmed;
            }
        }

        return processed;
    }
};

/// Query parser for boolean and phrase queries.
pub const QueryParser = struct {
    allocator: std.mem.Allocator,

    pub const QueryType = enum {
        term,
        phrase,
        boolean_and,
        boolean_or,
        boolean_not,
    };

    pub const ParsedQuery = struct {
        query_type: QueryType,
        terms: []const []const u8,
        subqueries: []const ParsedQuery,
    };

    pub fn init(allocator: std.mem.Allocator) QueryParser {
        return .{ .allocator = allocator };
    }

    /// Parse a query string.
    pub fn parse(self: *QueryParser, query: []const u8) !ParsedQuery {
        // Simple implementation - treat as term query
        _ = self;
        return .{
            .query_type = .term,
            .terms = &.{query},
            .subqueries = &.{},
        };
    }
};

test "inverted index basic" {
    const allocator = std.testing.allocator;
    var index = InvertedIndex.init(allocator, .{}, .{});
    defer index.deinit();

    try index.indexDocument(1, "hello world");
    try index.indexDocument(2, "hello zig");
    try index.indexDocument(3, "world peace");

    try std.testing.expectEqual(@as(u64, 3), index.documentCount());
    try std.testing.expect(index.termCount() > 0);
}

test "inverted index search" {
    const allocator = std.testing.allocator;
    var index = InvertedIndex.init(allocator, .{}, .{});
    defer index.deinit();

    try index.indexDocument(1, "the quick brown fox");
    try index.indexDocument(2, "the lazy dog");
    try index.indexDocument(3, "quick brown dog");

    const results = try index.search("quick", 10);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
}

test "bm25 idf calculation" {
    const allocator = std.testing.allocator;
    var index = InvertedIndex.init(allocator, .{}, .{});
    defer index.deinit();

    // Add some documents
    try index.indexDocument(1, "test document");
    try index.indexDocument(2, "another document");
    try index.indexDocument(3, "yet another");

    // IDF for rare term should be higher
    const common_df = index.getDocumentFrequency("document");
    const rare_df = index.getDocumentFrequency("test");

    try std.testing.expect(common_df >= rare_df);
}
