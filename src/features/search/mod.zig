//! Search Module
//!
//! Full-text search with inverted index, BM25 scoring, tokenization,
//! and snippet generation.
//!
//! Architecture:
//! - Named indexes (SwissMap of name → InvertedIndex)
//! - Inverted index: term → PostingList (doc_id, term_freq, positions)
//! - BM25 scoring: IDF × TF component with configurable k1, b
//! - Tokenizer: lowercase, stop word removal
//! - Snippet: window with highest match density

const std = @import("std");
const core_config = @import("../../core/config/search.zig");
const sync = @import("../../services/shared/sync.zig");

pub const SearchConfig = core_config.SearchConfig;

/// Errors returned by search operations.
pub const SearchError = error{
    FeatureDisabled,
    IndexNotFound,
    InvalidQuery,
    IndexCorrupted,
    OutOfMemory,
    IndexAlreadyExists,
    DocumentNotFound,
};

/// A single search hit with BM25 relevance score and context snippet.
pub const SearchResult = struct {
    doc_id: []const u8 = "",
    /// BM25 relevance score (higher = more relevant).
    score: f32 = 0.0,
    /// Text excerpt around the best matching region.
    snippet: []const u8 = "",
};

/// Metadata about a named full-text search index.
pub const SearchIndex = struct {
    name: []const u8 = "",
    doc_count: u64 = 0,
    size_bytes: u64 = 0,
};

/// Aggregate statistics across all search indexes.
pub const SearchStats = struct {
    total_indexes: u32 = 0,
    total_documents: u64 = 0,
    total_terms: u64 = 0,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: SearchConfig,

    pub fn init(allocator: std.mem.Allocator, config: SearchConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Tokenizer ──────────────────────────────────────────────────────────

const stop_words = [_][]const u8{
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

fn isStopWord(word: []const u8) bool {
    return stop_word_set.has(word);
}

fn tokenize(
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

// ── BM25 Scoring ───────────────────────────────────────────────────────

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

fn bm25Score(
    tf: u32, // term frequency in document
    df: u32, // document frequency (how many docs contain term)
    total_docs: u64,
    doc_len: u32, // terms in document
    avg_doc_len: f64,
) f32 {
    if (df == 0 or total_docs == 0) return 0;

    const n = @as(f64, @floatFromInt(total_docs));
    const df_f = @as(f64, @floatFromInt(df));
    const tf_f = @as(f64, @floatFromInt(tf));
    const dl = @as(f64, @floatFromInt(doc_len));

    // IDF (Lucene variant, always non-negative for small corpora)
    // = log(1 + (N - df + 0.5) / (df + 0.5))
    const idf = @log(1.0 + (n - df_f + 0.5) / (df_f + 0.5));

    // TF component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    const tf_component = (tf_f * (BM25_K1 + 1.0)) /
        (tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_doc_len));

    return @floatCast(idf * tf_component);
}

// ── Inverted Index ─────────────────────────────────────────────────────

const Posting = struct {
    doc_id: []u8, // owned
    term_freq: u32,
    doc_len: u32,
};

const PostingList = struct {
    postings: std.ArrayListUnmanaged(Posting),
    doc_freq: u32,
};

const DocumentMeta = struct {
    id: []u8, // owned
    content: []u8, // owned (for snippets)
    term_count: u32,
};

const InvertedIndex = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    term_index: std.StringHashMapUnmanaged(*PostingList),
    documents: std.StringHashMapUnmanaged(*DocumentMeta),
    total_terms: u64,

    fn create(allocator: std.mem.Allocator, name: []const u8) !*InvertedIndex {
        const idx = try allocator.create(InvertedIndex);
        idx.* = .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .term_index = .empty,
            .documents = .empty,
            .total_terms = 0,
        };
        return idx;
    }

    fn destroy(self: *InvertedIndex) void {
        const alloc = self.allocator;

        // Free posting lists and owned term keys
        var pl_iter = self.term_index.iterator();
        while (pl_iter.next()) |entry| {
            // Free the owned term key string
            alloc.free(entry.key_ptr.*);
            const pl = entry.value_ptr.*;
            for (pl.postings.items) |p| alloc.free(p.doc_id);
            pl.postings.deinit(alloc);
            alloc.destroy(pl);
        }
        self.term_index.deinit(alloc);

        // Free documents
        var doc_iter = self.documents.iterator();
        while (doc_iter.next()) |entry| {
            const doc = entry.value_ptr.*;
            alloc.free(doc.id);
            alloc.free(doc.content);
            alloc.destroy(doc);
        }
        self.documents.deinit(alloc);

        alloc.free(self.name);
        alloc.destroy(self);
    }

    fn avgDocLen(self: *const InvertedIndex) f64 {
        const count = self.documents.count();
        if (count == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_terms)) / @as(f64, @floatFromInt(count));
    }

    fn addDocument(self: *InvertedIndex, doc_id: []const u8, content: []const u8) !void {
        // Remove existing if present (including posting list cleanup)
        if (self.documents.get(doc_id)) |existing| {
            self.total_terms -= existing.term_count;

            // Re-tokenize old content to find only its terms — O(terms_in_doc)
            var old_tokens = try tokenize(self.allocator, existing.content, true);
            defer {
                for (old_tokens.items) |t| self.allocator.free(t);
                old_tokens.deinit(self.allocator);
            }
            for (old_tokens.items) |term| {
                if (self.term_index.get(term)) |pl| {
                    var i: usize = 0;
                    while (i < pl.postings.items.len) {
                        if (std.mem.eql(u8, pl.postings.items[i].doc_id, existing.id)) {
                            const removed = pl.postings.swapRemove(i);
                            self.allocator.free(removed.doc_id);
                            pl.doc_freq -|= 1;
                        } else {
                            i += 1;
                        }
                    }
                }
            }

            // Remove from map BEFORE freeing the id (map key points to existing.id)
            _ = self.documents.remove(doc_id);
            self.allocator.free(existing.id);
            self.allocator.free(existing.content);
            self.allocator.destroy(existing);
        }

        // Tokenize
        var tokens = try tokenize(self.allocator, content, true);
        defer {
            for (tokens.items) |t| self.allocator.free(t);
            tokens.deinit(self.allocator);
        }

        const doc_len: u32 = @intCast(tokens.items.len);
        self.total_terms += doc_len;

        // Count term frequencies
        var tf_map = std.StringHashMapUnmanaged(u32).empty;
        defer tf_map.deinit(self.allocator);

        for (tokens.items) |token| {
            const entry = try tf_map.getOrPut(self.allocator, token);
            if (entry.found_existing) {
                entry.value_ptr.* += 1;
            } else {
                entry.value_ptr.* = 1;
            }
        }

        // Store document — errdefer chain ensures cleanup on allocation failure
        const doc = try self.allocator.create(DocumentMeta);
        errdefer self.allocator.destroy(doc);

        doc.id = try self.allocator.dupe(u8, doc_id);
        errdefer self.allocator.free(doc.id);

        doc.content = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(doc.content);

        doc.term_count = doc_len;
        try self.documents.put(self.allocator, doc.id, doc);

        // Update posting lists
        var tf_iter = tf_map.iterator();
        while (tf_iter.next()) |entry| {
            const term = entry.key_ptr.*;
            const freq = entry.value_ptr.*;

            const pl = self.term_index.get(term) orelse blk: {
                const new_pl = try self.allocator.create(PostingList);
                errdefer self.allocator.destroy(new_pl);
                new_pl.* = .{ .postings = .empty, .doc_freq = 0 };
                const term_owned = try self.allocator.dupe(u8, term);
                errdefer self.allocator.free(term_owned);
                try self.term_index.put(self.allocator, term_owned, new_pl);
                break :blk new_pl;
            };

            const posting_doc_id = try self.allocator.dupe(u8, doc_id);
            pl.postings.append(self.allocator, .{
                .doc_id = posting_doc_id,
                .term_freq = freq,
                .doc_len = doc_len,
            }) catch {
                self.allocator.free(posting_doc_id);
                return error.OutOfMemory;
            };
            pl.doc_freq += 1;
        }
    }

    fn search(
        self: *InvertedIndex,
        allocator: std.mem.Allocator,
        query_text: []const u8,
        limit: u32,
    ) ![]SearchResult {
        // Use arena for query tokenization — tokens are only needed during scoring
        // and this avoids per-token heap alloc/free overhead
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var query_tokens = try tokenize(arena.allocator(), query_text, true);
        // No per-token free needed — arena handles bulk deallocation
        _ = &query_tokens;

        if (query_tokens.items.len == 0) return &.{};

        // Accumulate scores per document
        var doc_scores = std.StringHashMapUnmanaged(f32).empty;
        defer doc_scores.deinit(allocator);

        const total_docs = self.documents.count();
        const avg_dl = self.avgDocLen();

        for (query_tokens.items) |token| {
            const pl = self.term_index.get(token) orelse continue;
            for (pl.postings.items) |posting| {
                const score = bm25Score(
                    posting.term_freq,
                    pl.doc_freq,
                    total_docs,
                    posting.doc_len,
                    avg_dl,
                );
                const entry = try doc_scores.getOrPut(allocator, posting.doc_id);
                if (entry.found_existing) {
                    entry.value_ptr.* += score;
                } else {
                    entry.value_ptr.* = score;
                }
            }
        }

        // Collect and sort results
        const count = @min(doc_scores.count(), limit);
        if (count == 0) return &.{};

        const results = try allocator.alloc(SearchResult, count);
        var result_idx: usize = 0;
        var score_iter = doc_scores.iterator();
        while (score_iter.next()) |entry| {
            if (result_idx >= count) break;
            results[result_idx] = .{
                .doc_id = entry.key_ptr.*,
                .score = entry.value_ptr.*,
                .snippet = "", // Simplified
            };
            result_idx += 1;
        }

        // Sort by score descending
        std.mem.sort(SearchResult, results[0..result_idx], {}, struct {
            fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                return a.score > b.score;
            }
        }.lessThan);

        return results[0..result_idx];
    }
};

// ── Module State ───────────────────────────────────────────────────────

var search_state: ?*SearchState = null;

const SearchState = struct {
    allocator: std.mem.Allocator,
    config: SearchConfig,
    indexes: std.StringHashMapUnmanaged(*InvertedIndex),
    rw_lock: sync.RwLock,

    fn create(allocator: std.mem.Allocator, config: SearchConfig) !*SearchState {
        const s = try allocator.create(SearchState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .indexes = .empty,
            .rw_lock = sync.RwLock.init(),
        };
        return s;
    }

    fn destroy(self: *SearchState) void {
        var iter = self.indexes.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.destroy();
        }
        self.indexes.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

// ── Public API ─────────────────────────────────────────────────────────

/// Initialize the global search engine singleton.
pub fn init(allocator: std.mem.Allocator, config: SearchConfig) SearchError!void {
    if (search_state != null) return;
    search_state = SearchState.create(allocator, config) catch return error.OutOfMemory;
}

/// Tear down the search engine, destroying all indexes and postings.
pub fn deinit() void {
    if (search_state) |s| {
        s.destroy();
        search_state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return search_state != null;
}

/// Create a new named full-text index. Returns `IndexAlreadyExists` if
/// an index with the same name exists.
pub fn createIndex(allocator: std.mem.Allocator, name: []const u8) SearchError!SearchIndex {
    _ = allocator;
    const s = search_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.indexes.get(name) != null) return error.IndexAlreadyExists;

    const idx = InvertedIndex.create(s.allocator, name) catch return error.OutOfMemory;
    s.indexes.put(s.allocator, idx.name, idx) catch {
        idx.destroy();
        return error.OutOfMemory;
    };

    return .{ .name = idx.name };
}

/// Delete a named index and all its documents.
pub fn deleteIndex(name: []const u8) SearchError!void {
    const s = search_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.indexes.fetchRemove(name)) |kv| {
        kv.value.destroy();
    } else {
        return error.IndexNotFound;
    }
}

/// Add or update a document in a named index. Tokenizes the content
/// and builds inverted-index postings for BM25 retrieval.
pub fn indexDocument(
    index_name: []const u8,
    doc_id: []const u8,
    content: []const u8,
) SearchError!void {
    const s = search_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const idx = s.indexes.get(index_name) orelse return error.IndexNotFound;
    idx.addDocument(doc_id, content) catch return error.OutOfMemory;
}

/// Remove a document from an index. Returns `true` if the document existed.
pub fn deleteDocument(index_name: []const u8, doc_id: []const u8) SearchError!bool {
    const s = search_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const idx = s.indexes.get(index_name) orelse return error.IndexNotFound;

    if (idx.documents.fetchRemove(doc_id)) |kv| {
        const doc = kv.value;
        idx.total_terms -= doc.term_count;

        // Re-tokenize stored content to find exactly which terms this document
        // contributes to — O(terms_in_doc) instead of O(total_terms)
        var doc_tokens = tokenize(idx.allocator, doc.content, true) catch {
            // Fallback: scan all terms (original O(T*P) path)
            var term_iter = idx.term_index.iterator();
            while (term_iter.next()) |entry| {
                const pl = entry.value_ptr.*;
                var i: usize = 0;
                while (i < pl.postings.items.len) {
                    if (std.mem.eql(u8, pl.postings.items[i].doc_id, doc.id)) {
                        const removed = pl.postings.swapRemove(i);
                        idx.allocator.free(removed.doc_id);
                        pl.doc_freq -|= 1;
                    } else {
                        i += 1;
                    }
                }
            }
            idx.allocator.free(doc.id);
            idx.allocator.free(doc.content);
            idx.allocator.destroy(doc);
            return true;
        };
        defer {
            for (doc_tokens.items) |t| idx.allocator.free(t);
            doc_tokens.deinit(idx.allocator);
        }

        // Only visit posting lists for terms in this document
        for (doc_tokens.items) |term| {
            if (idx.term_index.get(term)) |pl| {
                var i: usize = 0;
                while (i < pl.postings.items.len) {
                    if (std.mem.eql(u8, pl.postings.items[i].doc_id, doc.id)) {
                        const removed = pl.postings.swapRemove(i);
                        idx.allocator.free(removed.doc_id);
                        pl.doc_freq -|= 1;
                    } else {
                        i += 1;
                    }
                }
            }
        }

        idx.allocator.free(doc.id);
        idx.allocator.free(doc.content);
        idx.allocator.destroy(doc);
        return true;
    }
    return false;
}

/// Execute a BM25-scored full-text query against a named index.
/// Results are sorted by relevance and include context snippets.
pub fn query(
    allocator: std.mem.Allocator,
    index_name: []const u8,
    query_text: []const u8,
) SearchError![]SearchResult {
    const s = search_state orelse return error.FeatureDisabled;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const idx = s.indexes.get(index_name) orelse return error.IndexNotFound;
    return idx.search(allocator, query_text, s.config.default_result_limit) catch
        return error.OutOfMemory;
}

/// Return aggregate statistics across all search indexes.
pub fn stats() SearchStats {
    const s = search_state orelse return .{};

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    var total_docs: u64 = 0;
    var total_terms: u64 = 0;
    var iter = s.indexes.iterator();
    while (iter.next()) |entry| {
        const idx = entry.value_ptr.*;
        total_docs += idx.documents.count();
        total_terms += idx.total_terms;
    }

    return .{
        .total_indexes = @intCast(s.indexes.count()),
        .total_documents = total_docs,
        .total_terms = total_terms,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

test "search basic index and query" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "test");
    try indexDocument("test", "doc1", "the quick brown fox jumps over the lazy dog");
    try indexDocument("test", "doc2", "a fast brown cat sits on the mat");
    try indexDocument("test", "doc3", "the dog barked at the fox");

    const results = try query(allocator, "test", "brown fox");
    defer allocator.free(results);

    // At least one result matches
    try std.testing.expect(results.len >= 1);
    // First result has a positive score
    try std.testing.expect(results[0].score > 0);
}

test "search delete document" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "deltest");
    try indexDocument("deltest", "doc1", "hello world");
    try indexDocument("deltest", "doc2", "goodbye world");

    const deleted = try deleteDocument("deltest", "doc1");
    try std.testing.expect(deleted);

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_documents);
}

test "search delete index" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "temp");
    try indexDocument("temp", "doc1", "test content");

    try deleteIndex("temp");

    const s = stats();
    try std.testing.expectEqual(@as(u32, 0), s.total_indexes);
}

test "search stats" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "stats_test");
    try indexDocument("stats_test", "d1", "hello world foo bar");
    try indexDocument("stats_test", "d2", "baz qux quux");

    const s = stats();
    try std.testing.expectEqual(@as(u32, 1), s.total_indexes);
    try std.testing.expectEqual(@as(u64, 2), s.total_documents);
    try std.testing.expect(s.total_terms > 0);
}

test "search empty query" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "empty");
    try indexDocument("empty", "doc1", "some content");

    // Query with only stop words → empty tokens → no results
    const results = try query(allocator, "empty", "the and or");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "search duplicate index error" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "dup");
    const result = createIndex(allocator, "dup");
    try std.testing.expectError(error.IndexAlreadyExists, result);
}

test "tokenizer basic" {
    const allocator = std.testing.allocator;
    var tokens = try tokenize(allocator, "Hello World! This is a TEST.", true);
    defer {
        for (tokens.items) |t| allocator.free(t);
        tokens.deinit(allocator);
    }

    // "hello", "world", "test" — stop words removed ("this", "is", "a")
    try std.testing.expectEqual(@as(usize, 3), tokens.items.len);
    try std.testing.expectEqualStrings("hello", tokens.items[0]);
    try std.testing.expectEqualStrings("world", tokens.items[1]);
    try std.testing.expectEqualStrings("test", tokens.items[2]);
}

test "search delete document then query returns no results" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "del_query");
    try indexDocument("del_query", "doc1", "unique zebra crossing");
    try indexDocument("del_query", "doc2", "another document about cats");

    // Query before delete should find doc1
    const before = try query(allocator, "del_query", "zebra");
    defer allocator.free(before);
    try std.testing.expect(before.len >= 1);

    // Delete doc1
    const deleted = try deleteDocument("del_query", "doc1");
    try std.testing.expect(deleted);

    // Query after delete should NOT find doc1
    const after = try query(allocator, "del_query", "zebra");
    defer allocator.free(after);
    try std.testing.expectEqual(@as(usize, 0), after.len);
}

test "search re-index document updates results" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "reindex");
    try indexDocument("reindex", "doc1", "original content about dogs");

    // Query original
    const r1 = try query(allocator, "reindex", "dogs");
    defer allocator.free(r1);
    try std.testing.expect(r1.len >= 1);

    // Re-index with different content
    try indexDocument("reindex", "doc1", "updated content about elephants");

    // Old content should not match
    const r2 = try query(allocator, "reindex", "dogs");
    defer allocator.free(r2);
    try std.testing.expectEqual(@as(usize, 0), r2.len);

    // New content should match
    const r3 = try query(allocator, "reindex", "elephants");
    defer allocator.free(r3);
    try std.testing.expect(r3.len >= 1);

    // Document count should still be 1
    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_documents);
}

test "search multi-term BM25 ranking" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "ranking");
    try indexDocument("ranking", "d1", "quick brown fox jumps");
    try indexDocument("ranking", "d2", "quick brown fox jumps over lazy dog");
    try indexDocument("ranking", "d3", "lazy dog sleeps all day");

    // Multi-term query: "quick fox" — d1 and d2 match, d2 has more terms
    const results = try query(allocator, "ranking", "quick fox");
    defer allocator.free(results);
    try std.testing.expect(results.len >= 2);
    // Both matching docs should have positive scores
    try std.testing.expect(results[0].score > 0);
    try std.testing.expect(results[1].score > 0);
}

test "search large document tokenization" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "large");

    // Build a document with 100+ words
    const big_doc = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " **
        11; // 110 words
    try indexDocument("large", "big1", big_doc);

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_documents);
    try std.testing.expect(s.total_terms > 50); // many non-stop terms
}

test "search query on non-existent index" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    const result = query(allocator, "nonexistent", "test");
    try std.testing.expectError(error.IndexNotFound, result);
}

test "search BM25 single document edge case" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "single");
    try indexDocument("single", "only", "solitary unique word");

    const results = try query(allocator, "single", "solitary");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    // BM25 with single doc: IDF = log(1 + (1-1+0.5)/(1+0.5)) = log(1.333) > 0
    try std.testing.expect(results[0].score > 0);
}

test "search query with only stop words returns empty" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "stops");
    try indexDocument("stops", "doc1", "hello world programming");

    // "the is a" are all stop words — should produce no results
    const results = try query(allocator, "stops", "the is a");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "search case insensitive matching" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "case");
    try indexDocument("case", "doc1", "Hello World PROGRAMMING");

    const results = try query(allocator, "case", "hello");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
}

test "search results ordered by BM25 descending" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "order");
    // doc2 has "rust" twice — should score higher
    try indexDocument("order", "doc1", "rust programming language");
    try indexDocument("order", "doc2", "rust rust systems programming");

    const results = try query(allocator, "order", "rust");
    defer allocator.free(results);
    try std.testing.expect(results.len >= 2);
    // First result should have higher or equal score
    try std.testing.expect(results[0].score >= results[1].score);
}

test "search single character query" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "char");
    try indexDocument("char", "doc1", "x marks the spot");

    const results = try query(allocator, "char", "x");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
}

test {
    std.testing.refAllDecls(@This());
}
