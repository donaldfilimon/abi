//! Search Module
//!
//! Full-text search with inverted index, BM25 scoring, tokenization,
//! and snippet generation.
//!
//! Architecture:
//! - Named indexes (SwissMap of name -> InvertedIndex)
//! - Inverted index: term -> PostingList (doc_id, term_freq, positions)
//! - BM25 scoring: IDF x TF component with configurable k1, b
//! - Tokenizer: lowercase, stop word removal
//! - Snippet: window with highest match density

const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
pub const types = @import("types.zig");

// ── Sub-modules ───────────────────────────────────────────────────────
pub const tokenizer = @import("tokenizer.zig");
pub const scoring = @import("scoring.zig");
pub const inverted_index = @import("index.zig");
pub const persistence = @import("persistence.zig");

// ── Internal sub-modules ────────────────────────────────────────────
const state_mod = @import("state.zig");
const document_ops = @import("document_ops.zig");
const index_lifecycle = @import("index_lifecycle.zig");

// ── Re-exported types ─────────────────────────────────────────────────
pub const SearchConfig = types.SearchConfig;
pub const SearchError = types.SearchError;
pub const Error = SearchError;
pub const SearchResult = types.SearchResult;
pub const SearchIndex = types.SearchIndex;
pub const SearchStats = types.SearchStats;
const SearchState = state_mod.SearchState;

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

// ── Module State ───────────────────────────────────────────────────────

var search_state: ?*SearchState = null;

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
    return index_lifecycle.createIndex(s, name);
}

/// Delete a named index and all its documents.
pub fn deleteIndex(name: []const u8) SearchError!void {
    const s = search_state orelse return error.FeatureDisabled;
    return index_lifecycle.deleteIndex(s, name);
}

/// Add or update a document in a named index. Tokenizes the content
/// and builds inverted-index postings for BM25 retrieval.
pub fn indexDocument(
    index_name: []const u8,
    doc_id: []const u8,
    content: []const u8,
) SearchError!void {
    const s = search_state orelse return error.FeatureDisabled;
    return document_ops.indexDocument(s, index_name, doc_id, content);
}

/// Remove a document from an index. Returns `true` if the document existed.
pub fn deleteDocument(index_name: []const u8, doc_id: []const u8) SearchError!bool {
    const s = search_state orelse return error.FeatureDisabled;
    return document_ops.deleteDocument(s, index_name, doc_id);
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

/// Serialize a named inverted index to disk at the given path.
pub fn saveIndex(allocator: std.mem.Allocator, name: []const u8, path: []const u8) SearchError!void {
    _ = allocator;
    const s = search_state orelse return error.FeatureDisabled;
    return index_lifecycle.saveIndex(s, name, path);
}

/// Deserialize a named index from disk. Creates a new index with the
/// given name (must not already exist) and populates it from the file.
pub fn loadIndex(allocator: std.mem.Allocator, name: []const u8, path: []const u8) SearchError!void {
    const s = search_state orelse return error.FeatureDisabled;
    return index_lifecycle.loadIndex(s, allocator, name, path);
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

    // Query with only stop words -> empty tokens -> no results
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
    var tokens = try tokenizer.tokenize(allocator, "Hello World! This is a TEST.", true);
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

test "search save and load index round-trip" {
    const allocator = std.testing.allocator;
    try init(allocator, SearchConfig.defaults());
    defer deinit();

    _ = try createIndex(allocator, "persist");
    try indexDocument("persist", "doc1", "the quick brown fox jumps over the lazy dog");
    try indexDocument("persist", "doc2", "a fast brown cat sits on the mat");
    try indexDocument("persist", "doc3", "the dog barked at the fox");

    // Query before save
    const before = try query(allocator, "persist", "brown fox");
    defer allocator.free(before);
    try std.testing.expect(before.len >= 1);
    const score_before = before[0].score;

    // Save to disk
<<<<<<< Updated upstream
    var path_buf: [128]u8 = undefined;
    const tmp_path = try std.fmt.bufPrint(&path_buf, "/tmp/abi_search_persist_test_{d}.idx", .{@import("../../foundation/mod.zig").time.unixMs()});

=======
    const tmp_path = "abi_search_persist_test.idx";
>>>>>>> Stashed changes
    try saveIndex(allocator, "persist", tmp_path);
    defer persistence.unlinkFile(tmp_path);

    // Delete original index
    try deleteIndex("persist");

    // Load from disk
    try loadIndex(allocator, "persist", tmp_path);

    // Query after load — should produce equivalent results
    const after = try query(allocator, "persist", "brown fox");
    defer allocator.free(after);
    try std.testing.expect(after.len >= 1);
    // Scores should be identical (same BM25 parameters)
    try std.testing.expectApproxEqAbs(score_before, after[0].score, 0.001);

    // Stats should match
    const s = stats();
    try std.testing.expectEqual(@as(u64, 3), s.total_documents);
}

test {
    std.testing.refAllDecls(@This());
}
