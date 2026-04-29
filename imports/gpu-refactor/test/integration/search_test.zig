//! Integration Tests: Search Module
//!
//! Tests the full-text search module exports, BM25 scoring types,
//! inverted index types, search result types, and basic API contracts.

const std = @import("std");
const abi = @import("abi");

const search = abi.search;

// ── Type Export Tests ──────────────────────────────────────────────────

test "search: module exports expected types" {
    const _config = search.SearchConfig{};
    try std.testing.expectEqual(@as(u32, 512), _config.max_index_size_mb);
    try std.testing.expectEqual(@as(u32, 100), _config.default_result_limit);
    try std.testing.expect(_config.enable_stemming);
    try std.testing.expect(_config.enable_fuzzy);

    const _err: search.SearchError = error.IndexNotFound;
    _ = _err;

    const _result = search.SearchResult{};
    try std.testing.expectEqualStrings("", _result.doc_id);
    try std.testing.expectEqual(@as(f32, 0.0), _result.score);

    const _index = search.SearchIndex{};
    try std.testing.expectEqualStrings("", _index.name);
    try std.testing.expectEqual(@as(u64, 0), _index.doc_count);

    const _stats = search.SearchStats{};
    try std.testing.expectEqual(@as(u32, 0), _stats.total_indexes);
    try std.testing.expectEqual(@as(u64, 0), _stats.total_documents);
    try std.testing.expectEqual(@as(u64, 0), _stats.total_terms);
}

test "search: config defaults factory" {
    const config = search.SearchConfig.defaults();
    try std.testing.expectEqual(@as(u32, 512), config.max_index_size_mb);
    try std.testing.expectEqual(@as(u32, 100), config.default_result_limit);
}

test "search: error type is aliased as Error" {
    const err1: search.Error = error.IndexNotFound;
    const err2: search.SearchError = err1;
    _ = err2;
}

// ── Module API Tests ───────────────────────────────────────────────────

test "search: isEnabled returns true" {
    try std.testing.expect(search.isEnabled());
}

test "search: init deinit lifecycle" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    try std.testing.expect(search.isInitialized());
}

// ── Context Lifecycle Tests ────────────────────────────────────────────

test "search: context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try search.Context.init(allocator, search.SearchConfig.defaults());
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u32, 100), ctx.config.default_result_limit);
}

test "search: context with custom config" {
    const allocator = std.testing.allocator;
    const ctx = try search.Context.init(allocator, .{
        .max_index_size_mb = 128,
        .default_result_limit = 50,
        .enable_stemming = false,
        .enable_fuzzy = false,
    });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u32, 128), ctx.config.max_index_size_mb);
    try std.testing.expectEqual(@as(u32, 50), ctx.config.default_result_limit);
    try std.testing.expect(!ctx.config.enable_stemming);
}

// ── Index and Query Tests ──────────────────────────────────────────────

test "search: create index and index documents" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    const idx = try search.createIndex(allocator, "integ-test");
    try std.testing.expectEqualStrings("integ-test", idx.name);

    try search.indexDocument("integ-test", "doc1", "the quick brown fox");
    try search.indexDocument("integ-test", "doc2", "a lazy brown dog");

    const s = search.stats();
    try std.testing.expectEqual(@as(u32, 1), s.total_indexes);
    try std.testing.expectEqual(@as(u64, 2), s.total_documents);
    try std.testing.expect(s.total_terms > 0);
}

test "search: query returns BM25-scored results" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    _ = try search.createIndex(allocator, "bm25-test");
    try search.indexDocument("bm25-test", "doc1", "the quick brown fox jumps over the lazy dog");
    try search.indexDocument("bm25-test", "doc2", "a fast brown cat sits on the mat");

    const results = try search.query(allocator, "bm25-test", "brown fox");
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expect(results[0].score > 0);
}

test "search: delete index" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    _ = try search.createIndex(allocator, "to-delete");
    try search.indexDocument("to-delete", "doc1", "content");
    try search.deleteIndex("to-delete");

    const s = search.stats();
    try std.testing.expectEqual(@as(u32, 0), s.total_indexes);
}

test "search: delete document" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    _ = try search.createIndex(allocator, "del-doc");
    try search.indexDocument("del-doc", "doc1", "hello world");
    try search.indexDocument("del-doc", "doc2", "goodbye world");

    const deleted = try search.deleteDocument("del-doc", "doc1");
    try std.testing.expect(deleted);

    const not_deleted = try search.deleteDocument("del-doc", "nonexistent");
    try std.testing.expect(!not_deleted);

    const s = search.stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_documents);
}

test "search: duplicate index returns error" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    _ = try search.createIndex(allocator, "dup-idx");
    try std.testing.expectError(error.IndexAlreadyExists, search.createIndex(allocator, "dup-idx"));
}

test "search: query non-existent index returns error" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    try std.testing.expectError(error.IndexNotFound, search.query(allocator, "missing", "test"));
}

test "search: query with only stop words returns empty" {
    const allocator = std.testing.allocator;
    try search.init(allocator, search.SearchConfig.defaults());
    defer search.deinit();

    _ = try search.createIndex(allocator, "stops-integ");
    try search.indexDocument("stops-integ", "doc1", "programming languages");

    const results = try search.query(allocator, "stops-integ", "the and or");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "search: sub-module exports accessible" {
    // Verify sub-modules are reachable through abi.search
    _ = search.tokenizer;
    _ = search.scoring;
    _ = search.inverted_index;
    _ = search.persistence;
    _ = search.types;
}

test {
    std.testing.refAllDecls(@This());
}
