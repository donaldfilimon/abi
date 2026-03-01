//! Full-Text Search Example
//!
//! Demonstrates the inverted index with BM25 ranking.
//! Shows index creation, document indexing, and search queries.
//!
//! Run with: `zig build run-search`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.App.builder(allocator);

    var framework = try builder
        .with(.search, abi.config.SearchConfig{})
        .build();
    defer framework.deinit();

    if (!abi.features.search.isEnabled()) {
        std.debug.print("Search feature is disabled. Enable with -Denable-search=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Full-Text Search Example ===\n\n", .{});

    // Create a search index
    _ = abi.features.search.createIndex(allocator, "articles") catch |err| {
        std.debug.print("Failed to create index: {t}\n", .{err});
        return;
    };
    std.debug.print("Created index: articles\n", .{});

    // Index documents
    const docs = [_]struct { id: []const u8, content: []const u8 }{
        .{ .id = "doc1", .content = "The quick brown fox jumps over the lazy dog" },
        .{ .id = "doc2", .content = "A fast red fox runs through the forest" },
        .{ .id = "doc3", .content = "Dogs and cats are popular household pets" },
    };

    for (docs) |doc| {
        abi.features.search.indexDocument("articles", doc.id, doc.content) catch |err| {
            std.debug.print("Failed to index {s}: {t}\n", .{ doc.id, err });
            continue;
        };
        std.debug.print("Indexed: {s}\n", .{doc.id});
    }

    // Search with BM25 ranking
    std.debug.print("\nSearching for 'fox':\n", .{});
    const results = abi.features.search.query(allocator, "articles", "fox") catch |err| {
        std.debug.print("Search failed: {t}\n", .{err});
        return;
    };
    defer allocator.free(results);

    if (results.len == 0) {
        std.debug.print("  No results found\n", .{});
    } else {
        for (results) |r| {
            std.debug.print("  {s} (score: {d:.3})\n", .{ r.doc_id, r.score });
        }
    }

    // Stats
    const s = abi.features.search.stats();
    std.debug.print("\nSearch stats: {} indexes, {} documents, {} terms\n", .{
        s.total_indexes, s.total_documents, s.total_terms,
    });
}
