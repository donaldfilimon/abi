//! Document Operations
//!
//! Document mutation logic: indexing (add/update) and deletion with
//! tokenization-based posting list cleanup.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const inverted_index = @import("index.zig");
const state_mod = @import("state.zig");

pub const SearchState = state_mod.SearchState;
pub const SearchError = @import("types.zig").SearchError;

/// Add or update a document in a named index. Tokenizes the content
/// and builds inverted-index postings for BM25 retrieval.
pub fn indexDocument(
    s: *SearchState,
    index_name: []const u8,
    doc_id: []const u8,
    content: []const u8,
) SearchError!void {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const idx = s.indexes.get(index_name) orelse return error.IndexNotFound;
    idx.addDocument(doc_id, content) catch return error.OutOfMemory;
}

/// Remove a document from an index. Returns `true` if the document existed.
pub fn deleteDocument(
    s: *SearchState,
    index_name: []const u8,
    doc_id: []const u8,
) SearchError!bool {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const idx = s.indexes.get(index_name) orelse return error.IndexNotFound;

    if (idx.documents.fetchRemove(doc_id)) |kv| {
        const doc = kv.value;
        idx.total_terms -= doc.term_count;

        // Re-tokenize stored content to find exactly which terms this document
        // contributes to — O(terms_in_doc) instead of O(total_terms)
        var doc_tokens = tokenizer.tokenize(idx.allocator, doc.content, true) catch {
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
