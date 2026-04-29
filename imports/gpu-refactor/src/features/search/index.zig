//! Inverted Index
//!
//! Core inverted index data structure with posting lists, document storage,
//! BM25-scored search, and document add/remove operations.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const scoring = @import("scoring.zig");
const types = @import("types.zig");

const SearchResult = types.SearchResult;

pub const Posting = struct {
    doc_id: []u8, // owned
    term_freq: u32,
    doc_len: u32,
};

pub const PostingList = struct {
    postings: std.ArrayListUnmanaged(Posting),
    doc_freq: u32,
};

pub const DocumentMeta = struct {
    id: []u8, // owned
    content: []u8, // owned (for snippets)
    term_count: u32,
};

pub const InvertedIndex = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    term_index: std.StringHashMapUnmanaged(*PostingList),
    documents: std.StringHashMapUnmanaged(*DocumentMeta),
    total_terms: u64,

    pub fn create(allocator: std.mem.Allocator, name: []const u8) !*InvertedIndex {
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

    pub fn destroy(self: *InvertedIndex) void {
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

    pub fn avgDocLen(self: *const InvertedIndex) f64 {
        const count = self.documents.count();
        if (count == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_terms)) / @as(f64, @floatFromInt(count));
    }

    pub fn addDocument(self: *InvertedIndex, doc_id: []const u8, content: []const u8) !void {
        // Remove existing if present (including posting list cleanup)
        if (self.documents.get(doc_id)) |existing| {
            self.total_terms -= existing.term_count;

            // Re-tokenize old content to find only its terms — O(terms_in_doc)
            var old_tokens = try tokenizer.tokenize(self.allocator, existing.content, true);
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
        var tokens = try tokenizer.tokenize(self.allocator, content, true);
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

    pub fn search(
        self: *InvertedIndex,
        allocator: std.mem.Allocator,
        query_text: []const u8,
        limit: u32,
    ) ![]SearchResult {
        // Use arena for query tokenization — tokens are only needed during scoring
        // and this avoids per-token heap alloc/free overhead
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var query_tokens = try tokenizer.tokenize(arena.allocator(), query_text, true);
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
                const score = scoring.bm25Score(
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
