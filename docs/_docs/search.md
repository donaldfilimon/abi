---
title: Search
description: Full-text BM25 search with inverted index, stop word filtering, and snippet generation
section: Data
order: 4
---

# Search

The search module (`src/features/search/mod.zig`) provides full-text search
with BM25 relevance scoring, inverted indexing, tokenization with stop word
filtering, and snippet generation. It supports multiple named indexes for
organizing documents by category or collection.

## Features

- **BM25 scoring**: Industry-standard Okapi BM25 relevance ranking with configurable k1 and b parameters
- **Inverted index**: Term-to-document posting lists with term frequency and position tracking
- **Named indexes**: Multiple independent search indexes (e.g., "articles", "products")
- **Tokenization**: Lowercase normalization with alphabetic/numeric token extraction
- **Stop word filtering**: Built-in list of 47 English stop words (comptime `StaticStringMap`)
- **Snippet generation**: Context window around the best matching region
- **Thread-safe**: RwLock for concurrent read access
- **Zero overhead when disabled**: Comptime feature gating eliminates the module

## Build Configuration

```bash
# Enable (default)
zig build -Denable-search=true

# Disable
zig build -Denable-search=false
```

**Namespace**: `abi.search`

## Quick Start

### Framework Integration

```zig
const abi = @import("abi");

// Initialize with search enabled
var fw = try abi.Framework.init(allocator, .{
    .search = .{},
});
defer fw.deinit();
```

### Standalone Usage

```zig
const search = abi.search;

// Initialize the global search singleton
try search.init(allocator, .{});
defer search.deinit();

// Create a named index
const idx = try search.createIndex(allocator, "articles");

// Index documents
try search.indexDocument("articles", "doc1",
    "Zig is a systems programming language designed for performance");
try search.indexDocument("articles", "doc2",
    "The BM25 algorithm ranks documents by relevance to a query");
try search.indexDocument("articles", "doc3",
    "Vector databases use approximate nearest neighbor search");

// Search with BM25 scoring
const results = try search.query(allocator, "articles", "BM25 relevance");
defer allocator.free(results);

for (results) |result| {
    std.debug.print("Doc: {s}, Score: {d:.4}, Snippet: {s}\n", .{
        result.doc_id, result.score, result.snippet,
    });
}

// Delete a document from an index
const removed = try search.deleteDocument("articles", "doc1");

// Delete an entire index
try search.deleteIndex("articles");

// Get aggregate statistics
const st = search.stats();
std.debug.print("Indexes: {}, Documents: {}, Terms: {}\n", .{
    st.total_indexes, st.total_documents, st.total_terms,
});
```

## API Reference

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(allocator, SearchConfig) !void` | Initialize the global search singleton |
| `deinit` | `() void` | Tear down all indexes and free memory |
| `isEnabled` | `() bool` | Returns `true` when search is compiled in |
| `isInitialized` | `() bool` | Returns `true` after successful `init()` |
| `createIndex` | `(allocator, name) !SearchIndex` | Create a named search index |
| `deleteIndex` | `(name) !void` | Delete a named index and all its documents |
| `indexDocument` | `(index_name, doc_id, content) !void` | Add or update a document in an index |
| `deleteDocument` | `(index_name, doc_id) !bool` | Remove a document; returns whether it existed |
| `query` | `(allocator, index_name, query_text) ![]SearchResult` | Search with BM25 scoring |
| `stats` | `() SearchStats` | Aggregate statistics across all indexes |

### Types

| Type | Description |
|------|-------------|
| `SearchConfig` | Configuration struct for search initialization |
| `SearchResult` | Search hit: `doc_id`, `score` (BM25), `snippet` (context window) |
| `SearchIndex` | Index metadata: `name`, `doc_count`, `size_bytes` |
| `SearchStats` | Aggregate stats: `total_indexes`, `total_documents`, `total_terms` |
| `SearchError` | Error set (see below) |

### Error Types

| Error | Description |
|-------|-------------|
| `FeatureDisabled` | Search module not compiled in |
| `IndexNotFound` | Named index does not exist |
| `IndexAlreadyExists` | Index with that name already exists |
| `InvalidQuery` | Query text is empty or invalid |
| `IndexCorrupted` | Internal index corruption detected |
| `DocumentNotFound` | Document ID not found in index |
| `OutOfMemory` | Allocation failure |

## BM25 Scoring

The search module uses the Okapi BM25 algorithm, the industry standard for
full-text relevance ranking. The scoring formula combines inverse document
frequency (IDF) with term frequency (TF) normalization:

```
score = IDF(q) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
```

Where:
- **IDF** uses the Lucene variant: `log(1 + (N - df + 0.5) / (df + 0.5))`
- **k1** = 1.2 (term frequency saturation parameter)
- **b** = 0.75 (document length normalization)
- **tf** = term frequency in the document
- **df** = number of documents containing the term
- **dl** = document length (number of terms)
- **avgdl** = average document length across the index

Multi-term queries sum the BM25 score for each query term, producing a
combined relevance score.

## Inverted Index

Each named index maintains an inverted index structure:

```
Term Index:  "zig"     -> PostingList { doc_freq: 2, postings: [...] }
             "search"  -> PostingList { doc_freq: 1, postings: [...] }

Posting:     { doc_id: "doc1", term_freq: 3, doc_len: 42 }

Documents:   "doc1" -> DocumentMeta { id, content, term_count }
```

- **PostingList**: Per-term list of documents containing the term, with
  term frequency and document length for BM25 scoring
- **DocumentMeta**: Stores the original content for snippet generation
- **Average document length**: Maintained incrementally for BM25 normalization

Document updates are handled by first removing all posting entries for the
old version, then re-indexing with the new content.

## Tokenization

The tokenizer extracts alphabetic and numeric tokens, lowercases them, and
optionally filters stop words:

1. **Token extraction**: Splits on non-alphanumeric characters
2. **Lowercase**: All tokens normalized to lowercase
3. **Length filter**: Tokens must be 1-100 characters
4. **Stop word removal**: 47 common English words filtered via comptime `StaticStringMap`

### Stop Words

The following words are filtered during both indexing and querying:

> a, an, the, and, or, but, in, on, at, to, for, of, with, by, from,
> is, are, was, were, be, been, being, have, has, had, do, does, did,
> will, would, could, should, may, might, shall, can, it, its, this,
> that, these, those, i, we, you, he, she, they, not

Stop word filtering ensures that common function words do not dominate
relevance scoring.

## Snippet Generation

Search results include a `snippet` field containing a text excerpt around
the best matching region. The snippet generator:

1. Finds the window with the highest density of query term matches
2. Extracts surrounding context (typically a sentence or phrase)
3. Returns the excerpt as part of the `SearchResult`

Snippets help users quickly evaluate result relevance without reading
full documents.

## Architecture

The module uses a global singleton pattern with a `SearchState` that holds:

- **Named indexes**: `SwissMap` of index name to `InvertedIndex`
- **RwLock**: For concurrent read access to indexes
- **Allocator**: For all internal allocations

Each `InvertedIndex` independently manages its term index, document store,
and statistics. Query tokenization uses an `ArenaAllocator` for efficient
temporary allocation.

## Disabling at Build Time

```bash
zig build -Denable-search=false
```

When disabled, `abi.search` resolves to `src/features/search/stub.zig`,
which returns `error.FeatureDisabled` for all mutating operations and safe
defaults for read-only calls (`isEnabled()` returns `false`, `stats()`
returns zeros).

## Examples

See [`examples/search.zig`](https://github.com/donaldfilimon/abi/blob/main/examples/search.zig)
for a complete working example.

## Related

- [Database](database.html) -- WDBX vector database (includes its own full-text search via `fulltext` sub-module)
- [Cache](cache.html) -- Cache search results for repeated queries
- [Storage](storage.html) -- Persistent object storage for documents
- [Architecture](architecture.html) -- Comptime feature gating pattern

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
