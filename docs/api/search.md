---
title: search API
purpose: Generated API reference for search
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# search

> Search Module

Full-text search with inverted index, BM25 scoring, tokenization,
and snippet generation.

Architecture:
- Named indexes (SwissMap of name → InvertedIndex)
- Inverted index: term → PostingList (doc_id, term_freq, positions)
- BM25 scoring: IDF × TF component with configurable k1, b
- Tokenizer: lowercase, stop word removal
- Snippet: window with highest match density

**Source:** [`src/features/search/mod.zig`](../../src/features/search/mod.zig)

**Build flag:** `-Dfeat_search=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-searchconfig-searcherror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: SearchConfig) SearchError!void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L415)

Initialize the global search engine singleton.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L421)

Tear down the search engine, destroying all indexes and postings.

### <a id="pub-fn-createindex-allocator-std-mem-allocator-name-const-u8-searcherror-searchindex"></a>`pub fn createIndex(allocator: std.mem.Allocator, name: []const u8) SearchError!SearchIndex`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L438)

Create a new named full-text index. Returns `IndexAlreadyExists` if
an index with the same name exists.

### <a id="pub-fn-deleteindex-name-const-u8-searcherror-void"></a>`pub fn deleteIndex(name: []const u8) SearchError!void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L457)

Delete a named index and all its documents.

### <a id="pub-fn-indexdocument-index-name-const-u8-doc-id-const-u8-content-const-u8-searcherror-void"></a>`pub fn indexDocument( index_name: []const u8, doc_id: []const u8, content: []const u8, ) SearchError!void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L472)

Add or update a document in a named index. Tokenizes the content
and builds inverted-index postings for BM25 retrieval.

### <a id="pub-fn-deletedocument-index-name-const-u8-doc-id-const-u8-searcherror-bool"></a>`pub fn deleteDocument(index_name: []const u8, doc_id: []const u8) SearchError!bool`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L487)

Remove a document from an index. Returns `true` if the document existed.

### <a id="pub-fn-query-allocator-std-mem-allocator-index-name-const-u8-query-text-const-u8-searcherror-searchresult"></a>`pub fn query( allocator: std.mem.Allocator, index_name: []const u8, query_text: []const u8, ) SearchError![]SearchResult`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L553)

Execute a BM25-scored full-text query against a named index.
Results are sorted by relevance and include context snippets.

### <a id="pub-fn-stats-searchstats"></a>`pub fn stats() SearchStats`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L569)

Return aggregate statistics across all search indexes.

### <a id="pub-fn-saveindex-allocator-std-mem-allocator-name-const-u8-path-const-u8-searcherror-void"></a>`pub fn saveIndex(allocator: std.mem.Allocator, name: []const u8, path: []const u8) SearchError!void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L691)

Serialize a named inverted index to disk at the given path.

File format:
```
[4B] Magic "SRCH"
[2B] Version (1)
[4B] term_count
[4B] doc_count
[4B] total_doc_length (for BM25 avgdl, truncated to u32)
--- Terms section ---
per term:
[2B] term_len
[term_len bytes] term string
[4B] posting_count
per posting:
[4B] doc_id_len  (length of doc_id string)
[doc_id_len bytes] doc_id
[4B] term_freq
[4B] doc_len
--- Documents section ---
per document:
[4B] doc_id_len
[doc_id_len bytes] doc_id
[4B] content_len
[content_len bytes] content
[4B] term_count
```

### <a id="pub-fn-loadindex-allocator-std-mem-allocator-name-const-u8-path-const-u8-searcherror-void"></a>`pub fn loadIndex(allocator: std.mem.Allocator, name: []const u8, path: []const u8) SearchError!void`

<sup>**fn**</sup> | [source](../../src/features/search/mod.zig#L745)

Deserialize a named index from disk. Creates a new index with the
given name (must not already exist) and populates it from the file.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
