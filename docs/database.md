---
title: "database"
tags: []
---
# Database (WDBX)
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Module-Database-orange?style=for-the-badge&logo=mongodb&logoColor=white" alt="Database Module"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/Type-Vector_DB-blue?style=for-the-badge" alt="Vector DB"/>
</p>

<p align="center">
  <a href="#usage">Usage</a> •
  <a href="#batch-operations">Batch Ops</a> •
  <a href="#full-text-search">Full-Text</a> •
  <a href="#metadata-filtering">Filtering</a> •
  <a href="#cli-commands">CLI</a>
</p>

---

**WDBX** is ABI's built-in vector database solution, optimized for high-dimensional embedding storage and retrieval.

## Feature Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Vector Search** | Dot product, Cosine, L2 Euclidean | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Zero-Copy** | Zig's memory management | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Backup/Restore** | Secure snapshotting | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Batch Operations** | High-throughput bulk ops | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Full-Text Search** | BM25 ranking | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Metadata Filtering** | Rich query operators | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Thread Safety** | Concurrent access | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Usage

```zig
const wdbx = abi.wdbx;

var db = try wdbx.createDatabase(allocator, .{ .dimension = 1536 });
defer db.deinit();

// Insert
try db.insertVector(id, embedding_slice);

// Search
const results = try db.searchVectors(query_embedding, 10);
```

## Security: Backup & Restore

> [!IMPORTANT]
> **Security Advisory**: Improper path validation in versions prior to 0.2.1 allowed directory traversal.

**Safe Practices**:

1.  **Restricted Directory**: All backup/restore operations are confined to the `backups/` directory.
2.  **Input Validation**: Filenames must **not** contain:
    - Path traversal sequences (`..`)
    - Absolute paths (`/etc/passwd`, `C:\Windows`)
    - Drive letters
3.  **Validation Error**: The API will return `PathValidationError` if an unsafe path is detected.

```zig
// GOOD
try db.restore("snapshot_2025.db");

// BAD (Will fail)
try db.restore("../../../secret.txt");
```

## Batch Operations

Efficient bulk operations for high-throughput scenarios.

```zig
const batch = @import("abi").wdbx.batch;

// Configure batch processing
const config = batch.BatchConfig{
    .batch_size = 1000,
    .parallel_workers = 4,
    .retry_failed = true,
    .report_progress = true,
};

// Prepare records
var records = std.ArrayListUnmanaged(batch.BatchRecord){};
defer records.deinit(allocator);

try records.append(allocator, .{
    .id = 1,
    .vector = embedding,
    .metadata = "{\"category\": \"tech\"}",
    .text = "Document content for full-text indexing",
});

// Execute batch insert
const result = try db.batchInsert(records.items, config);
std.debug.print("Processed: {}, Throughput: {d:.2} items/sec\n", .{
    result.total_processed,
    result.throughput,
});
```

## Full-Text Search

BM25-ranked full-text search with configurable tokenization.

```zig
const fulltext = @import("abi").wdbx.fulltext;

// Configure BM25 scoring
const bm25_config = fulltext.Bm25Config{
    .k1 = 1.2,          // Term frequency saturation
    .b = 0.75,          // Document length normalization
    .title_boost = 2.0, // Boost title matches
};

// Configure tokenizer
const tokenizer_config = fulltext.TokenizerConfig{
    .lowercase = true,
    .enable_stemming = true,
    .filter_stop_words = true,
    .min_token_length = 2,
};

// Search
const results = try db.textSearch("machine learning", .{
    .bm25 = bm25_config,
    .tokenizer = tokenizer_config,
    .max_results = 10,
});
```

## Metadata Filtering

Pre-filter and post-filter search with rich operators.

```zig
const filter = @import("abi").wdbx.filter;

// Build filter expression
const expr = filter.Filter.init()
    .field("category").eq(.{ .string = "tech" })
    .and()
    .field("year").gte(.{ .integer = 2023 })
    .and()
    .field("status").in_list(.{ .string_list = &.{ "published", "draft" } });

// Apply filter to vector search
const results = try db.searchVectors(query_embedding, 10, .{
    .filter = expr,
    .filter_strategy = .pre_filter, // or .post_filter
});
```

### Filter Operators

| Operator | Description |
|----------|-------------|
| `eq`, `ne` | Equal / Not equal |
| `gt`, `gte`, `lt`, `lte` | Numeric comparisons |
| `contains`, `starts_with`, `ends_with` | String matching |
| `in_list`, `not_in_list` | List membership |
| `exists`, `not_exists` | Field presence |
| `regex`, `between` | Pattern / Range matching |

---

## CLI Commands

```bash
# Database operations
zig build run -- db stats                              # Show statistics
zig build run -- db add --id 1 --embed "text"          # Add with embedding
zig build run -- db add --id 2 --vector "1.0,2.0,3.0"  # Add raw vector
zig build run -- db query --text "search term" --k 10  # Search
zig build run -- db optimize                           # Optimize indices

# Backup and restore
zig build run -- db backup --path snapshot.db
zig build run -- db restore --path snapshot.db

# HTTP API server
zig build run -- db serve --port 8080
```

```bash
# Dataset conversion
zig build run -- convert dataset --input data.bin --output data.wdbx --format to-wdbx
zig build run -- convert dataset --input data.wdbx --output data.bin --format to-tokenbin
```

---

## New in 2026.01

### Diagnostics

```zig
const diag = db.diagnostics();
if (!diag.isHealthy()) { try db.rebuildNormCache(); }
```

Fields: `vector_count`, `dimension`, `memory.total_bytes`, `index_health`, `norm_cache_health`

### Performance Optimizations

```zig
var db = try database.Database.initWithConfig(allocator, "fast-db", .{
    .cache_norms = true,       // Pre-computed L2 norms
    .initial_capacity = 10000, // Pre-allocate
    .thread_safe = true,       // Enable concurrent access
});

// O(1) lookup via hash index
const view = db.get(vector_id);

// Thread-safe variants
try db.insertThreadSafe(id, vector, metadata);
const results = try db.searchThreadSafe(allocator, query, top_k);

// Batch search
const all_results = try db.searchBatch(allocator, queries, top_k);
```

---

## API Reference

**Source:** `src/database/mod.zig`

### Quick Start

```zig
const database = @import("src/database/database.zig");

// Create a database with configuration
var db = try database.Database.initWithConfig(allocator, "my-vectors", .{
    .cache_norms = true,
    .initial_capacity = 1000,
    .thread_safe = true,
});
defer db.deinit();

// Insert vectors
try db.insert(1, &[_]f32{ 1.0, 0.0, 0.0 }, "metadata");
try db.insert(2, &[_]f32{ 0.0, 1.0, 0.0 }, null);

// Search for similar vectors
const results = try db.search(allocator, &[_]f32{ 0.9, 0.1, 0.0 }, 5);
defer allocator.free(results);

for (results) |result| {
    std.log.info("ID: {d}, Score: {d:.4}", .{ result.id, result.score });
}
```

### DiagnosticsInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `[]const u8` | Database name |
| `vector_count` | `usize` | Number of stored vectors |
| `dimension` | `usize` | Vector dimension |
| `memory` | `MemoryStats` | Memory usage breakdown |
| `config` | `ConfigStatus` | Configuration status |
| `index_health` | `f32` | Index integrity (1.0 = healthy) |
| `norm_cache_health` | `f32` | Norm cache integrity (1.0 = healthy) |

### MemoryStats Fields

| Field | Type | Description |
|-------|------|-------------|
| `vector_bytes` | `usize` | Memory for vector data |
| `norm_cache_bytes` | `usize` | Memory for norm cache |
| `metadata_bytes` | `usize` | Memory for metadata |
| `index_bytes` | `usize` | Memory for index structures |
| `total_bytes` | `usize` | Total memory footprint |
| `efficiency` | `f32` | Data bytes / total bytes ratio |

### Thread-Safe Operations

When `thread_safe` is enabled in config:

```zig
// Thread-safe variants automatically acquire locks
try db.insertThreadSafe(id, vector, metadata);
const view = db.getThreadSafe(id);
const results = try db.searchThreadSafe(allocator, query, top_k);
_ = db.deleteThreadSafe(id);
```

### Batch Search

Search multiple queries efficiently:

```zig
const queries = &[_][]const f32{
    &[_]f32{ 1.0, 0.0, 0.0 },
    &[_]f32{ 0.0, 1.0, 0.0 },
};

const all_results = try db.searchBatch(allocator, queries, 5);
defer {
    for (all_results) |res| allocator.free(res);
    allocator.free(all_results);
}
```

---

## See Also

<table>
<tr>
<td>

### Related Guides
- [AI & Agents](ai.md) — Embedding generation for vectors
- [GPU Acceleration](gpu.md) — GPU-accelerated search
- [Monitoring](monitoring.md) — Database metrics
- [Abbey-Aviva Research](research/abbey-aviva-abi-wdbx-framework.md) — WDBX architecture design

</td>
<td>

### Resources
- [Troubleshooting](troubleshooting.md) — Path validation and issues
- [API Reference](../API_REFERENCE.md) — Database API details
- [Examples](../examples/) — Database code samples

</td>
</tr>
</table>

---

<p align="center">
  <a href="gpu.md">← GPU Guide</a> •
  <a href="docs-index.md">Documentation Index</a> •
  <a href="network.md">Network Guide →</a>
</p>
