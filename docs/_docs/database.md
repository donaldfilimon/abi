---
title: Database (WDBX)
description: WDBX vector database with HNSW/IVF-PQ indexing, hybrid search, and GPU acceleration
section: Data
order: 1
---

# Database (WDBX)

The database module (`src/features/database/mod.zig`) provides WDBX, a
high-performance vector database for similarity search, full-text retrieval,
and hybrid queries. It supports HNSW and IVF-PQ indexing algorithms, metadata
filtering, batch operations, clustering, quantization, and optional GPU
acceleration for distance calculations.

## Features

- **Vector storage**: Store and retrieve high-dimensional vectors with metadata
- **Similarity search**: Cosine, euclidean, and dot product distance metrics
- **HNSW indexing**: Hierarchical Navigable Small World graphs for fast ANN search
- **IVF-PQ indexing**: Inverted File with Product Quantization for compressed search
- **Full-text search**: BM25-based text search with inverted index
- **Hybrid search**: Combine vector and text search with Reciprocal Rank Fusion
- **Metadata filtering**: Filter results by attribute conditions
- **Batch operations**: Efficient bulk insert, update, and delete
- **Clustering**: K-means clustering with silhouette scoring and elbow method
- **Quantization**: Scalar and product quantization for vector compression
- **GPU acceleration**: Optional GPU-accelerated distance calculations
- **Distributed**: Sharding, block exchange, and version vectors for multi-node deployment
- **Novel indexes**: DiskANN and ScaNN research implementations for billion-scale ANN
- **Parallel HNSW**: Multi-threaded index building

## Build Configuration

```bash
# Enable (default)
zig build -Denable-database=true

# Disable
zig build -Denable-database=false
```

**Namespace**: `abi.database`

## Quick Start

### Framework Integration

```zig
const abi = @import("abi");

// Initialize with database enabled
var fw = try abi.Framework.init(allocator, .{
    .database = .{ .path = "./vectors.db" },
});
defer fw.deinit();

// Get database context
const db_ctx = try fw.getDatabase();

// Insert a vector
try db_ctx.insertVector(1, &[_]f32{ 0.1, 0.2, 0.3, 0.4 }, "metadata");

// Search for similar vectors
const results = try db_ctx.searchVectors(&query_vector, 10);
defer allocator.free(results);
```

### Standalone Usage

```zig
const db = abi.database;

// Open or create a database
var handle = try db.open(allocator, "vectors.db");
defer db.close(&handle);

// Insert vectors
try db.insert(&handle, 1, &[_]f32{ 0.1, 0.2, 0.3 }, "doc1");
try db.insert(&handle, 2, &[_]f32{ 0.4, 0.5, 0.6 }, "doc2");

// Search
const results = try db.search(&handle, allocator, &query, 5);
defer allocator.free(results);

for (results) |result| {
    std.debug.print("ID: {}, Score: {d}\n", .{ result.id, result.score });
}
```

## API Reference

### Core Operations

| Function | Description |
|----------|-------------|
| `open(allocator, name)` | Open a database by name |
| `openOrCreate(allocator, path)` | Open or create a database file |
| `openFromFile(allocator, path)` | Open from a specific file path |
| `connect(allocator, name)` | Connect to a database |
| `close(handle)` | Close a database handle |
| `insert(handle, id, vector, metadata)` | Insert a vector with optional metadata |
| `search(handle, allocator, query, top_k)` | Search for similar vectors |
| `get(handle, id)` | Get a vector by ID |
| `update(handle, id, vector)` | Update an existing vector |
| `remove(handle, id)` | Remove a vector by ID |
| `list(handle, allocator, limit)` | List vectors up to a limit |
| `stats(handle)` | Get database statistics |
| `optimize(handle)` | Optimize index structures |
| `backup(handle, path)` | Backup database to file |
| `restore(handle, path)` | Restore database from backup |

### Types

| Type | Description |
|------|-------------|
| `DatabaseHandle` | Opaque handle to an open database |
| `SearchResult` | Search result with ID and similarity score |
| `VectorView` | Read-only view of a stored vector |
| `Stats` | Database statistics (vector count, dimensions, etc.) |
| `DiagnosticsInfo` | Detailed diagnostic information |
| `DatabaseError` | Error type for database operations |

### Hybrid Search

Combine vector similarity with full-text search for richer results:

```zig
var engine = try db.HybridSearchEngine.init(allocator, .{
    .vector_weight = 0.7,
    .text_weight = 0.3,
    .fusion = .rrf,  // Reciprocal Rank Fusion
});
defer engine.deinit();

const results = try engine.search(query_vector, "search text", 10);
```

Configuration via `HybridConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `vector_weight` | 0.7 | Weight for vector similarity scores |
| `text_weight` | 0.3 | Weight for BM25 text scores |
| `fusion` | `.rrf` | Fusion method (Reciprocal Rank Fusion) |

### Metadata Filtering

Filter search results by metadata attributes:

```zig
var filter = db.FilterBuilder.init()
    .eq("category", .{ .string = "tech" })
    .gte("year", .{ .int = 2020 })
    .build();

const results = try db.FilteredSearch.search(&handle, query, filter, 10);
```

### Batch Operations

Efficient bulk operations for large data loads:

```zig
var batch_proc = try db.BatchProcessor.init(allocator, .{
    .batch_size = 1000,
});
defer batch_proc.deinit();

// Add records to batch
try batch_proc.add(.{ .id = 1, .vector = vec1, .metadata = "doc1" });
try batch_proc.add(.{ .id = 2, .vector = vec2, .metadata = "doc2" });

// Flush batch to database
const result = try batch_proc.flush(&handle);
```

### Clustering

K-means clustering for data analysis:

```zig
var kmeans = try db.KMeans.init(allocator, .{
    .k = 5,
    .max_iterations = 100,
});
defer kmeans.deinit();

const fit_result = try kmeans.fit(vectors);
const score = try db.silhouetteScore(vectors, fit_result.labels);
```

### Quantization

Compress vectors to reduce memory usage:

```zig
// Scalar quantization
var sq = try db.ScalarQuantizer.init(allocator, vectors);
defer sq.deinit();

// Product quantization (PQ)
var pq = try db.ProductQuantizer.init(allocator, .{
    .num_subquantizers = 8,
    .bits_per_code = 8,
});
defer pq.deinit();
```

## Sub-Modules

| Module | Description |
|--------|-------------|
| `database` | Core database engine |
| `wdbx` | WDBX handle and storage types |
| `hnsw` | HNSW graph index |
| `index` | Index management |
| `fulltext` | BM25 full-text search with inverted index |
| `hybrid` | Hybrid vector + text search with RRF fusion |
| `filter` | Metadata filtering with builder pattern |
| `batch` | Batch insert/update/delete operations |
| `clustering` | K-means clustering and evaluation |
| `quantization` | Scalar and product quantization |
| `gpu_accel` | GPU-accelerated distance calculations |
| `storage` / `storage_v2` | Persistence layer (v2: bloom filter, CRC32) |
| `formats` | Import/export formats (GGUF, ZON, unified) |
| `distributed` | Sharding, block exchange, version vectors |
| `diskann` | DiskANN index (billion-scale ANN research) |
| `scann` | ScaNN index (billion-scale ANN research) |
| `parallel_search` | Parallel search utilities |
| `parallel_hnsw` | Multi-threaded HNSW index building |
| `block_chain` | Conversation block chain for persona routing |

## CLI Commands

```bash
zig build run -- db stats           # Show database statistics
zig build run -- db add             # Add vectors to database
zig build run -- db search          # Search for similar vectors
zig build run -- db backup          # Backup database to file
```

## Disabling at Build Time

```bash
zig build -Denable-database=false
```

When disabled, `abi.database` resolves to `src/features/database/stub.zig`,
which returns `error.DatabaseDisabled` for all operations. The stub preserves
the full type surface (all types and function signatures are available) so
that code can compile against the disabled module without conditional imports.

## Examples

See [`examples/database.zig`](https://github.com/donaldfilimon/abi/blob/main/examples/database.zig)
for a complete working example.

## Related

- [Search](search.html) -- Full-text BM25 search (standalone module)
- [Cache](cache.html) -- In-memory caching for query results
- [GPU](gpu.html) -- GPU-accelerated distance calculations
- [Storage](storage.html) -- Unified file/object storage

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
