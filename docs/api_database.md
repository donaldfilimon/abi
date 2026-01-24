# database API Reference

> Vector database (WDBX with HNSW/IVF-PQ)

**Source:** [`src/database/mod.zig`](../../src/database/mod.zig)

---

Database Module - Vector Database API

This module provides the WDBX vector database for high-performance similarity search.
It supports HNSW (Hierarchical Navigable Small World) and IVF-PQ (Inverted File with
Product Quantization) indexing algorithms.

## Features

- **Vector Storage**: Store and retrieve high-dimensional vectors with metadata
- **Similarity Search**: Find similar vectors using cosine, euclidean, or dot product
- **Full-text Search**: BM25-based text search with inverted index
- **Hybrid Search**: Combine vector and text search with configurable fusion
- **Metadata Filtering**: Filter results by metadata attributes
- **Batch Operations**: Efficient bulk insert/update/delete
- **Clustering**: K-means clustering for data analysis
- **Quantization**: Scalar and product quantization for compression
- **GPU Acceleration**: Optional GPU-accelerated distance calculations

## Quick Start

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

## Standalone Usage

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

## Advanced Features

### Hybrid Search

```zig
var engine = try db.HybridSearchEngine.init(allocator, .{
.vector_weight = 0.7,
.text_weight = 0.3,
.fusion = .rrf,  // Reciprocal Rank Fusion
});
defer engine.deinit();

const results = try engine.search(query_vector, "search text", 10);
```

### Metadata Filtering

```zig
var filter = db.FilterBuilder.init()
.eq("category", .{ .string = "tech" })
.gte("year", .{ .int = 2020 })
.build();

const results = try db.FilteredSearch.search(&handle, query, filter, 10);
```

---

## API

### `pub const Context`

<sup>**type**</sup>

Database Context for Framework integration.

The Context struct provides a high-level interface for database operations,
managing the database handle and providing convenient methods for common
operations like inserting and searching vectors.

## Thread Safety

The Context is not thread-safe. For concurrent access, use external
synchronization or create separate Context instances.

## Auto-open Behavior

If a path is provided in the configuration, the database will be automatically
opened during initialization. If no path is provided, the database must be
explicitly opened using `openDatabase()`.

## Example

```zig
var ctx = try Context.init(allocator, .{ .path = "./vectors.db" });
defer ctx.deinit();

// Insert vectors
try ctx.insertVector(1, &[_]f32{ 0.1, 0.2, 0.3 }, "metadata");

// Search
const results = try ctx.searchVectors(&query, 10);
defer allocator.free(results);
```

### `pub fn init(allocator: std.mem.Allocator, cfg: config_module.DatabaseConfig) !*Context`

<sup>**fn**</sup>

Initialize the database context.

## Parameters

- `allocator`: Memory allocator for database operations
- `cfg`: Database configuration (path, index settings, etc.)

## Returns

A pointer to the initialized Context.

## Errors

- `error.DatabaseDisabled`: Database feature is disabled at compile time
- `error.OutOfMemory`: Memory allocation failed

### `pub fn getHandle(self: *Context) !*DatabaseHandle`

<sup>**fn**</sup>

Get or create the database handle.

### `pub fn openDatabase(self: *Context, name: []const u8) !*DatabaseHandle`

<sup>**fn**</sup>

Open a database and attach it to the Context.
If a database is already open, it is closed first.
The returned handle is owned by the Context; do not close it directly.

### `pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void`

<sup>**fn**</sup>

Insert a vector into the database.

### `pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult`

<sup>**fn**</sup>

Search for similar vectors.

### `pub fn getStats(self: *Context) !Stats`

<sup>**fn**</sup>

Get database statistics.

### `pub fn optimize(self: *Context) !void`

<sup>**fn**</sup>

Optimize the database index.

---

*Generated automatically by `zig build gendocs`*
