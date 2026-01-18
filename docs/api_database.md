# Database API Reference

**Source:** `src/database/mod.zig`

The database module provides a high-performance vector database (WDBX) with HNSW indexing, hybrid search, full-text search, and comprehensive data management capabilities.

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with database enabled
    const config = abi.Config.init().withDatabase(true);
    var fw = try abi.Framework.init(allocator, config);
    defer fw.deinit();

    // Access database features
    if (fw.database()) |db| {
        try db.insertVector(1, &[_]f32{ 0.1, 0.2, 0.3 }, "document metadata");
        const results = try db.searchVectors(&[_]f32{ 0.1, 0.2, 0.3 }, 10);
        defer allocator.free(results);
    }
}
```

---

## Core Types

### `Database`

Main database interface.

```zig
pub const Database = struct {
    pub fn init(allocator: Allocator, config: DatabaseConfig) !Database;
    pub fn deinit(self: *Database) void;
    pub fn open(self: *Database, name: []const u8) !void;
    pub fn close(self: *Database) void;
    pub fn insert(self: *Database, id: u64, vector: []const f32, metadata: ?[]const u8) !void;
    pub fn search(self: *Database, query: []const f32, k: usize) ![]SearchResult;
    pub fn delete(self: *Database, id: u64) !void;
    pub fn update(self: *Database, id: u64, vector: []const f32, metadata: ?[]const u8) !void;
    pub fn get(self: *Database, id: u64) !?VectorEntry;
    pub fn count(self: *Database) usize;
    pub fn optimize(self: *Database) !void;
};
```

### `DatabaseHandle`

Low-level database handle for WDBX operations.

```zig
pub const DatabaseHandle = struct {
    pub fn create(allocator: Allocator, path: []const u8, config: WdbxConfig) !DatabaseHandle;
    pub fn open(allocator: Allocator, path: []const u8) !DatabaseHandle;
    pub fn close(self: *DatabaseHandle) void;
    pub fn sync(self: *DatabaseHandle) !void;
    pub fn compact(self: *DatabaseHandle) !void;
};
```

### `SearchResult`

Result from similarity search.

```zig
pub const SearchResult = struct {
    id: u64,
    score: f32,
    vector: ?[]const f32,
    metadata: ?[]const u8,
};
```

### `VectorView`

Read-only view of a stored vector.

```zig
pub const VectorView = struct {
    id: u64,
    data: []const f32,
    metadata: ?[]const u8,
    created_at: i64,
    updated_at: i64,
};
```

### `Stats`

Database statistics.

```zig
pub const Stats = struct {
    count: u64,
    dimension: u32,
    index_size_bytes: u64,
    storage_size_bytes: u64,
    memory_usage_bytes: u64,
    last_optimized: ?i64,
};
```

---

## Framework Context

### `Context`

Database Context for Framework integration.

```zig
pub const Context = struct {
    pub fn getHandle(self: *Context) !*DatabaseHandle;
    pub fn openDatabase(self: *Context, name: []const u8) !DatabaseHandle;
    pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void;
    pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult;
    pub fn getStats(self: *Context) !Stats;
    pub fn optimize(self: *Context) !void;
};
```

---

## Full-Text Search

### `InvertedIndex`

Full-text inverted index for text search.

```zig
pub const InvertedIndex = struct {
    pub fn init(allocator: Allocator, config: TokenizerConfig) !InvertedIndex;
    pub fn deinit(self: *InvertedIndex) void;
    pub fn index(self: *InvertedIndex, doc_id: u64, text: []const u8) !void;
    pub fn search(self: *InvertedIndex, query: []const u8, limit: usize) ![]TextSearchResult;
    pub fn remove(self: *InvertedIndex, doc_id: u64) !void;
};
```

### `Bm25Config`

BM25 ranking configuration.

```zig
pub const Bm25Config = struct {
    k1: f32 = 1.2,
    b: f32 = 0.75,
    delta: f32 = 0.5,
};
```

### `TokenizerConfig`

Text tokenization configuration.

```zig
pub const TokenizerConfig = struct {
    lowercase: bool = true,
    remove_stopwords: bool = true,
    stemming: bool = false,
    min_token_length: u32 = 2,
    max_token_length: u32 = 64,
    language: Language = .english,
};
```

### `TextSearchResult`

Full-text search result.

```zig
pub const TextSearchResult = struct {
    doc_id: u64,
    score: f32,
    matched_terms: []const []const u8,
    snippet: ?[]const u8,
};
```

### `QueryParser`

Parse and validate search queries.

```zig
pub const QueryParser = struct {
    pub fn init(allocator: Allocator) QueryParser;
    pub fn parse(self: *QueryParser, query: []const u8) !ParsedQuery;
    pub fn validate(self: *QueryParser, query: []const u8) !bool;
};
```

---

## Hybrid Search

### `HybridSearchEngine`

Combined vector + text search.

```zig
pub const HybridSearchEngine = struct {
    pub fn init(allocator: Allocator, config: HybridConfig) !HybridSearchEngine;
    pub fn deinit(self: *HybridSearchEngine) void;
    pub fn search(
        self: *HybridSearchEngine,
        vector_query: ?[]const f32,
        text_query: ?[]const u8,
        limit: usize,
    ) ![]HybridResult;
};
```

### `HybridConfig`

Hybrid search configuration.

```zig
pub const HybridConfig = struct {
    vector_weight: f32 = 0.5,
    text_weight: f32 = 0.5,
    fusion_method: FusionMethod = .reciprocal_rank,
    rerank: bool = true,
    min_score: f32 = 0.0,
};
```

### `HybridResult`

Combined search result.

```zig
pub const HybridResult = struct {
    id: u64,
    combined_score: f32,
    vector_score: ?f32,
    text_score: ?f32,
    metadata: ?[]const u8,
};
```

### `FusionMethod`

Score fusion methods.

```zig
pub const FusionMethod = enum {
    linear,
    reciprocal_rank,
    convex_combination,
    max,
    min,
};
```

---

## Filtering

### `FilterBuilder`

Build filter expressions for search.

```zig
pub const FilterBuilder = struct {
    pub fn init(allocator: Allocator) FilterBuilder;
    pub fn where(self: *FilterBuilder, field: []const u8) *FilterBuilder;
    pub fn eq(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn ne(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn gt(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn lt(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn gte(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn lte(self: *FilterBuilder, value: MetadataValue) *FilterBuilder;
    pub fn contains(self: *FilterBuilder, value: []const u8) *FilterBuilder;
    pub fn in(self: *FilterBuilder, values: []const MetadataValue) *FilterBuilder;
    pub fn andWhere(self: *FilterBuilder, field: []const u8) *FilterBuilder;
    pub fn orWhere(self: *FilterBuilder, field: []const u8) *FilterBuilder;
    pub fn build(self: *FilterBuilder) FilterExpression;
};
```

### `FilterExpression`

Compiled filter expression.

```zig
pub const FilterExpression = struct {
    pub fn matches(self: *FilterExpression, metadata: []const u8) bool;
    pub fn toSql(self: *FilterExpression) []const u8;
};
```

### `FilterOperator`

Filter comparison operators.

```zig
pub const FilterOperator = enum {
    eq,
    ne,
    gt,
    gte,
    lt,
    lte,
    contains,
    starts_with,
    ends_with,
    in,
    not_in,
    between,
    is_null,
    is_not_null,
};
```

### `MetadataValue`

Typed metadata value for filtering.

```zig
pub const MetadataValue = union(enum) {
    string: []const u8,
    int: i64,
    float: f64,
    bool: bool,
    null: void,
    array: []const MetadataValue,
};
```

### `MetadataStore`

Structured metadata storage.

```zig
pub const MetadataStore = struct {
    pub fn init(allocator: Allocator) MetadataStore;
    pub fn set(self: *MetadataStore, id: u64, key: []const u8, value: MetadataValue) !void;
    pub fn get(self: *MetadataStore, id: u64, key: []const u8) ?MetadataValue;
    pub fn delete(self: *MetadataStore, id: u64, key: []const u8) !void;
    pub fn query(self: *MetadataStore, filter: FilterExpression) ![]u64;
};
```

### `FilteredSearch`

Search with filters applied.

```zig
pub const FilteredSearch = struct {
    pub fn init(db: *Database, filter: FilterExpression) FilteredSearch;
    pub fn search(self: *FilteredSearch, query: []const f32, limit: usize) ![]FilteredResult;
};
```

---

## Batch Operations

### `BatchProcessor`

High-performance batch operations.

```zig
pub const BatchProcessor = struct {
    pub fn init(allocator: Allocator, config: BatchConfig) !BatchProcessor;
    pub fn deinit(self: *BatchProcessor) void;
    pub fn add(self: *BatchProcessor, record: BatchRecord) !void;
    pub fn flush(self: *BatchProcessor) !BatchResult;
    pub fn process(self: *BatchProcessor, records: []const BatchRecord) !BatchResult;
};
```

### `BatchConfig`

Batch processing configuration.

```zig
pub const BatchConfig = struct {
    batch_size: u32 = 1000,
    parallel_workers: u32 = 4,
    retry_on_failure: bool = true,
    max_retries: u32 = 3,
    timeout_ms: u32 = 30000,
};
```

### `BatchRecord`

Single record in a batch.

```zig
pub const BatchRecord = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
    operation: BatchOperation,
};

pub const BatchOperation = enum {
    insert,
    update,
    upsert,
    delete,
};
```

### `BatchResult`

Batch operation result.

```zig
pub const BatchResult = struct {
    success_count: u64,
    failure_count: u64,
    errors: []BatchError,
    duration_ms: u64,
};
```

### `BatchWriter`

Streaming batch writer.

```zig
pub const BatchWriter = struct {
    pub fn init(db: *Database, config: BatchConfig) !BatchWriter;
    pub fn write(self: *BatchWriter, record: BatchRecord) !void;
    pub fn commit(self: *BatchWriter) !BatchResult;
    pub fn abort(self: *BatchWriter) void;
};
```

### `BatchImporter`

Import data from external sources.

```zig
pub const BatchImporter = struct {
    pub fn fromCsv(allocator: Allocator, path: []const u8, config: CsvConfig) !BatchImporter;
    pub fn fromJson(allocator: Allocator, path: []const u8) !BatchImporter;
    pub fn fromParquet(allocator: Allocator, path: []const u8) !BatchImporter;
    pub fn import(self: *BatchImporter, db: *Database) !ImportResult;
};
```

---

## Clustering

### `KMeans`

K-means clustering for vector data.

```zig
pub const KMeans = struct {
    pub fn init(allocator: Allocator, k: u32, dimension: u32) !KMeans;
    pub fn deinit(self: *KMeans) void;
    pub fn fit(self: *KMeans, vectors: []const []const f32, options: FitOptions) !FitResult;
    pub fn predict(self: *KMeans, vector: []const f32) u32;
    pub fn getCentroids(self: *KMeans) []const []const f32;
};
```

### `ClusterStats`

Cluster statistics.

```zig
pub const ClusterStats = struct {
    cluster_id: u32,
    size: u64,
    centroid: []const f32,
    inertia: f32,
    silhouette_score: f32,
};
```

### `FitOptions`

K-means fitting options.

```zig
pub const FitOptions = struct {
    max_iterations: u32 = 300,
    tolerance: f32 = 1e-4,
    n_init: u32 = 10,
    init_method: InitMethod = .kmeans_plus_plus,
    random_seed: ?u64 = null,
};
```

### Distance Functions

```zig
pub fn euclideanDistance(a: []const f32, b: []const f32) f32;
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32;
pub fn dotProduct(a: []const f32, b: []const f32) f32;
pub fn manhattanDistance(a: []const f32, b: []const f32) f32;
```

---

## Quantization

### `quantization`

Vector quantization for memory efficiency.

```zig
pub const quantization = struct {
    pub const ProductQuantizer = struct {
        pub fn init(allocator: Allocator, config: PQConfig) !ProductQuantizer;
        pub fn train(self: *ProductQuantizer, vectors: []const []const f32) !void;
        pub fn encode(self: *ProductQuantizer, vector: []const f32) ![]u8;
        pub fn decode(self: *ProductQuantizer, codes: []const u8) ![]f32;
    };

    pub const ScalarQuantizer = struct {
        pub fn init(bits: u8) ScalarQuantizer;
        pub fn quantize(self: *ScalarQuantizer, value: f32) u8;
        pub fn dequantize(self: *ScalarQuantizer, code: u8) f32;
    };
};
```

---

## GPU Acceleration

### `gpu_accel`

GPU-accelerated database operations.

```zig
pub const gpu_accel = struct {
    pub fn init(db: *Database, gpu: *Gpu) !GpuAccelerator;
    pub fn searchBatch(self: *GpuAccelerator, queries: []const []const f32, k: usize) ![][]SearchResult;
    pub fn buildIndex(self: *GpuAccelerator) !void;
    pub fn isEnabled(self: *GpuAccelerator) bool;
};
```

---

## Data Formats

### `formats`

Import/export data formats.

```zig
pub const formats = struct {
    pub const csv = struct {
        pub fn read(allocator: Allocator, path: []const u8) ![]VectorRecord;
        pub fn write(path: []const u8, records: []const VectorRecord) !void;
    };

    pub const json = struct {
        pub fn read(allocator: Allocator, path: []const u8) ![]VectorRecord;
        pub fn write(path: []const u8, records: []const VectorRecord) !void;
    };

    pub const parquet = struct {
        pub fn read(allocator: Allocator, path: []const u8) ![]VectorRecord;
        pub fn write(path: []const u8, records: []const VectorRecord) !void;
    };
};
```

---

## Storage

### `storage`

Low-level storage operations.

```zig
pub const storage = struct {
    pub fn backup(db: *Database, path: []const u8) !void;
    pub fn restore(db: *Database, path: []const u8) !void;
    pub fn compact(db: *Database) !void;
    pub fn vacuum(db: *Database) !void;
    pub fn getStorageInfo(db: *Database) StorageInfo;
};
```

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-database` | true | Vector database (WDBX) |

---

## Security Notes

- Backup paths are restricted to `backups/` directory
- Path traversal (`../`) is blocked
- Absolute paths are rejected for security

---

## Related Documentation

- [Database Guide](database.md) - Comprehensive database guide
- [Vector Database Tutorial](tutorials/vector-database.md) - Step-by-step tutorial
- [WDBX Architecture](architecture/overview.md) - Internal architecture

---

*See also: [Framework API](api_abi.md) | [AI API](api_ai.md) | [GPU API](api_gpu.md)*
