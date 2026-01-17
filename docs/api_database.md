# database API Reference

**Source:** `src/features/database/mod.zig`

Database feature facade providing in-memory vector database with persistence helpers.

## Features

- Vector similarity search with cosine similarity
- SIMD-accelerated operations
- Cached L2 norms for fast similarity computation
- O(1) lookup index for vector IDs
- Thread-safe operations (optional)
- **Comprehensive diagnostics** (new in 2026.01)

## Quick Start

```zig
const database = @import("src/features/database/database.zig");

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

## Diagnostics (New)

Get comprehensive debugging information about database state:

```zig
// Get diagnostics
const diag = db.diagnostics();

// Check health
if (!diag.isHealthy()) {
    std.log.warn("Database health issue detected", .{});
}

// Format for logging
const diag_str = try diag.formatToString(allocator);
defer allocator.free(diag_str);
std.log.info("{s}", .{diag_str});
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

## Thread-Safe Operations

When `thread_safe` is enabled in config:

```zig
// Thread-safe variants automatically acquire locks
try db.insertThreadSafe(id, vector, metadata);
const view = db.getThreadSafe(id);
const results = try db.searchThreadSafe(allocator, query, top_k);
_ = db.deleteThreadSafe(id);
```

## Batch Search

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
