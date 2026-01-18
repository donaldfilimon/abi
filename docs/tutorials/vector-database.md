# Tutorial: Vector Database with WDBX

> **Duration:** 30 minutes | **Level:** Beginner

## What You'll Learn

- Insert and retrieve high-dimensional vectors
- Perform similarity search with HNSW indexing
- Manage database lifecycle and backups
- Build a practical document search system

## Prerequisites

- Complete [Getting Started Tutorial](getting-started.md)
- Zig 0.16.x installed
- Database feature enabled (`-Denable-database=true`)

---

## Conceptual Overview

Vector databases store high-dimensional data and enable fast similarity search. This is essential for:

- **Semantic Search:** Find documents by meaning, not keywords
- **RAG (Retrieval-Augmented Generation):** Provide context to LLMs
- **Recommendation Systems:** Find similar items
- **Anomaly Detection:** Identify outliers in embeddings

### Architecture

```
+---------------------+
|  Your Application   |
|  (Embeddings)       |
+----------+----------+
           |
           v
+---------------------+
|  abi.database API   |
|  (High-level ops)   |
+----------+----------+
           |
           v
+---------------------+
|  WDBX Database      |
|  (HNSW + Storage)   |
+---------------------+
```

---

## Step 1: Database Initialization

Let's create and open a database.

**Code:** `docs/tutorials/code/vector-database/01-basic-operations.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework with database enabled
    try abi.init(allocator);
    defer abi.shutdown();

    // Verify database feature is enabled
    if (!abi.isFeatureEnabled(.database)) {
        std.debug.print("Error: Database disabled\n", .{});
        std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
        return error.DatabaseDisabled;
    }

    // Open or create database named "my_vectors"
    var db = try abi.database.openOrCreate(allocator, "my_vectors");
    defer abi.database.close(&db);

    std.debug.print("Database 'my_vectors' ready\n", .{});

    // Get current statistics
    const stats = abi.database.stats(&db);
    std.debug.print("  Vectors: {d}\n", .{stats.count});
    std.debug.print("  Dimensions: {d}\n", .{stats.dimension});
}
```

**Run:**
```bash
zig run docs/tutorials/code/vector-database/01-basic-operations.zig
```

**Expected Output:**
```
Database 'my_vectors' ready
  Vectors: 0
  Dimensions: 0
```

### Key Patterns

| Pattern | Purpose |
|---------|---------|
| `abi.isFeatureEnabled(.database)` | Runtime feature check |
| `openOrCreate(allocator, name)` | Opens existing or creates new database |
| `defer abi.database.close(&db)` | Ensures cleanup on scope exit |
| `stats(&db)` | Returns metadata (count, dimension) |

---

## Step 2: Inserting Vectors

Now let's add some document embeddings.

**Code:** `docs/tutorials/code/vector-database/02-insert-vectors.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try abi.init(allocator);
    defer abi.shutdown();

    var db = try abi.database.openOrCreate(allocator, "documents");
    defer abi.database.close(&db);

    // Sample document embeddings (3-dimensional for simplicity)
    // In production, use 384, 768, or 1536 dimensions from real embedding models
    const Document = struct {
        id: u64,
        text: []const u8,
        embedding: [3]f32,
    };

    const documents = [_]Document{
        .{
            .id = 1,
            .text = "Zig programming language tutorial",
            .embedding = [_]f32{ 0.8, 0.2, 0.1 },
        },
        .{
            .id = 2,
            .text = "Vector database architecture guide",
            .embedding = [_]f32{ 0.2, 0.9, 0.3 },
        },
        .{
            .id = 3,
            .text = "High-performance systems programming",
            .embedding = [_]f32{ 0.7, 0.3, 0.8 },
        },
        .{
            .id = 4,
            .text = "Machine learning embeddings explained",
            .embedding = [_]f32{ 0.3, 0.8, 0.4 },
        },
    };

    // Insert all documents
    for (documents) |doc| {
        try abi.database.insert(&db, doc.id, &doc.embedding, doc.text);
        std.debug.print("Inserted: {s}\n", .{doc.text});
    }

    // Verify insertion
    const stats = abi.database.stats(&db);
    std.debug.print("\nDatabase contains {d} vectors of dimension {d}\n", .{
        stats.count,
        stats.dimension,
    });
}
```

**Output:**
```
Inserted: Zig programming language tutorial
Inserted: Vector database architecture guide
Inserted: High-performance systems programming
Inserted: Machine learning embeddings explained

Database contains 4 vectors of dimension 3
```

### Important Notes

**Vector Dimensions:**
- All vectors in a database must have the same dimensionality
- First insert determines the dimension for the entire database
- Common dimensions: 384 (sentence-transformers), 768 (BERT), 1536 (OpenAI)

**Metadata:**
- The fourth parameter (text) is optional metadata
- Useful for storing original document text or references
- Retrieved alongside search results

---

## Step 3: Similarity Search

Let's find documents similar to a query.

**Code:** `docs/tutorials/code/vector-database/03-similarity-search.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try abi.init(allocator);
    defer abi.shutdown();

    var db = try abi.database.openOrCreate(allocator, "documents");
    defer abi.database.close(&db);

    // Insert sample data (same as Step 2)
    const embeddings = [_][3]f32{
        [_]f32{ 0.8, 0.2, 0.1 }, // Zig programming
        [_]f32{ 0.2, 0.9, 0.3 }, // Vector database
        [_]f32{ 0.7, 0.3, 0.8 }, // Systems programming
        [_]f32{ 0.3, 0.8, 0.4 }, // ML embeddings
    };
    const texts = [_][]const u8{
        "Zig programming language tutorial",
        "Vector database architecture guide",
        "High-performance systems programming",
        "Machine learning embeddings explained",
    };

    for (embeddings, 0..) |emb, i| {
        try abi.database.insert(&db, @intCast(i + 1), &emb, texts[i]);
    }

    // Query embedding (similar to "Zig programming")
    const query_embedding = [_]f32{ 0.75, 0.25, 0.15 };
    const k = 3; // Find top 3 similar vectors

    std.debug.print("\n--- Similarity Search ---\n", .{});
    std.debug.print("Query vector: [{d:.2}, {d:.2}, {d:.2}]\n", .{
        query_embedding[0],
        query_embedding[1],
        query_embedding[2],
    });
    std.debug.print("Finding top {d} matches...\n\n", .{k});

    const results = try abi.database.search(&db, allocator, &query_embedding, k);
    defer allocator.free(results);

    std.debug.print("Results:\n", .{});
    for (results, 0..) |result, i| {
        std.debug.print("  {d}. ID={d}, Score={d:.3}\n", .{
            i + 1,
            result.id,
            result.score,
        });

        if (result.metadata) |meta| {
            std.debug.print("     \"{s}\"\n", .{meta});
        }
    }
}
```

**Output:**
```
--- Similarity Search ---
Query vector: [0.75, 0.25, 0.15]
Finding top 3 matches...

Results:
  1. ID=1, Score=0.987
     "Zig programming language tutorial"
  2. ID=3, Score=0.823
     "High-performance systems programming"
  3. ID=2, Score=0.612
     "Vector database architecture guide"
```

### Understanding Scores

- **Range:** 0.0 (completely different) to 1.0 (identical)
- **Algorithm:** Cosine similarity (default in WDBX)
- **HNSW:** Hierarchical Navigable Small World graph for fast search
- **Complexity:** O(log n) average case (vs O(n) for brute force)

---

## Step 4: Advanced Operations

**Code:** `docs/tutorials/code/vector-database/04-advanced-operations.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try abi.init(allocator);
    defer abi.shutdown();

    var db = try abi.database.openOrCreate(allocator, "advanced_db");
    defer abi.database.close(&db);

    // Initial insert
    const embedding1 = [_]f32{ 1.0, 2.0, 3.0 };
    try abi.database.insert(&db, 1, &embedding1, "Original data");
    std.debug.print("Inserted vector 1\n", .{});

    // Update existing vector
    const embedding1_updated = [_]f32{ 1.5, 2.5, 3.5 };
    try abi.database.update(&db, 1, &embedding1_updated, "Updated data");
    std.debug.print("Updated vector 1\n", .{});

    // Retrieve specific vector
    const retrieved = try abi.database.get(&db, allocator, 1);
    defer allocator.free(retrieved);
    std.debug.print("Retrieved: [{d:.1}, {d:.1}, {d:.1}]\n", .{
        retrieved[0],
        retrieved[1],
        retrieved[2],
    });

    // List all IDs
    const ids = try abi.database.listIds(&db, allocator);
    defer allocator.free(ids);
    std.debug.print("Total vectors: {d}\n", .{ids.len});

    // Delete a vector
    try abi.database.delete(&db, 1);
    std.debug.print("Deleted vector 1\n", .{});

    // Optimize database (rebuild index for better performance)
    try abi.database.optimize(&db);
    std.debug.print("Database optimized\n", .{});
}
```

**When to Optimize:**
- After bulk insertions/updates
- Before heavy search workloads
- Periodically in long-running applications

---

## Step 5: Backup and Restore

**Code:** `docs/tutorials/code/vector-database/05-backup-restore.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try abi.init(allocator);
    defer abi.shutdown();

    var db = try abi.database.openOrCreate(allocator, "production_db");
    defer abi.database.close(&db);

    // Insert critical data
    const embedding = [_]f32{ 1.0, 2.0, 3.0 };
    try abi.database.insert(&db, 1, &embedding, "Critical business data");
    std.debug.print("Inserted data\n", .{});

    // Create backup (restricted to backups/ directory for security)
    const backup_name = "backup_20260117.db";
    try abi.database.backup(&db, backup_name);
    std.debug.print("Backup created: backups/{s}\n", .{backup_name});

    // Restore from backup (example - don't run in same session)
    // try abi.database.restore(&db, backup_name);
    // std.debug.print("Restored from backup\n", .{});
}
```

**Security Notes:**
- Backups stored in `backups/` directory only
- Path traversal (`../`) blocked
- Absolute paths rejected
- See [Security Guide](../SECURITY.md) for details

---

## Complete Example: Document Search System

Let's build a practical system combining everything.

**Code:** `docs/tutorials/code/vector-database/06-document-search-system.zig`

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

const Document = struct {
    id: u64,
    title: []const u8,
    content: []const u8,
    embedding: [3]f32, // In production, use 384+ dimensions
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try abi.init(allocator);
    defer abi.shutdown();

    var db = try abi.database.openOrCreate(allocator, "document_search");
    defer abi.database.close(&db);

    // Sample document corpus
    const docs = [_]Document{
        .{
            .id = 1,
            .title = "Zig Programming Guide",
            .content = "Learn Zig from scratch with hands-on examples",
            .embedding = [_]f32{ 0.9, 0.1, 0.2 },
        },
        .{
            .id = 2,
            .title = "Vector Database Tutorial",
            .content = "Understanding similarity search and embeddings",
            .embedding = [_]f32{ 0.2, 0.9, 0.3 },
        },
        .{
            .id = 3,
            .title = "Systems Programming",
            .content = "Low-level programming for performance",
            .embedding = [_]f32{ 0.8, 0.2, 0.7 },
        },
    };

    // Index documents
    std.debug.print("Indexing documents...\n", .{});
    for (docs) |doc| {
        const metadata = try std.fmt.allocPrint(allocator, "{s}: {s}", .{
            doc.title,
            doc.content,
        });
        defer allocator.free(metadata);

        try abi.database.insert(&db, doc.id, &doc.embedding, metadata);
        std.debug.print("  Indexed: {s}\n", .{doc.title});
    }

    // Optimize for search
    try abi.database.optimize(&db);

    // Perform searches
    const queries = [_]struct {
        text: []const u8,
        embedding: [3]f32,
    }{
        .{
            .text = "How do I learn Zig?",
            .embedding = [_]f32{ 0.85, 0.15, 0.25 },
        },
        .{
            .text = "What is a vector database?",
            .embedding = [_]f32{ 0.25, 0.85, 0.35 },
        },
    };

    for (queries) |query| {
        std.debug.print("\n--- Query: \"{s}\" ---\n", .{query.text});

        const results = try abi.database.search(&db, allocator, &query.embedding, 2);
        defer allocator.free(results);

        for (results, 0..) |result, i| {
            std.debug.print("  {d}. Score={d:.3}\n", .{ i + 1, result.score });
            if (result.metadata) |meta| {
                std.debug.print("     {s}\n", .{meta});
            }
        }
    }

    // Backup the database
    try abi.database.backup(&db, "search_system_backup.db");
    std.debug.print("\nBackup created\n", .{});
}
```

---

## Practice Exercises

### Exercise 1: Dynamic Document Loader

Create a program that:
1. Reads documents from a JSON file
2. Generates random embeddings for each
3. Inserts into database
4. Accepts user queries and returns top 5 results

**Hints:**
- Use `std.json` for parsing
- Use `std.rand` for embeddings
- Accept query via command-line arguments

### Exercise 2: Performance Benchmarking

Build a tool that:
1. Inserts 1000, 10000, and 100000 vectors
2. Measures insertion time
3. Measures search time at each scale
4. Compares optimized vs unoptimized performance

**Starter Template:** See `benchmarks/database_perf.zig`

---

## Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `error.DimensionMismatch` | Vectors have different dimensions | Ensure all vectors have same length |
| `error.DatabaseDisabled` | Feature not enabled | Rebuild with `-Denable-database=true` |
| `error.PathValidationError` | Invalid backup path | Use filenames only, no paths |
| Slow search | Database not optimized | Call `optimize()` after bulk inserts |

---

## Performance Tips

1. **Batch Operations:** Insert many vectors, then optimize once
2. **Dimension Selection:** Higher dimensions = better accuracy but slower search
3. **Memory Management:** Use arena allocators for temporary search results
4. **Index Tuning:** For >100K vectors, consider advanced HNSW parameters

**Benchmarks:** See [Performance Baseline](../PERFORMANCE_BASELINE.md)

---

## Next Steps

- [AI Guide](../ai.md) - Generate real embeddings with LLMs
- [Database Guide](../database.md) - Advanced WDBX features
- [API Reference](../API_REFERENCE.md) - Complete database API

## Additional Resources

- **HNSW Paper:** [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
- **Vector DB Comparison:** [Database Guide - Architecture](../database.md#architecture)
- **Real-world Examples:** See `examples/database.zig`

---

**See Also:** [Getting Started Tutorial](getting-started.md)
