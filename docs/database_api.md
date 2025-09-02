# WDBX-AI Vector Database API Reference

## Overview

The WDBX-AI Vector Database is a high-performance, file-based vector database designed for storing and searching high-dimensional embeddings. It features a custom binary format, efficient memory management, SIMD-accelerated search, and extensible metadata support.

## Features

- **File-based Storage**: Portable, embeddable database files
- **SIMD Acceleration**: Optimized vector operations using CPU SIMD instructions
- **Memory Efficient**: Explicit memory management with Zig allocators
- **Robust Error Handling**: Comprehensive error types and validation
- **Extensible Format**: Future support for ANN search and metadata schemas
- **Cross-platform**: Works on all major operating systems

## File Format

The WDBX-AI format consists of:

- **Header (4096 bytes)**: Magic bytes, version, row count, dimensionality, and offset pointers
- **Records Section**: Densely packed float32 vectors, each record is `dim * sizeof(f32)` bytes
- **Future Extensions**: Index and schema blocks for ANN search and metadata

### Header Structure

```zig
pub const WdbxHeader = packed struct {
    magic0: u8,        // 'W'
    magic1: u8,        // 'D'
    magic2: u8,        // 'B'
    magic3: u8,        // 'X'
    magic4: u8,        // 'A'
    magic5: u8,        // 'I'
    magic6: u8,        // '\0'
    version: u16,      // Format version number
    row_count: u64,    // Number of records in the database
    dim: u16,          // Dimensionality of each vector
    page_size: u32,    // Page size used for file operations
    schema_off: u64,   // Offset to schema information
    index_off: u64,    // Offset to index data
    records_off: u64,  // Offset to records section
    freelist_off: u64, // Offset to freelist for deleted records
    _reserved: [4072]u8, // Reserved space for future use
};
```

## API Reference

### Database Instance

#### `Db.open(path: []const u8, create_if_missing: bool) DbError!*Db`

Opens an existing database file or creates a new one if it doesn't exist.

**Parameters:**
- `path`: File path to the database
- `create_if_missing`: If true, creates a new database file when it doesn't exist

**Returns:** A pointer to the database instance

**Example:**
```zig
var db = try database.Db.open("vectors.wdbx", true);
defer db.close();
```

#### `db.init(dim: u16) DbError!void`

Initializes the database with the specified vector dimensionality.

**Parameters:**
- `dim`: Vector dimensionality (1-4096)

**Example:**
```zig
try db.init(384); // Initialize for 384-dimensional vectors
```

#### `db.close() void`

Closes the database and frees all associated resources.

### Vector Operations

#### `db.addEmbedding(embedding: []const f32) DbError!u64`

Adds a single embedding vector to the database.

**Parameters:**
- `embedding`: Vector data (must match database dimensionality)

**Returns:** Row index of the added vector

**Example:**
```zig
const embedding = [_]f32{0.1, 0.2, 0.3, ...};
const row_id = try db.addEmbedding(&embedding);
```

#### `db.addEmbeddingsBatch(embeddings: []const []const f32) DbError![]u64`

Adds multiple embeddings in a single batch operation.

**Parameters:**
- `embeddings`: Array of vectors to add

**Returns:** Array of row indices

**Example:**
```zig
var batch = [_][]const f32{
    &[_]f32{0.1, 0.2, 0.3},
    &[_]f32{0.4, 0.5, 0.6},
    &[_]f32{0.7, 0.8, 0.9},
};
const indices = try db.addEmbeddingsBatch(&batch);
defer allocator.free(indices);
```

### Search Operations

#### `db.search(query: []const f32, top_k: usize, allocator: Allocator) DbError![]Result`

Searches for the most similar vectors to the query vector.

**Parameters:**
- `query`: Query vector (must match database dimensionality)
- `top_k`: Number of top results to return
- `allocator`: Memory allocator for results

**Returns:** Array of search results sorted by similarity

**Example:**
```zig
const query = [_]f32{0.15, 0.25, 0.35, ...};
const results = try db.search(&query, 10, allocator);
defer allocator.free(results);

for (results) |result| {
    std.debug.print("Index: {}, Score: {d}\n", .{result.index, result.score});
}
```

### Search Results

```zig
pub const Result = struct {
    index: u64,  // Row index of the result
    score: f32,  // Distance/similarity score (lower = more similar)
    
    pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
        return a.score < b.score;
    }
};
```

### Database Information

#### `db.getRowCount() u64`

Returns the total number of vectors in the database.

#### `db.getDimension() u16`

Returns the dimensionality of vectors in the database.

#### `db.getStats() DbStats`

Returns performance statistics.

```zig
pub const DbStats = struct {
    initialization_count: u64,
    write_count: u64,
    search_count: u64,
    total_search_time_us: u64,
    
    pub fn getAverageSearchTime(self: *const DbStats) u64 {
        if (self.search_count == 0) return 0;
        return self.total_search_time_us / self.search_count;
    }
};
```

## Error Handling

The database uses comprehensive error types:

```zig
pub const DatabaseError = error{
    InvalidFileFormat,
    CorruptedData,
    InvalidDimensions,
    IndexOutOfBounds,
    InsufficientMemory,
    FileSystemError,
    LockContention,
    InvalidOperation,
    VersionMismatch,
    ChecksumMismatch,
};

pub const DbError = error{
    AlreadyInitialized,
    DimensionMismatch,
    InvalidState,
    OutOfMemory,
    FileBusy,
    EndOfStream,
    InvalidMagic,
    UnsupportedVersion,
    CorruptedDatabase,
} || std.fs.File.SeekError || std.fs.File.WriteError ||
    std.fs.File.ReadError || std.fs.File.OpenError;
```

## Performance Characteristics

### Time Complexity
- **Insertion**: O(1) amortized
- **Search**: O(n) for brute-force search (future ANN support will improve this)
- **Batch Operations**: O(n) with reduced overhead

### Space Complexity
- **Header**: Fixed 4KB overhead
- **Vectors**: `n * dim * sizeof(f32)` bytes
- **Memory Buffer**: Configurable read buffer size

### SIMD Optimizations
The database automatically uses SIMD instructions when available:
- **f32x16**: For vectors with 16+ dimensions
- **f32x4**: For vectors with 8+ dimensions
- **Scalar**: Fallback for smaller vectors

## Best Practices

### 1. Memory Management
```zig
// Always use defer for cleanup
var db = try database.Db.open("vectors.wdbx", true);
defer db.close();

// Free search results
const results = try db.search(&query, 10, allocator);
defer allocator.free(results);
```

### 2. Batch Operations
```zig
// Use batch operations for multiple insertions
const batch_size = 100;
var batch = try allocator.alloc([]f32, batch_size);
defer {
    for (batch) |emb| allocator.free(emb);
    allocator.free(batch);
}

const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);
```

### 3. Error Handling
```zig
const result = db.addEmbedding(&embedding) catch |err| {
    switch (err) {
        error.DimensionMismatch => {
            std.debug.print("Vector dimension doesn't match database\n", .{});
            return;
        },
        error.InvalidState => {
            std.debug.print("Database not initialized\n", .{});
            return;
        },
        else => return err,
    }
};
```

### 4. Performance Tuning
```zig
// Use appropriate batch sizes
const optimal_batch_size = 64; // Adjust based on your use case

// Consider vector dimensionality
// Higher dimensions = more memory but potentially better accuracy
try db.init(384); // Common for modern embedding models
```

## Example Applications

### 1. Simple Vector Store
```zig
const std = @import("std");
const database = @import("database");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Create database
    var db = try database.Db.open("my_vectors.wdbx", true);
    defer db.close();
    
    try db.init(128);
    
    // Add vectors
    var embedding = try allocator.alloc(f32, 128);
    defer allocator.free(embedding);
    
    for (0..128) |i| {
        embedding[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }
    
    const row_id = try db.addEmbedding(embedding);
    std.debug.print("Added vector at row {}\n", .{row_id});
    
    // Search
    const results = try db.search(embedding, 5, allocator);
    defer allocator.free(results);
    
    for (results) |result| {
        std.debug.print("Result: index={}, score={d}\n", .{result.index, result.score});
    }
}
```

### 2. Batch Processing
```zig
pub fn processBatch(db: *database.Db, embeddings: []const []const f32) !void {
    const batch_size = 100;
    var i: usize = 0;
    
    while (i < embeddings.len) {
        const end = @min(i + batch_size, embeddings.len);
        const batch = embeddings[i..end];
        
        const indices = try db.addEmbeddingsBatch(batch);
        defer allocator.free(indices);
        
        std.debug.print("Processed batch {}-{}\n", .{i, end - 1});
        i = end;
    }
}
```

### 3. Similarity Search
```zig
pub fn findSimilar(db: *database.Db, query: []const f32, threshold: f32) ![]database.Db.Result {
    const candidates = try db.search(query, 100, allocator);
    defer allocator.free(candidates);
    
    var results = std.ArrayList(database.Db.Result).init(allocator);
    
    for (candidates) |candidate| {
        if (candidate.score < threshold) {
            try results.append(candidate);
        }
    }
    
    return results.toOwnedSlice();
}
```

## Future Enhancements

### Planned Features
1. **ANN Search**: HNSW and IVF index support for sub-linear search
2. **Concurrency**: Parallel search operations
3. **Metadata Support**: Schema-based metadata storage
4. **Import/Export**: Compatibility with Milvus, Faiss, and other formats
5. **Compression**: Vector compression for reduced storage
6. **Indexing**: Automatic index building and maintenance

### Performance Improvements
1. **GPU Acceleration**: CUDA/OpenCL support for large-scale operations
2. **Memory Mapping**: Zero-copy file access for read-heavy workloads
3. **Cache Optimization**: Intelligent caching strategies
4. **Vector Quantization**: Reduced precision for faster search

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
   - Ensure all vectors have the same dimensionality
   - Check database initialization with correct dimension

2. **File Permissions**
   - Verify write permissions for database directory
   - Check if file is locked by another process

3. **Memory Issues**
   - Monitor memory usage during large operations
   - Use appropriate allocators for your use case

4. **Performance Problems**
   - Enable SIMD optimizations
   - Use batch operations for multiple insertions
   - Consider vector dimensionality impact

### Debug Information
```zig
// Enable debug logging
const stats = db.getStats();
std.debug.print("Database stats: {any}\n", .{stats});

// Check database state
std.debug.print("Rows: {}, Dimension: {}\n", .{
    db.getRowCount(),
    db.getDimension(),
});
```

## Contributing

The WDBX-AI database is open for contributions. Areas of interest include:
- Performance optimizations
- Additional index types
- Format compatibility
- Testing and benchmarking
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.
