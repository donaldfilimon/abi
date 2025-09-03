# 🗄️ WDBX-AI Vector Database API Reference

> **Complete API reference for the high-performance vector database**

[![Vector Database](https://img.shields.io/badge/Vector-Database-blue.svg)](docs/database_api.md)
[![High Performance](https://img.shields.io/badge/High-Performance-brightgreen.svg)]()
[![SIMD Optimized](https://img.shields.io/badge/SIMD-Optimized-orange.svg)]()

The WDBX-AI Vector Database is a high-performance, file-based vector database designed for storing and searching high-dimensional embeddings. It features a custom binary format, efficient memory management, SIMD-accelerated search, and extensible metadata support.

## 📋 **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [File Format](#file-format)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## 🎯 **Overview**

The WDBX-AI Vector Database provides a robust, high-performance solution for vector storage and similarity search operations. Built with Zig's memory safety and performance characteristics, it offers enterprise-grade reliability with minimal resource overhead.

### **Key Design Principles**
- **Memory Safety**: Zero-copy operations where possible, explicit memory management
- **Performance**: SIMD-accelerated vector operations and efficient algorithms
- **Reliability**: Comprehensive error handling and data validation
- **Extensibility**: Future support for advanced indexing and metadata schemas
- **Portability**: Cross-platform compatibility with consistent behavior

---

## ✨ **Features**

### **Core Capabilities**
- **File-based Storage**: Portable, embeddable database files
- **SIMD Acceleration**: Optimized vector operations using CPU SIMD instructions
- **Memory Efficient**: Explicit memory management with Zig allocators
- **Robust Error Handling**: Comprehensive error types and validation
- **Extensible Format**: Future support for ANN search and metadata schemas
- **Cross-platform**: Works on all major operating systems

### **Performance Characteristics**
- **Vector Operations**: O(1) vector addition, O(n) linear search
- **Memory Usage**: Minimal overhead with efficient data structures
- **I/O Performance**: Optimized file operations with configurable page sizes
- **Scalability**: Support for millions of vectors with linear scaling

---

## 📁 **File Format**

### **Format Overview**

The WDBX-AI format consists of:

- **Header (4096 bytes)**: Magic bytes, version, row count, dimensionality, and offset pointers
- **Records Section**: Densely packed float32 vectors, each record is `dim * sizeof(f32)` bytes
- **Future Extensions**: Index and schema blocks for ANN search and metadata

### **Header Structure**

#### **Complete Header Definition**
```zig
pub const WdbxHeader = packed struct {
    // Magic bytes for file identification
    magic0: u8,        // 'W'
    magic1: u8,        // 'D'
    magic2: u8,        // 'B'
    magic3: u8,        // 'X'
    magic4: u8,        // 'A'
    magic5: u8,        // 'I'
    magic6: u8,        // '\0'
    
    // Version and metadata
    version: u16,      // Format version number
    row_count: u64,    // Number of records in the database
    dim: u16,          // Dimensionality of each vector
    page_size: u32,    // Page size used for file operations
    
    // Offset pointers for future extensions
    schema_off: u64,   // Offset to schema information
    index_off: u64,    // Offset to index data
    records_off: u64,  // Offset to records section
    freelist_off: u64, // Offset to freelist for deleted records
    
    // Reserved space for future use
    _reserved: [4072]u8,
};
```

#### **Header Validation**
```zig
const HeaderValidator = struct {
    pub fn validateHeader(header: WdbxHeader) !void {
        // Check magic bytes
        const expected_magic = [_]u8{ 'W', 'D', 'B', 'X', 'A', 'I', 0 };
        for (expected_magic, 0..) |expected, i| {
            const actual = switch (i) {
                0 => header.magic0,
                1 => header.magic1,
                2 => header.magic2,
                3 => header.magic3,
                4 => header.magic4,
                5 => header.magic5,
                6 => header.magic6,
                else => unreachable,
            };
            
            if (actual != expected) {
                return error.InvalidMagicBytes;
            }
        }
        
        // Validate version
        if (header.version < MIN_SUPPORTED_VERSION) {
            return error.UnsupportedVersion;
        }
        
        // Validate dimensionality
        if (header.dim == 0 or header.dim > MAX_DIMENSIONS) {
            return error.InvalidDimensions;
        }
        
        // Validate page size
        if (header.page_size == 0 or header.page_size % 4096 != 0) {
            return error.InvalidPageSize;
        }
    }
    
    const MIN_SUPPORTED_VERSION: u16 = 1;
    const MAX_DIMENSIONS: u16 = 4096;
};
```

### **Data Layout**

#### **Record Structure**
```zig
const RecordLayout = struct {
    pub fn getRecordSize(dimensions: u16) usize {
        return @intCast(usize, dimensions) * @sizeOf(f32);
    }
    
    pub fn getRecordOffset(record_index: u64, dimensions: u16) u64 {
        const header_size: u64 = 4096;
        const record_size = @intCast(u64, getRecordSize(dimensions));
        return header_size + (record_index * record_size);
    }
    
    pub fn getTotalFileSize(record_count: u64, dimensions: u16) u64 {
        const header_size: u64 = 4096;
        const records_size = record_count * @intCast(u64, getRecordSize(dimensions));
        return header_size + records_size;
    }
};
```

---

## 🔌 **API Reference**

### **Database Instance Management**

#### **Opening and Creating Databases**

##### `Db.open(path: []const u8, create_if_missing: bool) DbError!*Db`

Opens an existing database file or creates a new one if it doesn't exist.

**Parameters:**
- `path`: File path to the database
- `create_if_missing`: If true, creates a new database file when it doesn't exist

**Returns:** A pointer to the database instance

**Example:**
```zig
// Open existing database or create new one
var db = try database.Db.open("vectors.wdbx", true);
defer db.close();

// Initialize with specific dimensionality
try db.init(384); // 384-dimensional vectors
```

##### `db.init(dim: u16) DbError!void`

Initializes the database with the specified vector dimensionality.

**Parameters:**
- `dim`: Vector dimensionality (1-4096)

**Validation:**
- Ensures dimensionality is within valid range
- Allocates necessary data structures
- Initializes file headers and metadata

**Example:**
```zig
// Initialize for different vector dimensions
try db.init(128);   // 128-dimensional vectors
try db.init(256);   // 256-dimensional vectors
try db.init(512);   // 512-dimensional vectors
try db.init(1024);  // 1024-dimensional vectors
```

##### `db.close() void`

Closes the database and frees all associated resources.

**Cleanup Operations:**
- Flushes pending writes to disk
- Frees allocated memory
- Closes file handles
- Releases system resources

**Example:**
```zig
var db = try database.Db.open("vectors.wdbx", true);
defer db.close(); // Automatic cleanup when scope ends

// Use database...
// Database automatically closed here
```

### **Vector Operations**

#### **Adding Vectors**

##### `db.addEmbedding(embedding: []const f32) DbError!u64`

Adds a single embedding vector to the database.

**Parameters:**
- `embedding`: Vector data (must match database dimensionality)

**Returns:** Row index of the added vector

**Validation:**
- Checks vector dimensionality matches database
- Validates vector data (NaN, infinity checks)
- Ensures sufficient storage space

**Example:**
```zig
// Add single vector
const embedding = [_]f32{0.1, 0.2, 0.3, 0.4, 0.5};
const row_id = try db.addEmbedding(&embedding);

std.log.info("Added vector at row {}", .{row_id});
```

##### `db.addEmbeddingsBatch(embeddings: []const []const f32) DbError![]u64`

Adds multiple embedding vectors in a single operation.

**Parameters:**
- `embeddings`: Array of vectors to add

**Returns:** Array of row indices for added vectors

**Performance Benefits:**
- Reduced I/O operations
- Batch memory allocation
- Optimized file writes

**Example:**
```zig
// Add multiple vectors at once
const embeddings = [_][]const f32{
    &[_]f32{0.1, 0.2, 0.3, 0.4, 0.5},
    &[_]f32{0.6, 0.7, 0.8, 0.9, 1.0},
    &[_]f32{1.1, 1.2, 1.3, 1.4, 1.5},
};

const row_ids = try db.addEmbeddingsBatch(&embeddings);
defer allocator.free(row_ids);

std.log.info("Added {} vectors", .{row_ids.len});
```

#### **Retrieving Vectors**

##### `db.getEmbedding(row: u64) DbError![]f32`

Retrieves a vector from the database by row index.

**Parameters:**
- `row`: Row index of the vector to retrieve

**Returns:** Vector data as float32 array

**Error Handling:**
- Returns error if row index is out of bounds
- Handles file I/O errors gracefully
- Validates retrieved data integrity

**Example:**
```zig
// Retrieve vector by row index
const embedding = try db.getEmbedding(42);
defer allocator.free(embedding);

std.log.info("Retrieved vector: {any}", .{embedding});
```

##### `db.getEmbeddingsBatch(rows: []const u64) DbError![]const []const f32`

Retrieves multiple vectors by row indices.

**Parameters:**
- `rows`: Array of row indices to retrieve

**Returns:** Array of vectors

**Performance Benefits:**
- Batch I/O operations
- Reduced memory allocation overhead
- Optimized for multiple retrievals

**Example:**
```zig
// Retrieve multiple vectors
const row_indices = [_]u64{0, 5, 10, 15};
const embeddings = try db.getEmbeddingsBatch(&row_indices);
defer {
    for (embeddings) |embedding| {
        allocator.free(embedding);
    }
    allocator.free(embeddings);
}

std.log.info("Retrieved {} vectors", .{embeddings.len});
```

#### **Searching Vectors**

##### `db.findSimilar(query: []const f32, k: usize) DbError![]SearchResult`

Finds the k most similar vectors to the query vector.

**Parameters:**
- `query`: Query vector for similarity search
- `k`: Number of similar vectors to return

**Returns:** Array of search results with similarity scores

**Algorithm:**
- Linear search through all vectors
- SIMD-accelerated distance calculations
- Sorted results by similarity score

**Example:**
```zig
// Find similar vectors
const query = [_]f32{0.5, 0.5, 0.5, 0.5, 0.5};
const results = try db.findSimilar(&query, 10);
defer {
    for (results) |result| {
        allocator.free(result.embedding);
    }
    allocator.free(results);
}

// Process results
for (results, 0..) |result, i| {
    std.log.info("Result {}: row {}, similarity: {:.3}", .{
        i, result.row, result.similarity
    });
}
```

##### `db.findSimilarInRange(query: []const f32, k: usize, start_row: u64, end_row: u64) DbError![]SearchResult`

Finds similar vectors within a specific row range.

**Parameters:**
- `query`: Query vector for similarity search
- `k`: Number of similar vectors to return
- `start_row`: Starting row index (inclusive)
- `end_row`: Ending row index (exclusive)

**Use Cases:**
- Partitioned searches
- Incremental similarity search
- Parallel processing support

**Example:**
```zig
// Search in specific range
const query = [_]f32{0.5, 0.5, 0.5, 0.5, 0.5};
const results = try db.findSimilarInRange(&query, 5, 100, 200);
defer {
    for (results) |result| {
        allocator.free(result.embedding);
    }
    allocator.free(results);
}

std.log.info("Found {} similar vectors in range [100, 200)", .{results.len});
```

### **Database Information and Statistics**

#### **Metadata Queries**

##### `db.getRowCount() u64`

Returns the total number of vectors in the database.

**Example:**
```zig
const total_vectors = db.getRowCount();
std.log.info("Database contains {} vectors", .{total_vectors});
```

##### `db.getDimensions() u16`

Returns the dimensionality of vectors in the database.

**Example:**
```zig
const dimensions = db.getDimensions();
std.log.info("Vectors are {}-dimensional", .{dimensions});
```

##### `db.getFileSize() u64`

Returns the total size of the database file in bytes.

**Example:**
```zig
const file_size = db.getFileSize();
const size_mb = @intToFloat(f32, file_size) / (1024 * 1024);
std.log.info("Database file size: {:.2} MB", .{size_mb});
```

#### **Performance Statistics**

##### `db.getStats() DatabaseStats`

Returns comprehensive database statistics.

**Statistics Include:**
- Total vectors and dimensions
- File size and memory usage
- Operation counts and timing
- Error rates and performance metrics

**Example:**
```zig
const stats = db.getStats();
std.log.info("Database Statistics:", .{});
std.log.info("  Total vectors: {}", .{stats.total_vectors});
std.log.info("  Dimensions: {}", .{stats.dimensions});
std.log.info("  File size: {:.2} MB", .{@intToFloat(f32, stats.file_size) / (1024 * 1024)});
std.log.info("  Memory usage: {:.2} MB", .{@intToFloat(f32, stats.memory_usage) / (1024 * 1024)});
```

---

## 💡 **Usage Examples**

### **Basic Database Operations**

#### **Complete Database Lifecycle**
```zig
const DatabaseExample = struct {
    pub fn runExample() !void {
        const allocator = std.heap.page_allocator;
        
        // Create and initialize database
        var db = try database.Db.open("example.wdbx", true);
        defer db.close();
        
        try db.init(128); // 128-dimensional vectors
        
        // Add sample vectors
        const vectors = [_][]const f32{
            &[_]f32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
            &[_]f32{1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0},
            &[_]f32{2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0},
        };
        
        const row_ids = try db.addEmbeddingsBatch(&vectors);
        defer allocator.free(row_ids);
        
        // Perform similarity search
        const query = [_]f32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        const results = try db.findSimilar(&query, 3);
        defer {
            for (results) |result| {
                allocator.free(result.embedding);
            }
            allocator.free(results);
        }
        
        // Display results
        std.log.info("Similarity Search Results:", .{});
        for (results, 0..) |result, i| {
            std.log.info("  {}: Row {}, Similarity: {:.3}", .{
                i + 1, result.row, result.similarity
            });
        }
    }
};
```

### **Advanced Usage Patterns**

#### **Batch Processing with Error Handling**
```zig
const BatchProcessor = struct {
    pub fn processBatch(db: *database.Db, vectors: []const []const f32) !void {
        const allocator = std.heap.page_allocator;
        
        // Process vectors in batches
        const batch_size = 1000;
        var processed: usize = 0;
        
        while (processed < vectors.len) {
            const end = @min(processed + batch_size, vectors.len);
            const batch = vectors[processed..end];
            
            // Add batch to database
            const row_ids = db.addEmbeddingsBatch(batch) catch |err| {
                std.log.err("Failed to add batch {}-{}: {}", .{
                    processed, end - 1, err
                });
                return err;
            };
            defer allocator.free(row_ids);
            
            std.log.info("Processed batch {}-{}: {} vectors", .{
                processed, end - 1, row_ids.len
            });
            
            processed = end;
        }
        
        std.log.info("Successfully processed {} vectors", .{vectors.len});
    }
};
```

#### **Similarity Search with Filtering**
```zig
const SearchEngine = struct {
    pub fn findSimilarWithFilter(
        db: *database.Db,
        query: []const f32,
        k: usize,
        filter_fn: *const fn (row: u64) bool,
    ) ![]SearchResult {
        const allocator = std.heap.page_allocator;
        
        // Get all results first
        const all_results = try db.findSimilar(query, db.getRowCount());
        defer {
            for (all_results) |result| {
                allocator.free(result.embedding);
            }
            allocator.free(all_results);
        }
        
        // Filter results
        var filtered = std.ArrayList(SearchResult).init(allocator);
        defer filtered.deinit();
        
        for (all_results) |result| {
            if (filter_fn(result.row)) {
                try filtered.append(result);
                if (filtered.items.len >= k) break;
            }
        }
        
        return filtered.toOwnedSlice();
    }
    
    // Example filter function
    fn evenRowFilter(row: u64) bool {
        return row % 2 == 0;
    }
    
    fn oddRowFilter(row: u64) bool {
        return row % 2 == 1;
    }
};
```

---

## ⚡ **Performance Considerations**

### **Optimization Strategies**

#### **Memory Management**
```zig
const MemoryOptimizer = struct {
    pub fn optimizeForPerformance(db: *database.Db) !void {
        // Use appropriate page sizes
        const optimal_page_size = 64 * 1024; // 64KB pages
        
        // Pre-allocate buffers for batch operations
        const batch_buffer_size = 1024 * 1024; // 1MB buffer
        
        // Enable SIMD optimizations
        try db.enableSIMDOptimizations();
        
        // Configure memory pools
        try db.configureMemoryPools(.{
            .small_pool_size = 1024 * 1024,      // 1MB
            .medium_pool_size = 10 * 1024 * 1024, // 10MB
            .large_pool_size = 100 * 1024 * 1024, // 100MB
        });
    }
};
```

#### **Batch Operations**
```zig
const BatchOptimizer = struct {
    pub fn getOptimalBatchSize(dimensions: u16, available_memory: usize) usize {
        const vector_size = @intCast(usize, dimensions) * @sizeOf(f32);
        const optimal_batch_size = available_memory / (vector_size * 2); // 2x for safety
        
        // Clamp to reasonable bounds
        return @min(@max(optimal_batch_size, 100), 10000);
    }
    
    pub fn processWithOptimalBatch(
        db: *database.Db,
        vectors: []const []const f32,
    ) !void {
        const batch_size = getOptimalBatchSize(db.getDimensions(), 1024 * 1024 * 1024); // 1GB
        
        var processed: usize = 0;
        while (processed < vectors.len) {
            const end = @min(processed + batch_size, vectors.len);
            const batch = vectors[processed..end];
            
            try db.addEmbeddingsBatch(batch);
            processed = end;
        }
    }
};
```

### **Performance Monitoring**

#### **Benchmarking Tools**
```zig
const PerformanceBenchmark = struct {
    pub fn benchmarkDatabase(db: *database.Db) !BenchmarkResults {
        const allocator = std.heap.page_allocator;
        var results = BenchmarkResults.init(allocator);
        defer results.deinit();
        
        // Benchmark vector addition
        const add_time = try self.benchmarkAddition(db);
        try results.addMetric("vector_addition", add_time);
        
        // Benchmark similarity search
        const search_time = try self.benchmarkSearch(db);
        try results.addMetric("similarity_search", search_time);
        
        // Benchmark batch operations
        const batch_time = try self.benchmarkBatchOperations(db);
        try results.addMetric("batch_operations", batch_time);
        
        return results.toOwned();
    }
    
    const BenchmarkResults = struct {
        metrics: std.StringHashMap(u64),
        allocator: std.mem.Allocator,
        
        pub fn init(allocator: std.mem.Allocator) @This() {
            return @This(){
                .metrics = std.StringHashMap(u64).init(allocator),
                .allocator = allocator,
            };
        }
        
        pub fn addMetric(self: *@This(), name: []const u8, value: u64) !void {
            try self.metrics.put(name, value);
        }
        
        pub fn printResults(self: *@This()) void {
            std.log.info("Performance Benchmark Results:", .{});
            var iter = self.metrics.iterator();
            while (iter.next()) |entry| {
                std.log.info("  {}: {} ns", .{entry.key, entry.value});
            }
        }
    };
};
```

---

## 🛡️ **Error Handling**

### **Error Types**

#### **Comprehensive Error Definitions**
```zig
pub const DbError = error{
    // File operation errors
    FileNotFound,
    FilePermissionDenied,
    FileCorrupted,
    FileTooLarge,
    
    // Memory errors
    OutOfMemory,
    InvalidAllocation,
    MemoryCorruption,
    
    // Data validation errors
    InvalidDimensions,
    InvalidVectorData,
    VectorSizeMismatch,
    InvalidRowIndex,
    
    // Operation errors
    DatabaseNotInitialized,
    OperationNotSupported,
    InvalidParameters,
    
    // System errors
    SystemError,
    NetworkError,
    TimeoutError,
};
```

### **Error Handling Patterns**

#### **Graceful Error Recovery**
```zig
const ErrorHandler = struct {
    pub fn handleDatabaseError(err: DbError) void {
        switch (err) {
            error.FileNotFound => {
                std.log.err("Database file not found", .{});
                // Create new database or use default
            },
            error.FileCorrupted => {
                std.log.err("Database file is corrupted", .{});
                // Attempt recovery or restore from backup
            },
            error.OutOfMemory => {
                std.log.err("Out of memory", .{});
                // Free resources or reduce batch size
            },
            error.InvalidDimensions => {
                std.log.err("Invalid vector dimensions", .{});
                // Validate input data
            },
            else => {
                std.log.err("Unexpected error: {}", .{err});
                // Log and continue if possible
            },
        }
    }
    
    pub fn safeDatabaseOperation(operation: *const fn () error!void) !void {
        operation() catch |err| {
            handleDatabaseError(err);
            return err;
        };
    }
};
```

---

## 🎯 **Best Practices**

### **Database Design**

#### **Optimal Configuration**
```zig
const DatabaseConfig = struct {
    // Choose appropriate dimensionality
    dimensions: u16 = 128, // Balance between accuracy and performance
    
    // Use optimal page sizes
    page_size: u32 = 64 * 1024, // 64KB for most workloads
    
    // Enable performance features
    enable_simd: bool = true,
    enable_compression: bool = false, // For high-performance scenarios
    
    // Memory management
    preallocate_memory: bool = true,
    memory_pool_size: usize = 1024 * 1024 * 1024, // 1GB
};
```

#### **Vector Preparation**
```zig
const VectorPreprocessor = struct {
    pub fn normalizeVector(vector: []f32) void {
        // Calculate L2 norm
        var norm: f32 = 0.0;
        for (vector) |value| {
            norm += value * value;
        }
        norm = @sqrt(norm);
        
        // Normalize to unit length
        if (norm > 0.0) {
            for (vector) |*value| {
                value.* /= norm;
            }
        }
    }
    
    pub fn validateVector(vector: []const f32) !void {
        for (vector) |value| {
            if (std.math.isNan(value) or std.math.isInf(value)) {
                return error.InvalidVectorData;
            }
        }
    }
};
```

### **Performance Optimization**

#### **Efficient Search Patterns**
```zig
const SearchOptimizer = struct {
    pub fn optimizeSearchQuery(db: *database.Db, query: []const f32) ![]f32 {
        // Normalize query vector
        var normalized_query = try allocator.dupe(f32, query);
        defer allocator.free(normalized_query);
        
        VectorPreprocessor.normalizeVector(normalized_query);
        
        return normalized_query;
    }
    
    pub fn useOptimalSearchStrategy(
        db: *database.Db,
        query: []const f32,
        k: usize,
        dataset_size: usize,
    ) ![]SearchResult {
        // Use different strategies based on dataset size
        if (dataset_size < 1000) {
            // Small dataset: linear search is fine
            return db.findSimilar(query, k);
        } else if (dataset_size < 100000) {
            // Medium dataset: use range partitioning
            return self.searchWithPartitioning(db, query, k);
        } else {
            // Large dataset: use approximate search
            return self.approximateSearch(db, query, k);
        }
    }
};
```

---

## 🔗 **Additional Resources**

- **[Main Documentation](README.md)** - Start here for an overview
- **[Database Quickstart](docs/database_quickstart.md)** - Get started quickly
- **[Database Usage Guide](docs/database_usage_guide.md)** - Comprehensive usage guide
- **[Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)** - Production deployment guide
- **[WDBX Enhanced](docs/WDBX_ENHANCED.md)** - Enhanced features and capabilities

---

## 🎉 **WDBX-AI Vector Database: Production Ready**

✅ **WDBX-AI is production-ready** with:

- **High Performance**: SIMD-accelerated vector operations
- **Memory Efficient**: Explicit memory management and optimization
- **Robust Error Handling**: Comprehensive error types and recovery
- **Extensible Format**: Future support for advanced features
- **Cross-Platform**: Consistent behavior across all operating systems

**Ready for production use** 🚀

---

**🗄️ The WDBX-AI Vector Database provides a robust, high-performance solution for vector storage and similarity search operations!**

**⚡ With SIMD acceleration, efficient memory management, and comprehensive error handling, it's designed for enterprise-grade reliability and performance.**
