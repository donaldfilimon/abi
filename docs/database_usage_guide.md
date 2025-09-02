# WDBX-AI Database Usage Guide

## Quick Start

This guide will help you get up and running with the WDBX-AI vector database in minutes.

### Prerequisites

- Zig 0.12+ installed
- Basic understanding of Zig syntax
- Knowledge of vector embeddings

### Installation

The database is part of the Abi framework. Add it to your project:

```zig
// In your build.zig
const database_mod = b.addModule("database", .{
    .source_file = b.path("src/database.zig"),
});

// In your source files
const database = @import("database");
```

## Basic Usage

### 1. Create a Database

```zig
const std = @import("std");
const database = @import("database");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Create or open a database
    var db = try database.Db.open("my_vectors.wdbx", true);
    defer db.close();
    
    // Initialize with vector dimension
    try db.init(384); // For 384-dimensional vectors
    
    std.debug.print("Database created successfully!\n", .{});
}
```

### 2. Add Vectors

```zig
// Add a single vector
var embedding = try allocator.alloc(f32, 384);
defer allocator.free(embedding);

// Fill with your vector data
for (0..384) |i| {
    embedding[i] = @as(f32, @floatFromInt(i)) * 0.01;
}

const row_id = try db.addEmbedding(embedding);
std.debug.print("Added vector at row {}\n", .{row_id});

// Add multiple vectors at once
var batch = try allocator.alloc([]f32, 3);
defer {
    for (batch) |emb| allocator.free(emb);
    allocator.free(batch);
}

for (0..3) |i| {
    var emb = try allocator.alloc(f32, 384);
    for (0..384) |j| {
        emb[j] = @as(f32, @floatFromInt(i * 100 + j)) * 0.001;
    }
    batch[i] = emb;
}

const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);

for (indices, 0..) |index, i| {
    std.debug.print("Batch vector {} at row {}\n", .{i, index});
}
```

### 3. Search for Similar Vectors

```zig
// Create a query vector
var query = try allocator.alloc(f32, 384);
defer allocator.free(query);

for (0..384) |i| {
    query[i] = @as(f32, @floatFromInt(i)) * 0.015;
}

// Search for top 10 most similar vectors
const results = try db.search(query, 10, allocator);
defer allocator.free(results);

std.debug.print("Found {} similar vectors:\n", .{results.len});
for (results, 0..) |result, i| {
    std.debug.print("  {}. Index: {}, Score: {d}\n", .{
        i + 1, result.index, result.score
    });
}
```

## Real-World Examples

### Example 1: Document Similarity Search

```zig
const DocumentStore = struct {
    db: *database.Db,
    allocator: std.mem.Allocator,
    documents: std.StringHashMap([]const u8),
    
    pub fn init(db: *database.Db, allocator: std.mem.Allocator) DocumentStore {
        return .{
            .db = db,
            .allocator = allocator,
            .documents = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *DocumentStore) void {
        self.documents.deinit();
    }
    
    pub fn addDocument(self: *DocumentStore, id: []const u8, content: []const u8, embedding: []const f32) !void {
        const row_id = try self.db.addEmbedding(embedding);
        try self.documents.put(id, content);
        std.debug.print("Added document '{}' at row {}\n", .{id, row_id});
    }
    
    pub fn findSimilarDocuments(self: *DocumentStore, query_embedding: []const f32, top_k: usize) ![]DocumentResult {
        const results = try self.db.search(query_embedding, top_k, self.allocator);
        defer self.allocator.free(results);
        
        var doc_results = try self.allocator.alloc(DocumentResult, results.len);
        
        for (results, 0..) |result, i| {
            // In a real implementation, you'd map row_id back to document ID
            doc_results[i] = .{
                .document_id = "doc_" ++ std.fmt.bufPrint(&[_]u8{undefined} ** 10, "{}", .{result.index}) catch "unknown",
                .similarity_score = result.score,
                .content = "Document content would go here",
            };
        }
        
        return doc_results;
    }
    
    const DocumentResult = struct {
        document_id: []const u8,
        similarity_score: f32,
        content: []const u8,
    };
};

// Usage
pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var db = try database.Db.open("documents.wdbx", true);
    defer db.close();
    try db.init(384);
    
    var store = DocumentStore.init(&db, allocator);
    defer store.deinit();
    
    // Add some documents
    var embedding1 = try allocator.alloc(f32, 384);
    defer allocator.free(embedding1);
    for (0..384) |i| { embedding1[i] = @as(f32, @floatFromInt(i)) * 0.01; }
    
    try store.addDocument("doc1", "This is a document about AI", embedding1);
    
    // Search for similar documents
    var query = try allocator.alloc(f32, 384);
    defer allocator.free(query);
    for (0..384) |i| { query[i] = @as(f32, @floatFromInt(i)) * 0.01; }
    
    const similar = try store.findSimilarDocuments(query, 5);
    defer allocator.free(similar);
    
    for (similar) |doc| {
        std.debug.print("Document: {}, Score: {d}\n", .{doc.document_id, doc.similarity_score});
    }
}
```

### Example 2: Image Feature Database

```zig
const ImageDatabase = struct {
    db: *database.Db,
    allocator: std.mem.Allocator,
    image_metadata: std.AutoHashMap(u64, ImageInfo),
    
    pub fn init(db: *database.Db, allocator: std.mem.Allocator) ImageDatabase {
        return .{
            .db = db,
            .allocator = allocator,
            .image_metadata = std.AutoHashMap(u64, ImageInfo).init(allocator),
        };
    }
    
    pub fn deinit(self: *ImageDatabase) void {
        self.image_metadata.deinit();
    }
    
    pub fn addImage(self: *ImageDatabase, filename: []const u8, features: []const f32) !u64 {
        const row_id = try self.db.addEmbedding(features);
        
        try self.image_metadata.put(row_id, .{
            .filename = try self.allocator.dupe(u8, filename),
            .feature_count = @intCast(features.len),
            .added_timestamp = std.time.microTimestamp(),
        });
        
        return row_id;
    }
    
    pub fn findSimilarImages(self: *ImageDatabase, query_features: []const f32, top_k: usize) ![]ImageMatch {
        const results = try self.db.search(query_features, top_k, self.allocator);
        defer self.allocator.free(results);
        
        var matches = try self.allocator.alloc(ImageMatch, results.len);
        
        for (results, 0..) |result, i| {
            const metadata = self.image_metadata.get(result.index) orelse .{
                .filename = "unknown",
                .feature_count = 0,
                .added_timestamp = 0,
            };
            
            matches[i] = .{
                .filename = metadata.filename,
                .similarity_score = result.score,
                .feature_count = metadata.feature_count,
            };
        }
        
        return matches;
    }
    
    const ImageInfo = struct {
        filename: []const u8,
        feature_count: usize,
        added_timestamp: i64,
    };
    
    const ImageMatch = struct {
        filename: []const u8,
        similarity_score: f32,
        feature_count: usize,
    };
};
```

### Example 3: Recommendation System

```zig
const RecommendationEngine = struct {
    user_db: *database.Db,
    item_db: *database.Db,
    allocator: std.mem.Allocator,
    
    pub fn init(user_db: *database.Db, item_db: *database.Db, allocator: std.mem.Allocator) RecommendationEngine {
        return .{
            .user_db = user_db,
            .item_db = item_db,
            .allocator = allocator,
        };
    }
    
    pub fn addUser(self: *RecommendationEngine, user_id: u64, preferences: []const f32) !void {
        _ = try self.user_db.addEmbedding(preferences);
        std.debug.print("Added user {} with {} preferences\n", .{user_id, preferences.len});
    }
    
    pub fn addItem(self: *RecommendationEngine, item_id: u64, features: []const f32) !void {
        _ = try self.item_db.addEmbedding(features);
        std.debug.print("Added item {} with {} features\n", .{item_id, features.len});
    }
    
    pub fn getRecommendations(self: *RecommendationEngine, user_id: u64, top_k: usize) ![]Recommendation {
        // In a real system, you'd retrieve the user's preference vector
        var user_prefs = try self.allocator.alloc(f32, 128);
        defer self.allocator.free(user_prefs);
        
        // Simulate user preferences
        for (0..128) |i| {
            user_prefs[i] = @as(f32, @floatFromInt(i)) * 0.01;
        }
        
        const similar_items = try self.item_db.search(user_prefs, top_k, self.allocator);
        defer self.allocator.free(similar_items);
        
        var recommendations = try self.allocator.alloc(Recommendation, similar_items.len);
        
        for (similar_items, 0..) |item, i| {
            recommendations[i] = .{
                .item_id = item.index,
                .relevance_score = 1.0 - (item.score / 1000.0), // Convert distance to relevance
                .confidence = 0.8,
            };
        }
        
        return recommendations;
    }
    
    const Recommendation = struct {
        item_id: u64,
        relevance_score: f32,
        confidence: f32,
    };
};
```

## Performance Optimization

### 1. Batch Operations

Always use batch operations when inserting multiple vectors:

```zig
// Good: Batch insertion
const batch_size = 100;
var batch = try allocator.alloc([]f32, batch_size);
defer {
    for (batch) |emb| allocator.free(emb);
    allocator.free(batch);
}

// Fill batch...
const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);

// Avoid: Individual insertions
for (embeddings) |embedding| {
    _ = try db.addEmbedding(embedding); // Slower
}
```

### 2. Memory Management

Use appropriate allocators and always free memory:

```zig
// Use arena allocator for temporary operations
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const temp_allocator = arena.allocator();

// Use for temporary data
const temp_results = try db.search(&query, 100, temp_allocator);
// No need to free - arena handles it
```

### 3. Vector Dimensionality

Choose appropriate dimensions for your use case:

```zig
// For text embeddings (common choices)
try db.init(384);   // Sentence transformers
try db.init(768);   // BERT base
try db.init(1536);  // OpenAI embeddings

// For image features
try db.init(128);   // Lightweight features
try db.init(512);   // Medium features
try db.init(2048);  // Heavy features
```

## Error Handling Best Practices

### 1. Comprehensive Error Handling

```zig
const result = db.addEmbedding(&embedding) catch |err| {
    switch (err) {
        error.DimensionMismatch => {
            std.debug.print("Vector dimension {} doesn't match database dimension {}\n", 
                .{embedding.len, db.getDimension()});
            return error.InvalidInput;
        },
        error.InvalidState => {
            std.debug.print("Database not initialized\n", .{});
            return error.InvalidState;
        },
        error.OutOfMemory => {
            std.debug.print("Out of memory during vector insertion\n", .{});
            return error.OutOfMemory;
        },
        else => {
            std.debug.print("Unexpected error: {}\n", .{err});
            return err;
        },
    }
};
```

### 2. Resource Cleanup

```zig
var db = try database.Db.open("vectors.wdbx", true);
errdefer db.close(); // Ensure cleanup on error

try db.init(128);
errdefer db.close(); // Cleanup after successful init

// ... rest of your code ...
db.close(); // Normal cleanup
```

## Testing Your Database

### 1. Unit Tests

```zig
test "database basic operations" {
    const allocator = testing.allocator;
    const test_file = "test_db.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try database.Db.open(test_file, true);
    defer db.close();
    
    try db.init(64);
    try testing.expectEqual(@as(u16, 64), db.getDimension());
    try testing.expectEqual(@as(u64, 0), db.getRowCount());
    
    var embedding = try allocator.alloc(f32, 64);
    defer allocator.free(embedding);
    for (0..64) |i| { embedding[i] = @as(f32, @floatFromInt(i)) * 0.01; }
    
    const row_id = try db.addEmbedding(embedding);
    try testing.expectEqual(@as(u64, 0), row_id);
    try testing.expectEqual(@as(u64, 1), db.getRowCount());
}
```

### 2. Performance Tests

```zig
test "database performance" {
    const allocator = testing.allocator;
    const test_file = "perf_test.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try database.Db.open(test_file, true);
    defer db.close();
    
    try db.init(128);
    
    // Measure insertion performance
    const num_vectors = 1000;
    const start_time = std.time.microTimestamp();
    
    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);
        
        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }
        
        _ = try db.addEmbedding(embedding);
    }
    
    const end_time = std.time.microTimestamp();
    const total_time = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    
    std.debug.print("Inserted {} vectors in {d:.3f} seconds\n", .{num_vectors, total_time});
    try testing.expect(total_time < 10.0); // Should complete in under 10 seconds
}
```

## Common Patterns

### 1. Database Factory

```zig
const DatabaseFactory = struct {
    pub fn createDatabase(path: []const u8, dimension: u16, allocator: std.mem.Allocator) !*database.Db {
        var db = try database.Db.open(path, true);
        errdefer db.close();
        
        try db.init(dimension);
        return db;
    }
    
    pub fn openDatabase(path: []const u8, allocator: std.mem.Allocator) !*database.Db {
        return try database.Db.open(path, false);
    }
};
```

### 2. Connection Pool

```zig
const DatabasePool = struct {
    databases: std.ArrayList(*database.Db),
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: std.mem.Allocator) DatabasePool {
        return .{
            .databases = std.ArrayList(*database.Db).init(allocator),
            .allocator = allocator,
            .mutex = .{},
        };
    }
    
    pub fn deinit(self: *DatabasePool) void {
        for (self.databases.items) |db| {
            db.close();
        }
        self.databases.deinit();
    }
    
    pub fn getDatabase(self: *DatabasePool) ?*database.Db {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.databases.items.len > 0) {
            return self.databases.orderedRemove(0);
        }
        return null;
    }
    
    pub fn returnDatabase(self: *DatabasePool, db: *database.Db) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.databases.append(db) catch return;
    }
};
```

## Troubleshooting

### Common Issues and Solutions

1. **"Dimension Mismatch" Error**
   - Check that all vectors have the same length
   - Verify database initialization with correct dimension
   - Use `db.getDimension()` to check current setting

2. **"Invalid State" Error**
   - Ensure `db.init()` is called before adding vectors
   - Check that database file is properly opened

3. **Performance Issues**
   - Use batch operations for multiple insertions
   - Consider vector dimensionality impact
   - Monitor memory usage

4. **File Permission Errors**
   - Check write permissions for database directory
   - Ensure file isn't locked by another process
   - Use absolute paths if needed

### Debug Information

```zig
// Enable detailed logging
const stats = db.getStats();
std.debug.print("Database Statistics:\n", .{});
std.debug.print("  Initializations: {}\n", .{stats.initialization_count});
std.debug.print("  Writes: {}\n", .{stats.write_count});
std.debug.print("  Searches: {}\n", .{stats.search_count});
std.debug.print("  Avg Search Time: {}μs\n", .{stats.getAverageSearchTime()});

// Check database state
std.debug.print("Database State:\n", .{});
std.debug.print("  Rows: {}\n", .{db.getRowCount()});
std.debug.print("  Dimension: {}\n", .{db.getDimension()});
std.debug.print("  File: {s}\n", .{db.file.path});
```

## Next Steps

1. **Explore Advanced Features**: Look into SIMD optimizations and performance tuning
2. **Build Applications**: Create real-world applications using the examples above
3. **Contribute**: Help improve the database with performance optimizations or new features
4. **Learn More**: Read the full API reference for detailed information

The WDBX-AI database is designed to be simple to use while providing powerful vector storage and search capabilities. Start with the basic examples and gradually explore more advanced features as your needs grow.
