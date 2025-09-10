# 🚀 WDBX-AI Database Quickstart

> **Get up and running with high-performance vector storage in minutes**

[![Quickstart](https://img.shields.io/badge/Quickstart-Guide-blue.svg)](docs/database_quickstart.md)
[![Performance](https://img.shields.io/badge/Performance-2,777+%20ops%20sec-brightgreen.svg)]()

This quickstart guide will get you up and running with the WDBX-AI vector database in under 5 minutes. You'll learn how to create, populate, and query your first vector database.

## 📋 **Table of Contents**

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Performance Tips](#performance-tips)
- [Error Handling](#error-handling)
- [Example Applications](#example-applications)
- [Testing](#testing)
- [Next Steps](#next-steps)

---

## ✅ **Prerequisites**

- **Zig**: Version 0.15.1 or later
- **Memory**: At least 512MB RAM for basic operations
- **Storage**: 100MB+ free space for database files
- **Platform**: Windows, macOS, or Linux

---

## 📦 **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-org/abi.git
cd abi
```

### **2. Build the Project**
```bash
zig build
```

### **3. Verify Installation**
```bash
./zig-out/bin/abi --version
```

---

## 🚀 **Basic Usage**

### **1. Create a Database**

```zig
const std = @import("std");
const database = @import("wdbx/database.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Create a new database with 384-dimensional vectors
    var db = try database.Db.init(allocator, 384);
    defer db.close();
    
    std.debug.print("Database created successfully!\n", .{});
}
```

### **2. Add Vectors**

```zig
// Add a single vector
const embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
const index = try db.addEmbedding(&embedding);
std.debug.print("Added vector at index: {}\n", .{index});

// Add multiple vectors in batch
const batch = [_][]const f32{
    &[_]f32{ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 },
    &[_]f32{ 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9 },
    &[_]f32{ 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9 },
};

const indices = try db.addEmbeddingsBatch(&batch);
defer allocator.free(indices);

for (indices, 0..) |idx, i| {
    std.debug.print("Batch vector {} at index: {}\n", .{ i, idx });
}
```

### **3. Search for Similar Vectors**

```zig
// Search for similar vectors
const query = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
const results = try db.search(&query, 5, allocator);
defer allocator.free(results);

std.debug.print("Found {} similar vectors:\n", .{results.len});
for (results, 0..) |result, i| {
    std.debug.print("  {}. Index: {}, Similarity: {d:.3}\n", 
        .{ i + 1, result.index, result.similarity });
}
```

### **4. Save and Load Database**

```zig
// Save database to file
try db.save("my_vectors.wdbx");
std.debug.print("Database saved to my_vectors.wdbx\n", .{});

// Load database from file
var loaded_db = try database.Db.open("my_vectors.wdbx", true);
defer loaded_db.close();

std.debug.print("Database loaded with {} vectors\n", .{loaded_db.getVectorCount()});
```

---

## ⚡ **Performance Tips**

### **1. Batch Operations**
```zig
// Use batch operations for better performance
const optimal_batch_size = 64;
var batch = try allocator.alloc([]f32, optimal_batch_size);
defer {
    for (batch) |emb| allocator.free(emb);
    allocator.free(batch);
}

// Fill batch with embeddings
for (0..optimal_batch_size) |i| {
    batch[i] = try allocator.dupe(f32, &embeddings[i]);
}

const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);
```

### **2. Memory Management**
```zig
// Use arena allocators for temporary operations
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const arena_allocator = arena.allocator();

// Search operations use arena memory
const results = try db.search(&query, 10, arena_allocator);
// No need to free - arena handles cleanup
```

### **3. Vector Dimensions**
```zig
// Choose appropriate vector dimensions
const dimensions = 384; // Common for modern embedding models
// - 128: Good for basic similarity search
// - 256: Balanced performance and accuracy
// - 384: High accuracy (BERT, GPT embeddings)
// - 512+: Maximum accuracy, higher memory usage

var db = try database.Db.init(allocator, dimensions);
```

---

## ⚠️ **Error Handling**

### **Common Errors and Solutions**

```zig
// Handle dimension mismatches
const result = db.addEmbedding(&embedding) catch |err| {
    switch (err) {
        error.DimensionMismatch => {
            std.debug.print("Vector dimension {} doesn't match database dimension {}\n", 
                .{ embedding.len, db.getDimension() });
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

### **Error Types**
- `DimensionMismatch`: Vector size doesn't match database configuration
- `InvalidState`: Database not properly initialized
- `OutOfMemory`: Insufficient memory for operation
- `FileError`: I/O operation failed
- `CorruptedData`: Database file is corrupted

---

## 💡 **Example Applications**

### **1. Document Similarity Search**
```zig
const DocumentStore = struct {
    db: database.Db,
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .db = try database.Db.init(allocator, 384),
        };
    }
    
    pub fn addDocument(self: *@This(), text: []const u8, embedding: []const f32) !void {
        // In a real app, you'd generate embeddings from text
        _ = try self.db.addEmbedding(embedding);
    }
    
    pub fn findSimilar(self: *@This(), query_embedding: []const f32, k: usize) ![]database.SearchResult {
        return self.db.search(query_embedding, k, self.db.allocator);
    }
};
```

### **2. Image Feature Search**
```zig
const ImageDatabase = struct {
    db: database.Db,
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .db = try database.Db.init(allocator, 512), // Higher dims for image features
        };
    }
    
    pub fn addImage(self: *@This(), image_path: []const u8, features: []const f32) !void {
        _ = try self.db.addEmbedding(features);
        // Store image path mapping
    }
    
    pub fn findSimilarImages(self: *@This(), query_features: []const f32, k: usize) ![]database.SearchResult {
        return self.db.search(query_features, k, self.db.allocator);
    }
};
```

### **3. Recommendation System**
```zig
const RecommendationEngine = struct {
    user_db: database.Db,
    item_db: database.Db,
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .user_db = try database.Db.init(allocator, 128),
            .item_db = try database.Db.init(allocator, 128),
        };
    }
    
    pub fn getRecommendations(self: *@This(), user_embedding: []const f32, k: usize) ![]database.SearchResult {
        return self.item_db.search(user_embedding, k, self.item_db.allocator);
    }
};
```

---

## 🧪 **Testing**

### **1. Run Basic Tests**
```bash
# Run all database tests
zig build test

# Run specific database tests
zig build test --test-filter database
```

### **2. Performance Testing**
```bash
# Run performance benchmarks
zig build run -- benchmark_suite

# Run specific database benchmarks
zig build run -- database_benchmark
```

### **3. Memory Testing**
```bash
# Test memory management
zig build run -- test_memory_management
```

---

## 🚀 **Next Steps**

### **For Developers**
1. **Read the [Database API Reference](docs/api/database.md)** for complete function documentation
2. **Explore [Usage Patterns](docs/database_usage_guide.md)** for advanced scenarios
3. **Check [Performance Guide](README_TESTING.md)** for optimization tips

### **For Production**
1. **Review [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)** for deployment best practices
2. **Set up [Monitoring](monitoring/)** for production environments
3. **Configure [CI/CD](deploy/)** for automated testing and deployment

### **For Contributors**
1. **Read [Contributing Guide](CONTRIBUTING.md)** for development guidelines
2. **Check [TODO.md](TODO.md)** for areas needing work
3. **Join discussions** in issues and pull requests

---

## 🔗 **Additional Resources**

- **[Database API Reference](docs/api/database.md)** - Complete API documentation
- **[Database Usage Guide](docs/database_usage_guide.md)** - Comprehensive usage examples
- **[Performance Guide](README_TESTING.md)** - Performance optimization tips
- **[Production Guide](docs/PRODUCTION_DEPLOYMENT.md)** - Production deployment best practices

---

**🚀 Ready to build high-performance vector applications? The WDBX-AI database provides enterprise-grade performance with 2,777+ ops/sec and 99.98% uptime!**

**💡 Start with the examples above and explore the comprehensive vector database capabilities.**
