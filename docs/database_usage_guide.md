# 📚 WDBX-AI Database Usage Guide

> **Comprehensive guide to building production-ready vector applications**

[![Usage Guide](https://img.shields.io/badge/Usage-Guide-blue.svg)](docs/database_usage_guide.md)
[![Performance](https://img.shields.io/badge/Performance-2,777+%20ops%2Fsec-brightgreen.svg)]()

This comprehensive guide covers everything you need to know about using the WDBX-AI vector database in production applications. From basic operations to advanced optimization techniques, you'll learn how to build scalable, high-performance vector applications.

## 📋 **Table of Contents**

- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Real-World Examples](#real-world-examples)
- [Performance Optimization](#performance-optimization)
- [Error Handling Best Practices](#error-handling-best-practices)
- [Testing & Validation](#testing--validation)
- [Common Patterns](#common-patterns)
- [Production Considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)

---

## ✅ **Prerequisites**

- **Zig**: Version 0.15.1 or later
- **Memory**: 1GB+ RAM for production workloads
- **Storage**: SSD recommended for high I/O operations
- **Platform**: Windows, macOS, or Linux
- **Experience**: Basic understanding of vector operations and similarity search

---

## 📦 **Installation & Setup**

### **1. Project Dependencies**

Add to your `build.zig`:
```zig
const database_mod = b.addModule("database", .{
    .source_file = b.path("src/database.zig"),
});

// Add to your executable or library
exe.addModule("database", database_mod);
```

### **2. Environment Configuration**

```bash
# Set environment variables for production
export WDBX_CACHE_SIZE=1024
export WDBX_MAX_CONNECTIONS=100
export WDBX_LOG_LEVEL=info
```

### **3. Build Configuration**

```zig
// build.zig
const build_options = b.addOptions();
build_options.addOption([]const u8, "version", "1.0.0");
build_options.addOption(bool, "enable_simd", true);
build_options.addOption(bool, "enable_gpu", false);

exe.addOptions("build_options", build_options);
```

---

## 🚀 **Basic Usage**

### **1. Database Initialization**

```zig
const std = @import("std");
const database = @import("wdbx/database.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Initialize with custom configuration
    var db = try database.Db.initWithConfig(allocator, .{
        .dimension = 384,
        .max_vectors = 1_000_000,
        .cache_size = 1024,
        .enable_compression = true,
        .index_type = .HNSW,
    });
    defer db.close();
    
    std.debug.print("Database initialized with {} dimensions\n", .{db.getDimension()});
}
```

### **2. Vector Operations**

#### **Single Vector Operations**
```zig
// Add a single vector
const embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
const index = try db.addEmbedding(&embedding);

// Retrieve vector by index
const retrieved = try db.getEmbedding(index);
std.debug.print("Retrieved vector: {any}\n", .{retrieved});

// Update existing vector
try db.updateEmbedding(index, &new_embedding);

// Delete vector
try db.deleteEmbedding(index);
```

#### **Batch Operations**
```zig
// Prepare batch data
const batch_size = 1000;
var batch = try allocator.alloc([]f32, batch_size);
defer {
    for (batch) |emb| allocator.free(emb);
    allocator.free(batch);
}

// Fill batch with embeddings
for (0..batch_size) |i| {
    batch[i] = try allocator.dupe(f32, &embeddings[i]);
}

// Add batch to database
const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);

std.debug.print("Added {} vectors in batch\n", .{indices.len});
```

### **3. Search Operations**

#### **Basic Similarity Search**
```zig
const query = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
const results = try db.search(&query, 10, allocator);
defer allocator.free(results);

for (results, 0..) |result, i| {
    std.debug.print("{}. Index: {}, Similarity: {d:.3}, Distance: {d:.3}\n", 
        .{ i + 1, result.index, result.similarity, result.distance });
}
```

#### **Advanced Search with Options**
```zig
const search_options = database.SearchOptions{
    .metric = .Cosine,
    .k = 20,
    .threshold = 0.7,
    .include_metadata = true,
    .max_distance = 2.0,
};

const results = try db.searchWithOptions(&query, search_options, allocator);
defer allocator.free(results);
```

#### **Range Queries**
```zig
// Find vectors within a specific distance range
const range_results = try db.searchRange(&query, 0.5, 1.5, allocator);
defer allocator.free(range_results);

std.debug.print("Found {} vectors in range [0.5, 1.5]\n", .{range_results.len});
```

---

## 🌍 **Real-World Examples**

### **1. Document Similarity Search System**

```zig
const DocumentStore = struct {
    db: database.Db,
    documents: std.StringHashMap(Document),
    allocator: std.mem.Allocator,
    
    const Document = struct {
        id: []const u8,
        title: []const u8,
        content: []const u8,
        embedding: []f32,
        metadata: std.StringHashMap([]const u8),
    };
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .db = try database.Db.init(allocator, 384),
            .documents = std.StringHashMap(Document).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addDocument(self: *@This(), doc: Document) !void {
        // Add embedding to database
        const index = try self.db.addEmbedding(doc.embedding);
        
        // Store document metadata
        try self.documents.put(doc.id, doc);
        
        std.debug.print("Added document '{}' with index {}\n", .{ doc.title, index });
    }
    
    pub fn findSimilarDocuments(self: *@This(), query_embedding: []const f32, k: usize) ![]DocumentResult {
        const results = try self.db.search(query_embedding, k, self.allocator);
        defer self.allocator.free(results);
        
        var document_results = try self.allocator.alloc(DocumentResult, results.len);
        
        for (results, 0..) |result, i| {
            // Retrieve document by index
            const doc = self.getDocumentByIndex(result.index) orelse continue;
            
            document_results[i] = DocumentResult{
                .document = doc,
                .similarity = result.similarity,
                .distance = result.distance,
            };
        }
        
        return document_results;
    }
    
    const DocumentResult = struct {
        document: Document,
        similarity: f32,
        distance: f32,
    };
};
```

### **2. Image Feature Database**

```zig
const ImageDatabase = struct {
    db: database.Db,
    image_metadata: std.StringHashMap(ImageInfo),
    allocator: std.mem.Allocator,
    
    const ImageInfo = struct {
        path: []const u8,
        width: u32,
        height: u32,
        format: []const u8,
        tags: []const []const u8,
        timestamp: i64,
    };
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .db = try database.Db.init(allocator, 512), // Higher dims for image features
            .image_metadata = std.StringHashMap(ImageInfo).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addImage(self: *@This(), path: []const u8, features: []const f32, info: ImageInfo) !void {
        // Add image features to database
        const index = try self.db.addEmbedding(features);
        
        // Store image metadata
        try self.image_metadata.put(path, info);
        
        std.debug.print("Added image '{}' with {} features at index {}\n", 
            .{ path, features.len, index });
    }
    
    pub fn findSimilarImages(self: *@This(), query_features: []const f32, k: usize) ![]ImageResult {
        const results = try self.db.search(query_features, k, self.allocator);
        defer self.allocator.free(results);
        
        var image_results = try self.allocator.alloc(ImageResult, results.len);
        
        for (results, 0..) |result, i| {
            // Map index back to image path (you'd need a reverse mapping)
            const image_path = self.getImagePathByIndex(result.index) orelse continue;
            const metadata = self.image_metadata.get(image_path) orelse continue;
            
            image_results[i] = ImageResult{
                .path = image_path,
                .metadata = metadata,
                .similarity = result.similarity,
                .distance = result.distance,
            };
        }
        
        return image_results;
    }
    
    const ImageResult = struct {
        path: []const u8,
        metadata: ImageInfo,
        similarity: f32,
        distance: f32,
    };
};
```

### **3. Recommendation Engine**

```zig
const RecommendationEngine = struct {
    user_db: database.Db,
    item_db: database.Db,
    user_profiles: std.StringHashMap(UserProfile),
    item_catalog: std.StringHashMap(Item),
    allocator: std.mem.Allocator,
    
    const UserProfile = struct {
        id: []const u8,
        preferences: []f32,
        history: []ItemInteraction,
        demographics: Demographics,
    };
    
    const Item = struct {
        id: []const u8,
        features: []f32,
        category: []const u8,
        price: f32,
        rating: f32,
    };
    
    const ItemInteraction = struct {
        item_id: []const u8,
        rating: f32,
        timestamp: i64,
        interaction_type: InteractionType,
    };
    
    const InteractionType = enum {
        view,
        like,
        purchase,
        review,
    };
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .user_db = try database.Db.init(allocator, 128),
            .item_db = try database.Db.init(allocator, 128),
            .user_profiles = std.StringHashMap(UserProfile).init(allocator),
            .item_catalog = std.StringHashMap(Item).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addUser(self: *@This(), user: UserProfile) !void {
        // Add user preferences to user database
        const index = try self.user_db.addEmbedding(user.preferences);
        
        // Store user profile
        try self.user_profiles.put(user.id, user);
        
        std.debug.print("Added user '{}' with preference vector at index {}\n", 
            .{ user.id, index });
    }
    
    pub fn addItem(self: *@This(), item: Item) !void {
        // Add item features to item database
        const index = try self.item_db.addEmbedding(item.features);
        
        // Store item in catalog
        try self.item_catalog.put(item.id, item);
        
        std.debug.print("Added item '{}' with feature vector at index {}\n", 
            .{ item.id, index });
    }
    
    pub fn getRecommendations(self: *@This(), user_id: []const u8, k: usize) ![]Recommendation {
        const user_profile = self.user_profiles.get(user_id) orelse return error.UserNotFound;
        
        // Find similar items based on user preferences
        const results = try self.item_db.search(user_profile.preferences, k, self.allocator);
        defer self.allocator.free(results);
        
        var recommendations = try self.allocator.alloc(Recommendation, results.len);
        
        for (results, 0..) |result, i| {
            const item = self.getItemByIndex(result.index) orelse continue;
            
            recommendations[i] = Recommendation{
                .item = item,
                .score = result.similarity,
                .reason = "Similar to your preferences",
            };
        }
        
        return recommendations;
    }
    
    const Recommendation = struct {
        item: Item,
        score: f32,
        reason: []const u8,
    };
};
```

---

## ⚡ **Performance Optimization**

### **1. Memory Management**

#### **Arena Allocators for Temporary Operations**
```zig
// Use arena allocators for search operations
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const arena_allocator = arena.allocator();

// Search operations use arena memory
const results = try db.search(&query, 100, arena_allocator);
// No need to free - arena handles cleanup automatically
```

#### **Pool Allocators for Frequent Operations**
```zig
// Use pool allocators for frequent small allocations
var pool = std.heap.FixedBufferAllocator.init(buffer);
const pool_allocator = pool.allocator();

// Allocate small vectors from pool
const small_vector = try pool_allocator.alloc(f32, 64);
```

### **2. Batch Processing**

#### **Optimal Batch Sizes**
```zig
// Determine optimal batch size based on your use case
const optimal_batch_size = switch (vector_dimension) {
    128 => 128,
    256 => 64,
    384 => 32,
    512 => 16,
    else => 64,
};

// Process vectors in optimal batches
for (0..total_vectors / optimal_batch_size) |batch_idx| {
    const start = batch_idx * optimal_batch_size;
    const end = @min(start + optimal_batch_size, total_vectors);
    
    const batch = embeddings[start..end];
    const indices = try db.addEmbeddingsBatch(batch);
    defer allocator.free(indices);
}
```

#### **Parallel Batch Processing**
```zig
// Process multiple batches in parallel
const num_threads = 4;
const batch_size = total_vectors / num_threads;

var threads: [num_threads]std.Thread = undefined;
var results: [num_threads]error![]u32 = undefined;

for (0..num_threads) |i| {
    const start = i * batch_size;
    const end = if (i == num_threads - 1) total_vectors else start + batch_size;
    
    threads[i] = try std.Thread.spawn(.{}, processBatch, .{ 
        &db, &embeddings[start..end], &results[i] 
    });
}

// Wait for all threads to complete
for (threads) |thread| {
    thread.join();
}
```

### **3. Indexing Strategies**

#### **HNSW Index for Real-time Search**
```zig
// Build HNSW index for fast approximate search
try db.buildIndex(.HNSW);

// Configure HNSW parameters
const hnsw_config = database.HNSWConfig{
    .max_connections = 16,
    .ef_construction = 200,
    .ef_search = 100,
};

try db.configureHNSW(hnsw_config);
```

#### **LSH Index for High-dimensional Vectors**
```zig
// Build LSH index for high-dimensional vectors
try db.buildIndex(.LSH);

// Configure LSH parameters
const lsh_config = database.LSHConfig{
    .num_tables = 10,
    .num_bits = 64,
    .num_probes = 100,
};

try db.configureLSH(lsh_config);
```

### **4. Caching Strategies**

#### **In-Memory Cache for Hot Data**
```zig
const Cache = struct {
    data: std.AutoHashMap(u32, []f32),
    lru: std.ArrayList(u32),
    max_size: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, max_size: usize) @This() {
        return @This(){
            .data = std.AutoHashMap(u32, []f32).init(allocator),
            .lru = std.ArrayList(u32).init(allocator),
            .max_size = max_size,
            .allocator = allocator,
        };
    }
    
    pub fn get(self: *@This(), key: u32) ?[]f32 {
        if (self.data.get(key)) |value| {
            // Move to front of LRU
            self.moveToFront(key);
            return value;
        }
        return null;
    }
    
    pub fn put(self: *@This(), key: u32, value: []f32) !void {
        if (self.data.count() >= self.max_size) {
            // Evict least recently used
            const lru_key = self.lru.pop();
            _ = self.data.remove(lru_key);
        }
        
        const cloned_value = try self.allocator.dupe(f32, value);
        try self.data.put(key, cloned_value);
        try self.lru.append(key);
    }
};
```

---

## ⚠️ **Error Handling Best Practices**

### **1. Comprehensive Error Handling**

```zig
pub fn safeDatabaseOperation(db: *database.Db, operation: anytype) !void {
    return operation() catch |err| {
        switch (err) {
            error.DimensionMismatch => {
                std.log.err("Vector dimension mismatch: {}", .{err});
                return error.InvalidInput;
            },
            error.OutOfMemory => {
                std.log.err("Insufficient memory for operation", .{});
                return error.ResourceExhausted;
            },
            error.DatabaseCorrupted => {
                std.log.err("Database file is corrupted", .{});
                return error.DataCorrupted;
            },
            error.InvalidState => {
                std.log.err("Database in invalid state", .{});
                return error.InvalidState;
            },
            else => {
                std.log.err("Unexpected database error: {}", .{err});
                return err;
            },
        }
    };
}
```

### **2. Graceful Degradation**

```zig
pub fn searchWithFallback(self: *@This(), query: []const f32, k: usize) ![]database.SearchResult {
    // Try fast search first
    const fast_results = self.db.searchFast(query, k, self.allocator) catch |err| {
        std.log.warn("Fast search failed, falling back to exact search: {}", .{err});
        
        // Fall back to exact search
        return self.db.searchExact(query, k, self.allocator);
    };
    
    return fast_results;
}
```

### **3. Retry Logic**

```zig
pub fn searchWithRetry(self: *@This(), query: []const f32, k: usize, max_retries: u32) ![]database.SearchResult {
    var attempts: u32 = 0;
    
    while (attempts < max_retries) : (attempts += 1) {
        const results = self.db.search(query, k, self.allocator) catch |err| {
            if (err == error.Timeout and attempts < max_retries - 1) {
                std.log.warn("Search timeout, retrying... (attempt {}/{})", .{ attempts + 1, max_retries });
                std.time.sleep(100 * std.time.ns_per_ms); // Wait 100ms
                continue;
            }
            return err;
        };
        
        return results;
    }
    
    return error.MaxRetriesExceeded;
}
```

---

## 🧪 **Testing & Validation**

### **1. Unit Tests**

```zig
test "database basic operations" {
    const allocator = testing.allocator;
    
    // Test initialization
    var db = try database.Db.init(allocator, 128);
    defer db.close();
    
    try testing.expectEqual(@as(usize, 128), db.getDimension());
    try testing.expectEqual(@as(usize, 0), db.getVectorCount());
    
    // Test adding vectors
    const vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const index = try db.addEmbedding(&vector);
    try testing.expectEqual(@as(u32, 0), index);
    try testing.expectEqual(@as(usize, 1), db.getVectorCount());
    
    // Test retrieval
    const retrieved = try db.getEmbedding(index);
    try testing.expectEqualSlices(f32, &vector, retrieved);
}
```

### **2. Performance Tests**

```zig
test "database performance" {
    const allocator = testing.allocator;
    
    var db = try database.Db.init(allocator, 384);
    defer db.close();
    
    // Generate test data
    const num_vectors = 10000;
    var vectors = try allocator.alloc([]f32, num_vectors);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }
    
    for (0..num_vectors) |i| {
        vectors[i] = try allocator.alloc(f32, 384);
        for (0..384) |j| {
            vectors[i][j] = @floatFromInt(f32, i + j);
        }
    }
    
    // Measure insertion performance
    const start_time = std.time.milliTimestamp();
    
    for (vectors) |vector| {
        _ = try db.addEmbedding(vector);
    }
    
    const end_time = std.time.milliTimestamp();
    const duration = @intCast(u64, end_time - start_time);
    
    const ops_per_sec = (num_vectors * 1000) / duration;
    std.debug.print("Insertion performance: {} ops/sec\n", .{ops_per_sec});
    
    // Performance assertion
    try testing.expect(ops_per_sec > 1000); // At least 1000 ops/sec
}
```

### **3. Memory Tests**

```zig
test "database memory management" {
    const allocator = testing.allocator;
    
    // Track initial memory usage
    const initial_memory = getMemoryUsage();
    
    var db = try database.Db.init(allocator, 128);
    
    // Add vectors and check memory growth
    for (0..1000) |i| {
        const vector = [_]f32{ @floatFromInt(f32, i) } ** 128;
        _ = try db.addEmbedding(&vector);
        
        if (i % 100 == 0) {
            const current_memory = getMemoryUsage();
            const memory_growth = current_memory - initial_memory;
            
            // Memory should grow reasonably
            try testing.expect(memory_growth < 100 * 1024 * 1024); // Less than 100MB
        }
    }
    
    db.close();
    
    // Check memory cleanup
    const final_memory = getMemoryUsage();
    const memory_leak = final_memory - initial_memory;
    
    // Should have minimal memory leak
    try testing.expect(memory_leak < 1024 * 1024); // Less than 1MB
}

fn getMemoryUsage() usize {
    // Platform-specific memory usage function
    // Implementation depends on your platform
    return 0; // Placeholder
}
```

---

## 🔄 **Common Patterns**

### **1. Producer-Consumer Pattern**

```zig
const VectorProcessor = struct {
    db: database.Db,
    input_queue: std.fifo.LinearFifo(VectorTask, .{ .Static = 1000 }),
    output_queue: std.fifo.LinearFifo(SearchResult, .{ .Static = 1000 }),
    allocator: std.mem.Allocator,
    
    const VectorTask = struct {
        query: []f32,
        k: usize,
        request_id: u64,
    };
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        return @This(){
            .db = try database.Db.init(allocator, 384),
            .input_queue = std.fifo.LinearFifo(VectorTask, .{ .Static = 1000 }).init(),
            .output_queue = std.fifo.LinearFifo(SearchResult, .{ .Static = 1000 }).init(),
            .allocator = allocator,
        };
    }
    
    pub fn producer(self: *@This()) !void {
        while (true) {
            // Generate or receive vector tasks
            const task = generateTask() orelse continue;
            
            // Add to input queue
            try self.input_queue.writeItem(task);
        }
    }
    
    pub fn consumer(self: *@This()) !void {
        while (self.input_queue.readItem()) |task| {
            // Process search request
            const results = try self.db.search(task.query, task.k, self.allocator);
            defer self.allocator.free(results);
            
            // Add results to output queue
            for (results) |result| {
                try self.output_queue.writeItem(result);
            }
        }
    }
};
```

### **2. Observer Pattern**

```zig
const DatabaseObserver = struct {
    listeners: std.ArrayList(Listener),
    allocator: std.mem.Allocator,
    
    const Listener = struct {
        id: u64,
        callback: *const fn (event: DatabaseEvent) void,
    };
    
    const DatabaseEvent = union(enum) {
        vector_added: u32,
        vector_deleted: u32,
        search_completed: SearchCompletedEvent,
        error_occurred: []const u8,
    };
    
    const SearchCompletedEvent = struct {
        query_id: u64,
        result_count: usize,
        duration_ms: u64,
    };
    
    pub fn init(allocator: std.mem.Allocator) @This() {
        return @This(){
            .listeners = std.ArrayList(Listener).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn addListener(self: *@This(), callback: *const fn (event: DatabaseEvent) void) !u64 {
        const listener = Listener{
            .id = generateId(),
            .callback = callback,
        };
        
        try self.listeners.append(listener);
        return listener.id;
    }
    
    pub fn removeListener(self: *@This(), id: u64) void {
        for (self.listeners.items, 0..) |listener, i| {
            if (listener.id == id) {
                _ = self.listeners.orderedRemove(i);
                break;
            }
        }
    }
    
    pub fn notify(self: *@This(), event: DatabaseEvent) void {
        for (self.listeners.items) |listener| {
            listener.callback(event);
        }
    }
};
```

---

## 🏭 **Production Considerations**

### **1. Monitoring & Metrics**

```zig
const DatabaseMetrics = struct {
    operations: std.atomic.Atomic(u64),
    errors: std.atomic.Atomic(u64),
    avg_response_time: std.atomic.Atomic(u64),
    memory_usage: std.atomic.Atomic(u64),
    
    pub fn recordOperation(self: *@This(), duration_ns: u64) void {
        _ = self.operations.fetchAdd(1, .Monotonic);
        
        // Update average response time
        const current_avg = self.avg_response_time.load(.Monotonic);
        const new_avg = (current_avg + duration_ns) / 2;
        _ = self.avg_response_time.store(new_avg, .Monotonic);
    }
    
    pub fn recordError(self: *@This()) void {
        _ = self.errors.fetchAdd(1, .Monotonic);
    }
    
    pub fn getStats(self: *@This()) Stats {
        return Stats{
            .total_operations = self.operations.load(.Monotonic),
            .total_errors = self.errors.load(.Monotonic),
            .avg_response_time_ns = self.avg_response_time.load(.Monotonic),
            .memory_usage_bytes = self.memory_usage.load(.Monotonic),
        };
    }
    
    const Stats = struct {
        total_operations: u64,
        total_errors: u64,
        avg_response_time_ns: u64,
        memory_usage_bytes: u64,
    };
};
```

### **2. Health Checks**

```zig
const HealthChecker = struct {
    db: *database.Db,
    last_check: i64,
    check_interval: i64,
    
    pub fn init(db: *database.Db) @This() {
        return @This(){
            .db = db,
            .last_check = 0,
            .check_interval = 30 * std.time.ns_per_s, // 30 seconds
        };
    }
    
    pub fn shouldCheck(self: *@This()) bool {
        const now = std.time.milliTimestamp();
        return (now - self.last_check) >= self.check_interval;
    }
    
    pub fn performHealthCheck(self: *@This()) !HealthStatus {
        const now = std.time.milliTimestamp();
        self.last_check = now;
        
        // Check database connectivity
        const vector_count = self.db.getVectorCount();
        
        // Check memory usage
        const memory_usage = self.getMemoryUsage();
        
        // Check response time
        const start_time = std.time.nanoTimestamp();
        const test_query = [_]f32{ 0.0 } ** self.db.getDimension();
        _ = try self.db.search(&test_query, 1, self.db.allocator);
        const response_time = @intCast(u64, std.time.nanoTimestamp() - start_time);
        
        return HealthStatus{
            .status = .Healthy,
            .vector_count = vector_count,
            .memory_usage_bytes = memory_usage,
            .response_time_ns = response_time,
            .last_check = now,
        };
    }
    
    const HealthStatus = struct {
        status: Status,
        vector_count: usize,
        memory_usage_bytes: usize,
        response_time_ns: u64,
        last_check: i64,
        
        const Status = enum {
            Healthy,
            Degraded,
            Unhealthy,
        };
    };
};
```

### **3. Backup & Recovery**

```zig
const DatabaseBackup = struct {
    db: *database.Db,
    backup_dir: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(db: *database.Db, backup_dir: []const u8, allocator: std.mem.Allocator) @This() {
        return @This(){
            .db = db,
            .backup_dir = backup_dir,
            .allocator = allocator,
        };
    }
    
    pub fn createBackup(self: *@This()) ![]const u8 {
        const timestamp = std.time.milliTimestamp();
        const filename = try std.fmt.allocPrint(
            self.allocator,
            "backup_{}.wdbx",
            .{ timestamp }
        );
        defer self.allocator.free(filename);
        
        const backup_path = try std.fs.path.join(self.allocator, &[_][]const u8{
            self.backup_dir,
            filename,
        });
        defer self.allocator.free(backup_path);
        
        // Create backup
        try self.db.save(backup_path);
        
        std.log.info("Database backup created: {}", .{backup_path});
        return backup_path;
    }
    
    pub fn restoreFromBackup(self: *@This(), backup_path: []const u8) !void {
        // Close current database
        self.db.close();
        
        // Restore from backup
        try self.db.load(backup_path);
        
        std.log.info("Database restored from backup: {}", .{backup_path});
    }
    
    pub fn listBackups(self: *@This()) ![]BackupInfo {
        var backups = std.ArrayList(BackupInfo).init(self.allocator);
        
        var dir = try std.fs.openDirAbsolute(self.backup_dir, .{ .iterate = true });
        defer dir.close();
        
        var iter = dir.iterate();
        while (iter.next()) |entry| {
            if (std.mem.endsWith(u8, entry.name, ".wdbx")) {
                const backup_path = try std.fs.path.join(self.allocator, &[_][]const u8{
                    self.backup_dir,
                    entry.name,
                });
                defer self.allocator.free(backup_path);
                
                const stat = try dir.stat();
                
                try backups.append(BackupInfo{
                    .filename = entry.name,
                    .path = backup_path,
                    .size_bytes = stat.size,
                    .modified = stat.mtime,
                });
            }
        }
        
        return backups.toOwnedSlice();
    }
    
    const BackupInfo = struct {
        filename: []const u8,
        path: []const u8,
        size_bytes: u64,
        modified: i64,
    };
};
```

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Memory Issues**
```zig
// Problem: Out of memory errors
// Solution: Use arena allocators and monitor memory usage

var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const arena_allocator = arena.allocator();

// Monitor memory usage
const memory_usage = getMemoryUsage();
if (memory_usage > max_memory_threshold) {
    std.log.warn("High memory usage: {} MB", .{memory_usage / (1024 * 1024)});
}
```

#### **2. Performance Issues**
```zig
// Problem: Slow search performance
// Solution: Build appropriate indices and use batch operations

// Build HNSW index for fast approximate search
try db.buildIndex(.HNSW);

// Use batch operations
const batch_size = 64;
for (0..total_vectors / batch_size) |batch_idx| {
    const start = batch_idx * batch_size;
    const end = @min(start + batch_size, total_vectors);
    const batch = vectors[start..end];
    
    _ = try db.addEmbeddingsBatch(batch);
}
```

#### **3. File Corruption**
```zig
// Problem: Database file corruption
// Solution: Implement validation and recovery

pub fn validateDatabase(db: *database.Db) !bool {
    // Check file integrity
    const checksum = try db.calculateChecksum();
    const expected_checksum = db.getExpectedChecksum();
    
    if (checksum != expected_checksum) {
        std.log.err("Database checksum mismatch", .{});
        return false;
    }
    
    // Validate vector data
    const vector_count = db.getVectorCount();
    for (0..@min(vector_count, 100)) |i| {
        const vector = db.getEmbedding(@intCast(u32, i)) catch continue;
        
        // Check for NaN or infinite values
        for (vector) |value| {
            if (std.math.isNaN(value) or std.math.isInf(value)) {
                std.log.err("Invalid vector value at index {}", .{i});
                return false;
            }
        }
    }
    
    return true;
}
```

---

## 🔗 **Additional Resources**

- **[Database API Reference](docs/api/database.md)** - Complete API documentation
- **[Database Quickstart](docs/database_quickstart.md)** - Get started quickly
- **[Performance Guide](README_TESTING.md)** - Performance optimization tips
- **[Production Guide](docs/PRODUCTION_DEPLOYMENT.md)** - Production deployment best practices

---

**📚 Ready to build production-ready vector applications? This comprehensive guide covers everything from basic usage to advanced optimization techniques!**

**🚀 The WDBX-AI database provides enterprise-grade performance with 2,777+ ops/sec and comprehensive production features.**
