---
layout: documentation
title: "Module Reference"
description: "Comprehensive reference for all ABI modules and components"
---

# ABI Module Reference

## 📦 Core Modules

### `abi` - Main Module
The primary module containing all core functionality.

#### Key Components:
- **Database Engine**: High-performance vector database with HNSW indexing
- **AI System**: Neural networks and machine learning capabilities
- **SIMD Operations**: Optimized vector operations
- **Plugin System**: Extensible architecture for custom functionality

### `abi.database` - Database Module
Vector database operations and management.

#### Functions:
```zig
// Initialize database
pub fn init(allocator: Allocator, config: DatabaseConfig) !Database

// Insert vector
pub fn insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64

// Search vectors
pub fn search(self: *Database, query: []const f32, k: usize) ![]SearchResult

// Update vector
pub fn update(self: *Database, id: u64, vector: []const f32) !void

// Delete vector
pub fn delete(self: *Database, id: u64) !void
```

### `abi.ai` - AI Module
Artificial intelligence and machine learning capabilities.

#### Functions:
```zig
// Create neural network
pub fn createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork

// Train network
pub fn train(self: *NeuralNetwork, data: []const TrainingData) !f32

// Predict/Infer
pub fn predict(self: *NeuralNetwork, input: []const f32) ![]f32

// Enhanced agent operations
pub fn createAgent(allocator: Allocator, config: AgentConfig) !EnhancedAgent
```

### `abi.simd` - SIMD Module
SIMD-optimized vector operations.

#### Functions:
```zig
// Vector addition
pub fn add(result: []f32, a: []const f32, b: []const f32) void

// Vector subtraction
pub fn subtract(result: []f32, a: []const f32, b: []const f32) void

// Vector multiplication
pub fn multiply(result: []f32, a: []const f32, b: []const f32) void

// Vector normalization
pub fn normalize(result: []f32, input: []const f32) void
```

### `abi.plugins` - Plugin System
Extensible plugin architecture.

#### Functions:
```zig
// Load plugin
pub fn loadPlugin(path: []const u8) !Plugin

// Register plugin
pub fn registerPlugin(plugin: Plugin) !void

// Execute plugin function
pub fn executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8
```

## 🔧 Configuration Types

### DatabaseConfig
```zig
pub const DatabaseConfig = struct {
    max_vectors: usize = 1000000,
    vector_dimension: usize = 128,
    index_type: IndexType = .hnsw,
    storage_path: ?[]const u8 = null,
    enable_caching: bool = true,
    cache_size: usize = 1024 * 1024, // 1MB
};
```

### NetworkConfig
```zig
pub const NetworkConfig = struct {
    input_size: usize,
    hidden_sizes: []const usize,
    output_size: usize,
    activation: ActivationType = .relu,
    learning_rate: f32 = 0.01,
    batch_size: usize = 32,
};
```

## 📊 Performance Characteristics

| Operation | Performance | Memory Usage |
|-----------|-------------|--------------|
| Vector Insert | ~2.5ms (1000 vectors) | ~512 bytes/vector |
| Vector Search | ~13ms (10k vectors, k=10) | ~160 bytes/result |
| Neural Training | ~30μs/iteration | ~1MB/network |
| SIMD Operations | ~3μs (2048 elements) | ~16KB/batch |

## 🚀 Usage Examples

### Basic Database Usage
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize database
    const config = abi.DatabaseConfig{
        .max_vectors = 10000,
        .vector_dimension = 128,
    };
    var db = try abi.database.init(allocator, config);
    defer db.deinit();

    // Insert vectors
    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions
    const id = try db.insert(&vector, "sample_data");

    // Search for similar vectors
    const results = try db.search(&vector, 10);
    defer allocator.free(results);

    std.log.info("Found {} similar vectors", .{results.len});
}
```

### Neural Network Training
```zig
const config = abi.NetworkConfig{
    .input_size = 128,
    .hidden_sizes = &[_]usize{64, 32},
    .output_size = 10,
    .learning_rate = 0.01,
};

var network = try abi.ai.createNetwork(allocator, config);
defer network.deinit();

// Training data
const training_data = [_]abi.TrainingData{
    .{ .input = &input1, .output = &output1 },
    .{ .input = &input2, .output = &output2 },
};

// Train network
const loss = try network.train(&training_data);
std.log.info("Training loss: {}", .{loss});
```

## 🔍 Error Handling

All functions return appropriate error types:
- `DatabaseError` - Database-specific errors
- `AIError` - AI/ML operation errors
- `SIMDError` - SIMD operation errors
- `PluginError` - Plugin system errors

## 📈 Performance Tips

1. **Use appropriate vector dimensions** - 128-512 dimensions typically optimal
2. **Batch operations** - Group multiple operations for better performance
3. **Enable caching** - Significant performance improvement for repeated queries
4. **SIMD optimization** - Automatically enabled for supported operations
5. **Memory management** - Use arena allocators for bulk operations
