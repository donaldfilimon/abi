---
layout: documentation
title: "API Reference"
description: "Complete API reference for ABI with detailed function documentation"
---

# ABI API Reference

## 🗄️ Database API

### Database
Main database interface for vector operations.

#### Methods

##### `init(allocator: Allocator, config: DatabaseConfig) !Database`
Initialize a new database instance.

**Parameters:**
- `allocator`: Memory allocator to use
- `config`: Database configuration

**Returns:** Initialized database instance

**Errors:** `DatabaseError.OutOfMemory`, `DatabaseError.InvalidConfig`

##### `insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64`
Insert a vector into the database.

**Parameters:**
- `vector`: Vector data (must match configured dimension)
- `metadata`: Optional metadata string

**Returns:** Unique ID for the inserted vector

**Performance:** ~2.5ms for 1000 vectors

##### `search(self: *Database, query: []const f32, k: usize) ![]SearchResult`
Search for k nearest neighbors.

**Parameters:**
- `query`: Query vector
- `k`: Number of results to return

**Returns:** Array of search results (caller must free)

**Performance:** ~13ms for 10k vectors, k=10

## 🧠 AI API

### NeuralNetwork
Neural network for machine learning operations.

#### Methods

##### `createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork`
Create a new neural network.

**Parameters:**
- `allocator`: Memory allocator
- `config`: Network configuration

**Returns:** Initialized neural network

##### `train(self: *NeuralNetwork, data: []const TrainingData) !f32`
Train the neural network.

**Parameters:**
- `data`: Training data array

**Returns:** Final training loss

##### `predict(self: *NeuralNetwork, input: []const f32) ![]f32`
Make predictions using the trained network.

**Parameters:**
- `input`: Input vector

**Returns:** Prediction results (caller must free)

## ⚡ SIMD API

### Vector Operations
SIMD-optimized vector operations.

#### Functions

##### `add(result: []f32, a: []const f32, b: []const f32) void`
Add two vectors element-wise.

**Parameters:**
- `result`: Output vector (must be same size as inputs)
- `a`: First input vector
- `b`: Second input vector

**Performance:** ~3μs for 2048 elements

##### `normalize(result: []f32, input: []const f32) void`
Normalize a vector to unit length.

**Parameters:**
- `result`: Output normalized vector
- `input`: Input vector to normalize

## 🔌 Plugin API

### Plugin System
Extensible plugin architecture.

#### Functions

##### `loadPlugin(path: []const u8) !Plugin`
Load a plugin from file.

**Parameters:**
- `path`: Path to plugin file

**Returns:** Loaded plugin instance

##### `executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8`
Execute a plugin function.

**Parameters:**
- `plugin`: Plugin instance
- `function`: Function name to execute
- `args`: JSON-encoded arguments

**Returns:** JSON-encoded result (caller must free)

## 📊 Data Types

### SearchResult
```zig
pub const SearchResult = struct {
    id: u64,
    distance: f32,
    metadata: ?[]const u8,
};
```

### TrainingData
```zig
pub const TrainingData = struct {
    input: []const f32,
    output: []const f32,
};
```

## ⚠️ Error Types

### DatabaseError
```zig
pub const DatabaseError = error{
    OutOfMemory,
    InvalidConfig,
    VectorDimensionMismatch,
    IndexNotFound,
    StorageError,
};
```

### AIError
```zig
pub const AIError = error{
    InvalidNetworkConfig,
    TrainingDataEmpty,
    ConvergenceFailed,
    InvalidInputSize,
};
```
