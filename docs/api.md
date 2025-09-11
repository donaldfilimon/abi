---
layout: default
title: API Reference
---

# API Reference

This page provides a comprehensive reference for the ABI framework API.

## Core Modules

### Database API

#### `abi.Db`

The main vector database interface.

```zig
const db = try abi.Db.init(allocator, config);
defer db.deinit();

// Insert a vector
try db.insert(&vector, "unique_id");

// Search for similar vectors
const results = try db.search(&query_vector, k_neighbors);

// Batch operations
try db.batchInsert(&vectors, &ids);
```

**Configuration Options:**
- `dimensions`: Vector dimensionality (required)
- `max_elements`: Maximum number of vectors (default: 1M)
- `m`: HNSW parameter for index connectivity (default: 16)
- `ef_construction`: Build-time search parameter (default: 200)
- `ef_search`: Search-time parameter (default: 64)

#### `abi.DbConfig`

Database configuration structure.

```zig
const config = abi.DbConfig{
    .dimensions = 128,
    .max_elements = 1_000_000,
    .distance_metric = .cosine,
    .index_type = .hnsw,
};
```

### Vector Operations

#### `abi.VectorOps`

SIMD-accelerated vector operations.

```zig
// Distance calculations
const distance = abi.VectorOps.distance(&a, &b);
const similarity = abi.VectorOps.cosineSimilarity(&a, &b);

// Basic arithmetic
abi.VectorOps.add(&result, &a, &b);
abi.VectorOps.scale(&result, &vector, scalar);

// Normalization
abi.VectorOps.normalize(&result, &vector);
```

**Available Operations:**
- `add`: Element-wise addition
- `multiply`: Element-wise multiplication
- `scale`: Scalar multiplication
- `normalize`: L2 normalization
- `distance`: Euclidean distance
- `cosineSimilarity`: Cosine similarity
- `dotProduct`: Dot product

### GPU Operations

#### `abi.GPURenderer`

GPU-accelerated rendering and compute operations.

```zig
// Initialize GPU renderer
var renderer = try abi.GPURenderer.init(allocator, config);
defer renderer.deinit();

// Matrix operations
try renderer.matrixMultiply(&result, &a, &b, m, k, n);

// Buffer operations
const buffer = try renderer.createBuffer(.{
    .size = 1024 * 1024,
    .usage = .{ .storage = true, .copy_dst = true },
});
```

**Configuration Options:**
- `debug_validation`: Enable Vulkan validation layers
- `power_preference`: High performance vs low power
- `backend`: Auto, Vulkan, WebGPU, or CPU fallback

### AI Components

#### `abi.NeuralNetwork`

Neural network implementation.

```zig
// Create a neural network
var nn = try abi.NeuralNetwork.init(allocator, &[_]usize{784, 128, 64, 10});
defer nn.deinit();

// Add layers
try nn.addLayer(.{
    .type = .Dense,
    .input_size = 784,
    .output_size = 128,
    .activation = .ReLU,
});

// Training
const loss = try nn.train(&inputs, &targets, .{
    .learning_rate = 0.01,
    .epochs = 100,
    .batch_size = 32,
});
```

#### `abi.Layer`

Neural network layer configuration.

```zig
const layer = abi.Layer{
    .type = .Dense,
    .input_size = 784,
    .output_size = 128,
    .activation = .ReLU,
    .dropout_rate = 0.2,
    .weight_initializer = .xavier,
};
```

### Memory Management

#### `abi.MemoryPool`

Efficient memory pooling for frequent allocations.

```zig
// Initialize memory pool
var pool = try abi.MemoryPool.init(allocator, .{
    .enable_tracking = true,
    .initial_capacity = 1024,
    .max_buffer_size = 1024 * 1024,
});
defer pool.deinit();

// Allocate from pool
const buffer = try pool.allocBuffer(1024);
defer buffer.release();

// Use buffer...
pool.returnBuffer(buffer);
```

### Monitoring and Profiling

#### `abi.PerformanceProfiler`

Performance profiling and monitoring.

```zig
// Initialize profiler
var profiler = try abi.PerformanceProfiler.init(allocator);
defer profiler.deinit();

// Profile operations
profiler.start("operation_name");
// ... perform operation ...
const duration = profiler.end("operation_name");

// Get statistics
const stats = profiler.getStats();
```

#### `abi.MemoryTracker`

Memory usage tracking and leak detection.

```zig
// Initialize tracker
var tracker = try abi.MemoryTracker.init(allocator);
defer tracker.deinit();

// Track allocations
const ptr = try tracker.alloc(u8, 1024);
defer tracker.free(ptr);

// Get memory statistics
const stats = tracker.getStats();
```

## Error Types

### `abi.DbError`

Database operation errors.

```zig
pub const DbError = error{
    OutOfMemory,
    InvalidDimensions,
    ElementNotFound,
    DuplicateId,
    CapacityExceeded,
    InvalidConfig,
};
```

### `abi.GpuError`

GPU operation errors.

```zig
pub const GpuError = error{
    DeviceLost,
    OutOfMemory,
    UnsupportedOperation,
    CompilationFailed,
    ValidationError,
};
```

## Configuration Types

### `abi.PerformanceThresholds`

Performance monitoring thresholds.

```zig
const thresholds = abi.PerformanceThresholds{
    .max_search_time_ns = 20_000_000,  // 20ms
    .max_memory_usage_mb = 1024,       // 1GB
    .min_search_qps = 1000,            // 1000 queries/sec
    .max_regression_percent = 15.0,    // 15% regression threshold
};
```

## Utility Functions

### Memory Utilities

```zig
// Safe memory operations
const aligned_ptr = abi.mem.alignForward(usize, ptr, alignment);
const copied_data = try abi.mem.dupe(allocator, T, &data);

// Array utilities
const concatenated = try abi.mem.concat(allocator, T, &[_][]const T{ a, b });
```

### Math Utilities

```zig
// Vector math
const magnitude = abi.math.vectorMagnitude(&vec);
const normalized = try abi.math.normalizeVector(allocator, &vec);

// Statistical operations
const mean = abi.math.mean(&values);
const variance = abi.math.variance(&values);
```

### String Utilities

```zig
// String operations
const formatted = try abi.str.format(allocator, "Value: {}", .{value});
const trimmed = abi.str.trim(whitespace, input);
```

## Constants

### Default Configuration Values

```zig
pub const DEFAULT_DIMENSIONS = 128;
pub const DEFAULT_MAX_ELEMENTS = 1_000_000;
pub const DEFAULT_HNSW_M = 16;
pub const DEFAULT_EF_CONSTRUCTION = 200;
pub const DEFAULT_EF_SEARCH = 64;
pub const DEFAULT_LEARNING_RATE = 0.01;
pub const DEFAULT_BATCH_SIZE = 32;
```

### Performance Thresholds

```zig
pub const PERFORMANCE_THRESHOLDS = struct {
    pub const MAX_SEARCH_TIME_NS = 20_000_000;
    pub const MAX_MEMORY_USAGE_MB = 1024;
    pub const MIN_SEARCH_QPS = 1000;
    pub const MAX_REGRESSION_PERCENT = 15.0;
};
```

## Examples

### Complete Database Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize database
    var db = try abi.Db.init(allocator, .{
        .dimensions = 128,
        .max_elements = 10000,
    });
    defer db.deinit();

    // Create sample vectors
    var vectors: [100][128]f32 = undefined;
    // ... populate vectors ...

    // Insert vectors
    for (vectors, 0..) |vec, i| {
        const id = try std.fmt.allocPrint(allocator, "vec_{}", .{i});
        defer allocator.free(id);
        try db.insert(&vec, id);
    }

    // Search
    const query = [_]f32{0.1} ** 128;
    const results = try db.search(&query, 10);

    // Process results
    for (results.items) |result| {
        std.log.info("ID: {s}, Distance: {d}", .{
            result.id,
            result.distance,
        });
    }
}
```

### GPU-Accelerated Computation

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize GPU renderer
    var renderer = try abi.GPURenderer.init(allocator, .{
        .debug_validation = false,
        .power_preference = .high_performance,
    });
    defer renderer.deinit();

    // Create matrices
    const a = try allocator.alloc(f32, 1024 * 1024);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 1024 * 1024);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, 1024 * 1024);
    defer allocator.free(result);

    // GPU matrix multiplication
    try renderer.matrixMultiply(
        result,
        a, b,
        1024, 1024, 1024  // M, K, N
    );

    std.log.info("GPU matrix multiplication completed", .{});
}
```

---

*For more detailed examples and tutorials, see the [examples](/examples/) section.*
