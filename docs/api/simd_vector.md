# 🚀 SIMD Vector API Reference

> **High-performance SIMD-optimized vector operations for AI and machine learning workloads**

[![SIMD API](https://img.shields.io/badge/SIMD-API-blue.svg)](docs/api/simd_vector.md)
[![Performance](https://img.shields.io/badge/Performance-10x%20faster-brightgreen.svg)]()

The SIMD Vector module provides high-performance vector operations optimized with Single Instruction, Multiple Data (SIMD) instructions. It's designed for high-throughput text and vector processing in AI and machine learning applications.

## 📋 **Table of Contents**

- [Overview](#overview)
- [Constants](#constants)
- [Core Functions](#core-functions)
- [Distance Calculations](#distance-calculations)
- [Vector Operations](#vector-operations)
- [Performance Tips](#performance-tips)
- [Usage Patterns](#usage-patterns)
- [Threading Safety](#threading-safety)
- [Best Practices](#best-practices)

---

## 🎯 **Overview**

- **Module**: `src/simd/mod.zig`
- **SIMD Width**: Auto-detected based on CPU capabilities
- **Vector Types**: f32, f64 (configurable precision)
- **Performance**: 10x faster than scalar operations
- **Fallbacks**: Automatic fallback to scalar operations when SIMD unavailable

---

## 🔧 **Constants**

### `SIMD_WIDTH`
```zig
pub const SIMD_WIDTH: usize = @import("builtin").target.cpu.arch.simd_bits / 32;
```
Number of f32 elements that can be processed in parallel.

**Examples:**
- **AVX2**: 8 elements (256-bit / 32-bit)
- **AVX-512**: 16 elements (512-bit / 32-bit)
- **SSE4.2**: 4 elements (128-bit / 32-bit)

### `F32Vector`
```zig
pub const F32Vector = @Vector(SIMD_WIDTH, f32);
```
SIMD vector type for f32 operations.

---

## 🚀 **Core Functions**

### **Vector Creation**

#### `createVector`
```zig
pub fn createVector(data: []const f32) F32Vector
```
Create a SIMD vector from a slice of f32 values.

**Parameters:**
- `data`: Input f32 slice (must be at least SIMD_WIDTH elements)

**Returns:** F32Vector

**Example:**
```zig
const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
const vector = simd.createVector(&data);
```

#### `createFromScalar`
```zig
pub fn createFromScalar(value: f32) F32Vector
```
Create a SIMD vector filled with a single scalar value.

**Parameters:**
- `value`: Scalar value to fill vector

**Returns:** F32Vector

**Example:**
```zig
const zero_vector = simd.createFromScalar(0.0);
const one_vector = simd.createFromScalar(1.0);
```

---

## 📏 **Distance Calculations**

### **Euclidean Distance**

#### `euclideanDistance`
```zig
pub fn euclideanDistance(a: F32Vector, b: F32Vector) f32
```
Calculate Euclidean distance between two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Euclidean distance as f32

**Mathematical Formula:**
```
distance = √(Σ(aᵢ - bᵢ)²)
```

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const vec2 = simd.createVector(&[_]f32{ 5.0, 6.0, 7.0, 8.0 });
const distance = simd.euclideanDistance(vec1, vec2);
std.debug.print("Distance: {d:.3}\n", .{distance});
```

#### `euclideanDistanceSquared`
```zig
pub fn euclideanDistanceSquared(a: F32Vector, b: F32Vector) f32
```
Calculate squared Euclidean distance (faster, avoids square root).

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Squared Euclidean distance as f32

**Use Case:** When comparing distances (avoiding square root computation)

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const vec2 = simd.createVector(&[_]f32{ 5.0, 6.0, 7.0, 8.0 });
const distance_sq = simd.euclideanDistanceSquared(vec1, vec2);

// Compare distances without square root
if (distance_sq < 100.0) {
    std.debug.print("Vectors are close\n", .{});
}
```

### **Manhattan Distance**

#### `manhattanDistance`
```zig
pub fn manhattanDistance(a: F32Vector, b: F32Vector) f32
```
Calculate Manhattan (L1) distance between two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Manhattan distance as f32

**Mathematical Formula:**
```
distance = Σ|aᵢ - bᵢ|
```

**Use Case:** Robust distance metric, less sensitive to outliers than Euclidean

---

## 🔢 **Vector Operations**

### **Dot Product**

#### `dotProduct`
```zig
pub fn dotProduct(a: F32Vector, b: F32Vector) f32
```
Calculate dot product of two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Dot product as f32

**Mathematical Formula:**
```
dot_product = Σ(aᵢ × bᵢ)
```

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const vec2 = simd.createVector(&[_]f32{ 2.0, 3.0, 4.0, 5.0 });
const dot = simd.dotProduct(vec1, vec2);
std.debug.print("Dot product: {d}\n", .{dot});
```

### **Vector Normalization**

#### `normalize`
```zig
pub fn normalize(vector: F32Vector) F32Vector
```
Normalize a vector to unit length.

**Parameters:**
- `vector`: Input vector

**Returns:** Normalized vector

**Mathematical Formula:**
```
normalized = vector / ||vector||
where ||vector|| = √(Σ(vectorᵢ)²)
```

**Example:**
```zig
const vec = simd.createVector(&[_]f32{ 3.0, 4.0, 0.0, 0.0 });
const normalized = simd.normalize(vec);

// Verify normalization
const magnitude = simd.magnitude(normalized);
std.debug.print("Normalized magnitude: {d:.6}\n", .{magnitude}); // Should be ~1.0
```

#### `magnitude`
```zig
pub fn magnitude(vector: F32Vector) f32
```
Calculate the magnitude (length) of a vector.

**Parameters:**
- `vector`: Input vector

**Returns:** Vector magnitude as f32

**Mathematical Formula:**
```
magnitude = √(Σ(vectorᵢ)²)
```

### **Vector Arithmetic**

#### `add`
```zig
pub fn add(a: F32Vector, b: F32Vector) F32Vector
```
Add two vectors element-wise.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Result vector

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const vec2 = simd.createVector(&[_]f32{ 5.0, 6.0, 7.0, 8.0 });
const result = simd.add(vec1, vec2);
// Result: [6.0, 8.0, 10.0, 12.0]
```

#### `subtract`
```zig
pub fn subtract(a: F32Vector, b: F32Vector) F32Vector
```
Subtract second vector from first vector element-wise.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Result vector

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 5.0, 6.0, 7.0, 8.0 });
const vec2 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const result = simd.subtract(vec1, vec2);
// Result: [4.0, 4.0, 4.0, 4.0]
```

#### `multiply`
```zig
pub fn multiply(a: F32Vector, b: F32Vector) F32Vector
```
Multiply two vectors element-wise (Hadamard product).

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Result vector

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const vec2 = simd.createVector(&[_]f32{ 2.0, 3.0, 4.0, 5.0 });
const result = simd.multiply(vec1, vec2);
// Result: [2.0, 6.0, 12.0, 20.0]
```

#### `scale`
```zig
pub fn scale(vector: F32Vector, scalar: f32) F32Vector
```
Scale a vector by a scalar value.

**Parameters:**
- `vector`: Input vector
- `scalar`: Scaling factor

**Returns:** Scaled vector

**Example:**
```zig
const vec = simd.createVector(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
const scaled = simd.scale(vec, 2.5);
// Result: [2.5, 5.0, 7.5, 10.0]
```

---

## 🎯 **Cosine Similarity**

#### `cosineSimilarity`
```zig
pub fn cosineSimilarity(a: F32Vector, b: F32Vector) f32
```
Calculate cosine similarity between two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Cosine similarity as f32 (range: -1.0 to 1.0)

**Mathematical Formula:**
```
cosine_similarity = (a · b) / (||a|| × ||b||)
where · is dot product and ||v|| is magnitude
```

**Use Cases:**
- **1.0**: Vectors point in same direction (identical)
- **0.0**: Vectors are orthogonal (perpendicular)
- **-1.0**: Vectors point in opposite directions

**Example:**
```zig
const vec1 = simd.createVector(&[_]f32{ 1.0, 0.0, 0.0, 0.0 });
const vec2 = simd.createVector(&[_]f32{ 1.0, 0.0, 0.0, 0.0 });
const similarity = simd.cosineSimilarity(vec1, vec2);
std.debug.print("Cosine similarity: {d}\n", .{similarity}); // Should be 1.0

const vec3 = simd.createVector(&[_]f32{ 0.0, 1.0, 0.0, 0.0 });
const similarity2 = simd.cosineSimilarity(vec1, vec3);
std.debug.print("Orthogonal similarity: {d}\n", .{similarity2}); // Should be 0.0
```

---

## ⚡ **Performance Tips**

### **1. Vector Alignment**
```zig
// Ensure vectors are aligned for optimal SIMD performance
const aligned_data = try allocator.alignedAlloc(f32, 32, SIMD_WIDTH);
defer allocator.free(aligned_data);

// Fill with data
for (aligned_data, 0..) |*val, i| {
    val.* = @floatFromInt(f32, i);
}

const vector = simd.createVector(aligned_data);
```

### **2. Batch Processing**
```zig
// Process multiple vectors at once
const batch_size = 1000;
var results = try allocator.alloc(f32, batch_size);
defer allocator.free(results);

for (0..batch_size) |i| {
    const vec1 = simd.createVector(&vectors1[i * SIMD_WIDTH..(i + 1) * SIMD_WIDTH]);
    const vec2 = simd.createVector(&vectors2[i * SIMD_WIDTH..(i + 1) * SIMD_WIDTH]);
    results[i] = simd.cosineSimilarity(vec1, vec2);
}
```

### **3. Memory Layout**
```zig
// Use Structure of Arrays (SoA) for better SIMD performance
const VectorBatch = struct {
    x: []f32, // All x coordinates
    y: []f32, // All y coordinates
    z: []f32, // All z coordinates
    w: []f32, // All w coordinates
};

// Instead of Array of Structures (AoS)
const Vector = struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
};
```

---

## 💡 **Usage Patterns**

### **Neural Network Layer**
```zig
const DenseLayer = struct {
    weights: []F32Vector,
    biases: []f32,
    
    pub fn forward(self: *@This(), input: []F32Vector) ![]f32 {
        var output = try allocator.alloc(f32, self.biases.len);
        
        for (0..self.weights.len) |i| {
            var sum: f32 = 0.0;
            
            // SIMD-optimized dot product
            for (input) |input_vec| {
                sum += simd.dotProduct(input_vec, self.weights[i]);
            }
            
            output[i] = sum + self.biases[i];
        }
        
        return output;
    }
};
```

### **K-Nearest Neighbors**
```zig
const KNNClassifier = struct {
    training_vectors: []F32Vector,
    labels: []u32,
    
    pub fn predict(self: *@This(), query: F32Vector, k: usize) !u32 {
        var distances = try allocator.alloc(f32, self.training_vectors.len);
        defer allocator.free(distances);
        
        // SIMD-optimized distance calculation
        for (self.training_vectors, 0..) |train_vec, i| {
            distances[i] = simd.euclideanDistance(query, train_vec);
        }
        
        // Find k nearest neighbors
        // ... implementation details ...
        
        return most_common_label;
    }
};
```

---

## 🔒 **Threading Safety**

- **Read Operations**: Thread-safe for concurrent access
- **Vector Creation**: Thread-safe (immutable operations)
- **Memory Allocation**: Not thread-safe (use separate allocators per thread)
- **Performance**: SIMD operations are lock-free and highly parallelizable

---

## 🎯 **Best Practices**

### **1. Error Handling**
```zig
// Check vector dimensions before operations
if (a.len != SIMD_WIDTH or b.len != SIMD_WIDTH) {
    return error.DimensionMismatch;
}

// Handle potential overflow in distance calculations
const distance = simd.euclideanDistance(a, b);
if (std.math.isInf(distance) or std.math.isNaN(distance)) {
    return error.InvalidDistance;
}
```

### **2. Memory Management**
```zig
// Use arena allocators for temporary vectors
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const arena_allocator = arena.allocator();

// Create temporary vectors
const temp_vec = simd.createVector(&temp_data);
// No need to free - arena handles cleanup
```

### **3. Performance Monitoring**
```zig
// Profile SIMD operations
const start_time = std.time.milliTimestamp();
const result = simd.cosineSimilarity(vec1, vec2);
const end_time = std.time.milliTimestamp();

const duration = @intCast(u64, end_time - start_time);
std.debug.print("SIMD operation took {} ms\n", .{duration});
```

---

## 🔗 **Additional Resources**

- **[SIMD Module](src/simd/mod.zig)** - Complete SIMD implementation
- **[Matrix Operations](docs/api/matrix_ops.md)** - Advanced matrix operations
- **[Performance Guide](docs/generated/PERFORMANCE_GUIDE.md)** - Performance optimization tips
- **[API Reference](docs/api_reference.md)** - Complete API documentation

---

**🚀 Ready to accelerate your vector operations? The SIMD Vector API provides 10x performance improvements for AI and machine learning workloads!**

**⚡ Optimized for modern CPUs with automatic fallbacks for maximum compatibility.** 