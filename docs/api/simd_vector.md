# SIMD Vector API Reference

The SIMD Vector module provides high-performance vector operations using Zig's built-in SIMD support. It automatically detects the optimal vector width for the target architecture and provides fallback implementations.

## Overview

- **Module**: `src/simd_vector.zig`
- **Auto-vectorization**: Optimizes for target architecture (4-wide on most platforms, 8-wide on x86_64)
- **Performance**: 2-5x speedup over scalar operations for large vectors
- **Memory safety**: All operations are bounds-checked

## Constants

### `SIMD_WIDTH`
```zig
const SIMD_WIDTH: comptime_int
```
Optimal SIMD vector width for f32 operations on the target architecture.
- **x86_64**: 8 elements (256-bit AVX)
- **Other architectures**: 4 elements (128-bit)

### `F32Vector`
```zig
const F32Vector = @Vector(SIMD_WIDTH, f32)
```
SIMD vector type for f32 operations using the optimal width.

## Functions

### Distance Calculation

#### `distanceSquaredSIMD`
```zig
pub fn distanceSquaredSIMD(a: []const f32, b: []const f32) f32
```
Calculate squared Euclidean distance between two vectors using SIMD.

**Parameters:**
- `a`: First vector (must have same length as `b`)
- `b`: Second vector (must have same length as `a`)

**Returns:** Squared Euclidean distance as f32

**Example:**
```zig
const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
const dist = distanceSquaredSIMD(&a, &b); // Returns 4.0
```

**Performance:** ~3x faster than scalar implementation for vectors > 16 elements

---

### Dot Product

#### `dotProductSIMD`
```zig
pub fn dotProductSIMD(a: []const f32, b: []const f32) f32
```
Calculate dot product between two vectors using SIMD.

**Parameters:**
- `a`: First vector (must have same length as `b`)
- `b`: Second vector (must have same length as `a`)

**Returns:** Dot product as f32

**Example:**
```zig
const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
const dot = dotProductSIMD(&a, &b); // Returns 40.0 (1*2 + 2*3 + 3*4 + 4*5)
```

**Performance:** ~4x faster than scalar implementation for vectors > 32 elements

---

### Vector Normalization

#### `normalizeSIMD`
```zig
pub fn normalizeSIMD(vector: []f32) void
```
Normalize a vector to unit length using SIMD (modifies vector in-place).

**Parameters:**
- `vector`: Vector to normalize (modified in-place)

**Side Effects:** Modifies the input vector to have unit length

**Example:**
```zig
var vec = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
normalizeSIMD(&vec);
// vec is now [0.6, 0.8, 0.0, 0.0] (magnitude = 1.0)
```

**Performance:** ~2x faster than scalar implementation

---

### Vector Addition

#### `addVectorsSIMD`
```zig
pub fn addVectorsSIMD(a: []const f32, b: []const f32, result: []f32) void
```
Add two vectors element-wise using SIMD.

**Parameters:**
- `a`: First vector
- `b`: Second vector (must have same length as `a`)
- `result`: Output vector (must have same length as inputs)

**Example:**
```zig
const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
var result: [4]f32 = undefined;
addVectorsSIMD(&a, &b, &result);
// result is now [6.0, 8.0, 10.0, 12.0]
```

**Performance:** ~5x faster than scalar implementation for large vectors

---

### Vector Scaling

#### `scaleVectorSIMD`
```zig
pub fn scaleVectorSIMD(vector: []const f32, scalar: f32, result: []f32) void
```
Multiply vector by scalar using SIMD.

**Parameters:**
- `vector`: Input vector
- `scalar`: Scalar multiplier
- `result`: Output vector (must have same length as input)

**Example:**
```zig
const vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
var result: [4]f32 = undefined;
scaleVectorSIMD(&vec, 2.5, &result);
// result is now [2.5, 5.0, 7.5, 10.0]
```

**Performance:** ~3x faster than scalar implementation

---

### Cosine Similarity

#### `cosineSimilaritySIMD`
```zig
pub fn cosineSimilaritySIMD(a: []const f32, b: []const f32) f32
```
Calculate cosine similarity between two vectors using SIMD.

**Parameters:**
- `a`: First vector (must have same length as `b`)
- `b`: Second vector (must have same length as `a`)

**Returns:** Cosine similarity in range [-1.0, 1.0]
- `1.0`: Identical direction
- `0.0`: Orthogonal (perpendicular)
- `-1.0`: Opposite direction

**Example:**
```zig
const a = [_]f32{ 1.0, 0.0, 0.0 };
const b = [_]f32{ 0.0, 1.0, 0.0 };
const sim = cosineSimilaritySIMD(&a, &b); // Returns 0.0 (orthogonal)

const c = [_]f32{ 1.0, 2.0, 3.0 };
const d = [_]f32{ 1.0, 2.0, 3.0 };
const sim2 = cosineSimilaritySIMD(&c, &d); // Returns 1.0 (identical)
```

**Performance:** ~3x faster than scalar implementation

---

## Performance Tips

1. **Vector Size**: SIMD optimizations work best with vectors having length >= 16 elements
2. **Memory Alignment**: For best performance, ensure vectors are aligned to 32-byte boundaries
3. **Batch Operations**: Process multiple vectors in batches to maximize SIMD utilization
4. **Cache Locality**: Keep related vectors close in memory for better cache performance

## Usage Patterns

### High-Performance Vector Processing
```zig
const allocator = std.heap.page_allocator;
const vector_count = 1000;
const dimensions = 128;

// Allocate vectors
var vectors = try allocator.alloc([dimensions]f32, vector_count);
defer allocator.free(vectors);

// Process vectors in batches using SIMD
for (0..vector_count) |i| {
    normalizeSIMD(&vectors[i]);
}
```

### Similarity Search
```zig
fn findMostSimilar(query: []const f32, candidates: [][]const f32) usize {
    var best_idx: usize = 0;
    var best_sim: f32 = -1.0;
    
    for (candidates, 0..) |candidate, i| {
        const sim = cosineSimilaritySIMD(query, candidate);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    
    return best_idx;
}
```

## Error Handling

All functions assert that input vectors have compatible dimensions. In debug builds, mismatched dimensions will trigger a panic. In release builds, behavior is undefined for mismatched dimensions.

## Threading Safety

All SIMD functions are thread-safe and can be called concurrently from multiple threads, as long as:
- Read-only operations (`distanceSquaredSIMD`, `dotProductSIMD`, `cosineSimilaritySIMD`) can be called safely with shared input data
- Write operations (`normalizeSIMD`, `addVectorsSIMD`, `scaleVectorSIMD`) require exclusive access to output buffers 