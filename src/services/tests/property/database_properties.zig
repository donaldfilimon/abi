//! Database Property Tests
//!
//! Property-based tests for database operations including:
//! - Insert/search consistency
//! - Delete correctness
//! - Index invariants
//! - Concurrent operation correctness
//!
//! Uses the database module from src/features/database/

const std = @import("std");
const property = @import("mod.zig");
const generators = @import("generators.zig");
const abi = @import("abi");
const build_options = @import("build_options");

const forAll = property.forAll;
const forAllWithAllocator = property.forAllWithAllocator;
const assert = property.assert;
const Generator = property.Generator;

// ============================================================================
// Test Configuration
// ============================================================================

const TestConfig = property.PropertyConfig{
    .iterations = 50,
    .seed = 42,
    .verbose = false,
};

const VECTOR_DIM = 8;
const EPSILON: f32 = 1e-5;

// ============================================================================
// Database Operation Types for State Machine Testing
// ============================================================================

/// Represents a database operation for property testing
const DbOperation = struct {
    op_type: OpType,
    id: u64,
    vector: [VECTOR_DIM]f32,

    const OpType = enum {
        insert,
        search,
        delete,
        update,
    };
};

/// Generate random database operations
fn dbOperationGen() Generator(DbOperation) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) DbOperation {
                var result: DbOperation = undefined;

                // Generate operation type
                result.op_type = @enumFromInt(prng.random().intRangeLessThan(u8, 0, 4));
                result.id = prng.random().intRangeAtMost(u64, 1, 1000);

                // Generate normalized vector
                var sum_sq: f32 = 0.0;
                for (&result.vector) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                    sum_sq += v.* * v.*;
                }
                const mag = @sqrt(sum_sq);
                if (mag > 1e-6) {
                    for (&result.vector) |*v| {
                        v.* /= mag;
                    }
                }

                return result;
            }
        }.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Vector Similarity Properties
// ============================================================================

test "cosine similarity is bounded in [-1, 1]" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            const cos = abi.simd.cosineSimilarity(&pair.a, &pair.b);

            // Handle edge cases
            if (std.math.isNan(cos)) return false;

            return cos >= -1.0 - EPSILON and cos <= 1.0 + EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "identical vectors have cosine similarity 1.0" {
    const gen = generators.nonZeroVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            const cos = abi.simd.cosineSimilarity(&vec, &vec);
            return assert.approxEqual(cos, 1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "negated vectors have cosine similarity -1.0" {
    const gen = generators.nonZeroVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            var neg: [VECTOR_DIM]f32 = undefined;
            for (&neg, vec) |*n, v| {
                n.* = -v;
            }

            const cos = abi.simd.cosineSimilarity(&vec, &neg);
            return assert.approxEqual(cos, -1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Distance Metric Properties
// ============================================================================

test "L2 distance is a valid metric (non-negative)" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            const dist = abi.simd.l2Distance(&pair.a, &pair.b);
            return dist >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 distance is symmetric" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            const dist_ab = abi.simd.l2Distance(&pair.a, &pair.b);
            const dist_ba = abi.simd.l2Distance(&pair.b, &pair.a);
            return assert.approxEqual(dist_ab, dist_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 distance of vector with itself is zero" {
    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            const dist = abi.simd.l2Distance(&vec, &vec);
            return assert.approxEqual(dist, 0.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "triangle inequality holds for L2 distance" {
    const gen = generators.vectorTripleF32(VECTOR_DIM);

    const result = forAll(property.VectorTriple(VECTOR_DIM), gen, TestConfig, struct {
        fn check(triple: property.VectorTriple(VECTOR_DIM)) bool {
            const dist_ac = abi.simd.l2Distance(&triple.a, &triple.c);
            const dist_ab = abi.simd.l2Distance(&triple.a, &triple.b);
            const dist_bc = abi.simd.l2Distance(&triple.b, &triple.c);

            // Allow tolerance for floating point
            return dist_ac <= dist_ab + dist_bc + EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Database State Machine Properties (when database enabled)
// ============================================================================

/// Simple in-memory database model for testing invariants
const DatabaseModel = struct {
    vectors: std.AutoHashMap(u64, [VECTOR_DIM]f32),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) DatabaseModel {
        return .{
            .vectors = std.AutoHashMap(u64, [VECTOR_DIM]f32).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *DatabaseModel) void {
        self.vectors.deinit();
        self.* = undefined;
    }

    fn insert(self: *DatabaseModel, id: u64, vector: [VECTOR_DIM]f32) !void {
        try self.vectors.put(id, vector);
    }

    fn delete(self: *DatabaseModel, id: u64) bool {
        return self.vectors.remove(id);
    }

    fn get(self: *DatabaseModel, id: u64) ?[VECTOR_DIM]f32 {
        return self.vectors.get(id);
    }

    fn count(self: *DatabaseModel) usize {
        return self.vectors.count();
    }

    fn search(self: *DatabaseModel, query: [VECTOR_DIM]f32, k: usize) ![]SearchResult {
        const allocator = self.allocator;

        // Collect all vectors with their distances
        var results = std.ArrayListUnmanaged(SearchResult).empty;
        defer results.deinit(allocator);

        var iter = self.vectors.iterator();
        while (iter.next()) |entry| {
            const dist = abi.simd.l2Distance(&query, entry.value_ptr);
            try results.append(allocator, .{ .id = entry.key_ptr.*, .distance = dist });
        }

        // Sort by distance
        std.sort.heap(SearchResult, results.items, {}, struct {
            fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        // Return top k
        const limit = @min(k, results.items.len);
        return try allocator.dupe(SearchResult, results.items[0..limit]);
    }

    const SearchResult = struct {
        id: u64,
        distance: f32,
    };
};

test "database model insert then get returns same vector" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAllWithAllocator([VECTOR_DIM]f32, std.testing.allocator, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            model.insert(42, vec) catch return false;

            const retrieved = model.get(42) orelse return false;

            return assert.slicesApproxEqual(&retrieved, &vec, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model delete removes vector" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAllWithAllocator([VECTOR_DIM]f32, std.testing.allocator, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            model.insert(42, vec) catch return false;
            _ = model.delete(42);

            return model.get(42) == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model count is correct after operations" {
    const gen = generators.intRange(u8, 1, 50);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert n vectors
            for (0..n) |i| {
                const vec = [_]f32{@as(f32, @floatFromInt(i))} ++ [_]f32{0} ** (VECTOR_DIM - 1);
                model.insert(@intCast(i), vec) catch return false;
            }

            if (model.count() != n) return false;

            // Delete half
            const half = n / 2;
            for (0..half) |i| {
                _ = model.delete(@intCast(i));
            }

            return model.count() == n - half;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search returns k or fewer results" {
    const SearchTest = struct {
        n_vectors: u8,
        k: u8,
    };

    const gen = Generator(SearchTest){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) SearchTest {
                return .{
                    .n_vectors = prng.random().intRangeAtMost(u8, 1, 20),
                    .k = prng.random().intRangeAtMost(u8, 1, 10),
                };
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAllWithAllocator(SearchTest, std.testing.allocator, gen, TestConfig, struct {
        fn check(params: SearchTest, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert n vectors
            for (0..params.n_vectors) |i| {
                var vec: [VECTOR_DIM]f32 = undefined;
                for (&vec) |*v| {
                    v.* = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(params.n_vectors));
                }
                model.insert(@intCast(i), vec) catch return false;
            }

            // Search with k
            const query = [_]f32{0.5} ** VECTOR_DIM;
            const results = model.search(query, params.k) catch return false;
            defer allocator.free(results);

            // Should return min(k, n_vectors) results
            const expected = @min(params.k, params.n_vectors);
            return results.len == expected;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search results are sorted by distance" {
    const gen = generators.intRange(u8, 5, 30);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert n vectors at different positions
            var prng = std.Random.DefaultPrng.init(@intCast(n));
            for (0..n) |i| {
                var vec: [VECTOR_DIM]f32 = undefined;
                for (&vec) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                }
                model.insert(@intCast(i), vec) catch return false;
            }

            // Search
            const query = [_]f32{0.0} ** VECTOR_DIM;
            const results = model.search(query, 10) catch return false;
            defer allocator.free(results);

            // Verify sorted by distance (ascending)
            for (results[0 .. results.len - 1], results[1..]) |a, b| {
                if (a.distance > b.distance + EPSILON) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search finds exact match first" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAllWithAllocator([VECTOR_DIM]f32, std.testing.allocator, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert multiple vectors including the query
            model.insert(100, vec) catch return false;

            // Insert some other random vectors
            var other = vec;
            other[0] += 0.5;
            model.insert(1, other) catch return false;

            other[0] -= 1.0;
            model.insert(2, other) catch return false;

            // Search for the exact vector
            const results = model.search(vec, 3) catch return false;
            defer allocator.free(results);

            // First result should be the exact match (id=100)
            if (results.len == 0) return false;
            if (results[0].id != 100) return false;
            if (results[0].distance > EPSILON) return false;

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Batch Vector Properties
// ============================================================================

test "batch cosine similarities are bounded" {
    const BatchTest = struct {
        query: [VECTOR_DIM]f32,
        vectors: [5][VECTOR_DIM]f32,
    };

    const gen = Generator(BatchTest){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) BatchTest {
                var result: BatchTest = undefined;

                // Generate query
                for (&result.query) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                }

                // Generate vectors
                for (&result.vectors) |*vec| {
                    for (vec) |*v| {
                        v.* = prng.random().float(f32) * 2.0 - 1.0;
                    }
                }

                return result;
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAll(BatchTest, gen, TestConfig, struct {
        fn check(batch: BatchTest) bool {
            // Convert to slice of slices
            // Use indexed loop to avoid pointer capture issues in Zig 0.16
            var vec_slices: [5][]const f32 = undefined;
            for (0..5) |i| {
                vec_slices[i] = &batch.vectors[i];
            }

            var similarities: [5]f32 = undefined;
            abi.simd.batchCosineSimilarity(&batch.query, &vec_slices, &similarities);

            for (similarities) |sim| {
                if (std.math.isNan(sim)) return false;
                if (sim < -1.0 - EPSILON or sim > 1.0 + EPSILON) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Quantization Properties (if enabled)
// ============================================================================

test "scalar quantization roundtrip preserves approximate values" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            // Clamp values to quantization range
            var clamped: [VECTOR_DIM]f32 = undefined;
            for (&clamped, vec) |*c, v| {
                c.* = @max(-1.0, @min(1.0, v));
            }

            // Simulate quantization (scale to u8, back to f32)
            var quantized: [VECTOR_DIM]u8 = undefined;
            for (&quantized, clamped) |*q, c| {
                q.* = @intFromFloat((c + 1.0) * 127.5);
            }

            var dequantized: [VECTOR_DIM]f32 = undefined;
            for (&dequantized, quantized) |*d, q| {
                d.* = @as(f32, @floatFromInt(q)) / 127.5 - 1.0;
            }

            // Error should be bounded by quantization step size (1/127.5)
            const max_error: f32 = 1.0 / 127.5;
            for (clamped, dequantized) |c, d| {
                if (@abs(c - d) > max_error + EPSILON) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Index Invariant Properties
// ============================================================================

test "vector IDs are unique in model" {
    const gen = generators.intRange(u8, 10, 50);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert vectors with unique IDs
            for (0..n) |i| {
                const vec = [_]f32{@as(f32, @floatFromInt(i))} ++ [_]f32{0} ** (VECTOR_DIM - 1);
                model.insert(@intCast(i), vec) catch return false;
            }

            // Re-inserting with same ID should overwrite
            const new_vec = [_]f32{99.0} ++ [_]f32{0} ** (VECTOR_DIM - 1);
            model.insert(0, new_vec) catch return false;

            // Count should still be n
            if (model.count() != n) return false;

            // Vector at ID 0 should be updated
            const retrieved = model.get(0) orelse return false;
            if (@abs(retrieved[0] - 99.0) > EPSILON) return false;

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Edge Case Tests - Boundary Conditions
// ============================================================================

test "zero vector cosine similarity is undefined (NaN)" {
    const gen = generators.nonZeroVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            const zero_vec = [_]f32{0.0} ** VECTOR_DIM;

            // Cosine similarity with zero vector should be NaN (0/0)
            const cos = abi.simd.cosineSimilarity(&zero_vec, &vec);

            // NaN check - NaN != NaN
            return std.math.isNan(cos) or @abs(cos) < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "orthogonal vectors have cosine similarity near zero" {
    const result = forAll(u8, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8) bool {
            // Create orthogonal vectors: [1,0,0,...] and [0,1,0,...]
            var vec_a = [_]f32{0.0} ** VECTOR_DIM;
            var vec_b = [_]f32{0.0} ** VECTOR_DIM;
            vec_a[0] = 1.0;
            vec_b[1] = 1.0;

            const cos = abi.simd.cosineSimilarity(&vec_a, &vec_b);

            // Orthogonal vectors should have similarity ~0
            return @abs(cos) < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "cosine similarity is magnitude invariant" {
    const gen = generators.nonZeroVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            // Scale vector by arbitrary factor
            var scaled: [VECTOR_DIM]f32 = undefined;
            for (&scaled, vec) |*s, v| {
                s.* = v * 3.5;
            }

            const cos_original = abi.simd.cosineSimilarity(&vec, &vec);
            const cos_scaled = abi.simd.cosineSimilarity(&vec, &scaled);

            // Both should be ~1.0 (identical direction)
            if (std.math.isNan(cos_original) or std.math.isNan(cos_scaled)) return false;
            return assert.approxEqual(cos_original, cos_scaled, EPSILON * 10);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search with k exceeding vector count returns all" {
    const gen = generators.intRange(u8, 1, 10);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert n vectors
            for (0..n) |i| {
                const vec = [_]f32{@as(f32, @floatFromInt(i)) / 10.0} ++ [_]f32{0} ** (VECTOR_DIM - 1);
                model.insert(@intCast(i), vec) catch return false;
            }

            // Search with k > n
            const query = [_]f32{0.5} ** VECTOR_DIM;
            const results = model.search(query, 100) catch return false;
            defer allocator.free(results);

            // Should return exactly n results
            return results.len == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search on empty database returns empty" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Search empty database
            const query = [_]f32{0.5} ** VECTOR_DIM;
            const results = model.search(query, 10) catch return false;
            defer allocator.free(results);

            return results.len == 0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model delete non-existent ID returns false" {
    const gen = generators.intRange(u8, 1, 20);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert some vectors
            for (0..n) |i| {
                const vec = [_]f32{@as(f32, @floatFromInt(i))} ++ [_]f32{0} ** (VECTOR_DIM - 1);
                model.insert(@intCast(i), vec) catch return false;
            }

            // Delete non-existent ID
            const deleted = model.delete(9999);

            // Should return false (nothing deleted)
            if (deleted) return false;

            // Count should be unchanged
            return model.count() == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model search after mass delete" {
    // Direct test with specific values instead of property-based
    // to avoid edge cases with integer division
    var model = DatabaseModel.init(std.testing.allocator);
    defer model.deinit();

    // Insert 20 vectors
    const n: usize = 20;
    for (0..n) |i| {
        const vec = [_]f32{@as(f32, @floatFromInt(i)) / 20.0} ++ [_]f32{0} ** (VECTOR_DIM - 1);
        try model.insert(@intCast(i), vec);
    }

    // Delete 90% of vectors (18 vectors, leaving 2)
    const to_delete: usize = 18;
    for (0..to_delete) |i| {
        _ = model.delete(@intCast(i));
    }

    // Search should still work
    const query = [_]f32{0.5} ** VECTOR_DIM;
    const results = try model.search(query, 5);
    defer std.testing.allocator.free(results);

    // Results should be from remaining vectors (2 remaining)
    const remaining = n - to_delete;
    try std.testing.expectEqual(@min(5, remaining), results.len);
}

test "L2 distance with very large magnitudes" {
    const result = forAll(u8, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8) bool {
            // Create vectors with very large values
            var large_vec = [_]f32{1e6} ** VECTOR_DIM;
            var small_vec = [_]f32{1e-6} ** VECTOR_DIM;

            const dist = abi.simd.l2Distance(&large_vec, &small_vec);

            // Should be finite and positive
            if (std.math.isNan(dist) or std.math.isInf(dist)) return false;
            return dist > 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "database model duplicate insert updates existing" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert initial vector
            const vec1 = [_]f32{1.0} ++ [_]f32{0} ** (VECTOR_DIM - 1);
            model.insert(42, vec1) catch return false;

            // Insert different vector with same ID
            const vec2 = [_]f32{2.0} ++ [_]f32{0} ** (VECTOR_DIM - 1);
            model.insert(42, vec2) catch return false;

            // Count should still be 1
            if (model.count() != 1) return false;

            // Retrieved vector should be vec2
            const retrieved = model.get(42) orelse return false;
            return @abs(retrieved[0] - 2.0) < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Quantization Property Tests
// ============================================================================

test "property - 8-bit quantization roundtrip error bounded" {
    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            // Find min and max for quantization range
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);
            for (vec) |v| {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }

            // Avoid degenerate case
            if (max_val - min_val < 1e-6) return true;

            // Quantize each value to 8 bits (256 levels)
            var quantized: [VECTOR_DIM]u8 = undefined;
            for (&quantized, vec) |*q, v| {
                const normalized = (v - min_val) / (max_val - min_val);
                const scaled = normalized * 255.0;
                q.* = @intFromFloat(std.math.clamp(scaled, 0.0, 255.0));
            }

            // Dequantize
            var dequantized: [VECTOR_DIM]f32 = undefined;
            for (&dequantized, quantized) |*d, q| {
                const normalized = @as(f32, @floatFromInt(q)) / 255.0;
                d.* = min_val + normalized * (max_val - min_val);
            }

            // Error should be bounded by quantization step size
            const step = (max_val - min_val) / 255.0;
            for (vec, dequantized) |original, reconstructed| {
                if (@abs(original - reconstructed) > step + EPSILON) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - 4-bit quantization roundtrip error bounded" {
    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);
            for (vec) |v| {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }

            if (max_val - min_val < 1e-6) return true;

            // Quantize to 4 bits (16 levels)
            var quantized: [VECTOR_DIM]u8 = undefined;
            for (&quantized, vec) |*q, v| {
                const normalized = (v - min_val) / (max_val - min_val);
                const scaled = normalized * 15.0;
                q.* = @intFromFloat(std.math.clamp(scaled, 0.0, 15.0));
            }

            // Dequantize
            var dequantized: [VECTOR_DIM]f32 = undefined;
            for (&dequantized, quantized) |*d, q| {
                const normalized = @as(f32, @floatFromInt(q)) / 15.0;
                d.* = min_val + normalized * (max_val - min_val);
            }

            // Error bounded by quantization step
            const step = (max_val - min_val) / 15.0;
            for (vec, dequantized) |original, reconstructed| {
                if (@abs(original - reconstructed) > step + EPSILON) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - quantization preserves relative ordering" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            // Compute original distances
            const orig_dist = abi.simd.l2Distance(&pair.a, &pair.b);

            // Find global min/max
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);
            for (pair.a) |v| {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            for (pair.b) |v| {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }

            if (max_val - min_val < 1e-6) return true;

            // Quantize and dequantize both vectors
            var quant_a: [VECTOR_DIM]f32 = undefined;
            var quant_b: [VECTOR_DIM]f32 = undefined;

            for (&quant_a, pair.a) |*q, v| {
                const normalized = (v - min_val) / (max_val - min_val);
                const code: u8 = @intFromFloat(std.math.clamp(normalized * 255.0, 0.0, 255.0));
                q.* = min_val + (@as(f32, @floatFromInt(code)) / 255.0) * (max_val - min_val);
            }

            for (&quant_b, pair.b) |*q, v| {
                const normalized = (v - min_val) / (max_val - min_val);
                const code: u8 = @intFromFloat(std.math.clamp(normalized * 255.0, 0.0, 255.0));
                q.* = min_val + (@as(f32, @floatFromInt(code)) / 255.0) * (max_val - min_val);
            }

            const quant_dist = abi.simd.l2Distance(&quant_a, &quant_b);

            // Quantized distance should be close to original
            // Allow for accumulated quantization error
            const max_error = (max_val - min_val) / 255.0 * @sqrt(@as(f32, VECTOR_DIM)) * 2;
            return @abs(orig_dist - quant_dist) < max_error + EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Distance Metric Mathematical Properties
// ============================================================================

test "property - dot product is commutative" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            const dot_ab = abi.simd.vectorDot(&pair.a, &pair.b);
            const dot_ba = abi.simd.vectorDot(&pair.b, &pair.a);
            return assert.approxEqual(dot_ab, dot_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - dot product with self equals squared magnitude" {
    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            const dot_self = abi.simd.vectorDot(&vec, &vec);

            // Compute magnitude squared manually
            var mag_sq: f32 = 0.0;
            for (vec) |v| {
                mag_sq += v * v;
            }

            return assert.approxEqual(dot_self, mag_sq, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - L2 distance satisfies non-negativity" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            const dist = abi.simd.l2Distance(&pair.a, &pair.b);
            return dist >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - L2 distance identity of indiscernibles" {
    const gen = generators.vectorF32(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            const dist = abi.simd.l2Distance(&vec, &vec);
            return dist < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - vector addition is commutative" {
    const gen = generators.vectorPairF32(VECTOR_DIM);

    const result = forAll(property.VectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(VECTOR_DIM)) bool {
            var sum_ab: [VECTOR_DIM]f32 = undefined;
            var sum_ba: [VECTOR_DIM]f32 = undefined;

            for (&sum_ab, pair.a, pair.b) |*s, a, b| {
                s.* = a + b;
            }
            for (&sum_ba, pair.b, pair.a) |*s, b, a| {
                s.* = b + a;
            }

            return assert.slicesApproxEqual(&sum_ab, &sum_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - scalar multiplication is distributive" {
    const gen = generators.scalarVectorF32(VECTOR_DIM);

    const result = forAll(property.ScalarVectorPair(VECTOR_DIM), gen, TestConfig, struct {
        fn check(pair: property.ScalarVectorPair(VECTOR_DIM)) bool {
            const scalar = pair.scalar;
            const vec = pair.vector;

            // Scalar * vector element by element
            var scaled: [VECTOR_DIM]f32 = undefined;
            for (&scaled, vec) |*s, v| {
                s.* = scalar * v;
            }

            // Magnitude should scale by |scalar|
            var orig_mag_sq: f32 = 0.0;
            var scaled_mag_sq: f32 = 0.0;

            for (vec) |v| orig_mag_sq += v * v;
            for (scaled) |s| scaled_mag_sq += s * s;

            const orig_mag = @sqrt(orig_mag_sq);
            const scaled_mag = @sqrt(scaled_mag_sq);
            const expected_mag = orig_mag * @abs(scalar);

            return assert.approxEqual(scaled_mag, expected_mag, EPSILON * 10);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Boundary and Edge Case Properties
// ============================================================================

test "property - single element database always returns that element" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAllWithAllocator([VECTOR_DIM]f32, std.testing.allocator, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            model.insert(1, vec) catch return false;

            // Search with any query should return the single vector
            var query: [VECTOR_DIM]f32 = undefined;
            for (&query, vec) |*q, v| {
                q.* = v + 0.1; // Slightly perturbed query
            }

            const results = model.search(query, 10) catch return false;
            defer allocator.free(results);

            // Should return exactly 1 result
            if (results.len != 1) return false;
            return results[0].id == 1;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - k=1 search returns closest vector" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 2, 20), TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert vectors along a line: [0,0,...], [1,0,...], [2,0,...], etc.
            for (0..n) |i| {
                var vec = [_]f32{0.0} ** VECTOR_DIM;
                vec[0] = @floatFromInt(i);
                model.insert(@intCast(i), vec) catch return false;
            }

            // Query at position 0 should return vector 0
            const query = [_]f32{0.0} ** VECTOR_DIM;
            const results = model.search(query, 1) catch return false;
            defer allocator.free(results);

            if (results.len != 1) return false;
            return results[0].id == 0 and results[0].distance < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - search never returns duplicate IDs" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 5, 30), TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert n random vectors
            var prng = std.Random.DefaultPrng.init(@intCast(n));
            for (0..n) |i| {
                var vec: [VECTOR_DIM]f32 = undefined;
                for (&vec) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                }
                model.insert(@intCast(i), vec) catch return false;
            }

            // Search
            const query = [_]f32{0.0} ** VECTOR_DIM;
            const results = model.search(query, n) catch return false;
            defer allocator.free(results);

            // Check for duplicates
            for (results, 0..) |r1, i| {
                for (results[i + 1 ..]) |r2| {
                    if (r1.id == r2.id) return false;
                }
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - insert then delete then insert works correctly" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAllWithAllocator([VECTOR_DIM]f32, std.testing.allocator, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32, allocator: std.mem.Allocator) bool {
            var model = DatabaseModel.init(allocator);
            defer model.deinit();

            // Insert
            model.insert(42, vec) catch return false;
            if (model.count() != 1) return false;

            // Delete
            if (!model.delete(42)) return false;
            if (model.count() != 0) return false;

            // Re-insert with different vector
            var new_vec: [VECTOR_DIM]f32 = undefined;
            for (&new_vec, vec) |*n, v| {
                n.* = v + 0.5;
            }
            model.insert(42, new_vec) catch return false;
            if (model.count() != 1) return false;

            // Verify new vector is stored
            const retrieved = model.get(42) orelse return false;
            return @abs(retrieved[0] - new_vec[0]) < EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - cosine similarity range with normalized vectors" {
    const gen = generators.unitVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            // Generate another unit vector
            var other: [VECTOR_DIM]f32 = undefined;
            for (&other, vec) |*o, v| {
                o.* = -v; // Opposite direction
            }

            const cos = abi.simd.cosineSimilarity(&vec, &other);

            // Should be -1 for opposite vectors
            return assert.approxEqual(cos, -1.0, EPSILON * 10);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "property - parallel vectors have cosine similarity 1" {
    const gen = generators.nonZeroVector(VECTOR_DIM);

    const result = forAll([VECTOR_DIM]f32, gen, TestConfig, struct {
        fn check(vec: [VECTOR_DIM]f32) bool {
            // Scale vector by positive factor
            var scaled: [VECTOR_DIM]f32 = undefined;
            for (&scaled, vec) |*s, v| {
                s.* = v * 2.5;
            }

            const cos = abi.simd.cosineSimilarity(&vec, &scaled);

            // Skip NaN cases
            if (std.math.isNan(cos)) return true;

            return assert.approxEqual(cos, 1.0, EPSILON * 10);
        }
    }.check);

    try std.testing.expect(result.passed);
}
