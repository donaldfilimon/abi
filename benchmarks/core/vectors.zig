//! Unified Vector Generation for Benchmarks
//!
//! Provides consistent vector generation across all benchmark suites with
//! multiple distribution types for realistic testing scenarios.
//!
//! ## Supported Distributions
//!
//! - **uniform**: Random values in [-1, 1]
//! - **normalized**: Unit-length vectors (common for embeddings)
//! - **clustered**: Vectors grouped around cluster centers
//! - **gaussian**: Normal distribution around zero

const std = @import("std");

/// Vector distribution types for different benchmark scenarios
pub const VectorDistribution = enum {
    /// Uniform random values in [-1, 1]
    uniform,
    /// Normalized to unit length (L2 norm = 1)
    normalized,
    /// Clustered around randomly generated centers
    clustered,
    /// Gaussian distribution (mean=0, stddev=1)
    gaussian,
};

/// Configuration for vector generation
pub const VectorConfig = struct {
    /// Number of vectors to generate
    count: usize,
    /// Dimensionality of each vector
    dimension: usize,
    /// Distribution type
    distribution: VectorDistribution = .normalized,
    /// Number of clusters (only used for .clustered distribution)
    num_clusters: usize = 10,
    /// Random seed for reproducibility
    seed: u64 = 42,
    /// Cluster spread factor (only for .clustered, smaller = tighter clusters)
    cluster_spread: f32 = 0.2,
};

/// Generate vectors according to the specified configuration
pub fn generate(allocator: std.mem.Allocator, config: VectorConfig) ![][]f32 {
    return switch (config.distribution) {
        .uniform => generateUniform(allocator, config.count, config.dimension, config.seed),
        .normalized => generateNormalized(allocator, config.count, config.dimension, config.seed),
        .clustered => generateClustered(allocator, config.count, config.dimension, config.num_clusters, config.cluster_spread, config.seed),
        .gaussian => generateGaussian(allocator, config.count, config.dimension, config.seed),
    };
}

/// Free vectors allocated by generate()
pub fn free(allocator: std.mem.Allocator, vectors: [][]f32) void {
    for (vectors) |v| {
        allocator.free(v);
    }
    allocator.free(vectors);
}

/// Generate uniform random vectors in [-1, 1]
fn generateUniform(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const vectors = try allocator.alloc([]f32, count);
    var vectors_allocated: usize = 0;
    errdefer {
        for (vectors[0..vectors_allocated]) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        vectors_allocated += 1;
        for (vec.*) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
        }
    }

    return vectors;
}

/// Generate normalized (unit-length) vectors
pub fn generateNormalized(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const vectors = try allocator.alloc([]f32, count);
    var vectors_allocated: usize = 0;
    errdefer {
        for (vectors[0..vectors_allocated]) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        vectors_allocated += 1;
        var norm: f32 = 0;

        for (vec.*) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
            norm += val.* * val.*;
        }

        // Normalize to unit length
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*val| {
                val.* /= norm;
            }
        }
    }

    return vectors;
}

/// Generate clustered vectors around random centers
pub fn generateClustered(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    num_clusters: usize,
    spread: f32,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Generate cluster centers
    const centers = try allocator.alloc([]f32, num_clusters);
    var centers_allocated: usize = 0;
    defer {
        for (centers[0..centers_allocated]) |c| {
            allocator.free(c);
        }
        allocator.free(centers);
    }

    for (centers) |*center| {
        center.* = try allocator.alloc(f32, dim);
        centers_allocated += 1;
        for (center.*) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
        }
    }

    // Generate points around clusters
    const vectors = try allocator.alloc([]f32, count);
    var vectors_allocated: usize = 0;
    errdefer {
        for (vectors[0..vectors_allocated]) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        vectors_allocated += 1;
        const cluster_idx = rand.intRangeLessThan(usize, 0, num_clusters);
        const center = centers[cluster_idx];

        var norm: f32 = 0;
        for (vec.*, center) |*val, c| {
            val.* = c + (rand.float(f32) - 0.5) * spread;
            norm += val.* * val.*;
        }

        // Normalize
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*val| {
                val.* /= norm;
            }
        }
    }

    return vectors;
}

/// Generate vectors with Gaussian distribution
fn generateGaussian(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const vectors = try allocator.alloc([]f32, count);
    var vectors_allocated: usize = 0;
    errdefer {
        for (vectors[0..vectors_allocated]) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        vectors_allocated += 1;
        var i: usize = 0;
        while (i < dim) {
            // Box-Muller transform for Gaussian distribution
            const rand_u1 = rand.float(f32);
            const rand_u2 = rand.float(f32);

            const radius = @sqrt(-2.0 * @log(rand_u1 + 1e-10));
            const theta = 2.0 * std.math.pi * rand_u2;

            vec.*[i] = radius * @cos(theta);
            i += 1;

            if (i < dim) {
                vec.*[i] = radius * @sin(theta);
                i += 1;
            }
        }
    }

    return vectors;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick generation of normalized vectors (most common use case)
pub fn normalized(allocator: std.mem.Allocator, count: usize, dim: usize) ![][]f32 {
    return generate(allocator, .{
        .count = count,
        .dimension = dim,
        .distribution = .normalized,
    });
}

/// Quick generation of clustered vectors
pub fn clustered(allocator: std.mem.Allocator, count: usize, dim: usize, num_clusters: usize) ![][]f32 {
    return generate(allocator, .{
        .count = count,
        .dimension = dim,
        .distribution = .clustered,
        .num_clusters = num_clusters,
    });
}

// ============================================================================
// Tests
// ============================================================================

test "generate normalized vectors" {
    const allocator = std.testing.allocator;

    const vectors = try generateNormalized(allocator, 10, 64, 42);
    defer free(allocator, vectors);

    try std.testing.expectEqual(@as(usize, 10), vectors.len);
    try std.testing.expectEqual(@as(usize, 64), vectors[0].len);

    // Check normalization
    var norm: f32 = 0;
    for (vectors[0]) |v| {
        norm += v * v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}

test "generate clustered vectors" {
    const allocator = std.testing.allocator;

    const vectors = try generateClustered(allocator, 100, 32, 5, 0.2, 42);
    defer free(allocator, vectors);

    try std.testing.expectEqual(@as(usize, 100), vectors.len);
    try std.testing.expectEqual(@as(usize, 32), vectors[0].len);
}

test "generate with config" {
    const allocator = std.testing.allocator;

    const vectors = try generate(allocator, .{
        .count = 50,
        .dimension = 128,
        .distribution = .gaussian,
        .seed = 123,
    });
    defer free(allocator, vectors);

    try std.testing.expectEqual(@as(usize, 50), vectors.len);
    try std.testing.expectEqual(@as(usize, 128), vectors[0].len);
}

test "convenience functions" {
    const allocator = std.testing.allocator;

    const norm_vecs = try normalized(allocator, 20, 64);
    defer free(allocator, norm_vecs);
    try std.testing.expectEqual(@as(usize, 20), norm_vecs.len);

    const clust_vecs = try clustered(allocator, 30, 64, 3);
    defer free(allocator, clust_vecs);
    try std.testing.expectEqual(@as(usize, 30), clust_vecs.len);
}
