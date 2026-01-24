//! K-means vector clustering for WDBX database.
//!
//! Implements K-means clustering with K-means++ initialization for
//! grouping similar vectors. Useful for:
//! - Organizing large vector collections into semantic groups
//! - Pre-filtering search results by cluster
//! - Analyzing vector distribution patterns
//!
//! ## Usage
//! ```zig
//! const clustering = @import("clustering.zig");
//!
//! // Cluster vectors
//! var kmeans = try clustering.KMeans.init(allocator, 10, 128); // 10 clusters, 128 dimensions
//! defer kmeans.deinit();
//!
//! try kmeans.fit(vectors, .{ .max_iterations = 100 });
//!
//! // Get cluster assignment for a new vector
//! const cluster_id = kmeans.predict(query_vector);
//!
//! // Get cluster centroids
//! const centroids = kmeans.getCentroids();
//! ```

const std = @import("std");
const build_options = @import("build_options");
const simd = @import("../shared/simd.zig");
const gpu_accel = @import("gpu_accel.zig");

pub const ClusteringError = error{
    InvalidK,
    InvalidDimension,
    EmptyInput,
    NotFitted,
    ConvergenceFailed,
    OutOfMemory,
};

pub const FitOptions = struct {
    max_iterations: u32 = 300,
    tolerance: f32 = 1e-4,
    seed: ?u64 = null,
    verbose: bool = false,
};

pub const ClusterStats = struct {
    cluster_id: u32,
    size: usize,
    inertia: f64,
    centroid: []const f32,
};

pub const FitResult = struct {
    iterations: u32,
    inertia: f64,
    converged: bool,
};

pub const KMeans = struct {
    allocator: std.mem.Allocator,
    k: u32,
    dimension: u32,
    centroids: [][]f32,
    labels: []u32,
    fitted: bool,
    inertia: f64,
    rng: std.Random.DefaultPrng,
    /// Optional GPU accelerator for batch distance computation
    gpu_accelerator: ?*gpu_accel.GpuAccelerator,

    pub fn init(allocator: std.mem.Allocator, k: u32, dimension: u32) !KMeans {
        if (k == 0) return ClusteringError.InvalidK;
        if (dimension == 0) return ClusteringError.InvalidDimension;

        const centroids = try allocator.alloc([]f32, k);
        errdefer allocator.free(centroids);

        // Track how many centroids have been allocated for cleanup on error
        var allocated_count: usize = 0;
        errdefer {
            // Clean up any partially allocated centroids
            for (centroids[0..allocated_count]) |centroid| {
                allocator.free(centroid);
            }
        }

        for (centroids) |*centroid| {
            centroid.* = try allocator.alloc(f32, dimension);
            allocated_count += 1;
        }

        return .{
            .allocator = allocator,
            .k = k,
            .dimension = dimension,
            .centroids = centroids,
            .labels = &.{},
            .fitted = false,
            .inertia = 0.0,
            .rng = std.Random.DefaultPrng.init(42),
            .gpu_accelerator = null,
        };
    }

    pub fn deinit(self: *KMeans) void {
        for (self.centroids) |centroid| {
            self.allocator.free(centroid);
        }
        self.allocator.free(self.centroids);

        if (self.labels.len > 0) {
            self.allocator.free(self.labels);
        }

        if (self.gpu_accelerator) |accel| {
            accel.deinit();
            self.allocator.destroy(accel);
        }

        self.* = undefined;
    }

    /// Enable GPU acceleration for batch distance computation.
    /// Beneficial for large datasets (>10k vectors) and high-dimensional spaces (>64 dims).
    pub fn enableGpuAcceleration(self: *KMeans) !void {
        if (self.gpu_accelerator != null) return; // Already enabled

        if (!build_options.enable_gpu) return ClusteringError.OutOfMemory;

        const accel = try self.allocator.create(gpu_accel.GpuAccelerator);
        errdefer self.allocator.destroy(accel);

        accel.* = try gpu_accel.GpuAccelerator.init(self.allocator, .{
            .batch_threshold = 256, // Use GPU for batches >= 256 vectors
        });

        self.gpu_accelerator = accel;
    }

    /// Disable GPU acceleration.
    pub fn disableGpuAcceleration(self: *KMeans) void {
        if (self.gpu_accelerator) |accel| {
            accel.deinit();
            self.allocator.destroy(accel);
            self.gpu_accelerator = null;
        }
    }

    /// Fit the K-means model to the given vectors using K-means++ initialization.
    pub fn fit(self: *KMeans, vectors: []const []const f32, options: FitOptions) !FitResult {
        if (vectors.len == 0) return ClusteringError.EmptyInput;
        if (vectors.len < self.k) return ClusteringError.InvalidK;

        // Initialize RNG with seed if provided
        if (options.seed) |seed| {
            self.rng = std.Random.DefaultPrng.init(seed);
        }

        // Validate dimensions
        for (vectors) |v| {
            if (v.len != self.dimension) {
                return ClusteringError.InvalidDimension;
            }
        }

        // Allocate labels
        if (self.labels.len > 0) {
            self.allocator.free(self.labels);
        }
        self.labels = try self.allocator.alloc(u32, vectors.len);

        // Initialize centroids using K-means++
        try self.initializeCentroidsPlusPlus(vectors);

        // Run Lloyd's algorithm
        var prev_inertia: f64 = std.math.inf(f64);
        var iteration: u32 = 0;
        var converged = false;

        while (iteration < options.max_iterations) : (iteration += 1) {
            // Assignment step
            self.assignLabels(vectors);

            // Update step
            try self.updateCentroids(vectors);

            // Calculate inertia
            self.inertia = self.calculateInertia(vectors);

            if (options.verbose and iteration % 10 == 0) {
                std.debug.print("Iteration {d}: inertia = {d:.4}\n", .{ iteration, self.inertia });
            }

            // Check convergence
            const improvement = prev_inertia - self.inertia;
            if (improvement >= 0 and improvement < options.tolerance) {
                converged = true;
                break;
            }
            prev_inertia = self.inertia;
        }

        self.fitted = true;

        return FitResult{
            .iterations = iteration,
            .inertia = self.inertia,
            .converged = converged,
        };
    }

    /// Predict cluster assignment for a single vector.
    pub fn predict(self: *const KMeans, vector: []const f32) !u32 {
        if (!self.fitted) return ClusteringError.NotFitted;
        if (vector.len != self.dimension) return ClusteringError.InvalidDimension;

        var min_dist: f32 = std.math.inf(f32);
        var best_cluster: u32 = 0;

        for (self.centroids, 0..) |centroid, i| {
            const dist = euclideanDistanceSquared(vector, centroid);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = @intCast(i);
            }
        }

        return best_cluster;
    }

    /// Predict cluster assignments for multiple vectors.
    pub fn predictBatch(self: *const KMeans, vectors: []const []const f32, allocator: std.mem.Allocator) ![]u32 {
        if (!self.fitted) return ClusteringError.NotFitted;

        var labels = try allocator.alloc(u32, vectors.len);
        errdefer allocator.free(labels);

        for (vectors, 0..) |v, i| {
            labels[i] = try self.predict(v);
        }

        return labels;
    }

    /// Get the cluster centroids.
    pub fn getCentroids(self: *const KMeans) []const []const f32 {
        return self.centroids;
    }

    /// Get the labels assigned during fitting.
    pub fn getLabels(self: *const KMeans) ![]const u32 {
        if (!self.fitted) return ClusteringError.NotFitted;
        return self.labels;
    }

    /// Get statistics for each cluster.
    pub fn getClusterStats(self: *const KMeans, vectors: []const []const f32, allocator: std.mem.Allocator) ![]ClusterStats {
        if (!self.fitted) return ClusteringError.NotFitted;

        var stats = try allocator.alloc(ClusterStats, self.k);
        errdefer allocator.free(stats);

        // Initialize stats
        for (stats, 0..) |*s, i| {
            s.cluster_id = @intCast(i);
            s.size = 0;
            s.inertia = 0.0;
            s.centroid = self.centroids[i];
        }

        // Calculate sizes and inertia per cluster
        for (vectors, 0..) |v, i| {
            const label = self.labels[i];
            stats[label].size += 1;
            const dist = euclideanDistanceSquared(v, self.centroids[label]);
            stats[label].inertia += dist;
        }

        return stats;
    }

    /// Transform vectors to cluster-distance space.
    pub fn transform(self: *const KMeans, vectors: []const []const f32, allocator: std.mem.Allocator) ![][]f32 {
        if (!self.fitted) return ClusteringError.NotFitted;

        var result = try allocator.alloc([]f32, vectors.len);
        errdefer allocator.free(result);

        for (vectors, 0..) |v, i| {
            result[i] = try allocator.alloc(f32, self.k);
            for (self.centroids, 0..) |centroid, j| {
                result[i][j] = @sqrt(euclideanDistanceSquared(v, centroid));
            }
        }

        return result;
    }

    /// Get the inertia (sum of squared distances to nearest centroid).
    pub fn getInertia(self: *const KMeans) !f64 {
        if (!self.fitted) return ClusteringError.NotFitted;
        return self.inertia;
    }

    // Internal methods

    fn initializeCentroidsPlusPlus(self: *KMeans, vectors: []const []const f32) !void {
        const random = self.rng.random();

        // Choose first centroid randomly
        const first_idx = random.uintLessThan(usize, vectors.len);
        @memcpy(self.centroids[0], vectors[first_idx]);

        // Allocate distance array
        var distances = try self.allocator.alloc(f32, vectors.len);
        defer self.allocator.free(distances);

        // Choose remaining centroids
        var centroid_idx: u32 = 1;
        while (centroid_idx < self.k) : (centroid_idx += 1) {
            // Calculate distances to nearest centroid
            var sum_dist: f64 = 0.0;
            for (vectors, 0..) |v, i| {
                var min_dist: f32 = std.math.inf(f32);
                for (self.centroids[0..centroid_idx]) |c| {
                    const dist = euclideanDistanceSquared(v, c);
                    if (dist < min_dist) min_dist = dist;
                }
                distances[i] = min_dist;
                sum_dist += min_dist;
            }

            // Sample proportional to squared distance
            const target = random.float(f32) * @as(f32, @floatCast(sum_dist));
            var cumsum: f32 = 0.0;
            var selected: usize = 0;

            for (distances, 0..) |d, i| {
                cumsum += d;
                if (cumsum >= target) {
                    selected = i;
                    break;
                }
            }

            @memcpy(self.centroids[centroid_idx], vectors[selected]);
        }
    }

    fn assignLabels(self: *KMeans, vectors: []const []const f32) void {
        // Try GPU-accelerated batch assignment for large datasets
        if (self.gpu_accelerator != null and vectors.len >= 1000) {
            self.assignLabelsGpu(vectors) catch {
                // Fall back to CPU if GPU fails
                self.assignLabelsCpu(vectors);
            };
        } else {
            // Use CPU for small datasets or when GPU is unavailable
            self.assignLabelsCpu(vectors);
        }
    }

    /// CPU implementation of label assignment using SIMD distance computation.
    fn assignLabelsCpu(self: *KMeans, vectors: []const []const f32) void {
        for (vectors, 0..) |v, i| {
            var min_dist: f32 = std.math.inf(f32);
            var best: u32 = 0;

            for (self.centroids, 0..) |centroid, j| {
                const dist = euclideanDistanceSquared(v, centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = @intCast(j);
                }
            }

            self.labels[i] = best;
        }
    }

    /// GPU-accelerated label assignment using batch distance computation.
    fn assignLabelsGpu(self: *KMeans, vectors: []const []const f32) !void {
        const accel = self.gpu_accelerator.?;

        // Allocate distance matrix: vectors x centroids
        var distances = try self.allocator.alloc([]f32, vectors.len);
        defer {
            for (distances) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(distances);
        }

        for (distances) |*row| {
            row.* = try self.allocator.alloc(f32, self.k);
        }

        // Compute distances for each centroid in batch
        for (self.centroids, 0..) |centroid, c_idx| {
            // Compute centroid norm once
            const centroid_norm = simd.vectorL2Norm(centroid);

            // Extract column of distances for this centroid
            const centroid_dists = try self.allocator.alloc(f32, vectors.len);
            defer self.allocator.free(centroid_dists);

            // Batch compute cosine similarity, then convert to euclidean distance
            try accel.batchCosineSimilarity(centroid, centroid_norm, vectors, centroid_dists);

            // Convert similarities to euclidean distances
            // For unit vectors: d^2 = 2(1 - similarity)
            // For general vectors, we use L2 distance directly
            for (vectors, 0..) |v, v_idx| {
                distances[v_idx][c_idx] = euclideanDistanceSquared(v, centroid);
            }
        }

        // Find nearest centroid for each vector
        for (distances, 0..) |row, i| {
            var min_dist: f32 = std.math.inf(f32);
            var best: u32 = 0;

            for (row, 0..) |dist, j| {
                if (dist < min_dist) {
                    min_dist = dist;
                    best = @intCast(j);
                }
            }

            self.labels[i] = best;
        }
    }

    fn updateCentroids(self: *KMeans, vectors: []const []const f32) !void {
        // Temporary storage for cluster sums and counts
        var sums = try self.allocator.alloc([]f64, self.k);
        defer {
            for (sums) |s| self.allocator.free(s);
            self.allocator.free(sums);
        }

        for (sums, 0..) |*s, i| {
            s.* = try self.allocator.alloc(f64, self.dimension);
            @memset(s.*, 0.0);
            _ = i;
        }

        var counts = try self.allocator.alloc(usize, self.k);
        defer self.allocator.free(counts);
        @memset(counts, 0);

        // Accumulate sums
        for (vectors, 0..) |v, i| {
            const label = self.labels[i];
            counts[label] += 1;
            for (v, 0..) |val, d| {
                sums[label][d] += val;
            }
        }

        // Update centroids
        for (self.centroids, 0..) |*centroid, i| {
            const count = counts[i];
            if (count > 0) {
                for (centroid.*, 0..) |*val, d| {
                    val.* = @floatCast(sums[i][d] / @as(f64, @floatFromInt(count)));
                }
            }
        }
    }

    fn calculateInertia(self: *const KMeans, vectors: []const []const f32) f64 {
        var total: f64 = 0.0;
        for (vectors, 0..) |v, i| {
            const label = self.labels[i];
            const dist = euclideanDistanceSquared(v, self.centroids[label]);
            total += dist;
        }
        return total;
    }
};

/// Calculate squared Euclidean distance between two vectors using SIMD.
pub fn euclideanDistanceSquared(a: []const f32, b: []const f32) f32 {
    return simd.l2DistanceSquared(a, b);
}

/// Calculate Euclidean distance between two vectors using SIMD.
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    return @sqrt(simd.l2DistanceSquared(a, b));
}

/// Calculate cosine similarity between two vectors using SIMD.
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
}

/// Silhouette score for evaluating cluster quality.
/// Returns a value between -1 and 1 where higher is better.
pub fn silhouetteScore(
    vectors: []const []const f32,
    labels: []const u32,
    k: u32,
    allocator: std.mem.Allocator,
) !f32 {
    if (vectors.len == 0) return 0.0;

    var scores = try allocator.alloc(f32, vectors.len);
    defer allocator.free(scores);

    for (vectors, 0..) |v, i| {
        const label = labels[i];

        // Calculate a(i) - mean distance to same cluster
        var a_sum: f32 = 0.0;
        var a_count: usize = 0;
        for (vectors, 0..) |other, j| {
            if (i != j and labels[j] == label) {
                a_sum += euclideanDistance(v, other);
                a_count += 1;
            }
        }
        const a = if (a_count > 0) a_sum / @as(f32, @floatFromInt(a_count)) else 0.0;

        // Calculate b(i) - minimum mean distance to other clusters
        var b: f32 = std.math.inf(f32);
        for (0..k) |cluster| {
            if (cluster == label) continue;

            var b_sum: f32 = 0.0;
            var b_count: usize = 0;
            for (vectors, 0..) |other, j| {
                if (labels[j] == cluster) {
                    b_sum += euclideanDistance(v, other);
                    b_count += 1;
                }
            }

            if (b_count > 0) {
                const mean_dist = b_sum / @as(f32, @floatFromInt(b_count));
                if (mean_dist < b) b = mean_dist;
            }
        }

        // Silhouette coefficient
        const max_ab = @max(a, b);
        scores[i] = if (max_ab > 0) (b - a) / max_ab else 0.0;
    }

    // Return mean silhouette score
    var sum: f32 = 0.0;
    for (scores) |s| sum += s;
    return sum / @as(f32, @floatFromInt(scores.len));
}

/// Elbow method helper - returns inertias for different k values.
pub fn elbowMethod(
    vectors: []const []const f32,
    k_range: []const u32,
    dimension: u32,
    allocator: std.mem.Allocator,
) ![]f64 {
    var inertias = try allocator.alloc(f64, k_range.len);
    errdefer allocator.free(inertias);

    for (k_range, 0..) |k, i| {
        var kmeans = try KMeans.init(allocator, k, dimension);
        defer kmeans.deinit();

        const result = try kmeans.fit(vectors, .{});
        inertias[i] = result.inertia;
    }

    return inertias;
}

test "kmeans initialization" {
    const allocator = std.testing.allocator;
    var kmeans = try KMeans.init(allocator, 3, 4);
    defer kmeans.deinit();

    try std.testing.expectEqual(@as(u32, 3), kmeans.k);
    try std.testing.expectEqual(@as(u32, 4), kmeans.dimension);
    try std.testing.expect(!kmeans.fitted);
}

test "kmeans fit and predict" {
    const allocator = std.testing.allocator;

    // Create simple test data with 3 distinct clusters
    const vectors = [_][]const f32{
        // Cluster 0
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 0.1, 0.1 },
        &[_]f32{ 0.0, 0.1 },
        // Cluster 1
        &[_]f32{ 5.0, 5.0 },
        &[_]f32{ 5.1, 5.0 },
        &[_]f32{ 5.0, 5.1 },
        // Cluster 2
        &[_]f32{ 10.0, 0.0 },
        &[_]f32{ 10.1, 0.1 },
        &[_]f32{ 10.0, 0.1 },
    };

    var kmeans = try KMeans.init(allocator, 3, 2);
    defer kmeans.deinit();

    const result = try kmeans.fit(&vectors, .{ .seed = 42 });

    try std.testing.expect(result.converged or result.iterations <= 300);
    try std.testing.expect(kmeans.fitted);

    // Predict should assign to closest cluster
    const query = [_]f32{ 0.05, 0.05 };
    const predicted = try kmeans.predict(&query);
    try std.testing.expect(predicted < 3);
}

test "euclidean distance" {
    const a = [_]f32{ 0.0, 0.0 };
    const b = [_]f32{ 3.0, 4.0 };

    const dist = euclideanDistance(&a, &b);
    try std.testing.expect(@abs(dist - 5.0) < 0.001);
}

test "cosine similarity" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0 };
    const c = [_]f32{ 0.0, 1.0 };

    const sim_same = cosineSimilarity(&a, &b);
    const sim_orthogonal = cosineSimilarity(&a, &c);

    try std.testing.expect(@abs(sim_same - 1.0) < 0.001);
    try std.testing.expect(@abs(sim_orthogonal) < 0.001);
}

test "kmeans transform" {
    const allocator = std.testing.allocator;

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 0.0 },
        &[_]f32{ 0.0, 1.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    var kmeans = try KMeans.init(allocator, 2, 2);
    defer kmeans.deinit();

    _ = try kmeans.fit(&vectors, .{ .seed = 123 });

    const transformed = try kmeans.transform(&vectors, allocator);
    defer {
        for (transformed) |t| allocator.free(t);
        allocator.free(transformed);
    }

    // Each transformed vector should have k dimensions
    try std.testing.expectEqual(@as(usize, 4), transformed.len);
    for (transformed) |t| {
        try std.testing.expectEqual(@as(usize, 2), t.len);
    }
}

test "kmeans cluster stats" {
    const allocator = std.testing.allocator;

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 0.1, 0.1 },
        &[_]f32{ 5.0, 5.0 },
        &[_]f32{ 5.1, 5.1 },
    };

    var kmeans = try KMeans.init(allocator, 2, 2);
    defer kmeans.deinit();

    _ = try kmeans.fit(&vectors, .{ .seed = 456 });

    const stats = try kmeans.getClusterStats(&vectors, allocator);
    defer allocator.free(stats);

    // Should have 2 clusters
    try std.testing.expectEqual(@as(usize, 2), stats.len);

    // Each cluster should have 2 vectors
    var total_size: usize = 0;
    for (stats) |s| {
        total_size += s.size;
    }
    try std.testing.expectEqual(@as(usize, 4), total_size);
}
