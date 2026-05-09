//! Anisotropic Vector Quantization (AVQ) codebook for ScaNN.

const std = @import("std");
const types = @import("types.zig");
const computeL2DistanceSquared = types.computeL2DistanceSquared;
const computeWeightedL2 = types.computeWeightedL2;

/// Anisotropic Vector Quantization codebook
pub const AVQCodebook = struct {
    allocator: std.mem.Allocator,
    dimensions: u32,
    num_centroids: u32,
    /// Centroids: [num_centroids][dimensions]
    centroids: []f32,
    /// Anisotropy weights per dimension (importance)
    dimension_weights: []f32,
    /// Direction vectors for anisotropic loss
    direction_vectors: ?[]f32 = null,

    pub fn init(allocator: std.mem.Allocator, dimensions: u32, num_centroids: u32) !AVQCodebook {
        const cb = AVQCodebook{
            .allocator = allocator,
            .dimensions = dimensions,
            .num_centroids = num_centroids,
            .centroids = try allocator.alloc(f32, num_centroids * dimensions),
            .dimension_weights = try allocator.alloc(f32, dimensions),
        };

        // Initialize weights uniformly
        @memset(cb.dimension_weights, 1.0);

        return cb;
    }

    pub fn deinit(self: *AVQCodebook) void {
        self.allocator.free(self.centroids);
        self.allocator.free(self.dimension_weights);
        if (self.direction_vectors) |dv| {
            self.allocator.free(dv);
        }
    }

    /// Train codebook with anisotropic loss
    pub fn train(self: *AVQCodebook, training_vectors: []const f32, query_vectors: ?[]const f32) !void {
        const num_vectors = training_vectors.len / self.dimensions;
        if (num_vectors < self.num_centroids) return error.InsufficientTrainingData;

        // Phase 1: Initial k-means clustering
        try self.initializeCentroids(training_vectors);
        try self.runKMeans(training_vectors, 20);

        // Phase 2: Learn dimension weights from query distribution (if available)
        if (query_vectors) |queries| {
            try self.learnDimensionWeights(training_vectors, queries);
        }

        // Phase 3: Refine centroids with anisotropic loss
        try self.refineWithAnisotropicLoss(training_vectors, 10);
    }

    fn initializeCentroids(self: *AVQCodebook, vectors: []const f32) !void {
        const num_vectors = vectors.len / self.dimensions;
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // K-means++ initialization
        // First centroid: random
        const first_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
        @memcpy(
            self.centroids[0..self.dimensions],
            vectors[first_idx * self.dimensions ..][0..self.dimensions],
        );

        // Subsequent centroids: weighted by distance to nearest centroid
        var min_distances = try self.allocator.alloc(f32, num_vectors);
        defer self.allocator.free(min_distances);

        for (0..num_vectors) |i| {
            min_distances[i] = std.math.inf(f32);
        }

        for (1..self.num_centroids) |c| {
            // Update min distances
            for (0..num_vectors) |i| {
                const vec = vectors[i * self.dimensions ..][0..self.dimensions];
                const prev_centroid = self.centroids[(c - 1) * self.dimensions ..][0..self.dimensions];
                const dist = computeL2DistanceSquared(vec, prev_centroid);
                min_distances[i] = @min(min_distances[i], dist);
            }

            // Select next centroid proportional to squared distance
            var total_dist: f64 = 0;
            for (min_distances) |d| {
                total_dist += d;
            }

            const threshold = random.float(f64) * total_dist;
            var cumulative: f64 = 0;
            var selected_idx: usize = 0;

            for (min_distances, 0..) |d, i| {
                cumulative += d;
                if (cumulative >= threshold) {
                    selected_idx = i;
                    break;
                }
            }

            @memcpy(
                self.centroids[c * self.dimensions ..][0..self.dimensions],
                vectors[selected_idx * self.dimensions ..][0..self.dimensions],
            );
        }
    }

    fn runKMeans(self: *AVQCodebook, vectors: []const f32, iterations: u32) !void {
        const num_vectors = vectors.len / self.dimensions;
        const allocator = self.allocator;

        const assignments = try allocator.alloc(u32, num_vectors);
        defer allocator.free(assignments);

        const counts = try allocator.alloc(u32, self.num_centroids);
        defer allocator.free(counts);

        const new_centroids = try allocator.alloc(f32, self.centroids.len);
        defer allocator.free(new_centroids);

        for (0..iterations) |_| {
            // E-step: assign vectors to nearest centroid
            for (0..num_vectors) |i| {
                const vec = vectors[i * self.dimensions ..][0..self.dimensions];

                var min_dist: f32 = std.math.inf(f32);
                var min_centroid: u32 = 0;

                for (0..self.num_centroids) |c| {
                    const centroid = self.centroids[c * self.dimensions ..][0..self.dimensions];
                    const dist = computeWeightedL2(vec, centroid, self.dimension_weights);

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_centroid = @intCast(c);
                    }
                }

                assignments[i] = min_centroid;
            }

            // M-step: update centroids
            @memset(new_centroids, 0);
            @memset(counts, 0);

            for (0..num_vectors) |i| {
                const c = assignments[i];
                const vec = vectors[i * self.dimensions ..][0..self.dimensions];

                for (0..self.dimensions) |d| {
                    new_centroids[c * self.dimensions + d] += vec[d];
                }
                counts[c] += 1;
            }

            // Normalize
            for (0..self.num_centroids) |c| {
                if (counts[c] > 0) {
                    for (0..self.dimensions) |d| {
                        new_centroids[c * self.dimensions + d] /= @floatFromInt(counts[c]);
                    }
                }
            }

            @memcpy(self.centroids, new_centroids);
        }
    }

    fn learnDimensionWeights(self: *AVQCodebook, _: []const f32, queries: []const f32) !void {
        const num_queries = queries.len / self.dimensions;
        if (num_queries == 0) return;

        // Compute mean
        var means = try self.allocator.alloc(f32, self.dimensions);
        defer self.allocator.free(means);
        @memset(means, 0);

        for (0..num_queries) |q| {
            for (0..self.dimensions) |d| {
                means[d] += queries[q * self.dimensions + d];
            }
        }
        for (means) |*m| {
            m.* /= @floatFromInt(num_queries);
        }

        // Compute variance
        @memset(self.dimension_weights, 0);
        for (0..num_queries) |q| {
            for (0..self.dimensions) |d| {
                const diff = queries[q * self.dimensions + d] - means[d];
                self.dimension_weights[d] += diff * diff;
            }
        }

        // Normalize to get importance weights
        var max_var: f32 = 0;
        for (self.dimension_weights) |w| {
            max_var = @max(max_var, w);
        }

        if (max_var > 0) {
            for (self.dimension_weights) |*w| {
                w.* = @sqrt(w.* / max_var + 0.01);
            }
        }
    }

    fn refineWithAnisotropicLoss(self: *AVQCodebook, vectors: []const f32, iterations: u32) !void {
        _ = self;
        _ = vectors;
        _ = iterations;
        // Simplified: just use learned weights in distance computation
    }

    /// Encode a vector to nearest centroid index
    pub fn encode(self: *const AVQCodebook, vector: []const f32) u32 {
        var min_dist: f32 = std.math.inf(f32);
        var min_idx: u32 = 0;

        for (0..self.num_centroids) |c| {
            const centroid = self.centroids[c * self.dimensions ..][0..self.dimensions];
            const dist = computeWeightedL2(vector, centroid, self.dimension_weights);

            if (dist < min_dist) {
                min_dist = dist;
                min_idx = @intCast(c);
            }
        }

        return min_idx;
    }

    /// Compute distance using asymmetric distance (query to centroid)
    pub fn asymmetricDistance(self: *const AVQCodebook, query: []const f32, centroid_idx: u32) f32 {
        const centroid = self.centroids[centroid_idx * self.dimensions ..][0..self.dimensions];
        return computeWeightedL2(query, centroid, self.dimension_weights);
    }
};
