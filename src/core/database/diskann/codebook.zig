//! Product Quantization codebook for DiskANN compression.

const std = @import("std");
const types = @import("types.zig");
const computeL2DistanceSquared = types.computeL2DistanceSquared;

/// Product Quantization codebook for compression
pub const PQCodebook = struct {
    allocator: std.mem.Allocator,
    num_subspaces: u32,
    subspace_dim: u32,
    num_centroids: u32,
    /// Centroids: [num_subspaces][num_centroids][subspace_dim]
    centroids: []f32,
    /// Precomputed distance tables for query
    distance_tables: ?[]f32 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        num_subspaces: u32,
        subspace_dim: u32,
        num_centroids: u32,
    ) !PQCodebook {
        const total_centroids = num_subspaces * num_centroids * subspace_dim;

        return PQCodebook{
            .allocator = allocator,
            .num_subspaces = num_subspaces,
            .subspace_dim = subspace_dim,
            .num_centroids = num_centroids,
            .centroids = try allocator.alloc(f32, total_centroids),
        };
    }

    pub fn deinit(self: *PQCodebook) void {
        self.allocator.free(self.centroids);
        if (self.distance_tables) |tables| {
            self.allocator.free(tables);
        }
    }

    /// Train codebook using k-means clustering on training vectors
    pub fn train(self: *PQCodebook, training_vectors: []const f32, dim: u32) !void {
        const num_vectors = training_vectors.len / dim;
        if (num_vectors < self.num_centroids) return error.InsufficientTrainingData;

        // Simple k-means initialization (random selection)
        // Production would use k-means++ initialization
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        for (0..self.num_subspaces) |s| {
            const subspace_offset = s * self.subspace_dim;
            const centroid_offset = s * self.num_centroids * self.subspace_dim;

            // Initialize centroids from random training vectors
            for (0..self.num_centroids) |c| {
                const vec_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
                const src_offset = vec_idx * dim + subspace_offset;
                const dst_offset = centroid_offset + c * self.subspace_dim;

                @memcpy(
                    self.centroids[dst_offset..][0..self.subspace_dim],
                    training_vectors[src_offset..][0..self.subspace_dim],
                );
            }
        }

        // Run k-means iterations (simplified - production uses convergence check)
        try self.runKMeansIterations(training_vectors, dim, 10);
    }

    fn runKMeansIterations(self: *PQCodebook, vectors: []const f32, dim: u32, iterations: u32) !void {
        const num_vectors = vectors.len / dim;
        const allocator = self.allocator;

        // Allocations for k-means
        const assignments = try allocator.alloc(u8, num_vectors * self.num_subspaces);
        defer allocator.free(assignments);

        const counts = try allocator.alloc(u32, self.num_subspaces * self.num_centroids);
        defer allocator.free(counts);

        const new_centroids = try allocator.alloc(f32, self.centroids.len);
        defer allocator.free(new_centroids);

        for (0..iterations) |_| {
            // E-step: assign vectors to nearest centroids
            for (0..num_vectors) |v| {
                for (0..self.num_subspaces) |s| {
                    const vec_offset = v * dim + s * self.subspace_dim;
                    const subvec = vectors[vec_offset..][0..self.subspace_dim];

                    var min_dist: f32 = std.math.inf(f32);
                    var min_centroid: u8 = 0;

                    for (0..self.num_centroids) |c| {
                        const c_offset = s * self.num_centroids * self.subspace_dim + c * self.subspace_dim;
                        const centroid = self.centroids[c_offset..][0..self.subspace_dim];

                        const dist = computeL2DistanceSquared(subvec, centroid);
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_centroid = @intCast(c);
                        }
                    }

                    assignments[v * self.num_subspaces + s] = min_centroid;
                }
            }

            // M-step: update centroids
            @memset(new_centroids, 0);
            @memset(counts, 0);

            for (0..num_vectors) |v| {
                for (0..self.num_subspaces) |s| {
                    const c = assignments[v * self.num_subspaces + s];
                    const vec_offset = v * dim + s * self.subspace_dim;
                    const c_offset = s * self.num_centroids * self.subspace_dim + @as(usize, c) * self.subspace_dim;

                    for (0..self.subspace_dim) |d| {
                        new_centroids[c_offset + d] += vectors[vec_offset + d];
                    }
                    counts[s * self.num_centroids + c] += 1;
                }
            }

            // Normalize centroids
            for (0..self.num_subspaces) |s| {
                for (0..self.num_centroids) |c| {
                    const count = counts[s * self.num_centroids + c];
                    if (count > 0) {
                        const c_offset = s * self.num_centroids * self.subspace_dim + c * self.subspace_dim;
                        for (0..self.subspace_dim) |d| {
                            new_centroids[c_offset + d] /= @floatFromInt(count);
                        }
                    }
                }
            }

            @memcpy(self.centroids, new_centroids);
        }
    }

    /// Encode a vector using PQ codes
    pub fn encode(self: *const PQCodebook, vector: []const f32, codes: []u8) void {
        std.debug.assert(codes.len == self.num_subspaces);

        for (0..self.num_subspaces) |s| {
            const subvec_offset = s * self.subspace_dim;
            const subvec = vector[subvec_offset..][0..self.subspace_dim];

            var min_dist: f32 = std.math.inf(f32);
            var min_centroid: u8 = 0;

            for (0..self.num_centroids) |c| {
                const c_offset = s * self.num_centroids * self.subspace_dim + c * self.subspace_dim;
                const centroid = self.centroids[c_offset..][0..self.subspace_dim];

                const dist = computeL2DistanceSquared(subvec, centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_centroid = @intCast(c);
                }
            }

            codes[s] = min_centroid;
        }
    }

    /// Precompute distance table for a query vector
    pub fn computeDistanceTable(self: *PQCodebook, query: []const f32) !void {
        if (self.distance_tables == null) {
            self.distance_tables = try self.allocator.alloc(f32, self.num_subspaces * self.num_centroids);
        }

        for (0..self.num_subspaces) |s| {
            const query_offset = s * self.subspace_dim;
            const query_subvec = query[query_offset..][0..self.subspace_dim];

            for (0..self.num_centroids) |c| {
                const c_offset = s * self.num_centroids * self.subspace_dim + c * self.subspace_dim;
                const centroid = self.centroids[c_offset..][0..self.subspace_dim];

                const dist = computeL2DistanceSquared(query_subvec, centroid);
                self.distance_tables.?[s * self.num_centroids + c] = dist;
            }
        }
    }

    /// Compute asymmetric distance using precomputed table
    pub fn computeAsymmetricDistance(self: *const PQCodebook, codes: []const u8) f32 {
        var total_dist: f32 = 0;

        for (0..self.num_subspaces) |s| {
            const code = codes[s];
            total_dist += self.distance_tables.?[s * self.num_centroids + code];
        }

        return total_dist;
    }
};
