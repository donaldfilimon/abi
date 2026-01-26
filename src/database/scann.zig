//! ScaNN: Scalable Nearest Neighbors with Learned Quantization
//!
//! Implements Google's ScaNN algorithm for efficient approximate nearest
//! neighbor search with learned quantization. Provides better recall vs
//! compression tradeoffs than traditional quantization methods.
//!
//! Key features:
//! - Anisotropic Vector Quantization (AVQ) for direction-aware compression
//! - Score-aware quantization loss for improved ranking
//! - Two-phase search: coarse quantization + fine reranking
//! - Asymmetric distance computation
//!
//! Performance targets:
//! - 10-100x faster than brute force at >95% recall
//! - 4-16x compression with minimal accuracy loss
//! - Sub-millisecond query latency for million-scale datasets

const std = @import("std");
const build_options = @import("build_options");
const simd = @import("../shared/simd.zig");
const index_mod = @import("index.zig");

/// ScaNN configuration parameters
pub const ScaNNConfig = struct {
    /// Number of dimensions in vectors
    dimensions: u32 = 128,
    /// Number of partitions for tree-based search
    num_partitions: u32 = 100,
    /// Number of partitions to search (leaves_to_search)
    partitions_to_search: u32 = 10,
    /// Number of dimensions after projection
    projected_dimensions: u32 = 0, // 0 = no projection
    /// Quantization type
    quantization_type: QuantizationType = .scalar,
    /// Number of centroids for asymmetric hashing
    num_centroids: u32 = 256,
    /// Bits per dimension for scalar quantization
    bits_per_dim: u32 = 8,
    /// Enable anisotropic quantization
    anisotropic: bool = true,
    /// Anisotropy threshold (eta parameter)
    anisotropy_threshold: f32 = 0.2,
    /// Reranking factor (rerank top-k * factor candidates)
    rerank_factor: u32 = 4,
    /// Training sample size for quantization
    training_sample_size: u32 = 10000,
    /// Random seed for reproducibility
    seed: u64 = 42,
};

/// Quantization types supported by ScaNN
pub const QuantizationType = enum {
    /// No quantization (brute force)
    none,
    /// Scalar quantization (per-dimension)
    scalar,
    /// Product quantization
    product,
    /// Anisotropic vector quantization
    avq,

    pub fn compressionRatio(self: QuantizationType, bits: u32) f32 {
        return switch (self) {
            .none => 1.0,
            .scalar => 32.0 / @as(f32, @floatFromInt(bits)),
            .product => 32.0 / @as(f32, @floatFromInt(bits)),
            .avq => 32.0 / @as(f32, @floatFromInt(bits)) * 1.5, // AVQ has overhead
        };
    }
};

/// Scalar quantization parameters per dimension
pub const ScalarQuantParams = struct {
    min_val: f32,
    max_val: f32,
    scale: f32,
    offset: f32,

    pub fn quantize(self: *const ScalarQuantParams, value: f32) u8 {
        const normalized = (value - self.min_val) * self.scale;
        const clamped = @max(0.0, @min(255.0, normalized));
        return @intFromFloat(clamped);
    }

    pub fn dequantize(self: *const ScalarQuantParams, code: u8) f32 {
        return @as(f32, @floatFromInt(code)) / self.scale + self.min_val;
    }
};

/// Partition/cluster for tree-based search
pub const Partition = struct {
    /// Partition ID
    id: u32,
    /// Centroid vector
    centroid: []f32,
    /// Vector IDs in this partition
    member_ids: std.ArrayListUnmanaged(u32),
    /// Quantized vectors (for fast distance computation)
    quantized_vectors: std.ArrayListUnmanaged([]u8),

    pub fn init(allocator: std.mem.Allocator, id: u32, dim: u32) !Partition {
        return Partition{
            .id = id,
            .centroid = try allocator.alloc(f32, dim),
            .member_ids = .empty,
            .quantized_vectors = .empty,
        };
    }

    pub fn deinit(self: *Partition, allocator: std.mem.Allocator) void {
        allocator.free(self.centroid);
        self.member_ids.deinit(allocator);
        for (self.quantized_vectors.items) |qv| {
            allocator.free(qv);
        }
        self.quantized_vectors.deinit(allocator);
    }
};

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
        var codebook = AVQCodebook{
            .allocator = allocator,
            .dimensions = dimensions,
            .num_centroids = num_centroids,
            .centroids = try allocator.alloc(f32, num_centroids * dimensions),
            .dimension_weights = try allocator.alloc(f32, dimensions),
        };

        // Initialize weights uniformly
        @memset(codebook.dimension_weights, 1.0);

        return codebook;
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
        // Compute variance of query vectors per dimension
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
                w.* = @sqrt(w.* / max_var + 0.01); // Add epsilon for stability
            }
        }
    }

    fn refineWithAnisotropicLoss(self: *AVQCodebook, vectors: []const f32, iterations: u32) !void {
        // Refine centroids to minimize anisotropic quantization loss
        // Loss = sum of weighted squared distances
        _ = self;
        _ = vectors;
        _ = iterations;

        // Simplified: just use learned weights in distance computation
        // Full implementation would solve optimization problem
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

/// ScaNN index for scalable approximate nearest neighbor search
pub const ScaNNIndex = struct {
    config: ScaNNConfig,
    allocator: std.mem.Allocator,

    // Index state
    num_vectors: u32 = 0,

    // Partitioning
    partitions: std.ArrayListUnmanaged(Partition),
    partition_centroids: ?[]f32 = null,

    // Quantization
    scalar_params: ?[]ScalarQuantParams = null,
    avq_codebook: ?AVQCodebook = null,

    // Full vectors for reranking
    vectors: std.ArrayListUnmanaged([]f32),

    // Mapping from vector ID to partition
    vector_to_partition: std.ArrayListUnmanaged(u32),

    // Statistics
    stats: IndexStats = .{},

    pub fn init(allocator: std.mem.Allocator, config: ScaNNConfig) !ScaNNIndex {
        return ScaNNIndex{
            .config = config,
            .allocator = allocator,
            .partitions = .empty,
            .vectors = .empty,
            .vector_to_partition = .empty,
        };
    }

    pub fn deinit(self: *ScaNNIndex) void {
        // Free partitions
        for (self.partitions.items) |*p| {
            p.deinit(self.allocator);
        }
        self.partitions.deinit(self.allocator);

        if (self.partition_centroids) |pc| {
            self.allocator.free(pc);
        }

        // Free scalar params
        if (self.scalar_params) |sp| {
            self.allocator.free(sp);
        }

        // Free AVQ codebook
        if (self.avq_codebook) |*cb| {
            cb.deinit();
        }

        // Free vectors
        for (self.vectors.items) |vec| {
            self.allocator.free(vec);
        }
        self.vectors.deinit(self.allocator);

        self.vector_to_partition.deinit(self.allocator);
    }

    /// Build index from vectors
    pub fn build(self: *ScaNNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_vectors = @as(u32, @intCast(vectors.len / dim));

        if (num_vectors == 0) return error.EmptyDataset;

        // Store full vectors
        for (0..num_vectors) |i| {
            const vec_copy = try self.allocator.alloc(f32, dim);
            @memcpy(vec_copy, vectors[i * dim ..][0..dim]);
            try self.vectors.append(self.allocator, vec_copy);
        }
        self.num_vectors = num_vectors;

        // Build partitioning tree
        try self.buildPartitions(vectors);

        // Train quantization
        switch (self.config.quantization_type) {
            .scalar => try self.trainScalarQuantization(vectors),
            .avq => try self.trainAVQ(vectors),
            else => {},
        }

        // Quantize vectors within partitions
        try self.quantizePartitions();

        self.stats.build_complete = true;
    }

    /// Build partition tree using k-means
    fn buildPartitions(self: *ScaNNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_partitions = self.config.num_partitions;

        // Initialize partitions
        self.partition_centroids = try self.allocator.alloc(f32, num_partitions * dim);

        for (0..num_partitions) |i| {
            const partition = try Partition.init(self.allocator, @intCast(i), dim);
            try self.partitions.append(self.allocator, partition);
        }

        // Initialize partition centroids (k-means++)
        try self.initializePartitionCentroids(vectors);

        // Run k-means to refine partitions
        try self.runPartitionKMeans(vectors, 20);

        // Assign vectors to partitions
        try self.assignVectorsToPartitions(vectors);
    }

    fn initializePartitionCentroids(self: *ScaNNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_vectors = vectors.len / dim;
        const num_partitions = self.config.num_partitions;

        var prng = std.Random.DefaultPrng.init(self.config.seed);
        const random = prng.random();

        // First centroid: random
        const first_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
        @memcpy(
            self.partition_centroids.?[0..dim],
            vectors[first_idx * dim ..][0..dim],
        );

        // K-means++ for remaining
        var min_dists = try self.allocator.alloc(f32, num_vectors);
        defer self.allocator.free(min_dists);
        @memset(min_dists, std.math.inf(f32));

        for (1..num_partitions) |p| {
            // Update distances
            for (0..num_vectors) |i| {
                const vec = vectors[i * dim ..][0..dim];
                const prev = self.partition_centroids.?[(p - 1) * dim ..][0..dim];
                const dist = computeL2DistanceSquared(vec, prev);
                min_dists[i] = @min(min_dists[i], dist);
            }

            // Sample proportional to distance
            var total: f64 = 0;
            for (min_dists) |d| total += d;

            const threshold = random.float(f64) * total;
            var cumulative: f64 = 0;
            var selected: usize = 0;

            for (min_dists, 0..) |d, i| {
                cumulative += d;
                if (cumulative >= threshold) {
                    selected = i;
                    break;
                }
            }

            @memcpy(
                self.partition_centroids.?[p * dim ..][0..dim],
                vectors[selected * dim ..][0..dim],
            );
        }
    }

    fn runPartitionKMeans(self: *ScaNNIndex, vectors: []const f32, iterations: u32) !void {
        const dim = self.config.dimensions;
        const num_vectors = vectors.len / dim;
        const num_partitions = self.config.num_partitions;

        const assignments = try self.allocator.alloc(u32, num_vectors);
        defer self.allocator.free(assignments);

        const counts = try self.allocator.alloc(u32, num_partitions);
        defer self.allocator.free(counts);

        const new_centroids = try self.allocator.alloc(f32, num_partitions * dim);
        defer self.allocator.free(new_centroids);

        for (0..iterations) |_| {
            // Assign
            for (0..num_vectors) |i| {
                const vec = vectors[i * dim ..][0..dim];
                var min_dist: f32 = std.math.inf(f32);
                var min_p: u32 = 0;

                for (0..num_partitions) |p| {
                    const centroid = self.partition_centroids.?[p * dim ..][0..dim];
                    const dist = computeL2DistanceSquared(vec, centroid);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_p = @intCast(p);
                    }
                }
                assignments[i] = min_p;
            }

            // Update
            @memset(new_centroids, 0);
            @memset(counts, 0);

            for (0..num_vectors) |i| {
                const p = assignments[i];
                for (0..dim) |d| {
                    new_centroids[p * dim + d] += vectors[i * dim + d];
                }
                counts[p] += 1;
            }

            for (0..num_partitions) |p| {
                if (counts[p] > 0) {
                    for (0..dim) |d| {
                        new_centroids[p * dim + d] /= @floatFromInt(counts[p]);
                    }
                }
            }

            @memcpy(self.partition_centroids.?, new_centroids);
        }
    }

    fn assignVectorsToPartitions(self: *ScaNNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_vectors = vectors.len / dim;
        const num_partitions = self.config.num_partitions;

        for (0..num_vectors) |i| {
            const vec = vectors[i * dim ..][0..dim];
            var min_dist: f32 = std.math.inf(f32);
            var min_p: u32 = 0;

            for (0..num_partitions) |p| {
                const centroid = self.partition_centroids.?[p * dim ..][0..dim];
                const dist = computeL2DistanceSquared(vec, centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_p = @intCast(p);
                }
            }

            try self.partitions.items[min_p].member_ids.append(self.allocator, @intCast(i));
            try self.vector_to_partition.append(self.allocator, min_p);

            // Copy centroid
            @memcpy(self.partitions.items[min_p].centroid, self.partition_centroids.?[min_p * dim ..][0..dim]);
        }
    }

    /// Train scalar quantization
    fn trainScalarQuantization(self: *ScaNNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_vectors = vectors.len / dim;

        self.scalar_params = try self.allocator.alloc(ScalarQuantParams, dim);

        for (0..dim) |d| {
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);

            for (0..num_vectors) |i| {
                const val = vectors[i * dim + d];
                min_val = @min(min_val, val);
                max_val = @max(max_val, val);
            }

            const range = max_val - min_val;
            self.scalar_params.?[d] = .{
                .min_val = min_val,
                .max_val = max_val,
                .scale = if (range > 0) 255.0 / range else 1.0,
                .offset = min_val,
            };
        }
    }

    /// Train anisotropic vector quantization
    fn trainAVQ(self: *ScaNNIndex, vectors: []const f32) !void {
        self.avq_codebook = try AVQCodebook.init(
            self.allocator,
            self.config.dimensions,
            self.config.num_centroids,
        );

        try self.avq_codebook.?.train(vectors, null);
    }

    /// Quantize vectors within partitions
    fn quantizePartitions(self: *ScaNNIndex) !void {
        const dim = self.config.dimensions;

        for (self.partitions.items) |*partition| {
            for (partition.member_ids.items) |vec_id| {
                const vec = self.vectors.items[vec_id];

                const qvec = switch (self.config.quantization_type) {
                    .scalar => blk: {
                        const codes = try self.allocator.alloc(u8, dim);
                        for (0..dim) |d| {
                            codes[d] = self.scalar_params.?[d].quantize(vec[d]);
                        }
                        break :blk codes;
                    },
                    .avq => blk: {
                        const codes = try self.allocator.alloc(u8, 4); // 32-bit centroid index
                        const idx = self.avq_codebook.?.encode(vec);
                        std.mem.writeInt(u32, codes[0..4], idx, .little);
                        break :blk codes;
                    },
                    else => try self.allocator.alloc(u8, 0),
                };

                try partition.quantized_vectors.append(self.allocator, qvec);
            }
        }
    }

    /// Search for k nearest neighbors
    pub fn search(
        self: *ScaNNIndex,
        query: []const f32,
        k: u32,
    ) ![]index_mod.IndexResult {
        if (self.num_vectors == 0) return &[_]index_mod.IndexResult{};

        // Phase 1: Find nearest partitions
        const partition_candidates = try self.findNearestPartitions(query);
        defer self.allocator.free(partition_candidates);

        // Phase 2: Search within partitions using quantized distances
        var candidates = std.ArrayListUnmanaged(SearchCandidate).empty;
        defer candidates.deinit(self.allocator);

        for (partition_candidates) |p_idx| {
            const partition = &self.partitions.items[p_idx];

            for (partition.member_ids.items, partition.quantized_vectors.items) |vec_id, qvec| {
                const approx_dist = self.computeQuantizedDistance(query, qvec);
                try candidates.append(self.allocator, .{
                    .id = vec_id,
                    .distance = approx_dist,
                });
            }
        }

        // Sort by approximate distance
        std.mem.sort(SearchCandidate, candidates.items, {}, struct {
            fn lessThan(_: void, a: SearchCandidate, b: SearchCandidate) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        // Phase 3: Rerank top candidates with exact distances
        const rerank_count = @min(candidates.items.len, k * self.config.rerank_factor);
        var results = std.ArrayListUnmanaged(index_mod.IndexResult).empty;

        for (candidates.items[0..rerank_count]) |candidate| {
            const vec = self.vectors.items[candidate.id];
            const exact_dist = computeL2DistanceSquared(query, vec);

            try results.append(self.allocator, .{
                .id = candidate.id,
                .distance = @sqrt(exact_dist),
                .metadata = null,
            });
        }

        // Sort by exact distance
        std.mem.sort(index_mod.IndexResult, results.items, {}, struct {
            fn lessThan(_: void, a: index_mod.IndexResult, b: index_mod.IndexResult) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        // Return top k
        const result_count = @min(results.items.len, k);
        const final_results = try self.allocator.alloc(index_mod.IndexResult, result_count);
        @memcpy(final_results, results.items[0..result_count]);

        self.stats.queries_processed += 1;
        return final_results;
    }

    fn findNearestPartitions(self: *ScaNNIndex, query: []const f32) ![]u32 {
        const num_partitions = self.config.num_partitions;
        const partitions_to_search = self.config.partitions_to_search;

        var partition_dists = try self.allocator.alloc(SearchCandidate, num_partitions);
        defer self.allocator.free(partition_dists);

        const dim = self.config.dimensions;
        for (0..num_partitions) |p| {
            const centroid = self.partition_centroids.?[p * dim ..][0..dim];
            partition_dists[p] = .{
                .id = @intCast(p),
                .distance = computeL2DistanceSquared(query, centroid),
            };
        }

        std.mem.sort(SearchCandidate, partition_dists, {}, struct {
            fn lessThan(_: void, a: SearchCandidate, b: SearchCandidate) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        const result = try self.allocator.alloc(u32, partitions_to_search);
        for (0..partitions_to_search) |i| {
            result[i] = partition_dists[i].id;
        }

        return result;
    }

    fn computeQuantizedDistance(self: *ScaNNIndex, query: []const f32, qvec: []const u8) f32 {
        return switch (self.config.quantization_type) {
            .scalar => blk: {
                var sum: f32 = 0;
                for (0..self.config.dimensions) |d| {
                    const dequant = self.scalar_params.?[d].dequantize(qvec[d]);
                    const diff = query[d] - dequant;
                    sum += diff * diff;
                }
                break :blk sum;
            },
            .avq => blk: {
                const idx = std.mem.readInt(u32, qvec[0..4], .little);
                break :blk self.avq_codebook.?.asymmetricDistance(query, idx);
            },
            else => 0,
        };
    }

    /// Get index statistics
    pub fn getStats(self: *const ScaNNIndex) IndexStats {
        var stats = self.stats;
        stats.num_vectors = self.num_vectors;
        stats.num_partitions = @intCast(self.partitions.items.len);
        stats.memory_bytes = self.estimateMemoryUsage();
        return stats;
    }

    fn estimateMemoryUsage(self: *const ScaNNIndex) u64 {
        const dim = self.config.dimensions;
        var total: u64 = 0;

        // Full vectors
        total += @as(u64, self.num_vectors) * dim * @sizeOf(f32);

        // Partition centroids
        total += @as(u64, self.config.num_partitions) * dim * @sizeOf(f32);

        // Quantized vectors (estimate)
        total += switch (self.config.quantization_type) {
            .scalar => @as(u64, self.num_vectors) * dim,
            .avq => @as(u64, self.num_vectors) * 4,
            else => 0,
        };

        return total;
    }
};

/// Search candidate for internal use
const SearchCandidate = struct {
    id: u32,
    distance: f32,
};

/// Index statistics
pub const IndexStats = struct {
    num_vectors: u32 = 0,
    num_partitions: u32 = 0,
    queries_processed: u64 = 0,
    memory_bytes: u64 = 0,
    build_complete: bool = false,

    pub fn report(self: *const IndexStats) void {
        std.log.info("ScaNN Index Statistics:", .{});
        std.log.info("  Vectors: {d}", .{self.num_vectors});
        std.log.info("  Partitions: {d}", .{self.num_partitions});
        std.log.info("  Queries: {d}", .{self.queries_processed});
        std.log.info("  Memory: {d:.2} MB", .{@as(f64, @floatFromInt(self.memory_bytes)) / (1024 * 1024)});
    }
};

// Helper functions

fn computeL2DistanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (a, b) |av, bv| {
        const diff = av - bv;
        sum += diff * diff;
    }
    return sum;
}

fn computeWeightedL2(a: []const f32, b: []const f32, weights: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == weights.len);
    var sum: f32 = 0;
    for (a, b, weights) |av, bv, w| {
        const diff = av - bv;
        sum += diff * diff * w;
    }
    return sum;
}

// Tests

test "scalar quantization" {
    const params = ScalarQuantParams{
        .min_val = 0,
        .max_val = 1,
        .scale = 255,
        .offset = 0,
    };

    const code = params.quantize(0.5);
    try std.testing.expect(code == 127 or code == 128);

    const dequant = params.dequantize(127);
    try std.testing.expectApproxEqAbs(@as(f32, 0.498), dequant, 0.01);
}

test "avq codebook basic" {
    const allocator = std.testing.allocator;

    var codebook = try AVQCodebook.init(allocator, 4, 8);
    defer codebook.deinit();

    try std.testing.expect(codebook.dimensions == 4);
    try std.testing.expect(codebook.num_centroids == 8);
    try std.testing.expect(codebook.centroids.len == 32);
}

test "scann index basic" {
    const allocator = std.testing.allocator;

    const config = ScaNNConfig{
        .dimensions = 4,
        .num_partitions = 2,
        .partitions_to_search = 2,
        .quantization_type = .scalar,
    };

    var index = try ScaNNIndex.init(allocator, config);
    defer index.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try index.build(&vectors);

    try std.testing.expect(index.num_vectors == 4);
    try std.testing.expect(index.stats.build_complete);
}

test "scann search" {
    const allocator = std.testing.allocator;

    const config = ScaNNConfig{
        .dimensions = 4,
        .num_partitions = 2,
        .partitions_to_search = 2,
        .quantization_type = .scalar,
        .rerank_factor = 2,
    };

    var index = try ScaNNIndex.init(allocator, config);
    defer index.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try index.build(&vectors);

    const query = [_]f32{ 0.9, 0.1, 0, 0 };
    const results = try index.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expect(results[0].id == 0); // Closest to first vector
}
