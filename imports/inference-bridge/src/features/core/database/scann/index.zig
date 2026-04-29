//! ScaNN index for scalable approximate nearest neighbor search.

const std = @import("std");
const index_mod = @import("../index.zig");
const types = @import("types.zig");
const codebook_mod = @import("codebook.zig");

const ScaNNConfig = types.ScaNNConfig;
const QuantizationType = types.QuantizationType;
const ScalarQuantParams = types.ScalarQuantParams;
const Partition = types.Partition;
const SearchCandidate = types.SearchCandidate;
const IndexStats = types.IndexStats;
const AVQCodebook = codebook_mod.AVQCodebook;
const computeL2DistanceSquared = types.computeL2DistanceSquared;

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
        for (self.partitions.items) |*p| {
            p.deinit(self.allocator);
        }
        self.partitions.deinit(self.allocator);

        if (self.partition_centroids) |pc| {
            self.allocator.free(pc);
        }

        if (self.scalar_params) |sp| {
            self.allocator.free(sp);
        }

        if (self.avq_codebook) |*cb| {
            cb.deinit();
        }

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

        self.partition_centroids = try self.allocator.alloc(f32, num_partitions * dim);

        for (0..num_partitions) |i| {
            const partition = try Partition.init(self.allocator, @intCast(i), dim);
            try self.partitions.append(self.allocator, partition);
        }

        try self.initializePartitionCentroids(vectors);
        try self.runPartitionKMeans(vectors, 20);
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
            for (0..num_vectors) |i| {
                const vec = vectors[i * dim ..][0..dim];
                const prev = self.partition_centroids.?[(p - 1) * dim ..][0..dim];
                const dist = computeL2DistanceSquared(vec, prev);
                min_dists[i] = @min(min_dists[i], dist);
            }

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
                        const codes = try self.allocator.alloc(u8, 4);
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
        defer results.deinit(self.allocator);

        for (candidates.items[0..rerank_count]) |candidate| {
            const vec = self.vectors.items[candidate.id];
            const exact_dist = computeL2DistanceSquared(query, vec);

            try results.append(self.allocator, .{
                .id = @as(u64, candidate.id),
                .score = @sqrt(exact_dist),
            });
        }

        // Sort by exact distance
        std.mem.sort(index_mod.IndexResult, results.items, {}, struct {
            fn lessThan(_: void, a: index_mod.IndexResult, b: index_mod.IndexResult) bool {
                return a.score < b.score;
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

        total += @as(u64, self.num_vectors) * dim * @sizeOf(f32);
        total += @as(u64, self.config.num_partitions) * dim * @sizeOf(f32);

        total += switch (self.config.quantization_type) {
            .scalar => @as(u64, self.num_vectors) * dim,
            .avq => @as(u64, self.num_vectors) * 4,
            else => 0,
        };

        return total;
    }
};
