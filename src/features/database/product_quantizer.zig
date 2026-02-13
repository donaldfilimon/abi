//! Product Quantizer for high compression ratios.
//!
//! Divides vectors into M subvectors, clusters each subspace with K centroids,
//! and stores only centroid IDs. Achieves 97% compression (768-dim f32 -> ~96 bytes).
//!
//! Compression: M subvectors * ceil(log2(K)) bits per subvector
//! Typical config: M=8, K=256 -> 8 bytes per vector (vs 3072 bytes for 768-dim f32)

const std = @import("std");
const quantization = @import("quantization.zig");
const QuantizationError = quantization.QuantizationError;
const computeL2DistanceSimd = quantization.computeL2DistanceSimd;
const packBits = quantization.packBits;
const unpackBits = quantization.unpackBits;

pub const ProductQuantizer = struct {
    dim: usize,
    num_subvectors: u8,
    bits_per_code: u8,
    subvector_dim: usize,
    codebooks: [][]f32, // [num_subvectors][num_centroids * subvector_dim]
    num_centroids: usize,
    trained: bool,
    allocator: std.mem.Allocator,

    /// Configuration for product quantization.
    pub const Config = struct {
        /// Number of subvectors (M). Must divide dimension evenly.
        num_subvectors: u8 = 8,
        /// Bits per code (log2 of number of centroids). Typically 8 (256 centroids).
        bits_per_code: u8 = 8,
        /// Number of k-means iterations for training.
        kmeans_iterations: u8 = 25,
    };

    /// Statistics about the quantizer.
    pub const Stats = struct {
        dimension: usize,
        num_subvectors: u8,
        bits_per_code: u8,
        num_centroids: usize,
        compression_ratio: f32,
        bytes_per_vector: usize,
    };

    /// Initialize a product quantizer.
    pub fn init(allocator: std.mem.Allocator, dim: usize, config: Config) QuantizationError!ProductQuantizer {
        if (dim == 0) return QuantizationError.InvalidDimension;
        if (config.num_subvectors == 0) return QuantizationError.InvalidSubvectors;
        if (dim % config.num_subvectors != 0) return QuantizationError.InvalidSubvectors;
        if (config.bits_per_code == 0 or config.bits_per_code > 16) {
            return QuantizationError.InvalidBits;
        }

        const subvector_dim = dim / config.num_subvectors;
        const num_centroids = @as(usize, 1) << @as(u4, @intCast(config.bits_per_code));

        // Allocate codebooks
        const codebooks = allocator.alloc([]f32, config.num_subvectors) catch return QuantizationError.OutOfMemory;
        errdefer allocator.free(codebooks);

        var initialized: usize = 0;
        errdefer {
            for (codebooks[0..initialized]) |cb| {
                allocator.free(cb);
            }
        }

        for (codebooks) |*cb| {
            cb.* = allocator.alloc(f32, num_centroids * subvector_dim) catch return QuantizationError.OutOfMemory;
            @memset(cb.*, 0);
            initialized += 1;
        }

        return .{
            .dim = dim,
            .num_subvectors = config.num_subvectors,
            .bits_per_code = config.bits_per_code,
            .subvector_dim = subvector_dim,
            .codebooks = codebooks,
            .num_centroids = num_centroids,
            .trained = false,
            .allocator = allocator,
        };
    }

    /// Release resources.
    pub fn deinit(self: *ProductQuantizer) void {
        for (self.codebooks) |cb| {
            self.allocator.free(cb);
        }
        self.allocator.free(self.codebooks);
        self.* = undefined;
    }

    /// Train the quantizer using k-means clustering on each subspace.
    pub fn train(self: *ProductQuantizer, vectors: []const []const f32, config: Config) QuantizationError!void {
        if (vectors.len == 0) return QuantizationError.EmptyData;

        // Validate dimensions
        for (vectors) |v| {
            if (v.len != self.dim) return QuantizationError.DimensionMismatch;
        }

        const sample_size = @min(vectors.len, 10000); // Limit training samples

        // Train each subspace independently
        for (0..self.num_subvectors) |m| {
            try self.trainSubspace(vectors[0..sample_size], m, config.kmeans_iterations);
        }

        self.trained = true;
    }

    /// Train a single subspace using k-means.
    fn trainSubspace(self: *ProductQuantizer, vectors: []const []const f32, subspace_idx: usize, iterations: u8) QuantizationError!void {
        const offset = subspace_idx * self.subvector_dim;
        const codebook = self.codebooks[subspace_idx];

        // Initialize centroids from random samples
        var prng = std.Random.DefaultPrng.init(@as(u64, subspace_idx) * 12345);
        const random = prng.random();

        for (0..self.num_centroids) |k| {
            const sample_idx = random.intRangeLessThan(usize, 0, vectors.len);
            const src = vectors[sample_idx][offset..][0..self.subvector_dim];
            const dst = codebook[k * self.subvector_dim ..][0..self.subvector_dim];
            @memcpy(dst, src);
        }

        // K-means iterations
        const assignments = self.allocator.alloc(usize, vectors.len) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(assignments);

        const counts = self.allocator.alloc(usize, self.num_centroids) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(counts);

        const sums = self.allocator.alloc(f32, self.num_centroids * self.subvector_dim) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(sums);

        var iter: u8 = 0;
        while (iter < iterations) : (iter += 1) {
            // Assign vectors to nearest centroids
            for (vectors, 0..) |vector, i| {
                const subvec = vector[offset..][0..self.subvector_dim];
                assignments[i] = self.findNearestCentroid(subvec, codebook);
            }

            // Update centroids
            @memset(counts, 0);
            @memset(sums, 0);

            for (vectors, assignments) |vector, cluster_id| {
                counts[cluster_id] += 1;
                const subvec = vector[offset..][0..self.subvector_dim];
                const sum_offset = cluster_id * self.subvector_dim;
                for (subvec, 0..) |value, j| {
                    sums[sum_offset + j] += value;
                }
            }

            // Compute new centroids
            for (0..self.num_centroids) |k| {
                if (counts[k] == 0) continue;
                const centroid = codebook[k * self.subvector_dim ..][0..self.subvector_dim];
                const sum_slice = sums[k * self.subvector_dim ..][0..self.subvector_dim];
                const count_f: f32 = @floatFromInt(counts[k]);
                for (centroid, sum_slice) |*c, s| {
                    c.* = s / count_f;
                }
            }
        }
    }

    /// Find the nearest centroid for a subvector.
    /// Uses SIMD-accelerated L2 distance computation.
    fn findNearestCentroid(self: *const ProductQuantizer, subvec: []const f32, codebook: []const f32) usize {
        var best_idx: usize = 0;
        var best_dist: f32 = std.math.inf(f32);

        const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

        for (0..self.num_centroids) |k| {
            const centroid = codebook[k * self.subvector_dim ..][0..self.subvector_dim];
            var dist: f32 = 0.0;

            // SIMD-accelerated L2 distance
            if (VectorSize > 1 and self.subvector_dim >= VectorSize) {
                const Vec = @Vector(VectorSize, f32);
                var vec_sum: Vec = @splat(0.0);
                var i: usize = 0;

                while (i + VectorSize <= self.subvector_dim) : (i += VectorSize) {
                    const sv: Vec = subvec[i..][0..VectorSize].*;
                    const cv: Vec = centroid[i..][0..VectorSize].*;
                    const diff = sv - cv;
                    vec_sum += diff * diff;
                }

                // Horizontal sum using @reduce
                dist = @reduce(.Add, vec_sum);

                // Scalar remainder
                while (i < self.subvector_dim) : (i += 1) {
                    const diff = subvec[i] - centroid[i];
                    dist += diff * diff;
                }
            } else {
                // Scalar fallback
                for (subvec, centroid) |s, c| {
                    const diff = s - c;
                    dist += diff * diff;
                }
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = k;
            }
        }

        return best_idx;
    }

    /// Encode a vector to PQ codes.
    pub fn encode(self: *const ProductQuantizer, vector: []const f32, output: []u8) QuantizationError!usize {
        if (!self.trained) return QuantizationError.NotTrained;
        if (vector.len != self.dim) return QuantizationError.DimensionMismatch;

        const bytes_needed = self.bytesPerVector();
        if (output.len < bytes_needed) return QuantizationError.OutOfMemory;

        // For 8-bit codes, one byte per subvector
        if (self.bits_per_code == 8) {
            for (0..self.num_subvectors) |m| {
                const off = m * self.subvector_dim;
                const subvec = vector[off..][0..self.subvector_dim];
                output[m] = @intCast(self.findNearestCentroid(subvec, self.codebooks[m]));
            }
        } else {
            // For other bit widths, pack codes
            var bit_offset: usize = 0;
            for (0..self.num_subvectors) |m| {
                const off = m * self.subvector_dim;
                const subvec = vector[off..][0..self.subvector_dim];
                const code = self.findNearestCentroid(subvec, self.codebooks[m]);
                packBits(output, bit_offset, @intCast(code), self.bits_per_code);
                bit_offset += self.bits_per_code;
            }
        }

        return bytes_needed;
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(self: *const ProductQuantizer, codes: []const u8, output: []f32) QuantizationError!void {
        if (!self.trained) return QuantizationError.NotTrained;
        if (output.len != self.dim) return QuantizationError.DimensionMismatch;

        if (self.bits_per_code == 8) {
            for (0..self.num_subvectors) |m| {
                const code = codes[m];
                const centroid = self.codebooks[m][code * self.subvector_dim ..][0..self.subvector_dim];
                const out_offset = m * self.subvector_dim;
                @memcpy(output[out_offset..][0..self.subvector_dim], centroid);
            }
        } else {
            var bit_offset: usize = 0;
            for (0..self.num_subvectors) |m| {
                const code = unpackBits(codes, bit_offset, self.bits_per_code);
                bit_offset += self.bits_per_code;
                const centroid = self.codebooks[m][@as(usize, code) * self.subvector_dim ..][0..self.subvector_dim];
                const out_offset = m * self.subvector_dim;
                @memcpy(output[out_offset..][0..self.subvector_dim], centroid);
            }
        }
    }

    /// Compute asymmetric distance using precomputed distance tables.
    /// This is the fast path for PQ search.
    pub fn asymmetricDistanceWithTable(
        self: *const ProductQuantizer,
        distance_table: []const f32,
        codes: []const u8,
    ) f32 {
        var dist: f32 = 0.0;

        if (self.bits_per_code == 8) {
            for (0..self.num_subvectors) |m| {
                const code = codes[m];
                dist += distance_table[m * self.num_centroids + code];
            }
        } else {
            var bit_offset: usize = 0;
            for (0..self.num_subvectors) |m| {
                const code = unpackBits(codes, bit_offset, self.bits_per_code);
                bit_offset += self.bits_per_code;
                dist += distance_table[m * self.num_centroids + code];
            }
        }

        return dist;
    }

    /// Batch compute asymmetric distances using SIMD acceleration.
    /// Processes multiple codes at once for better throughput.
    pub fn batchAsymmetricDistance(
        self: *const ProductQuantizer,
        distance_table: []const f32,
        codes_batch: []const []const u8,
        distances: []f32,
    ) void {
        std.debug.assert(codes_batch.len == distances.len);

        // SIMD-friendly batch processing
        const VectorSize = 8; // Process 8 at a time for better SIMD utilization
        const batch_count = codes_batch.len / VectorSize;
        const remainder = codes_batch.len % VectorSize;

        // Process in batches of VectorSize
        var batch_idx: usize = 0;
        while (batch_idx < batch_count) : (batch_idx += 1) {
            const start = batch_idx * VectorSize;
            var batch_dists: [VectorSize]f32 = .{0} ** VectorSize;

            // Process all subvectors for this batch
            if (self.bits_per_code == 8) {
                for (0..self.num_subvectors) |m| {
                    const table_offset = m * self.num_centroids;
                    inline for (0..VectorSize) |i| {
                        const code = codes_batch[start + i][m];
                        batch_dists[i] += distance_table[table_offset + code];
                    }
                }
            } else {
                for (0..self.num_subvectors) |m| {
                    const bit_off = m * self.bits_per_code;
                    const table_offset = m * self.num_centroids;
                    inline for (0..VectorSize) |i| {
                        const code = unpackBits(codes_batch[start + i], bit_off, self.bits_per_code);
                        batch_dists[i] += distance_table[table_offset + code];
                    }
                }
            }

            // Write results
            for (0..VectorSize) |i| {
                distances[start + i] = batch_dists[i];
            }
        }

        // Handle remainder
        const remainder_start = batch_count * VectorSize;
        for (0..remainder) |i| {
            distances[remainder_start + i] = self.asymmetricDistanceWithTable(
                distance_table,
                codes_batch[remainder_start + i],
            );
        }
    }

    /// Compute distance table with SIMD acceleration.
    /// Returns a table of size [num_subvectors * num_centroids].
    pub fn computeDistanceTableSimd(self: *const ProductQuantizer, query: []const f32) QuantizationError![]f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        const table = self.allocator.alloc(f32, self.num_subvectors * self.num_centroids) catch return QuantizationError.OutOfMemory;

        for (0..self.num_subvectors) |m| {
            const query_offset = m * self.subvector_dim;
            const query_subvec = query[query_offset..][0..self.subvector_dim];

            for (0..self.num_centroids) |k| {
                const centroid = self.codebooks[m][k * self.subvector_dim ..][0..self.subvector_dim];
                // Use SIMD for distance computation
                table[m * self.num_centroids + k] = computeL2DistanceSimd(query_subvec, centroid);
            }
        }

        return table;
    }

    /// Precompute distance table for a query.
    /// Returns a table of size [num_subvectors * num_centroids].
    pub fn computeDistanceTable(self: *const ProductQuantizer, query: []const f32) QuantizationError![]f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        const table = self.allocator.alloc(f32, self.num_subvectors * self.num_centroids) catch return QuantizationError.OutOfMemory;

        for (0..self.num_subvectors) |m| {
            const query_offset = m * self.subvector_dim;
            const query_subvec = query[query_offset..][0..self.subvector_dim];

            for (0..self.num_centroids) |k| {
                const centroid = self.codebooks[m][k * self.subvector_dim ..][0..self.subvector_dim];
                var dist: f32 = 0.0;
                for (query_subvec, centroid) |q, c| {
                    const diff = q - c;
                    dist += diff * diff;
                }
                table[m * self.num_centroids + k] = dist;
            }
        }

        return table;
    }

    /// Number of bytes needed to store one encoded vector.
    pub fn bytesPerVector(self: *const ProductQuantizer) usize {
        const total_bits = @as(usize, self.num_subvectors) * @as(usize, self.bits_per_code);
        return (total_bits + 7) / 8;
    }

    /// Get statistics about this quantizer.
    pub fn getStats(self: *const ProductQuantizer) Stats {
        const original_bytes = self.dim * @sizeOf(f32);
        const quantized_bytes = self.bytesPerVector();
        return .{
            .dimension = self.dim,
            .num_subvectors = self.num_subvectors,
            .bits_per_code = self.bits_per_code,
            .num_centroids = self.num_centroids,
            .compression_ratio = @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(quantized_bytes)),
            .bytes_per_vector = quantized_bytes,
        };
    }
};
