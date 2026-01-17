//! Vector Quantization Module
//!
//! Provides scalar and product quantization for compressing high-dimensional vectors
//! while maintaining search quality. Based on techniques from:
//! - Scalar Quantization: Simple min-max scaling to fixed-bit codes
//! - Product Quantization (PQ): Subvector clustering for 97% compression
//!
//! References:
//! - Pinecone PQ Guide: https://www.pinecone.io/learn/series/faiss/product-quantization/
//! - Milvus IVF_PQ: https://milvus.io/docs/ivf-pq.md
//! - Zilliz Quantization: https://zilliz.com/learn/scalar-quantization-and-product-quantization

const std = @import("std");
const simd = @import("../shared/simd.zig");

/// Errors that can occur during quantization operations.
pub const QuantizationError = error{
    InvalidDimension,
    InvalidBits,
    InvalidSubvectors,
    EmptyData,
    DimensionMismatch,
    NotTrained,
    OutOfMemory,
};

/// Scalar Quantizer using min-max scaling.
///
/// Compresses f32 vectors to fixed-bit codes (4, 8, or 16 bits).
/// Memory savings: 4-bit = 8x, 8-bit = 4x, 16-bit = 2x compression.
/// Typical recall loss: <1% for 8-bit quantization.
pub const ScalarQuantizer = struct {
    dim: usize,
    bits: u8,
    min_values: []f32,
    max_values: []f32,
    trained: bool,
    allocator: std.mem.Allocator,

    /// Configuration for scalar quantization.
    pub const Config = struct {
        /// Number of bits per dimension (4, 8, or 16).
        bits: u8 = 8,
    };

    /// Statistics about the quantizer.
    pub const Stats = struct {
        dimension: usize,
        bits: u8,
        compression_ratio: f32,
        bytes_per_vector: usize,
    };

    /// Initialize a scalar quantizer for the given dimension.
    pub fn init(allocator: std.mem.Allocator, dim: usize, config: Config) QuantizationError!ScalarQuantizer {
        if (dim == 0) return QuantizationError.InvalidDimension;
        if (config.bits != 4 and config.bits != 8 and config.bits != 16) {
            return QuantizationError.InvalidBits;
        }

        const min_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;
        errdefer allocator.free(min_values);
        const max_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;

        // Initialize to extreme values
        @memset(min_values, std.math.inf(f32));
        @memset(max_values, -std.math.inf(f32));

        return .{
            .dim = dim,
            .bits = config.bits,
            .min_values = min_values,
            .max_values = max_values,
            .trained = false,
            .allocator = allocator,
        };
    }

    /// Release resources.
    pub fn deinit(self: *ScalarQuantizer) void {
        self.allocator.free(self.min_values);
        self.allocator.free(self.max_values);
        self.* = undefined;
    }

    /// Train the quantizer on a set of vectors to compute min/max statistics.
    pub fn train(self: *ScalarQuantizer, vectors: []const []const f32) QuantizationError!void {
        if (vectors.len == 0) return QuantizationError.EmptyData;

        // Reset statistics
        @memset(self.min_values, std.math.inf(f32));
        @memset(self.max_values, -std.math.inf(f32));

        // Compute global min/max per dimension
        for (vectors) |vector| {
            if (vector.len != self.dim) return QuantizationError.DimensionMismatch;

            for (vector, 0..) |value, i| {
                if (value < self.min_values[i]) self.min_values[i] = value;
                if (value > self.max_values[i]) self.max_values[i] = value;
            }
        }

        self.trained = true;
    }

    /// Encode a vector to quantized codes.
    /// Returns the number of bytes written to output.
    pub fn encode(self: *const ScalarQuantizer, vector: []const f32, output: []u8) QuantizationError!usize {
        if (!self.trained) return QuantizationError.NotTrained;
        if (vector.len != self.dim) return QuantizationError.DimensionMismatch;

        const bytes_needed = self.bytesPerVector();
        if (output.len < bytes_needed) return QuantizationError.OutOfMemory;

        const num_levels = self.levels();

        switch (self.bits) {
            8 => {
                for (vector, 0..) |value, i| {
                    output[i] = quantizeValue(value, self.min_values[i], self.max_values[i], num_levels);
                }
            },
            4 => {
                // Pack two 4-bit values per byte
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const code1 = quantizeValue(vector[i], self.min_values[i], self.max_values[i], num_levels);
                    const code2 = if (i + 1 < self.dim)
                        quantizeValue(vector[i + 1], self.min_values[i + 1], self.max_values[i + 1], num_levels)
                    else
                        0;
                    output[byte_idx] = (code1 << 4) | code2;
                    byte_idx += 1;
                    i += 2;
                }
            },
            16 => {
                // Two bytes per dimension
                for (vector, 0..) |value, i| {
                    const code = quantizeValue16(value, self.min_values[i], self.max_values[i]);
                    output[i * 2] = @truncate(code >> 8);
                    output[i * 2 + 1] = @truncate(code);
                }
            },
            else => unreachable,
        }

        return bytes_needed;
    }

    /// Decode quantized codes back to a vector.
    pub fn decode(self: *const ScalarQuantizer, codes: []const u8, output: []f32) QuantizationError!void {
        if (!self.trained) return QuantizationError.NotTrained;
        if (output.len != self.dim) return QuantizationError.DimensionMismatch;

        const num_levels = self.levels();

        switch (self.bits) {
            8 => {
                for (output, 0..) |*value, i| {
                    value.* = dequantizeValue(codes[i], self.min_values[i], self.max_values[i], num_levels);
                }
            },
            4 => {
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const byte = codes[byte_idx];
                    output[i] = dequantizeValue(byte >> 4, self.min_values[i], self.max_values[i], num_levels);
                    if (i + 1 < self.dim) {
                        output[i + 1] = dequantizeValue(byte & 0x0F, self.min_values[i + 1], self.max_values[i + 1], num_levels);
                    }
                    byte_idx += 1;
                    i += 2;
                }
            },
            16 => {
                for (output, 0..) |*value, i| {
                    const code = (@as(u16, codes[i * 2]) << 8) | @as(u16, codes[i * 2 + 1]);
                    value.* = dequantizeValue16(code, self.min_values[i], self.max_values[i]);
                }
            },
            else => unreachable,
        }
    }

    /// Compute approximate distance between a query and encoded vector.
    /// Uses asymmetric distance computation (ADC) for better accuracy.
    pub fn asymmetricDistance(self: *const ScalarQuantizer, query: []const f32, codes: []const u8) QuantizationError!f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        // Decode and compute distance
        const decoded = self.allocator.alloc(f32, self.dim) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(decoded);

        try self.decode(codes, decoded);

        // Return L2 distance squared
        var dist: f32 = 0.0;
        for (query, decoded) |q, d| {
            const diff = q - d;
            dist += diff * diff;
        }
        return dist;
    }

    /// Compute cosine similarity between query and encoded vector.
    pub fn cosineSimilarity(self: *const ScalarQuantizer, query: []const f32, codes: []const u8) QuantizationError!f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        const decoded = self.allocator.alloc(f32, self.dim) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(decoded);

        try self.decode(codes, decoded);
        return simd.cosineSimilarity(query, decoded);
    }

    /// Number of bytes needed to store one encoded vector.
    pub fn bytesPerVector(self: *const ScalarQuantizer) usize {
        return switch (self.bits) {
            4 => (self.dim + 1) / 2,
            8 => self.dim,
            16 => self.dim * 2,
            else => unreachable,
        };
    }

    /// Get statistics about this quantizer.
    pub fn getStats(self: *const ScalarQuantizer) Stats {
        const original_bytes = self.dim * @sizeOf(f32);
        const quantized_bytes = self.bytesPerVector();
        return .{
            .dimension = self.dim,
            .bits = self.bits,
            .compression_ratio = @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(quantized_bytes)),
            .bytes_per_vector = quantized_bytes,
        };
    }

    fn levels(self: *const ScalarQuantizer) u16 {
        return (@as(u16, 1) << @as(u4, @intCast(self.bits))) - 1;
    }
};

/// Product Quantizer for high compression ratios.
///
/// Divides vectors into M subvectors, clusters each subspace with K centroids,
/// and stores only centroid IDs. Achieves 97% compression (768-dim f32 -> ~96 bytes).
///
/// Compression: M subvectors * ceil(log2(K)) bits per subvector
/// Typical config: M=8, K=256 -> 8 bytes per vector (vs 3072 bytes for 768-dim f32)
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
    fn findNearestCentroid(self: *const ProductQuantizer, subvec: []const f32, codebook: []const f32) usize {
        var best_idx: usize = 0;
        var best_dist: f32 = std.math.inf(f32);

        for (0..self.num_centroids) |k| {
            const centroid = codebook[k * self.subvector_dim ..][0..self.subvector_dim];
            var dist: f32 = 0.0;
            for (subvec, centroid) |s, c| {
                const diff = s - c;
                dist += diff * diff;
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
                const offset = m * self.subvector_dim;
                const subvec = vector[offset..][0..self.subvector_dim];
                output[m] = @intCast(self.findNearestCentroid(subvec, self.codebooks[m]));
            }
        } else {
            // For other bit widths, pack codes
            var bit_offset: usize = 0;
            for (0..self.num_subvectors) |m| {
                const offset = m * self.subvector_dim;
                const subvec = vector[offset..][0..self.subvector_dim];
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
                    const bit_offset = m * self.bits_per_code;
                    const table_offset = m * self.num_centroids;
                    inline for (0..VectorSize) |i| {
                        const code = unpackBits(codes_batch[start + i], bit_offset, self.bits_per_code);
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

// ============================================================================
// Helper functions
// ============================================================================

/// SIMD-accelerated L2 distance squared computation.
fn computeL2DistanceSimd(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

    var sum: f32 = 0.0;
    var i: usize = 0;

    if (VectorSize > 1 and a.len >= VectorSize) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            vec_sum += diff * diff;
        }

        // Horizontal sum
        const sums: [VectorSize]f32 = vec_sum;
        for (sums) |s| {
            sum += s;
        }
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

/// Thread-local decode buffer for avoiding allocation in hot path.
pub const DecodeBuffer = struct {
    buffer: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: usize) !DecodeBuffer {
        return .{
            .buffer = try allocator.alloc(f32, dim),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DecodeBuffer) void {
        self.allocator.free(self.buffer);
    }

    pub fn get(self: *DecodeBuffer) []f32 {
        return self.buffer;
    }
};

fn quantizeValue(value: f32, min_val: f32, max_val: f32, levels: u16) u8 {
    if (max_val <= min_val) return 0;
    const normalized = (value - min_val) / (max_val - min_val);
    const scaled = normalized * @as(f32, @floatFromInt(levels));
    const clamped = std.math.clamp(scaled, 0.0, @as(f32, @floatFromInt(levels)));
    return @intCast(@min(@as(u16, @intFromFloat(clamped)), levels));
}

fn quantizeValue16(value: f32, min_val: f32, max_val: f32) u16 {
    if (max_val <= min_val) return 0;
    const normalized = (value - min_val) / (max_val - min_val);
    const scaled = normalized * 65535.0;
    const clamped = std.math.clamp(scaled, 0.0, 65535.0);
    return @intFromFloat(clamped);
}

fn dequantizeValue(code: u8, min_val: f32, max_val: f32, levels: u16) f32 {
    if (max_val <= min_val or levels == 0) return min_val;
    const fraction = @as(f32, @floatFromInt(code)) / @as(f32, @floatFromInt(levels));
    return min_val + fraction * (max_val - min_val);
}

fn dequantizeValue16(code: u16, min_val: f32, max_val: f32) f32 {
    if (max_val <= min_val) return min_val;
    const fraction = @as(f32, @floatFromInt(code)) / 65535.0;
    return min_val + fraction * (max_val - min_val);
}

fn packBits(output: []u8, bit_offset: usize, value: u16, bits: u8) void {
    const byte_offset = bit_offset / 8;
    const bit_pos = @as(u4, @intCast(bit_offset % 8));

    if (bit_pos + bits <= 8) {
        // Fits in one byte
        const mask = (@as(u8, 1) << @as(u3, @intCast(bits))) - 1;
        output[byte_offset] &= ~(mask << (8 - bit_pos - bits));
        output[byte_offset] |= @as(u8, @truncate(value)) << (8 - bit_pos - bits);
    } else {
        // Spans two bytes
        const first_bits = 8 - bit_pos;
        const second_bits = bits - first_bits;
        output[byte_offset] |= @as(u8, @truncate(value >> second_bits));
        const mask2 = (@as(u8, 1) << @as(u3, @intCast(second_bits))) - 1;
        output[byte_offset + 1] = (@as(u8, @truncate(value)) & mask2) << (8 - second_bits);
    }
}

fn unpackBits(input: []const u8, bit_offset: usize, bits: u8) u16 {
    const byte_offset = bit_offset / 8;
    const bit_pos = @as(u4, @intCast(bit_offset % 8));

    if (bit_pos + bits <= 8) {
        const shift = 8 - bit_pos - bits;
        const mask = (@as(u8, 1) << @as(u3, @intCast(bits))) - 1;
        return @as(u16, (input[byte_offset] >> @as(u3, @intCast(shift))) & mask);
    } else {
        const first_bits = 8 - bit_pos;
        const second_bits = bits - first_bits;
        const first_mask = (@as(u8, 1) << @as(u3, @intCast(first_bits))) - 1;
        const first_val = @as(u16, input[byte_offset] & first_mask) << @as(u4, @intCast(second_bits));
        const second_val = @as(u16, input[byte_offset + 1]) >> @as(u3, @intCast(8 - second_bits));
        return first_val | second_val;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "scalar quantizer 8-bit basic" {
    const allocator = std.testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 4, .{ .bits = 8 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 1.0, -1.0, 0.5 },
        &[_]f32{ 0.5, 0.0, 0.0, 1.0 },
        &[_]f32{ 1.0, -1.0, 1.0, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.dimension);
    try std.testing.expectEqual(@as(u8, 8), stats.bits);
    try std.testing.expectEqual(@as(f32, 4.0), stats.compression_ratio);

    // Encode and decode
    var encoded: [4]u8 = undefined;
    const bytes_written = try sq.encode(&vectors[0], &encoded);
    try std.testing.expectEqual(@as(usize, 4), bytes_written);

    var decoded: [4]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // Check approximate reconstruction
    for (vectors[0], decoded) |orig, dec| {
        try std.testing.expect(@abs(orig - dec) < 0.02); // Allow small quantization error
    }
}

test "scalar quantizer 4-bit compression" {
    const allocator = std.testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 8, .{ .bits = 4 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 },
        &[_]f32{ 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.bytes_per_vector); // 8 dims / 2 = 4 bytes
    try std.testing.expectEqual(@as(f32, 8.0), stats.compression_ratio);

    var encoded: [4]u8 = undefined;
    _ = try sq.encode(&vectors[0], &encoded);

    var decoded: [8]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // 4-bit has larger quantization error
    for (vectors[0], decoded) |orig, dec| {
        try std.testing.expect(@abs(orig - dec) < 0.1);
    }
}

test "product quantizer basic" {
    const allocator = std.testing.allocator;

    // 8-dim vectors with 2 subvectors of 4 dims each
    var pq = try ProductQuantizer.init(allocator, 8, .{
        .num_subvectors = 2,
        .bits_per_code = 8,
        .kmeans_iterations = 5,
    });
    defer pq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 },
        &[_]f32{ 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 },
        &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
        &[_]f32{ 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 },
    };

    try pq.train(&vectors, .{ .num_subvectors = 2, .bits_per_code = 8, .kmeans_iterations = 5 });

    const stats = pq.getStats();
    try std.testing.expectEqual(@as(usize, 8), stats.dimension);
    try std.testing.expectEqual(@as(u8, 2), stats.num_subvectors);
    try std.testing.expectEqual(@as(usize, 2), stats.bytes_per_vector); // 2 subvectors * 1 byte each
    try std.testing.expectEqual(@as(f32, 16.0), stats.compression_ratio); // 32 bytes -> 2 bytes

    // Encode and decode
    var encoded: [2]u8 = undefined;
    _ = try pq.encode(&vectors[0], &encoded);

    var decoded: [8]f32 = undefined;
    try pq.decode(&encoded, &decoded);

    // PQ has larger reconstruction error but should be in reasonable range
    var total_error: f32 = 0.0;
    for (vectors[0], decoded) |orig, dec| {
        total_error += @abs(orig - dec);
    }
    try std.testing.expect(total_error / 8.0 < 0.5); // Average error < 0.5
}

test "product quantizer distance table" {
    const allocator = std.testing.allocator;

    var pq = try ProductQuantizer.init(allocator, 4, .{
        .num_subvectors = 2,
        .bits_per_code = 8,
        .kmeans_iterations = 3,
    });
    defer pq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.0, 1.0, 1.0 },
        &[_]f32{ 1.0, 1.0, 0.0, 0.0 },
    };

    try pq.train(&vectors, .{ .num_subvectors = 2, .bits_per_code = 8, .kmeans_iterations = 3 });

    const query = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const dist_table = try pq.computeDistanceTable(&query);
    defer allocator.free(dist_table);

    var encoded: [2]u8 = undefined;
    _ = try pq.encode(&vectors[0], &encoded);

    const dist = pq.asymmetricDistanceWithTable(dist_table, &encoded);
    try std.testing.expect(dist >= 0.0); // Distance should be non-negative
}
