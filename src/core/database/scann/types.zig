//! Shared types for the ScaNN index implementation.

const std = @import("std");

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

/// Search candidate for internal use
pub const SearchCandidate = struct {
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

pub const computeL2DistanceSquared = @import("../../../foundation/mod.zig").simd.distances.l2DistanceSquared;

pub fn computeWeightedL2(a: []const f32, b: []const f32, weights: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == weights.len);
    var sum: f32 = 0;
    for (a, b, weights) |av, bv, w| {
        const diff = av - bv;
        sum += diff * diff * w;
    }
    return sum;
}
