//! Compression utilities — delta encoding and vector quantization.

const std = @import("std");

// ============================================================================
// Delta Encoding for Vectors
// ============================================================================

/// Delta encoding for sorted vector IDs (improves compression)
pub fn deltaEncode(ids: []const u64, allocator: std.mem.Allocator) ![]u64 {
    if (ids.len == 0) return &.{};

    var encoded = try allocator.alloc(u64, ids.len);
    errdefer allocator.free(encoded);

    encoded[0] = ids[0];
    for (1..ids.len) |i| {
        encoded[i] = ids[i] - ids[i - 1];
    }

    return encoded;
}

pub fn deltaDecode(encoded: []const u64, allocator: std.mem.Allocator) ![]u64 {
    if (encoded.len == 0) return &.{};

    var decoded = try allocator.alloc(u64, encoded.len);
    errdefer allocator.free(decoded);

    decoded[0] = encoded[0];
    for (1..encoded.len) |i| {
        decoded[i] = decoded[i - 1] + encoded[i];
    }

    return decoded;
}

/// Simple vector quantization for compression
pub fn quantizeVectors(
    vectors: []const []const f32,
    allocator: std.mem.Allocator,
) !QuantizedVectors {
    if (vectors.len == 0) return .{
        .scales = &.{},
        .offsets = &.{},
        .data = &.{},
        .dimension = 0,
        .allocator = allocator,
    };

    const dim = vectors[0].len;
    const num_vectors = vectors.len;

    // Calculate min/max per dimension for quantization
    var mins = try allocator.alloc(f32, dim);
    defer allocator.free(mins);
    var maxs = try allocator.alloc(f32, dim);
    defer allocator.free(maxs);

    @memset(mins, std.math.inf(f32));
    @memset(maxs, -std.math.inf(f32));

    for (vectors) |vec| {
        for (0..dim) |d| {
            mins[d] = @min(mins[d], vec[d]);
            maxs[d] = @max(maxs[d], vec[d]);
        }
    }

    // Calculate scales and offsets
    const scales = try allocator.alloc(f32, dim);
    errdefer allocator.free(scales);
    const offsets = try allocator.alloc(f32, dim);
    errdefer allocator.free(offsets);

    for (0..dim) |d| {
        const range = maxs[d] - mins[d];
        scales[d] = if (range > 0) range / 255.0 else 1.0;
        offsets[d] = mins[d];
    }

    // Quantize to uint8
    const data = try allocator.alloc(u8, num_vectors * dim);
    errdefer allocator.free(data);

    for (vectors, 0..) |vec, vi| {
        const base = vi * dim;
        for (0..dim) |d| {
            const normalized = (vec[d] - offsets[d]) / scales[d];
            data[base + d] = @intFromFloat(@min(255.0, @max(0.0, normalized)));
        }
    }

    return .{
        .scales = scales,
        .offsets = offsets,
        .data = data,
        .dimension = dim,
        .allocator = allocator,
    };
}

pub const QuantizedVectors = struct {
    scales: []f32,
    offsets: []f32,
    data: []u8,
    dimension: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *QuantizedVectors) void {
        if (self.scales.len > 0) self.allocator.free(self.scales);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.data.len > 0) self.allocator.free(self.data);
    }

    pub fn dequantize(self: *const QuantizedVectors, vector_idx: usize) []f32 {
        _ = self;
        _ = vector_idx;
        // Would need allocator - simplified for now
        return &.{};
    }

    pub fn getVector(self: *const QuantizedVectors, vector_idx: usize, out: []f32) void {
        const base = vector_idx * self.dimension;
        for (0..self.dimension) |d| {
            out[d] = @as(f32, @floatFromInt(self.data[base + d])) * self.scales[d] + self.offsets[d];
        }
    }

    pub fn compressionRatio(self: *const QuantizedVectors, num_vectors: usize) f64 {
        const original_size = num_vectors * self.dimension * @sizeOf(f32);
        const compressed_size = self.data.len + self.scales.len * @sizeOf(f32) + self.offsets.len * @sizeOf(f32);
        return @as(f64, @floatFromInt(original_size)) / @as(f64, @floatFromInt(compressed_size));
    }
};
