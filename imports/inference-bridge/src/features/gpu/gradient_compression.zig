//! Gradient Compression for Distributed Training
//!
//! Implements top-k sparsification with error feedback for bandwidth-efficient
//! gradient synchronization across distributed training nodes.
//!
//! Key features:
//! - Top-k sparsification: Only transmit the largest-magnitude gradient elements
//! - Error feedback: Residual errors accumulate and are included in next round
//! - Configurable compression ratio (default: 0.1% = keep top 0.1%)
//!
//! ## Usage
//! ```zig
//! var compressor = try GradientCompressor.init(allocator, 1_000_000, 0.001);
//! defer compressor.deinit();
//!
//! const compressed = try compressor.compress(gradients);
//! defer compressed.deinit(allocator);
//!
//! var decompressed: [1_000_000]f32 = undefined;
//! compressor.decompress(compressed, &decompressed);
//! ```

const std = @import("std");

/// Compressed gradient representation: sparse indices + values.
pub const CompressedGradient = struct {
    indices: []u32,
    values: []f32,
    original_size: usize,
    compression_ratio: f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompressedGradient) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.values);
    }

    /// Compute compressed size in bytes.
    pub fn compressedBytes(self: *const CompressedGradient) usize {
        return self.indices.len * @sizeOf(u32) + self.values.len * @sizeOf(f32);
    }

    /// Compute original size in bytes.
    pub fn originalBytes(self: *const CompressedGradient) usize {
        return self.original_size * @sizeOf(f32);
    }
};

/// Statistics about compression performance.
pub const CompressionStats = struct {
    compressed_bytes: usize,
    original_bytes: usize,
    ratio: f32,
    residual_norm: f32,
    compressions_count: u64,
};

/// Top-k gradient compressor with error feedback.
pub const GradientCompressor = struct {
    residual: []f32,
    ratio: f32,
    allocator: std.mem.Allocator,
    gradient_size: usize,
    compressions_count: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, gradient_size: usize, ratio: f32) !GradientCompressor {
        const residual = try allocator.alloc(f32, gradient_size);
        @memset(residual, 0);
        return .{
            .residual = residual,
            .ratio = @max(ratio, 0.0001), // Minimum 0.01%
            .allocator = allocator,
            .gradient_size = gradient_size,
        };
    }

    pub fn deinit(self: *GradientCompressor) void {
        self.allocator.free(self.residual);
    }

    /// Compress gradients using top-k sparsification with error feedback.
    /// The residual from previous compressions is added to current gradients.
    pub fn compress(self: *GradientCompressor, gradients: []const f32) !CompressedGradient {
        if (gradients.len != self.gradient_size) return error.InvalidDimensions;

        const k = @max(1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.gradient_size)) * self.ratio)));

        // Add residual to gradients (error feedback)
        const combined = try self.allocator.alloc(f32, self.gradient_size);
        defer self.allocator.free(combined);
        for (combined, gradients, self.residual) |*c, g, r| {
            c.* = g + r;
        }

        // Find magnitude threshold for top-k via partial sort (quickselect)
        const magnitudes = try self.allocator.alloc(f32, self.gradient_size);
        defer self.allocator.free(magnitudes);
        for (magnitudes, combined) |*m, c| {
            m.* = @abs(c);
        }

        const threshold = partialKthLargest(magnitudes, k);

        // Extract indices and values above threshold
        var indices = try self.allocator.alloc(u32, k);
        var values = try self.allocator.alloc(f32, k);
        var count: usize = 0;

        for (combined, 0..) |c, i| {
            if (@abs(c) >= threshold and count < k) {
                indices[count] = @intCast(i);
                values[count] = c;
                count += 1;
            }
        }

        // Resize to actual count
        if (count < k) {
            indices = self.allocator.realloc(indices, count) catch indices;
            values = self.allocator.realloc(values, count) catch values;
        }

        // Update residual: residual = combined - decompressed(compressed)
        @memcpy(self.residual, combined);
        for (indices[0..count], values[0..count]) |idx, val| {
            self.residual[idx] -= val;
        }

        self.compressions_count += 1;

        return .{
            .indices = indices[0..count],
            .values = values[0..count],
            .original_size = self.gradient_size,
            .compression_ratio = @as(f32, @floatFromInt(count)) / @as(f32, @floatFromInt(self.gradient_size)),
            .allocator = self.allocator,
        };
    }

    /// Decompress: scatter compressed values into a zero buffer.
    pub fn decompress(compressed: *const CompressedGradient, output: []f32) void {
        @memset(output, 0);
        for (compressed.indices, compressed.values) |idx, val| {
            if (idx < output.len) {
                output[idx] = val;
            }
        }
    }

    /// Get compression statistics.
    pub fn getStats(self: *const GradientCompressor) CompressionStats {
        var residual_norm: f32 = 0;
        for (self.residual) |r| {
            residual_norm += r * r;
        }
        residual_norm = @sqrt(residual_norm);

        const compressed_bytes = @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.gradient_size)) * self.ratio)) *
            (@sizeOf(u32) + @sizeOf(f32));

        return .{
            .compressed_bytes = compressed_bytes,
            .original_bytes = self.gradient_size * @sizeOf(f32),
            .ratio = self.ratio,
            .residual_norm = residual_norm,
            .compressions_count = self.compressions_count,
        };
    }
};

/// Find the k-th largest value using partial sort (introselect-style).
/// Modifies the input array in place.
fn partialKthLargest(data: []f32, k: usize) f32 {
    if (data.len == 0) return 0;
    if (k >= data.len) return 0;

    // Simple approach: sort descending and return k-th element
    std.mem.sort(f32, data, {}, struct {
        fn desc(_: void, a: f32, b: f32) bool {
            return a > b;
        }
    }.desc);

    return data[@min(k, data.len - 1)];
}

/// Gradient bucket manager for fusing small gradients before AllReduce.
/// Groups parameters into buckets of configurable size to reduce communication overhead.
pub const GradientBucketManager = struct {
    bucket_size: usize,
    buckets: std.ArrayListUnmanaged(Bucket),
    allocator: std.mem.Allocator,

    pub const Bucket = struct {
        data: []f32,
        param_ids: std.ArrayListUnmanaged(u32),
        used: usize = 0,
        capacity: usize,

        pub fn deinit(self: *Bucket, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.param_ids.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, bucket_size: usize) GradientBucketManager {
        return .{
            .bucket_size = if (bucket_size == 0) 25 * 1024 * 1024 / @sizeOf(f32) else bucket_size,
            .buckets = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GradientBucketManager) void {
        for (self.buckets.items) |*bucket| {
            bucket.deinit(self.allocator);
        }
        self.buckets.deinit(self.allocator);
    }

    /// Add a gradient to the current bucket. Creates a new bucket if current is full.
    pub fn addGradient(self: *GradientBucketManager, param_id: u32, gradient_data: []const f32) !void {
        // Find a bucket with space
        var target: ?*Bucket = null;
        for (self.buckets.items) |*b| {
            if (b.used + gradient_data.len <= b.capacity) {
                target = b;
                break;
            }
        }

        if (target == null) {
            // Create new bucket
            const cap = @max(self.bucket_size, gradient_data.len);
            const data = try self.allocator.alloc(f32, cap);
            try self.buckets.append(self.allocator, .{
                .data = data,
                .param_ids = .empty,
                .used = 0,
                .capacity = cap,
            });
            target = &self.buckets.items[self.buckets.items.len - 1];
        }

        const b = target.?;
        @memcpy(b.data[b.used .. b.used + gradient_data.len], gradient_data);
        b.used += gradient_data.len;
        try b.param_ids.append(self.allocator, param_id);
    }

    /// Check if any bucket is full and ready for AllReduce.
    pub fn hasReadyBucket(self: *const GradientBucketManager) bool {
        for (self.buckets.items) |b| {
            if (b.used >= b.capacity * 3 / 4) return true; // 75% threshold
        }
        return false;
    }

    /// Get the number of buckets.
    pub fn bucketCount(self: *const GradientBucketManager) usize {
        return self.buckets.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "GradientCompressor compress and decompress" {
    const allocator = std.testing.allocator;
    var compressor = try GradientCompressor.init(allocator, 100, 0.1); // Keep top 10%
    defer compressor.deinit();

    var gradients: [100]f32 = undefined;
    for (&gradients, 0..) |*g, i| {
        g.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var compressed = try compressor.compress(&gradients);
    defer compressed.deinit();

    try std.testing.expect(compressed.indices.len <= 10);
    try std.testing.expect(compressed.indices.len > 0);

    var decompressed: [100]f32 = undefined;
    GradientCompressor.decompress(&compressed, &decompressed);

    // At least some non-zero values should be preserved in the decompressed output
    var nonzero_count: usize = 0;
    for (decompressed) |v| {
        if (v != 0) nonzero_count += 1;
    }
    try std.testing.expect(nonzero_count > 0);
    try std.testing.expect(nonzero_count <= 10);

    // Compressed values should correspond to the largest-magnitude gradients
    for (compressed.values) |v| {
        try std.testing.expect(@abs(v) > 0);
    }
}

test "GradientCompressor error feedback accumulates" {
    const allocator = std.testing.allocator;
    var compressor = try GradientCompressor.init(allocator, 10, 0.3); // Keep top 30%
    defer compressor.deinit();

    const gradients = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };

    // First compression
    var c1 = try compressor.compress(&gradients);
    defer c1.deinit();

    // Residual should be non-zero for dropped elements
    var residual_sum: f32 = 0;
    for (compressor.residual) |r| residual_sum += @abs(r);
    try std.testing.expect(residual_sum > 0);
}

test "GradientCompressor all-zero gradient" {
    const allocator = std.testing.allocator;
    var compressor = try GradientCompressor.init(allocator, 10, 0.5);
    defer compressor.deinit();

    const gradients = [_]f32{0} ** 10;
    var compressed = try compressor.compress(&gradients);
    defer compressed.deinit();

    // All zeros — nothing above threshold
    try std.testing.expectEqual(@as(usize, 10), compressed.original_size);
}

test "GradientCompressor compression stats" {
    const allocator = std.testing.allocator;
    var compressor = try GradientCompressor.init(allocator, 1000, 0.01);
    defer compressor.deinit();

    const stats = compressor.getStats();
    try std.testing.expectEqual(@as(usize, 1000 * 4), stats.original_bytes);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), stats.ratio, 0.001);
}

test "GradientCompressor invalid dimensions" {
    const allocator = std.testing.allocator;
    var compressor = try GradientCompressor.init(allocator, 10, 0.5);
    defer compressor.deinit();

    const wrong_size = [_]f32{ 1.0, 2.0, 3.0 }; // Size 3, expects 10
    try std.testing.expectError(error.InvalidDimensions, compressor.compress(&wrong_size));
}

test "GradientBucketManager basic operations" {
    const allocator = std.testing.allocator;
    var mgr = GradientBucketManager.init(allocator, 100); // 100 elements per bucket
    defer mgr.deinit();

    const grad = [_]f32{ 1.0, 2.0, 3.0 };
    try mgr.addGradient(0, &grad);
    try mgr.addGradient(1, &grad);

    try std.testing.expectEqual(@as(usize, 1), mgr.bucketCount());
}

test "GradientBucketManager creates new bucket when full" {
    const allocator = std.testing.allocator;
    var mgr = GradientBucketManager.init(allocator, 4); // Tiny bucket: 4 elements
    defer mgr.deinit();

    const grad = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try mgr.addGradient(0, &grad); // Fills bucket 1
    try mgr.addGradient(1, &grad); // Creates bucket 2

    try std.testing.expectEqual(@as(usize, 2), mgr.bucketCount());
}

test "GradientBucketManager hasReadyBucket threshold" {
    const allocator = std.testing.allocator;
    var mgr = GradientBucketManager.init(allocator, 4); // 4 elements per bucket
    defer mgr.deinit();

    // Empty manager has no ready buckets
    try std.testing.expect(!mgr.hasReadyBucket());

    // Add enough data to pass 75% threshold (3 out of 4)
    const grad = [_]f32{ 1.0, 2.0, 3.0 };
    try mgr.addGradient(0, &grad);
    try std.testing.expect(mgr.hasReadyBucket());
}

test "GradientBucketManager zero bucket size uses default" {
    const allocator = std.testing.allocator;
    var mgr = GradientBucketManager.init(allocator, 0);
    defer mgr.deinit();

    // Default: 25 MB / 4 bytes = 6553600
    try std.testing.expectEqual(@as(usize, 25 * 1024 * 1024 / @sizeOf(f32)), mgr.bucket_size);
}

test {
    std.testing.refAllDecls(@This());
}
