//! Pooling Layers for Vision
//!
//! Implements MaxPool2D, AvgPool2D, and AdaptiveAvgPool2D for spatial downsampling.
//! Supports forward and backward passes for training.

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

/// Result from max pooling forward pass
pub const PoolResult = struct {
    output: []f32,
    indices: []u32, // Indices of max values for backward pass
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PoolResult) void {
        self.allocator.free(self.output);
        self.allocator.free(self.indices);
    }
};

// ============================================================================
// MaxPool2D
// ============================================================================

/// 2D Max Pooling Layer
///
/// Downsamples by taking the maximum value in each pooling window.
/// Stores indices of max values for backward pass gradient routing.
///
/// Input shape: [batch, channels, height, width]
/// Output shape: [batch, channels, out_height, out_width]
pub const MaxPool2D = struct {
    kernel_size: u32,
    stride: u32,
    padding: u32,
    allocator: std.mem.Allocator,

    // Cached for backward pass
    last_indices: ?[]u32 = null,
    last_batch: u32 = 0,
    last_channels: u32 = 0,
    last_h: u32 = 0,
    last_w: u32 = 0,

    const Self = @This();

    /// Initialize MaxPool2D layer
    /// Returns error.InvalidDimensions if kernel_size or stride is 0
    pub fn init(allocator: std.mem.Allocator, kernel_size: u32, stride: u32, padding: u32) !MaxPool2D {
        if (kernel_size == 0) return error.InvalidDimensions;
        if (stride == 0) return error.InvalidDimensions;
        return MaxPool2D{
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *MaxPool2D) void {
        if (self.last_indices) |li| {
            self.allocator.free(li);
        }
    }

    /// Calculate output dimensions
    pub fn outputSize(self: *const MaxPool2D, h: u32, w: u32) struct { h: u32, w: u32 } {
        const out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        const out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        return .{ .h = out_h, .w = out_w };
    }

    /// Forward pass
    ///
    /// input: [batch * channels * h * w] flattened tensor
    /// returns: PoolResult with output and indices for backward pass
    pub fn forward(
        self: *MaxPool2D,
        input: []const f32,
        batch: u32,
        channels: u32,
        h: u32,
        w: u32,
    ) !PoolResult {
        const out_size = self.outputSize(h, w);
        const out_h = out_size.h;
        const out_w = out_size.w;

        const output_len = batch * channels * out_h * out_w;
        const output = try self.allocator.alloc(f32, output_len);
        errdefer self.allocator.free(output);

        const indices = try self.allocator.alloc(u32, output_len);
        errdefer self.allocator.free(indices);

        // Cache dimensions for backward
        self.last_batch = batch;
        self.last_channels = channels;
        self.last_h = h;
        self.last_w = w;

        // Store indices copy for backward pass
        if (self.last_indices) |li| {
            self.allocator.free(li);
        }

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        var max_val: f32 = -std.math.inf(f32);
                        var max_idx: u32 = 0;

                        // Find max in pooling window
                        for (0..self.kernel_size) |ky| {
                            for (0..self.kernel_size) |kx| {
                                const ih_signed: i64 = @as(i64, @intCast(oh * self.stride + ky)) - @as(i64, @intCast(self.padding));
                                const iw_signed: i64 = @as(i64, @intCast(ow * self.stride + kx)) - @as(i64, @intCast(self.padding));

                                if (ih_signed >= 0 and ih_signed < h and iw_signed >= 0 and iw_signed < w) {
                                    const ih: usize = @intCast(ih_signed);
                                    const iw: usize = @intCast(iw_signed);
                                    const input_idx = b * channels * h * w + c * h * w + ih * w + iw;
                                    const val = input[input_idx];

                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = @intCast(input_idx);
                                    }
                                }
                            }
                        }

                        const output_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[output_idx] = max_val;
                        indices[output_idx] = max_idx;
                    }
                }
            }
        }

        self.last_indices = try self.allocator.dupe(u32, indices);

        return PoolResult{
            .output = output,
            .indices = indices,
            .allocator = self.allocator,
        };
    }

    /// Backward pass - route gradients to max positions
    ///
    /// grad_output: gradient from next layer [batch * channels * out_h * out_w]
    /// indices: indices of max values from forward pass
    /// returns: gradient w.r.t. input [batch * channels * h * w]
    pub fn backward(self: *MaxPool2D, grad_output: []const f32, indices: []const u32) ![]f32 {
        const batch = self.last_batch;
        const channels = self.last_channels;
        const h = self.last_h;
        const w = self.last_w;

        const input_grad = try self.allocator.alloc(f32, batch * channels * h * w);
        @memset(input_grad, 0);

        // Route gradients to max positions
        for (grad_output, 0..) |grad, i| {
            const idx = indices[i];
            input_grad[idx] += grad;
        }

        return input_grad;
    }
};

// ============================================================================
// AvgPool2D
// ============================================================================

/// 2D Average Pooling Layer
///
/// Downsamples by averaging values in each pooling window.
///
/// Input shape: [batch, channels, height, width]
/// Output shape: [batch, channels, out_height, out_width]
pub const AvgPool2D = struct {
    kernel_size: u32,
    stride: u32,
    padding: u32,
    allocator: std.mem.Allocator,

    // Cached for backward pass
    last_batch: u32 = 0,
    last_channels: u32 = 0,
    last_h: u32 = 0,
    last_w: u32 = 0,

    const Self = @This();

    /// Initialize AvgPool2D layer
    /// Returns error.InvalidDimensions if kernel_size or stride is 0
    pub fn init(allocator: std.mem.Allocator, kernel_size: u32, stride: u32, padding: u32) !AvgPool2D {
        if (kernel_size == 0) return error.InvalidDimensions;
        if (stride == 0) return error.InvalidDimensions;
        return AvgPool2D{
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *AvgPool2D) void {
        _ = self;
        // No allocations to free
    }

    /// Calculate output dimensions
    pub fn outputSize(self: *const AvgPool2D, h: u32, w: u32) struct { h: u32, w: u32 } {
        const out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        const out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        return .{ .h = out_h, .w = out_w };
    }

    /// Forward pass
    ///
    /// input: [batch * channels * h * w] flattened tensor
    /// returns: [batch * channels * out_h * out_w] flattened tensor
    pub fn forward(
        self: *AvgPool2D,
        input: []const f32,
        batch: u32,
        channels: u32,
        h: u32,
        w: u32,
    ) ![]f32 {
        const out_size = self.outputSize(h, w);
        const out_h = out_size.h;
        const out_w = out_size.w;

        const output = try self.allocator.alloc(f32, batch * channels * out_h * out_w);
        errdefer self.allocator.free(output);

        // Cache dimensions for backward
        self.last_batch = batch;
        self.last_channels = channels;
        self.last_h = h;
        self.last_w = w;

        const pool_area: f32 = @floatFromInt(self.kernel_size * self.kernel_size);

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        var sum: f32 = 0;

                        // Sum values in pooling window
                        for (0..self.kernel_size) |ky| {
                            for (0..self.kernel_size) |kx| {
                                const ih_signed: i64 = @as(i64, @intCast(oh * self.stride + ky)) - @as(i64, @intCast(self.padding));
                                const iw_signed: i64 = @as(i64, @intCast(ow * self.stride + kx)) - @as(i64, @intCast(self.padding));

                                if (ih_signed >= 0 and ih_signed < h and iw_signed >= 0 and iw_signed < w) {
                                    const ih: usize = @intCast(ih_signed);
                                    const iw: usize = @intCast(iw_signed);
                                    const input_idx = b * channels * h * w + c * h * w + ih * w + iw;
                                    sum += input[input_idx];
                                }
                            }
                        }

                        const output_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        // Use full pool area for average (includes padding zeros)
                        output[output_idx] = sum / pool_area;
                    }
                }
            }
        }

        return output;
    }

    /// Backward pass - distribute gradients evenly
    ///
    /// grad_output: gradient from next layer [batch * channels * out_h * out_w]
    /// returns: gradient w.r.t. input [batch * channels * h * w]
    pub fn backward(self: *AvgPool2D, grad_output: []const f32) ![]f32 {
        const batch = self.last_batch;
        const channels = self.last_channels;
        const h = self.last_h;
        const w = self.last_w;

        const out_size = self.outputSize(h, w);
        const out_h = out_size.h;
        const out_w = out_size.w;

        const input_grad = try self.allocator.alloc(f32, batch * channels * h * w);
        @memset(input_grad, 0);

        const pool_area: f32 = @floatFromInt(self.kernel_size * self.kernel_size);
        const grad_scale = 1.0 / pool_area;

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const output_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        const grad = grad_output[output_idx] * grad_scale;

                        // Distribute gradient to all positions in window
                        for (0..self.kernel_size) |ky| {
                            for (0..self.kernel_size) |kx| {
                                const ih_signed: i64 = @as(i64, @intCast(oh * self.stride + ky)) - @as(i64, @intCast(self.padding));
                                const iw_signed: i64 = @as(i64, @intCast(ow * self.stride + kx)) - @as(i64, @intCast(self.padding));

                                if (ih_signed >= 0 and ih_signed < h and iw_signed >= 0 and iw_signed < w) {
                                    const ih: usize = @intCast(ih_signed);
                                    const iw: usize = @intCast(iw_signed);
                                    const input_idx = b * channels * h * w + c * h * w + ih * w + iw;
                                    input_grad[input_idx] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        return input_grad;
    }
};

// ============================================================================
// AdaptiveAvgPool2D
// ============================================================================

/// Adaptive Average Pooling Layer
///
/// Produces fixed output size regardless of input size by automatically
/// computing kernel size and stride.
///
/// Input shape: [batch, channels, height, width]
/// Output shape: [batch, channels, output_size[0], output_size[1]]
pub const AdaptiveAvgPool2D = struct {
    output_size: [2]u32, // Target [height, width]
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize AdaptiveAvgPool2D layer
    pub fn init(allocator: std.mem.Allocator, output_size: [2]u32) AdaptiveAvgPool2D {
        return AdaptiveAvgPool2D{
            .output_size = output_size,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *AdaptiveAvgPool2D) void {
        _ = self;
        // No allocations to free
    }

    /// Forward pass
    ///
    /// input: [batch * channels * h * w] flattened tensor
    /// returns: [batch * channels * out_h * out_w] flattened tensor
    pub fn forward(
        self: *AdaptiveAvgPool2D,
        input: []const f32,
        batch: u32,
        channels: u32,
        h: u32,
        w: u32,
    ) ![]f32 {
        const out_h = self.output_size[0];
        const out_w = self.output_size[1];

        const output = try self.allocator.alloc(f32, batch * channels * out_h * out_w);
        errdefer self.allocator.free(output);

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        // Compute input region for this output position
                        const ih_start = (oh * h) / out_h;
                        const ih_end = ((oh + 1) * h + out_h - 1) / out_h;
                        const iw_start = (ow * w) / out_w;
                        const iw_end = ((ow + 1) * w + out_w - 1) / out_w;

                        var sum: f32 = 0;
                        var count: u32 = 0;

                        for (ih_start..ih_end) |ih| {
                            for (iw_start..iw_end) |iw| {
                                const input_idx = b * channels * h * w + c * h * w + ih * w + iw;
                                sum += input[input_idx];
                                count += 1;
                            }
                        }

                        const output_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[output_idx] = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
                    }
                }
            }
        }

        return output;
    }
};

// ============================================================================
// Global Average Pooling (special case of adaptive)
// ============================================================================

/// Global Average Pooling
///
/// Reduces spatial dimensions to 1x1 by averaging all values per channel.
/// Equivalent to AdaptiveAvgPool2D with output_size = [1, 1].
///
/// Input shape: [batch, channels, height, width]
/// Output shape: [batch, channels] (flattened)
pub fn globalAvgPool2D(
    allocator: std.mem.Allocator,
    input: []const f32,
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
) ![]f32 {
    const output = try allocator.alloc(f32, batch * channels);
    errdefer allocator.free(output);

    const spatial_size: f32 = @floatFromInt(h * w);

    for (0..batch) |b| {
        for (0..channels) |c| {
            var sum: f32 = 0;
            const base_idx = b * channels * h * w + c * h * w;

            for (0..h * w) |i| {
                sum += input[base_idx + i];
            }

            output[b * channels + c] = sum / spatial_size;
        }
    }

    return output;
}

// ============================================================================
// Tests
// ============================================================================

test "maxpool2d output dimensions" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2, 0);
    defer pool.deinit();

    const out_size = pool.outputSize(4, 4);
    try std.testing.expectEqual(@as(u32, 2), out_size.h);
    try std.testing.expectEqual(@as(u32, 2), out_size.w);
}

test "maxpool2d forward" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2, 0);
    defer pool.deinit();

    // 1x1x4x4 input
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var result = try pool.forward(&input, 1, 1, 4, 4);
    defer result.deinit();

    // Should get 1x1x2x2 output with max values
    try std.testing.expectEqual(@as(usize, 4), result.output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result.output[0], 0.001); // max(1,2,5,6)
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), result.output[1], 0.001); // max(3,4,7,8)
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), result.output[2], 0.001); // max(9,10,13,14)
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), result.output[3], 0.001); // max(11,12,15,16)
}

test "avgpool2d forward" {
    const allocator = std.testing.allocator;

    var pool = AvgPool2D.init(allocator, 2, 2, 0);
    defer pool.deinit();

    // 1x1x4x4 input
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var output = try pool.forward(&input, 1, 1, 4, 4);
    defer allocator.free(output);

    // Should get 1x1x2x2 output with average values
    try std.testing.expectEqual(@as(usize, 4), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), output[0], 0.001); // avg(1,2,5,6)
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), output[1], 0.001); // avg(3,4,7,8)
    try std.testing.expectApproxEqAbs(@as(f32, 11.5), output[2], 0.001); // avg(9,10,13,14)
    try std.testing.expectApproxEqAbs(@as(f32, 13.5), output[3], 0.001); // avg(11,12,15,16)
}

test "adaptive avgpool2d forward" {
    const allocator = std.testing.allocator;

    var pool = AdaptiveAvgPool2D.init(allocator, .{ 1, 1 });
    defer pool.deinit();

    // 1x2x4x4 input (2 channels)
    const input = [_]f32{
        // Channel 0
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        // Channel 1
        2,  4,  6,  8,
        10, 12, 14, 16,
        18, 20, 22, 24,
        26, 28, 30, 32,
    };

    var output = try pool.forward(&input, 1, 2, 4, 4);
    defer allocator.free(output);

    // Should get 1x2x1x1 = 2 values
    try std.testing.expectEqual(@as(usize, 2), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), output[0], 0.001); // avg of 1..16
    try std.testing.expectApproxEqAbs(@as(f32, 17.0), output[1], 0.001); // avg of 2,4,...,32
}

test "global avg pool" {
    const allocator = std.testing.allocator;

    // 1x1x2x2 input
    const input = [_]f32{ 1, 2, 3, 4 };

    var output = try globalAvgPool2D(allocator, &input, 1, 1, 2, 2);
    defer allocator.free(output);

    try std.testing.expectEqual(@as(usize, 1), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output[0], 0.001);
}

test "maxpool2d backward" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2, 0);
    defer pool.deinit();

    // 1x1x4x4 input
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var result = try pool.forward(&input, 1, 1, 4, 4);
    defer result.deinit();

    // Gradient from next layer
    const grad_output = [_]f32{ 1, 1, 1, 1 };

    var input_grad = try pool.backward(&grad_output, result.indices);
    defer allocator.free(input_grad);

    // Gradient should only be non-zero at max positions
    try std.testing.expectEqual(@as(usize, 16), input_grad.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), input_grad[5], 0.001); // position of 6
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), input_grad[7], 0.001); // position of 8
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), input_grad[0], 0.001); // not a max
}
