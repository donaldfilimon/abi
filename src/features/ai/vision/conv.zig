//! 2D Convolution Layer for Vision
//!
//! Implements Conv2D using im2col + GEMM approach for efficient computation.
//! Supports forward and backward passes for training.

const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");

// ============================================================================
// Time utility for seeding (platform-aware)
// ============================================================================

fn getTimeSeed() u64 {
    // Use getSeed() for proper PRNG seeding (crypto random on WASM)
    return platform_time.getSeed();
}

// ============================================================================
// Types
// ============================================================================

/// Gradients computed during backward pass
pub const ConvGradients = struct {
    weight_grad: []f32,
    bias_grad: ?[]f32,
    input_grad: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ConvGradients) void {
        self.allocator.free(self.weight_grad);
        if (self.bias_grad) |bg| {
            self.allocator.free(bg);
        }
        self.allocator.free(self.input_grad);
    }
};

// ============================================================================
// Conv2D Layer
// ============================================================================

/// 2D Convolutional Layer
///
/// Performs spatial convolution over input images using learned kernels.
/// Uses im2col approach: unfold input into matrix, then use GEMM for efficiency.
///
/// Input shape: [batch, in_channels, height, width]
/// Output shape: [batch, out_channels, out_height, out_width]
/// where:
///   out_height = (height + 2*padding - kernel_size) / stride + 1
///   out_width  = (width + 2*padding - kernel_size) / stride + 1
pub const Conv2D = struct {
    in_channels: u32,
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    weights: []f32, // [out_channels, in_channels, kernel_size, kernel_size]
    bias: ?[]f32, // [out_channels] or null
    allocator: std.mem.Allocator,

    // Cached for backward pass
    last_input: ?[]f32 = null,
    last_col: ?[]f32 = null,
    last_batch: u32 = 0,
    last_h: u32 = 0,
    last_w: u32 = 0,

    const Self = @This();

    /// Initialize a Conv2D layer with random weights
    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
        use_bias: bool,
    ) !Conv2D {
        // Validate parameters to prevent division-by-zero and invalid allocations
        if (stride == 0) return error.InvalidDimensions;
        if (kernel_size == 0) return error.InvalidDimensions;
        if (in_channels == 0 or out_channels == 0) return error.InvalidDimensions;

        const weight_size = out_channels * in_channels * kernel_size * kernel_size;

        // Kaiming/He initialization for ReLU activations
        const fan_in = in_channels * kernel_size * kernel_size;
        const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));

        const weights = try allocator.alloc(f32, weight_size);
        errdefer allocator.free(weights);

        // Initialize weights with scaled random values
        var prng = std.Random.DefaultPrng.init(getTimeSeed());
        const rand = prng.random();

        for (weights) |*w| {
            // Box-Muller transform for normal distribution
            const rand_u1 = rand.float(f32);
            const rand_u2 = rand.float(f32);
            const z = @sqrt(-2.0 * @log(rand_u1 + 1e-10)) * @cos(2.0 * std.math.pi * rand_u2);
            w.* = z * std_dev;
        }

        var bias: ?[]f32 = null;
        if (use_bias) {
            bias = try allocator.alloc(f32, out_channels);
            @memset(bias.?, 0);
        }

        return Conv2D{
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .weights = weights,
            .bias = bias,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Conv2D) void {
        self.allocator.free(self.weights);
        if (self.bias) |b| {
            self.allocator.free(b);
        }
        if (self.last_input) |li| {
            self.allocator.free(li);
        }
        if (self.last_col) |lc| {
            self.allocator.free(lc);
        }
    }

    /// Calculate output dimensions
    pub fn outputSize(self: *const Conv2D, h: u32, w: u32) struct { h: u32, w: u32 } {
        const out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        const out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        return .{ .h = out_h, .w = out_w };
    }

    /// Forward pass
    ///
    /// input: [batch * in_channels * h * w] flattened tensor
    /// returns: [batch * out_channels * out_h * out_w] flattened tensor
    pub fn forward(self: *Conv2D, input: []const f32, batch: u32, h: u32, w: u32) ![]f32 {
        const out_size = self.outputSize(h, w);
        const out_h = out_size.h;
        const out_w = out_size.w;

        // Cache input for backward pass
        if (self.last_input) |li| {
            self.allocator.free(li);
        }
        self.last_input = try self.allocator.dupe(f32, input);
        self.last_batch = batch;
        self.last_h = h;
        self.last_w = w;

        // im2col dimensions
        const col_h = self.in_channels * self.kernel_size * self.kernel_size;
        const col_w = out_h * out_w;

        // Allocate column matrix for all batches
        const col_size = batch * col_h * col_w;
        const col = try self.allocator.alloc(f32, col_size);
        errdefer self.allocator.free(col);

        // Store col for backward pass
        if (self.last_col) |lc| {
            self.allocator.free(lc);
        }

        // Perform im2col for each batch
        for (0..batch) |b| {
            const input_offset = b * self.in_channels * h * w;
            const col_offset = b * col_h * col_w;

            im2col(
                input[input_offset..],
                1,
                self.in_channels,
                h,
                w,
                self.kernel_size,
                self.stride,
                self.padding,
                col[col_offset .. col_offset + col_h * col_w],
            );
        }

        self.last_col = col;

        // Allocate output
        const output_size = batch * self.out_channels * out_h * out_w;
        const output = try self.allocator.alloc(f32, output_size);
        errdefer self.allocator.free(output);

        // GEMM: output = weights @ col
        // weights: [out_channels, in_channels * kernel_size * kernel_size]
        // col: [in_channels * kernel_size * kernel_size, out_h * out_w]
        // output: [out_channels, out_h * out_w]

        const m = self.out_channels;
        const k = col_h;
        const n = col_w;

        for (0..batch) |b| {
            const col_offset = b * col_h * col_w;
            const out_offset = b * m * n;

            // Matrix multiply
            for (0..m) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0;
                    for (0..k) |l| {
                        sum += self.weights[i * k + l] * col[col_offset + l * n + j];
                    }
                    output[out_offset + i * n + j] = sum;
                }
            }

            // Add bias
            if (self.bias) |bias| {
                for (0..m) |i| {
                    for (0..n) |j| {
                        output[out_offset + i * n + j] += bias[i];
                    }
                }
            }
        }

        return output;
    }

    /// Backward pass - compute gradients
    ///
    /// grad_output: gradient from next layer [batch * out_channels * out_h * out_w]
    /// input: original input from forward pass
    /// returns: ConvGradients containing weight, bias, and input gradients
    pub fn backward(self: *Conv2D, grad_output: []const f32, input: []const f32) !ConvGradients {
        const batch = self.last_batch;
        const h = self.last_h;
        const w = self.last_w;
        const out_size = self.outputSize(h, w);
        const out_h = out_size.h;
        const out_w = out_size.w;

        const col_h = self.in_channels * self.kernel_size * self.kernel_size;
        const col_w = out_h * out_w;

        // Allocate gradients
        const weight_grad = try self.allocator.alloc(f32, self.weights.len);
        errdefer self.allocator.free(weight_grad);
        @memset(weight_grad, 0);

        var bias_grad: ?[]f32 = null;
        if (self.bias != null) {
            bias_grad = try self.allocator.alloc(f32, self.out_channels);
            @memset(bias_grad.?, 0);
        }
        errdefer if (bias_grad) |bg| self.allocator.free(bg);

        const input_grad = try self.allocator.alloc(f32, input.len);
        errdefer self.allocator.free(input_grad);
        @memset(input_grad, 0);

        // Get cached col matrix
        const col = self.last_col orelse return error.NoForwardPass;

        const m = self.out_channels;
        const k = col_h;
        const n = col_w;

        // Allocate col gradient
        const col_grad = try self.allocator.alloc(f32, col.len);
        defer self.allocator.free(col_grad);
        @memset(col_grad, 0);

        for (0..batch) |b| {
            const col_offset = b * col_h * col_w;
            const grad_offset = b * m * n;

            // Compute weight gradient: dW = grad_output @ col^T
            for (0..m) |i| {
                for (0..k) |j| {
                    var sum: f32 = 0;
                    for (0..n) |l| {
                        sum += grad_output[grad_offset + i * n + l] * col[col_offset + j * n + l];
                    }
                    weight_grad[i * k + j] += sum;
                }
            }

            // Compute bias gradient: db = sum(grad_output, axis=(0,2,3))
            if (bias_grad) |bg| {
                for (0..m) |i| {
                    for (0..n) |j| {
                        bg[i] += grad_output[grad_offset + i * n + j];
                    }
                }
            }

            // Compute col gradient: dcol = W^T @ grad_output
            for (0..k) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0;
                    for (0..m) |l| {
                        sum += self.weights[l * k + i] * grad_output[grad_offset + l * n + j];
                    }
                    col_grad[col_offset + i * n + j] = sum;
                }
            }
        }

        // Convert col gradient back to input gradient using col2im
        for (0..batch) |b| {
            const col_offset = b * col_h * col_w;
            const input_offset = b * self.in_channels * h * w;

            col2im(
                col_grad[col_offset .. col_offset + col_h * col_w],
                1,
                self.in_channels,
                h,
                w,
                self.kernel_size,
                self.stride,
                self.padding,
                input_grad[input_offset .. input_offset + self.in_channels * h * w],
            );
        }

        return ConvGradients{
            .weight_grad = weight_grad,
            .bias_grad = bias_grad,
            .input_grad = input_grad,
            .allocator = self.allocator,
        };
    }

    /// Update weights using gradients (SGD)
    pub fn updateWeights(self: *Conv2D, weight_grad: []const f32, bias_grad: ?[]const f32, learning_rate: f32) void {
        for (self.weights, 0..) |*w, i| {
            w.* -= learning_rate * weight_grad[i];
        }
        if (self.bias) |bias| {
            if (bias_grad) |bg| {
                for (bias, 0..) |*b, i| {
                    b.* -= learning_rate * bg[i];
                }
            }
        }
    }

    /// Get number of trainable parameters
    pub fn paramCount(self: *const Conv2D) usize {
        var count: usize = self.weights.len;
        if (self.bias) |b| {
            count += b.len;
        }
        return count;
    }
};

// ============================================================================
// im2col / col2im Helper Functions
// ============================================================================

/// Transform image to column matrix for efficient convolution
///
/// Unfolds input patches into columns so convolution becomes matrix multiplication.
///
/// input: [batch, channels, h, w] flattened
/// output: [channels * kernel * kernel, out_h * out_w] for each batch element
pub fn im2col(
    input: []const f32,
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
    kernel: u32,
    stride: u32,
    padding: u32,
    output: []f32,
) void {
    _ = batch; // Process one batch at a time in caller

    const out_h = (h + 2 * padding - kernel) / stride + 1;
    const out_w = (w + 2 * padding - kernel) / stride + 1;

    var col_idx: usize = 0;

    for (0..channels) |c| {
        for (0..kernel) |ky| {
            for (0..kernel) |kx| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const ih_signed: i64 = @as(i64, @intCast(oh * stride + ky)) - @as(i64, @intCast(padding));
                        const iw_signed: i64 = @as(i64, @intCast(ow * stride + kx)) - @as(i64, @intCast(padding));

                        if (ih_signed >= 0 and ih_signed < h and iw_signed >= 0 and iw_signed < w) {
                            const ih: usize = @intCast(ih_signed);
                            const iw: usize = @intCast(iw_signed);
                            const input_idx = c * h * w + ih * w + iw;
                            output[col_idx] = input[input_idx];
                        } else {
                            output[col_idx] = 0; // Zero padding
                        }
                        col_idx += 1;
                    }
                }
            }
        }
    }
}

/// Transform column matrix back to image (for gradient computation)
///
/// Inverse of im2col - accumulates values back to original positions.
/// Used during backward pass to compute input gradients.
///
/// col: [channels * kernel * kernel, out_h * out_w] for each batch element
/// output: [batch, channels, h, w] flattened (accumulates into output)
pub fn col2im(
    col: []const f32,
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
    kernel: u32,
    stride: u32,
    padding: u32,
    output: []f32,
) void {
    _ = batch; // Process one batch at a time in caller

    const out_h = (h + 2 * padding - kernel) / stride + 1;
    const out_w = (w + 2 * padding - kernel) / stride + 1;

    var col_idx: usize = 0;

    for (0..channels) |c| {
        for (0..kernel) |ky| {
            for (0..kernel) |kx| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const ih_signed: i64 = @as(i64, @intCast(oh * stride + ky)) - @as(i64, @intCast(padding));
                        const iw_signed: i64 = @as(i64, @intCast(ow * stride + kx)) - @as(i64, @intCast(padding));

                        if (ih_signed >= 0 and ih_signed < h and iw_signed >= 0 and iw_signed < w) {
                            const ih: usize = @intCast(ih_signed);
                            const iw: usize = @intCast(iw_signed);
                            const output_idx = c * h * w + ih * w + iw;
                            output[output_idx] += col[col_idx];
                        }
                        col_idx += 1;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "conv2d output dimensions" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 3, 16, 3, 1, 1, true);
    defer conv.deinit();

    const out_size = conv.outputSize(32, 32);
    try std.testing.expectEqual(@as(u32, 32), out_size.h); // Same padding
    try std.testing.expectEqual(@as(u32, 32), out_size.w);
}

test "conv2d output dimensions stride 2" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 3, 16, 3, 2, 1, true);
    defer conv.deinit();

    const out_size = conv.outputSize(32, 32);
    try std.testing.expectEqual(@as(u32, 16), out_size.h);
    try std.testing.expectEqual(@as(u32, 16), out_size.w);
}

test "conv2d forward" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 1, 1, 3, 1, 0, false);
    defer conv.deinit();

    // Set known weights for verification
    @memset(conv.weights, 1.0);

    // 1x1x4x4 input filled with 1s
    const input = [_]f32{1.0} ** 16;

    const output = try conv.forward(&input, 1, 4, 4);
    defer allocator.free(output);

    // Output should be 1x1x2x2 = 4 elements
    // Each output is sum of 3x3=9 ones = 9
    try std.testing.expectEqual(@as(usize, 4), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), output[0], 0.001);
}

test "im2col basic" {
    const allocator = std.testing.allocator;

    // Simple 1-channel 3x3 input
    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // 2x2 kernel, stride 1, no padding -> 2x2 output
    // col: [1*2*2, 2*2] = [4, 4]
    const col = try allocator.alloc(f32, 16);
    defer allocator.free(col);

    im2col(&input, 1, 1, 3, 3, 2, 1, 0, col);

    // First column should be patch at (0,0): [1,2,4,5]
    // We store as row-major per kernel position
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), col[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), col[1], 0.001);
}

test "conv2d param count" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 3, 16, 3, 1, 1, true);
    defer conv.deinit();

    // 16 * 3 * 3 * 3 + 16 = 432 + 16 = 448
    try std.testing.expectEqual(@as(usize, 448), conv.paramCount());
}

test "conv2d no bias" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 3, 16, 3, 1, 1, false);
    defer conv.deinit();

    // 16 * 3 * 3 * 3 = 432
    try std.testing.expectEqual(@as(usize, 432), conv.paramCount());
    try std.testing.expect(conv.bias == null);
}

test {
    std.testing.refAllDecls(@This());
}
