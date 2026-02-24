//! Batch Normalization for Vision
//!
//! Implements BatchNorm2D for normalizing activations in convolutional networks.
//! Supports training mode with running statistics and inference mode.

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

/// Gradients computed during backward pass
pub const BatchNormGradients = struct {
    gamma_grad: []f32, // Scale gradient
    beta_grad: []f32, // Shift gradient
    input_grad: []f32, // Gradient w.r.t. input
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BatchNormGradients) void {
        self.allocator.free(self.gamma_grad);
        self.allocator.free(self.beta_grad);
        self.allocator.free(self.input_grad);
    }
};

// ============================================================================
// BatchNorm2D
// ============================================================================

/// 2D Batch Normalization Layer
///
/// Normalizes activations across batch and spatial dimensions for each channel.
/// During training, computes mean/variance from mini-batch and updates running statistics.
/// During inference, uses running statistics for normalization.
///
/// Input shape: [batch, channels, height, width]
/// Output shape: [batch, channels, height, width] (same as input)
///
/// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
pub const BatchNorm2D = struct {
    num_features: u32, // Number of channels
    eps: f32, // Epsilon for numerical stability
    momentum: f32, // Momentum for running stats update
    running_mean: []f32, // [num_features]
    running_var: []f32, // [num_features]
    gamma: []f32, // Scale parameters [num_features]
    beta: []f32, // Shift parameters [num_features]
    training: bool,
    allocator: std.mem.Allocator,

    // Cached for backward pass
    last_input: ?[]f32 = null,
    last_normalized: ?[]f32 = null,
    last_mean: ?[]f32 = null,
    last_var: ?[]f32 = null,
    last_batch: u32 = 0,
    last_h: u32 = 0,
    last_w: u32 = 0,

    const Self = @This();

    /// Initialize BatchNorm2D layer
    pub fn init(
        allocator: std.mem.Allocator,
        num_features: u32,
        eps: f32,
        momentum: f32,
    ) !BatchNorm2D {
        const running_mean = try allocator.alloc(f32, num_features);
        errdefer allocator.free(running_mean);
        @memset(running_mean, 0);

        const running_var = try allocator.alloc(f32, num_features);
        errdefer allocator.free(running_var);
        @memset(running_var, 1.0);

        const gamma = try allocator.alloc(f32, num_features);
        errdefer allocator.free(gamma);
        @memset(gamma, 1.0);

        const beta = try allocator.alloc(f32, num_features);
        errdefer allocator.free(beta);
        @memset(beta, 0);

        return BatchNorm2D{
            .num_features = num_features,
            .eps = eps,
            .momentum = momentum,
            .running_mean = running_mean,
            .running_var = running_var,
            .gamma = gamma,
            .beta = beta,
            .training = true,
            .allocator = allocator,
        };
    }

    /// Initialize with default eps=1e-5 and momentum=0.1
    pub fn initDefault(allocator: std.mem.Allocator, num_features: u32) !BatchNorm2D {
        return init(allocator, num_features, 1e-5, 0.1);
    }

    /// Clean up resources
    pub fn deinit(self: *BatchNorm2D) void {
        self.allocator.free(self.running_mean);
        self.allocator.free(self.running_var);
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);

        if (self.last_input) |li| self.allocator.free(li);
        if (self.last_normalized) |ln| self.allocator.free(ln);
        if (self.last_mean) |lm| self.allocator.free(lm);
        if (self.last_var) |lv| self.allocator.free(lv);
    }

    /// Set training/evaluation mode
    pub fn setTraining(self: *BatchNorm2D, training: bool) void {
        self.training = training;
    }

    /// Forward pass
    ///
    /// input: [batch * channels * h * w] flattened tensor
    /// returns: normalized output [batch * channels * h * w]
    pub fn forward(
        self: *BatchNorm2D,
        input: []const f32,
        batch: u32,
        channels: u32,
        h: u32,
        w: u32,
    ) ![]f32 {
        if (channels != self.num_features) {
            return error.ChannelMismatch;
        }

        const spatial_size = h * w;
        const n: f32 = @floatFromInt(batch * spatial_size);

        const output = try self.allocator.alloc(f32, input.len);
        errdefer self.allocator.free(output);

        // Cache for backward
        self.last_batch = batch;
        self.last_h = h;
        self.last_w = w;

        if (self.last_input) |li| self.allocator.free(li);
        self.last_input = try self.allocator.dupe(f32, input);

        if (self.training) {
            // Compute batch statistics
            const batch_mean = try self.allocator.alloc(f32, channels);
            defer self.allocator.free(batch_mean);
            @memset(batch_mean, 0);

            const batch_var = try self.allocator.alloc(f32, channels);
            defer self.allocator.free(batch_var);
            @memset(batch_var, 0);

            // Compute mean for each channel
            for (0..batch) |b| {
                for (0..channels) |c| {
                    const base_idx = b * channels * spatial_size + c * spatial_size;
                    for (0..spatial_size) |i| {
                        batch_mean[c] += input[base_idx + i];
                    }
                }
            }
            for (0..channels) |c| {
                batch_mean[c] /= n;
            }

            // Compute variance for each channel
            for (0..batch) |b| {
                for (0..channels) |c| {
                    const base_idx = b * channels * spatial_size + c * spatial_size;
                    for (0..spatial_size) |i| {
                        const diff = input[base_idx + i] - batch_mean[c];
                        batch_var[c] += diff * diff;
                    }
                }
            }
            for (0..channels) |c| {
                batch_var[c] /= n;
            }

            // Update running statistics
            for (0..channels) |c| {
                self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] + self.momentum * batch_mean[c];
                self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] + self.momentum * batch_var[c];
            }

            // Allocate and save normalized values for backward
            const normalized = try self.allocator.alloc(f32, input.len);
            errdefer self.allocator.free(normalized);

            // Normalize and apply affine transform
            for (0..batch) |b| {
                for (0..channels) |c| {
                    const base_idx = b * channels * spatial_size + c * spatial_size;
                    const std_dev = @sqrt(batch_var[c] + self.eps);

                    for (0..spatial_size) |i| {
                        const idx = base_idx + i;
                        const norm_val = (input[idx] - batch_mean[c]) / std_dev;
                        normalized[idx] = norm_val;
                        output[idx] = self.gamma[c] * norm_val + self.beta[c];
                    }
                }
            }

            // Cache for backward pass
            if (self.last_normalized) |ln| self.allocator.free(ln);
            self.last_normalized = normalized;

            if (self.last_mean) |lm| self.allocator.free(lm);
            self.last_mean = try self.allocator.dupe(f32, batch_mean);

            if (self.last_var) |lv| self.allocator.free(lv);
            self.last_var = try self.allocator.dupe(f32, batch_var);
        } else {
            // Inference mode: use running statistics
            for (0..batch) |b| {
                for (0..channels) |c| {
                    const base_idx = b * channels * spatial_size + c * spatial_size;
                    const std_dev = @sqrt(self.running_var[c] + self.eps);

                    for (0..spatial_size) |i| {
                        const idx = base_idx + i;
                        const norm_val = (input[idx] - self.running_mean[c]) / std_dev;
                        output[idx] = self.gamma[c] * norm_val + self.beta[c];
                    }
                }
            }
        }

        return output;
    }

    /// Backward pass - compute gradients
    ///
    /// grad_output: gradient from next layer [batch * channels * h * w]
    /// input: original input from forward pass
    /// returns: BatchNormGradients containing gamma, beta, and input gradients
    pub fn backward(self: *BatchNorm2D, grad_output: []const f32, input: []const f32) !BatchNormGradients {
        _ = input; // Use cached values

        const batch = self.last_batch;
        const h = self.last_h;
        const w = self.last_w;
        const channels = self.num_features;
        const spatial_size = h * w;
        const n: f32 = @floatFromInt(batch * spatial_size);

        const normalized = self.last_normalized orelse return error.NoForwardPass;
        _ = self.last_mean orelse return error.NoForwardPass; // Verify forward was called
        const batch_var = self.last_var orelse return error.NoForwardPass;
        _ = self.last_input orelse return error.NoForwardPass; // Verify forward was called

        // Allocate gradients
        const gamma_grad = try self.allocator.alloc(f32, channels);
        errdefer self.allocator.free(gamma_grad);
        @memset(gamma_grad, 0);

        const beta_grad = try self.allocator.alloc(f32, channels);
        errdefer self.allocator.free(beta_grad);
        @memset(beta_grad, 0);

        const input_grad = try self.allocator.alloc(f32, grad_output.len);
        errdefer self.allocator.free(input_grad);
        @memset(input_grad, 0);

        // Compute gamma and beta gradients
        // d_gamma = sum(grad_output * normalized)
        // d_beta = sum(grad_output)
        for (0..batch) |b| {
            for (0..channels) |c| {
                const base_idx = b * channels * spatial_size + c * spatial_size;
                for (0..spatial_size) |i| {
                    const idx = base_idx + i;
                    gamma_grad[c] += grad_output[idx] * normalized[idx];
                    beta_grad[c] += grad_output[idx];
                }
            }
        }

        // Compute input gradient using chain rule
        // d_input = (1/std) * (grad_output * gamma - mean(grad_output * gamma)
        //           - normalized * mean(grad_output * gamma * normalized))
        for (0..channels) |c| {
            const std_dev = @sqrt(batch_var[c] + self.eps);
            const inv_std = 1.0 / std_dev;

            // Compute intermediate sums
            var sum_grad_gamma: f32 = 0;
            var sum_grad_gamma_norm: f32 = 0;

            for (0..batch) |b| {
                const base_idx = b * channels * spatial_size + c * spatial_size;
                for (0..spatial_size) |i| {
                    const idx = base_idx + i;
                    const grad_gamma = grad_output[idx] * self.gamma[c];
                    sum_grad_gamma += grad_gamma;
                    sum_grad_gamma_norm += grad_gamma * normalized[idx];
                }
            }

            const mean_grad_gamma = sum_grad_gamma / n;
            const mean_grad_gamma_norm = sum_grad_gamma_norm / n;

            // Apply gradient formula
            for (0..batch) |b| {
                const base_idx = b * channels * spatial_size + c * spatial_size;
                for (0..spatial_size) |i| {
                    const idx = base_idx + i;
                    const grad_gamma = grad_output[idx] * self.gamma[c];
                    input_grad[idx] = inv_std * (grad_gamma - mean_grad_gamma - normalized[idx] * mean_grad_gamma_norm);
                }
            }
        }

        return BatchNormGradients{
            .gamma_grad = gamma_grad,
            .beta_grad = beta_grad,
            .input_grad = input_grad,
            .allocator = self.allocator,
        };
    }

    /// Update parameters using gradients (SGD)
    pub fn updateParams(self: *BatchNorm2D, gamma_grad: []const f32, beta_grad: []const f32, learning_rate: f32) void {
        for (self.gamma, 0..) |*g, i| {
            g.* -= learning_rate * gamma_grad[i];
        }
        for (self.beta, 0..) |*b, i| {
            b.* -= learning_rate * beta_grad[i];
        }
    }

    /// Get number of trainable parameters
    pub fn paramCount(self: *const BatchNorm2D) usize {
        return self.num_features * 2; // gamma + beta
    }

    /// Zero gradients (for use with external gradient accumulators)
    pub fn zeroGrad(self: *BatchNorm2D) void {
        // BatchNorm doesn't store gradients internally
        _ = self;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "batchnorm2d initialization" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.initDefault(allocator, 16);
    defer bn.deinit();

    try std.testing.expectEqual(@as(u32, 16), bn.num_features);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), bn.eps, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), bn.momentum, 1e-6);

    // Check initial gamma = 1, beta = 0
    for (bn.gamma) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
    for (bn.beta) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), b, 1e-6);
    }
}

test "batchnorm2d forward training" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.initDefault(allocator, 2);
    defer bn.deinit();

    // 2 batch, 2 channels, 2x2 spatial
    const input = [_]f32{
        // Batch 0, Channel 0
        1,  2,  3,  4,
        // Batch 0, Channel 1
        5,  6,  7,  8,
        // Batch 1, Channel 0
        9,  10, 11, 12,
        // Batch 1, Channel 1
        13, 14, 15, 16,
    };

    const output = try bn.forward(&input, 2, 2, 2, 2);
    defer allocator.free(output);

    // After normalization, each channel should have mean ~0 and var ~1
    // Check channel 0 (positions 0-3, 8-11)
    var sum0: f32 = 0;
    var sum_sq0: f32 = 0;
    const ch0_positions = [_]usize{ 0, 1, 2, 3, 8, 9, 10, 11 };
    for (ch0_positions) |pos| {
        sum0 += output[pos];
        sum_sq0 += output[pos] * output[pos];
    }
    const mean0 = sum0 / 8.0;
    const var0 = sum_sq0 / 8.0 - mean0 * mean0;

    try std.testing.expect(@abs(mean0) < 0.01);
    try std.testing.expect(@abs(var0 - 1.0) < 0.01);
}

test "batchnorm2d forward inference" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.initDefault(allocator, 1);
    defer bn.deinit();

    // Set known running stats
    bn.running_mean[0] = 5.0;
    bn.running_var[0] = 4.0;
    bn.setTraining(false);

    // 1 batch, 1 channel, 2x2 spatial
    const input = [_]f32{ 1, 3, 7, 9 };

    const output = try bn.forward(&input, 1, 1, 2, 2);
    defer allocator.free(output);

    // Expected: (x - 5) / 2
    // 1 -> -2, 3 -> -1, 7 -> 1, 9 -> 2
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), output[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[3], 0.01);
}

test "batchnorm2d param count" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.initDefault(allocator, 64);
    defer bn.deinit();

    // 64 gamma + 64 beta = 128
    try std.testing.expectEqual(@as(usize, 128), bn.paramCount());
}

test "batchnorm2d running stats update" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.init(allocator, 1, 1e-5, 0.5);
    defer bn.deinit();

    // Initial running_mean = 0, running_var = 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bn.running_mean[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bn.running_var[0], 1e-6);

    // Run forward pass
    const input = [_]f32{ 2, 4, 6, 8 }; // mean=5, var=5

    const output = try bn.forward(&input, 1, 1, 2, 2);
    defer allocator.free(output);

    // With momentum=0.5: running_mean = 0.5*0 + 0.5*5 = 2.5
    //                    running_var = 0.5*1 + 0.5*5 = 3.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), bn.running_mean[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), bn.running_var[0], 0.01);
}

test "batchnorm2d backward" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm2D.initDefault(allocator, 1);
    defer bn.deinit();

    // Simple input
    const input = [_]f32{ 1, 2, 3, 4 };

    const output = try bn.forward(&input, 1, 1, 2, 2);
    defer allocator.free(output);

    // Gradient from next layer
    const grad_output = [_]f32{ 1, 1, 1, 1 };

    var grads = try bn.backward(&grad_output, &input);
    defer grads.deinit();

    // Check gradient shapes
    try std.testing.expectEqual(@as(usize, 1), grads.gamma_grad.len);
    try std.testing.expectEqual(@as(usize, 1), grads.beta_grad.len);
    try std.testing.expectEqual(@as(usize, 4), grads.input_grad.len);

    // Beta gradient should be sum of grad_output = 4
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grads.beta_grad[0], 0.01);
}

test {
    std.testing.refAllDecls(@This());
}
