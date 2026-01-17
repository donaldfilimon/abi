//! Abbey Neural Network Layers
//!
//! Building blocks for Abbey's learning architecture.
//! Supports online learning with gradient tracking.

const std = @import("std");
const tensor = @import("tensor.zig");
const types = @import("../core/types.zig");

const F32Tensor = tensor.F32Tensor;

// ============================================================================
// Layer Interface
// ============================================================================

/// Error set for layer operations.
/// Layer implementations may encounter various errors during forward/backward passes.
pub const LayerError = error{
    /// No forward pass was performed before backward
    NoForwardPass,
    /// Index is out of bounds
    IndexOutOfBounds,
    /// Shape mismatch between tensors
    ShapeMismatch,
    /// Tensor shape is invalid for the requested operation
    InvalidShape,
    /// Tensor dimensions are incompatible
    IncompatibleDimensions,
    /// Numerical overflow or underflow
    NumericalInstability,
} || std.mem.Allocator.Error;

/// Abstract layer interface
pub const Layer = struct {
    vtable: *const VTable,
    ptr: *anyopaque,

    /// VTable uses LayerError to keep the interface explicit across implementations.
    pub const VTable = struct {
        forward: *const fn (*anyopaque, *const F32Tensor) LayerError!F32Tensor,
        backward: *const fn (*anyopaque, *const F32Tensor) LayerError!F32Tensor,
        updateParams: *const fn (*anyopaque, f32) LayerError!void,
        zeroGrad: *const fn (*anyopaque) void,
        deinit: *const fn (*anyopaque) void,
        paramCount: *const fn (*anyopaque) usize,
    };

    pub fn forward(self: Layer, input: *const F32Tensor) LayerError!F32Tensor {
        return self.vtable.forward(self.ptr, input);
    }

    pub fn backward(self: Layer, grad_output: *const F32Tensor) LayerError!F32Tensor {
        return self.vtable.backward(self.ptr, grad_output);
    }

    pub fn updateParams(self: Layer, learning_rate: f32) LayerError!void {
        return self.vtable.updateParams(self.ptr, learning_rate);
    }

    pub fn zeroGrad(self: Layer) void {
        self.vtable.zeroGrad(self.ptr);
    }

    pub fn deinit(self: Layer) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn paramCount(self: Layer) usize {
        return self.vtable.paramCount(self.ptr);
    }
};

// ============================================================================
// Linear Layer
// ============================================================================

/// Fully connected linear layer: y = xW + b
pub const LinearLayer = struct {
    allocator: std.mem.Allocator,
    in_features: usize,
    out_features: usize,
    weights: F32Tensor,
    bias: F32Tensor,
    weight_grad: F32Tensor,
    bias_grad: F32Tensor,
    last_input: ?F32Tensor = null,
    use_bias: bool = true,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Self {
        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_features + out_features)));

        var weights = try F32Tensor.random(allocator, &.{ in_features, out_features }, -scale, scale);
        errdefer weights.deinit();

        var bias = try F32Tensor.zeros(allocator, &.{out_features});
        errdefer bias.deinit();

        var weight_grad = try F32Tensor.zeros(allocator, &.{ in_features, out_features });
        errdefer weight_grad.deinit();

        var bias_grad = try F32Tensor.zeros(allocator, &.{out_features});
        errdefer bias_grad.deinit();

        return Self{
            .allocator = allocator,
            .in_features = in_features,
            .out_features = out_features,
            .weights = weights,
            .bias = bias,
            .weight_grad = weight_grad,
            .bias_grad = bias_grad,
        };
    }

    pub fn deinit(self: *Self) void {
        self.weights.deinit();
        self.bias.deinit();
        self.weight_grad.deinit();
        self.bias_grad.deinit();
        if (self.last_input) |*li| {
            li.deinit();
        }
    }

    pub fn forward(self: *Self, input: *const F32Tensor) LayerError!F32Tensor {
        // Store input for backward pass
        if (self.last_input) |*li| {
            li.deinit();
        }
        self.last_input = try input.clone();

        // y = xW + b
        var output = try input.matmul(&self.weights);

        if (self.use_bias) {
            // Add bias to each row
            const batch_size = output.shape[0];
            for (0..batch_size) |b| {
                for (0..self.out_features) |o| {
                    output.data[b * self.out_features + o] += self.bias.data[o];
                }
            }
        }

        return output;
    }

    pub fn backward(self: *Self, grad_output: *const F32Tensor) LayerError!F32Tensor {
        const input = self.last_input orelse return error.NoForwardPass;

        // Gradient w.r.t. weights: dW = x^T @ dL/dy
        var input_t = try input.transpose();
        defer input_t.deinit();
        var dw = try input_t.matmul(grad_output);
        defer dw.deinit();
        try self.weight_grad.addInPlace(&dw);

        // Gradient w.r.t. bias: db = sum(dL/dy, axis=0)
        if (self.use_bias) {
            const batch_size = grad_output.shape[0];
            for (0..self.out_features) |o| {
                var sum: f32 = 0;
                for (0..batch_size) |b| {
                    sum += grad_output.data[b * self.out_features + o];
                }
                self.bias_grad.data[o] += sum;
            }
        }

        // Gradient w.r.t. input: dx = dL/dy @ W^T
        var weights_t = try self.weights.transpose();
        defer weights_t.deinit();
        return grad_output.matmul(&weights_t);
    }

    pub fn updateParams(self: *Self, learning_rate: f32) LayerError!void {
        // SGD update: W = W - lr * dW
        for (self.weights.data, 0..) |*w, i| {
            w.* -= learning_rate * self.weight_grad.data[i];
        }
        if (self.use_bias) {
            for (self.bias.data, 0..) |*b, i| {
                b.* -= learning_rate * self.bias_grad.data[i];
            }
        }
    }

    pub fn zeroGrad(self: *Self) void {
        @memset(self.weight_grad.data, 0);
        @memset(self.bias_grad.data, 0);
    }

    pub fn paramCount(self: *const Self) usize {
        var count = self.in_features * self.out_features;
        if (self.use_bias) count += self.out_features;
        return count;
    }

    pub fn layer(self: *Self) Layer {
        return .{
            .vtable = &.{
                .forward = @ptrCast(&forward),
                .backward = @ptrCast(&backward),
                .updateParams = @ptrCast(&updateParams),
                .zeroGrad = @ptrCast(&zeroGrad),
                .deinit = @ptrCast(&deinit),
                .paramCount = @ptrCast(&paramCount),
            },
            .ptr = self,
        };
    }
};

// ============================================================================
// Embedding Layer
// ============================================================================

/// Lookup table for discrete tokens
pub const EmbeddingLayer = struct {
    allocator: std.mem.Allocator,
    num_embeddings: usize,
    embedding_dim: usize,
    embeddings: F32Tensor,
    embedding_grad: F32Tensor,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_embeddings: usize, embedding_dim: usize) !Self {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(embedding_dim)));

        var embeddings = try F32Tensor.random(allocator, &.{ num_embeddings, embedding_dim }, -scale, scale);
        errdefer embeddings.deinit();

        var embedding_grad = try F32Tensor.zeros(allocator, &.{ num_embeddings, embedding_dim });
        errdefer embedding_grad.deinit();

        return Self{
            .allocator = allocator,
            .num_embeddings = num_embeddings,
            .embedding_dim = embedding_dim,
            .embeddings = embeddings,
            .embedding_grad = embedding_grad,
        };
    }

    pub fn deinit(self: *Self) void {
        self.embeddings.deinit();
        self.embedding_grad.deinit();
    }

    /// Look up embeddings for token indices
    pub fn lookup(self: *const Self, indices: []const usize) !F32Tensor {
        var result = try F32Tensor.init(self.allocator, &.{ indices.len, self.embedding_dim });

        for (indices, 0..) |idx, i| {
            if (idx >= self.num_embeddings) return error.IndexOutOfBounds;
            const start = idx * self.embedding_dim;
            @memcpy(
                result.data[i * self.embedding_dim .. (i + 1) * self.embedding_dim],
                self.embeddings.data[start .. start + self.embedding_dim],
            );
        }

        return result;
    }

    /// Accumulate gradients for indices
    pub fn accumulateGrad(self: *Self, indices: []const usize, grad: *const F32Tensor) void {
        for (indices, 0..) |idx, i| {
            if (idx >= self.num_embeddings) continue;
            const emb_start = idx * self.embedding_dim;
            const grad_start = i * self.embedding_dim;
            for (0..self.embedding_dim) |d| {
                self.embedding_grad.data[emb_start + d] += grad.data[grad_start + d];
            }
        }
    }

    pub fn updateParams(self: *Self, learning_rate: f32) void {
        for (self.embeddings.data, 0..) |*e, i| {
            e.* -= learning_rate * self.embedding_grad.data[i];
        }
    }

    pub fn zeroGrad(self: *Self) void {
        @memset(self.embedding_grad.data, 0);
    }
};

// ============================================================================
// Layer Normalization
// ============================================================================

/// Layer normalization for stable training
pub const LayerNorm = struct {
    allocator: std.mem.Allocator,
    normalized_shape: usize,
    gamma: F32Tensor, // Scale
    beta: F32Tensor, // Shift
    gamma_grad: F32Tensor,
    beta_grad: F32Tensor,
    epsilon: f32 = 1e-5,
    last_input: ?F32Tensor = null,
    last_normalized: ?F32Tensor = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, normalized_shape: usize) !Self {
        var gamma = try F32Tensor.ones(allocator, &.{normalized_shape});
        errdefer gamma.deinit();

        var beta = try F32Tensor.zeros(allocator, &.{normalized_shape});
        errdefer beta.deinit();

        var gamma_grad = try F32Tensor.zeros(allocator, &.{normalized_shape});
        errdefer gamma_grad.deinit();

        var beta_grad = try F32Tensor.zeros(allocator, &.{normalized_shape});
        errdefer beta_grad.deinit();

        return Self{
            .allocator = allocator,
            .normalized_shape = normalized_shape,
            .gamma = gamma,
            .beta = beta,
            .gamma_grad = gamma_grad,
            .beta_grad = beta_grad,
        };
    }

    pub fn deinit(self: *Self) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.gamma_grad.deinit();
        self.beta_grad.deinit();
        if (self.last_input) |*li| li.deinit();
        if (self.last_normalized) |*ln| ln.deinit();
    }

    pub fn forward(self: *Self, input: *const F32Tensor) !F32Tensor {
        // Store for backward
        if (self.last_input) |*li| li.deinit();
        self.last_input = try input.clone();

        const batch_size = input.data.len / self.normalized_shape;
        var output = try F32Tensor.init(self.allocator, input.shape);

        for (0..batch_size) |b| {
            const start = b * self.normalized_shape;
            const end = start + self.normalized_shape;
            const slice = input.data[start..end];

            // Compute mean
            var sum: f32 = 0;
            for (slice) |v| sum += v;
            const mean_val = sum / @as(f32, @floatFromInt(self.normalized_shape));

            // Compute variance
            var var_sum: f32 = 0;
            for (slice) |v| {
                const diff = v - mean_val;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(self.normalized_shape));
            const std_dev = @sqrt(variance + self.epsilon);

            // Normalize and scale
            for (0..self.normalized_shape) |i| {
                const normalized = (slice[i] - mean_val) / std_dev;
                output.data[start + i] = normalized * self.gamma.data[i] + self.beta.data[i];
            }
        }

        if (self.last_normalized) |*ln| ln.deinit();
        self.last_normalized = try output.clone();

        return output;
    }

    pub fn zeroGrad(self: *Self) void {
        @memset(self.gamma_grad.data, 0);
        @memset(self.beta_grad.data, 0);
    }
};

// ============================================================================
// Dropout Layer
// ============================================================================

/// Dropout for regularization
pub const Dropout = struct {
    allocator: std.mem.Allocator,
    p: f32, // Dropout probability
    training: bool = true,
    mask: ?F32Tensor = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, p: f32) Self {
        return Self{
            .allocator = allocator,
            .p = p,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.mask) |*m| m.deinit();
    }

    pub fn forward(self: *Self, input: *const F32Tensor) !F32Tensor {
        if (!self.training or self.p == 0) {
            return input.clone();
        }

        var output = try F32Tensor.init(self.allocator, input.shape);

        // Generate mask
        if (self.mask) |*m| m.deinit();
        self.mask = try F32Tensor.init(self.allocator, input.shape);

        var prng = std.Random.DefaultPrng.init(@intCast(types.getTimestampNs() & 0xFFFFFFFFFFFFFFFF));
        const rand = prng.random();
        const scale = 1.0 / (1.0 - self.p);

        for (0..input.data.len) |i| {
            if (rand.float(f32) > self.p) {
                self.mask.?.data[i] = scale;
                output.data[i] = input.data[i] * scale;
            } else {
                self.mask.?.data[i] = 0;
                output.data[i] = 0;
            }
        }

        return output;
    }

    pub fn setTraining(self: *Self, training: bool) void {
        self.training = training;
    }
};

// ============================================================================
// ReLU Activation Layer
// ============================================================================

pub const ReLU = struct {
    last_input: ?F32Tensor = null,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.last_input) |*li| li.deinit();
    }

    pub fn forward(self: *Self, input: *const F32Tensor) !F32Tensor {
        if (self.last_input) |*li| li.deinit();
        self.last_input = try input.clone();
        return input.relu();
    }

    pub fn backward(self: *Self, grad_output: *const F32Tensor) !F32Tensor {
        const input = self.last_input orelse return error.NoForwardPass;

        var grad_input = try F32Tensor.init(self.allocator, input.shape);
        for (0..input.data.len) |i| {
            grad_input.data[i] = if (input.data[i] > 0) grad_output.data[i] else 0;
        }
        return grad_input;
    }
};

// ============================================================================
// GELU Activation (used in transformers)
// ============================================================================

pub const GELU = struct {
    last_input: ?F32Tensor = null,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.last_input) |*li| li.deinit();
    }

    pub fn forward(self: *Self, input: *const F32Tensor) !F32Tensor {
        if (self.last_input) |*li| li.deinit();
        self.last_input = try input.clone();

        var output = try F32Tensor.init(self.allocator, input.shape);

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const sqrt_2_over_pi: f32 = 0.7978845608;

        for (0..input.data.len) |i| {
            const x = input.data[i];
            const x3 = x * x * x;
            const inner = sqrt_2_over_pi * (x + 0.044715 * x3);
            output.data[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
        }

        return output;
    }
};

// ============================================================================
// Sequential Container
// ============================================================================

/// Sequential container for stacking layers
pub const Sequential = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayListUnmanaged(Layer),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .layers = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);
    }

    pub fn add(self: *Self, layer: Layer) !void {
        try self.layers.append(self.allocator, layer);
    }

    pub fn forward(self: *Self, input: *const F32Tensor) !F32Tensor {
        var current = try input.clone();

        for (self.layers.items) |layer| {
            const next = try layer.forward(&current);
            current.deinit();
            current = next;
        }

        return current;
    }

    pub fn backward(self: *Self, grad_output: *const F32Tensor) !F32Tensor {
        var current_grad = try grad_output.clone();

        // Backward pass in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const prev_grad = try self.layers.items[i].backward(&current_grad);
            current_grad.deinit();
            current_grad = prev_grad;
        }

        return current_grad;
    }

    pub fn updateParams(self: *Self, learning_rate: f32) !void {
        for (self.layers.items) |layer| {
            try layer.updateParams(learning_rate);
        }
    }

    pub fn zeroGrad(self: *Self) void {
        for (self.layers.items) |layer| {
            layer.zeroGrad();
        }
    }

    pub fn paramCount(self: *const Self) usize {
        var count: usize = 0;
        for (self.layers.items) |layer| {
            count += layer.paramCount();
        }
        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "linear layer forward" {
    const allocator = std.testing.allocator;

    var layer = try LinearLayer.init(allocator, 4, 3);
    defer layer.deinit();

    var input = try F32Tensor.fill(allocator, &.{ 2, 4 }, 1.0);
    defer input.deinit();

    var output = try layer.forward(&input);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[1]);
}

test "embedding lookup" {
    const allocator = std.testing.allocator;

    var emb = try EmbeddingLayer.init(allocator, 100, 16);
    defer emb.deinit();

    const indices = [_]usize{ 5, 10, 15 };
    var output = try emb.lookup(&indices);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 3), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 16), output.shape[1]);
}

test "layer norm" {
    const allocator = std.testing.allocator;

    var ln = try LayerNorm.init(allocator, 4);
    defer ln.deinit();

    var input = try F32Tensor.fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 4 });
    defer input.deinit();

    var output = try ln.forward(&input);
    defer output.deinit();

    // Check that each row is approximately normalized (mean ~0, var ~1)
    const first_row_mean = (output.data[0] + output.data[1] + output.data[2] + output.data[3]) / 4.0;
    try std.testing.expect(@abs(first_row_mean) < 0.01);
}
