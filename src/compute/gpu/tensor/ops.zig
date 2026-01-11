//! Tensor operations.
//!
//! Core tensor operations including arithmetic, activations,
//! and linear algebra.

const std = @import("std");
const types = @import("types.zig");
const Tensor = types.Tensor;
const TensorShape = types.TensorShape;
const TensorError = types.TensorError;
const DataType = types.DataType;

/// Matrix multiplication: C = A @ B
/// A: (M, K), B: (K, N) -> C: (M, N)
pub fn matmul(a: *const Tensor, b: *const Tensor, out: *Tensor) !void {
    // Validate shapes
    if (a.shape.ndim < 2 or b.shape.ndim < 2) {
        return TensorError.DimensionMismatch;
    }

    const m = a.shape.dim(a.shape.ndim - 2);
    const k1 = a.shape.dim(a.shape.ndim - 1);
    const k2 = b.shape.dim(b.shape.ndim - 2);
    const n = b.shape.dim(b.shape.ndim - 1);

    if (k1 != k2) return TensorError.DimensionMismatch;

    const expected_m = out.shape.dim(out.shape.ndim - 2);
    const expected_n = out.shape.dim(out.shape.ndim - 1);
    if (m != expected_m or n != expected_n) return TensorError.ShapeMismatch;

    // CPU fallback
    switch (a.dtype) {
        .f32 => matmulF32(a, b, out, m, k1, n),
        .f64 => matmulF64(a, b, out, m, k1, n),
        else => return TensorError.InvalidDataType,
    }
}

fn matmulF32(a: *const Tensor, b: *const Tensor, out: *Tensor, m: usize, k: usize, n: usize) void {
    const a_data = a.asConstSlice(f32);
    const b_data = b.asConstSlice(f32);
    const out_data = out.asSlice(f32);

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                sum += a_data[i * k + kk] * b_data[kk * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
}

fn matmulF64(a: *const Tensor, b: *const Tensor, out: *Tensor, m: usize, k: usize, n: usize) void {
    const a_data = a.asConstSlice(f64);
    const b_data = b.asConstSlice(f64);
    const out_data = out.asSlice(f64);

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |ki| {
                sum += a_data[i * k + ki] * b_data[ki * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
}

/// Element-wise addition: out = a + b
pub fn add(a: *const Tensor, b: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(b.shape) or !a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const b_data = b.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av + bv;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const b_data = b.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av + bv;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise multiplication: out = a * b
pub fn mul(a: *const Tensor, b: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(b.shape) or !a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const b_data = b.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av * bv;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const b_data = b.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av * bv;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// ReLU activation: out = max(0, input)
pub fn relu(input: *const Tensor, out: *Tensor) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (input.dtype) {
        .f32 => {
            const in_data = input.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (in_data, out_data) |v, *o| {
                o.* = @max(0.0, v);
            }
        },
        .f64 => {
            const in_data = input.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (in_data, out_data) |v, *o| {
                o.* = @max(0.0, v);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Sigmoid activation: out = 1 / (1 + exp(-input))
pub fn sigmoid(input: *const Tensor, out: *Tensor) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (input.dtype) {
        .f32 => {
            const in_data = input.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (in_data, out_data) |v, *o| {
                o.* = 1.0 / (1.0 + @exp(-v));
            }
        },
        .f64 => {
            const in_data = input.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (in_data, out_data) |v, *o| {
                o.* = 1.0 / (1.0 + @exp(-v));
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Tanh activation
pub fn tanh(input: *const Tensor, out: *Tensor) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (input.dtype) {
        .f32 => {
            const in_data = input.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (in_data, out_data) |v, *o| {
                o.* = std.math.tanh(v);
            }
        },
        .f64 => {
            const in_data = input.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (in_data, out_data) |v, *o| {
                o.* = std.math.tanh(v);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// GELU activation (Gaussian Error Linear Unit)
pub fn gelu(input: *const Tensor, out: *Tensor) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const gelu_const: f32 = 0.044715;

    switch (input.dtype) {
        .f32 => {
            const in_data = input.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (in_data, out_data) |x, *o| {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                const x3 = x * x * x;
                const inner = sqrt_2_over_pi * (x + gelu_const * x3);
                o.* = 0.5 * x * (1.0 + std.math.tanh(inner));
            }
        },
        .f64 => {
            const in_data = input.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (in_data, out_data) |x, *o| {
                const x3 = x * x * x;
                const inner = 0.7978845608028654 * (x + 0.044715 * x3);
                o.* = 0.5 * x * (1.0 + std.math.tanh(inner));
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Softmax activation along last dimension
pub fn softmax(input: *const Tensor, out: *Tensor) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }
    if (input.shape.ndim == 0) return TensorError.InvalidShape;

    const last_dim = input.shape.dim(input.shape.ndim - 1);
    const batch_size = input.numel() / last_dim;

    switch (input.dtype) {
        .f32 => softmaxF32(input, out, batch_size, last_dim),
        .f64 => softmaxF64(input, out, batch_size, last_dim),
        else => return TensorError.InvalidDataType,
    }
}

fn softmaxF32(input: *const Tensor, out: *Tensor, batch_size: usize, dim: usize) void {
    const in_data = input.asConstSlice(f32);
    const out_data = out.asSlice(f32);

    for (0..batch_size) |b| {
        const offset = b * dim;
        const slice = in_data[offset .. offset + dim];
        const out_slice = out_data[offset .. offset + dim];

        // Find max for numerical stability
        var max_val: f32 = slice[0];
        for (slice[1..]) |v| {
            max_val = @max(max_val, v);
        }

        // Compute exp and sum
        var sum: f32 = 0.0;
        for (slice, out_slice) |v, *o| {
            o.* = @exp(v - max_val);
            sum += o.*;
        }

        // Normalize
        for (out_slice) |*o| {
            o.* /= sum;
        }
    }
}

fn softmaxF64(input: *const Tensor, out: *Tensor, batch_size: usize, dim: usize) void {
    const in_data = input.asConstSlice(f64);
    const out_data = out.asSlice(f64);

    for (0..batch_size) |b| {
        const offset = b * dim;
        const slice = in_data[offset .. offset + dim];
        const out_slice = out_data[offset .. offset + dim];

        var max_val: f64 = slice[0];
        for (slice[1..]) |v| {
            max_val = @max(max_val, v);
        }

        var sum: f64 = 0.0;
        for (slice, out_slice) |v, *o| {
            o.* = @exp(v - max_val);
            sum += o.*;
        }

        for (out_slice) |*o| {
            o.* /= sum;
        }
    }
}

/// Layer normalization
pub fn layerNorm(input: *const Tensor, gamma: *const Tensor, beta: *const Tensor, out: *Tensor, eps: f32) !void {
    if (!input.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }
    if (input.shape.ndim == 0) return TensorError.InvalidShape;

    const norm_dim = input.shape.dim(input.shape.ndim - 1);
    const batch_size = input.numel() / norm_dim;

    switch (input.dtype) {
        .f32 => layerNormF32(input, gamma, beta, out, batch_size, norm_dim, eps),
        else => return TensorError.InvalidDataType,
    }
}

fn layerNormF32(input: *const Tensor, gamma: *const Tensor, beta: *const Tensor, out: *Tensor, batch_size: usize, dim: usize, eps: f32) void {
    const in_data = input.asConstSlice(f32);
    const out_data = out.asSlice(f32);
    const gamma_data = gamma.asConstSlice(f32);
    const beta_data = beta.asConstSlice(f32);

    for (0..batch_size) |b| {
        const offset = b * dim;
        const slice = in_data[offset .. offset + dim];
        const out_slice = out_data[offset .. offset + dim];

        // Compute mean
        var mean: f32 = 0.0;
        for (slice) |v| {
            mean += v;
        }
        mean /= @floatFromInt(dim);

        // Compute variance
        var variance: f32 = 0.0;
        for (slice) |v| {
            const diff = v - mean;
            variance += diff * diff;
        }
        variance /= @floatFromInt(dim);

        // Normalize
        const inv_std = 1.0 / @sqrt(variance + eps);
        for (slice, out_slice, gamma_data, beta_data) |v, *o, g, be| {
            o.* = (v - mean) * inv_std * g + be;
        }
    }
}

/// Transpose last two dimensions
pub fn transpose(input: *const Tensor, out: *Tensor) !void {
    if (input.shape.ndim < 2) return TensorError.DimensionMismatch;

    // Validate output shape
    const in_rows = input.shape.dim(input.shape.ndim - 2);
    const in_cols = input.shape.dim(input.shape.ndim - 1);
    const out_rows = out.shape.dim(out.shape.ndim - 2);
    const out_cols = out.shape.dim(out.shape.ndim - 1);

    if (in_rows != out_cols or in_cols != out_rows) {
        return TensorError.ShapeMismatch;
    }

    switch (input.dtype) {
        .f32 => transposeF32(input, out, in_rows, in_cols),
        .f64 => transposeF64(input, out, in_rows, in_cols),
        else => return TensorError.InvalidDataType,
    }
}

fn transposeF32(input: *const Tensor, out: *Tensor, rows: usize, cols: usize) void {
    const in_data = input.asConstSlice(f32);
    const out_data = out.asSlice(f32);

    for (0..rows) |i| {
        for (0..cols) |j| {
            out_data[j * rows + i] = in_data[i * cols + j];
        }
    }
}

fn transposeF64(input: *const Tensor, out: *Tensor, rows: usize, cols: usize) void {
    const in_data = input.asConstSlice(f64);
    const out_data = out.asSlice(f64);

    for (0..rows) |i| {
        for (0..cols) |j| {
            out_data[j * rows + i] = in_data[i * cols + j];
        }
    }
}

/// Reshape tensor (creates a view if possible).
pub fn reshape(input: *Tensor, new_shape: []const usize) !*Tensor {
    return input.view(new_shape);
}

test "relu" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ -2, -1, 0, 1, 2 };
    var input = try Tensor.fromSlice(allocator, f32, &data, &.{5});
    defer input.deinit();

    var out = try Tensor.zeros(allocator, &.{5}, .f32);
    defer out.deinit();

    try relu(input, out);

    const result = out.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 0.0), result[0]);
    try std.testing.expectEqual(@as(f32, 0.0), result[1]);
    try std.testing.expectEqual(@as(f32, 0.0), result[2]);
    try std.testing.expectEqual(@as(f32, 1.0), result[3]);
    try std.testing.expectEqual(@as(f32, 2.0), result[4]);
}

test "sigmoid" {
    const allocator = std.testing.allocator;

    const data = [_]f32{0};
    var input = try Tensor.fromSlice(allocator, f32, &data, &.{1});
    defer input.deinit();

    var out = try Tensor.zeros(allocator, &.{1}, .f32);
    defer out.deinit();

    try sigmoid(input, out);

    const result = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result[0], 0.0001);
}

test "softmax" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 1, 2, 3 };
    var input = try Tensor.fromSlice(allocator, f32, &data, &.{3});
    defer input.deinit();

    var out = try Tensor.zeros(allocator, &.{3}, .f32);
    defer out.deinit();

    try softmax(input, out);

    const result = out.asSlice(f32);

    // Sum should be 1
    var sum: f32 = 0.0;
    for (result) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.0001);

    // Values should be increasing
    try std.testing.expect(result[0] < result[1]);
    try std.testing.expect(result[1] < result[2]);
}

test "add" {
    const allocator = std.testing.allocator;

    const a_data = [_]f32{ 1, 2, 3 };
    const b_data = [_]f32{ 4, 5, 6 };

    var a = try Tensor.fromSlice(allocator, f32, &a_data, &.{3});
    defer a.deinit();

    var b = try Tensor.fromSlice(allocator, f32, &b_data, &.{3});
    defer b.deinit();

    var out = try Tensor.zeros(allocator, &.{3}, .f32);
    defer out.deinit();

    try add(a, b, out);

    const result = out.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 5.0), result[0]);
    try std.testing.expectEqual(@as(f32, 7.0), result[1]);
    try std.testing.expectEqual(@as(f32, 9.0), result[2]);
}
