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

// ============================================================================
// Additional tensor operations
// ============================================================================

/// Element-wise subtraction: out = a - b
pub fn sub(a: *const Tensor, b: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(b.shape) or !a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const b_data = b.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av - bv;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const b_data = b.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av - bv;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise division: out = a / b
pub fn div(a: *const Tensor, b: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(b.shape) or !a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const b_data = b.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av / bv;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const b_data = b.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, b_data, out_data) |av, bv, *ov| {
                ov.* = av / bv;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Add scalar to tensor: out = a + scalar
pub fn addScalar(a: *const Tensor, scalar: f64, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const s: f32 = @floatCast(scalar);
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = av + s;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = av + scalar;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Multiply tensor by scalar: out = a * scalar
pub fn mulScalar(a: *const Tensor, scalar: f64, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const s: f32 = @floatCast(scalar);
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = av * s;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = av * scalar;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise power: out = a ^ exponent
pub fn pow(a: *const Tensor, exponent: f64, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const e: f32 = @floatCast(exponent);
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = std.math.pow(f32, av, e);
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = std.math.pow(f64, av, exponent);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise square root: out = sqrt(a)
pub fn sqrt(a: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = @sqrt(av);
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = @sqrt(av);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise exponential: out = exp(a)
pub fn exp(a: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = @exp(av);
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = @exp(av);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise natural log: out = ln(a)
pub fn log(a: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = @log(av);
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = @log(av);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise absolute value: out = |a|
pub fn abs(a: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = @abs(av);
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = @abs(av);
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise negation: out = -a
pub fn neg(a: *const Tensor, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = -av;
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = -av;
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

/// Element-wise clamp: out = clamp(a, min_val, max_val)
pub fn clamp(a: *const Tensor, min_val: f64, max_val: f64, out: *Tensor) !void {
    if (!a.shape.eql(out.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const min_f32: f32 = @floatCast(min_val);
            const max_f32: f32 = @floatCast(max_val);
            const a_data = a.asConstSlice(f32);
            const out_data = out.asSlice(f32);
            for (a_data, out_data) |av, *ov| {
                ov.* = @max(min_f32, @min(max_f32, av));
            }
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const out_data = out.asSlice(f64);
            for (a_data, out_data) |av, *ov| {
                ov.* = @max(min_val, @min(max_val, av));
            }
        },
        else => return TensorError.InvalidDataType,
    }
}

// ============================================================================
// Reduction operations
// ============================================================================

/// Sum all elements in tensor
pub fn sum(a: *const Tensor) f64 {
    switch (a.dtype) {
        .f32 => {
            var total: f64 = 0.0;
            for (a.asConstSlice(f32)) |v| {
                total += v;
            }
            return total;
        },
        .f64 => {
            var total: f64 = 0.0;
            for (a.asConstSlice(f64)) |v| {
                total += v;
            }
            return total;
        },
        else => return 0.0,
    }
}

/// Mean of all elements in tensor
pub fn mean(a: *const Tensor) f64 {
    const n = a.numel();
    if (n == 0) return 0.0;
    return sum(a) / @as(f64, @floatFromInt(n));
}

/// Variance of all elements in tensor
pub fn variance(a: *const Tensor) f64 {
    const n = a.numel();
    if (n == 0) return 0.0;

    const m = mean(a);

    switch (a.dtype) {
        .f32 => {
            var var_sum: f64 = 0.0;
            for (a.asConstSlice(f32)) |v| {
                const diff = @as(f64, v) - m;
                var_sum += diff * diff;
            }
            return var_sum / @as(f64, @floatFromInt(n));
        },
        .f64 => {
            var var_sum: f64 = 0.0;
            for (a.asConstSlice(f64)) |v| {
                const diff = v - m;
                var_sum += diff * diff;
            }
            return var_sum / @as(f64, @floatFromInt(n));
        },
        else => return 0.0,
    }
}

/// Standard deviation of all elements in tensor
pub fn std_dev(a: *const Tensor) f64 {
    return @sqrt(variance(a));
}

/// Maximum element in tensor
pub fn max(a: *const Tensor) f64 {
    switch (a.dtype) {
        .f32 => {
            const data = a.asConstSlice(f32);
            if (data.len == 0) return 0.0;
            var max_val: f32 = data[0];
            for (data[1..]) |v| {
                if (v > max_val) max_val = v;
            }
            return max_val;
        },
        .f64 => {
            const data = a.asConstSlice(f64);
            if (data.len == 0) return 0.0;
            var max_val: f64 = data[0];
            for (data[1..]) |v| {
                if (v > max_val) max_val = v;
            }
            return max_val;
        },
        else => return 0.0,
    }
}

/// Minimum element in tensor
pub fn min(a: *const Tensor) f64 {
    switch (a.dtype) {
        .f32 => {
            const data = a.asConstSlice(f32);
            if (data.len == 0) return 0.0;
            var min_val: f32 = data[0];
            for (data[1..]) |v| {
                if (v < min_val) min_val = v;
            }
            return min_val;
        },
        .f64 => {
            const data = a.asConstSlice(f64);
            if (data.len == 0) return 0.0;
            var min_val: f64 = data[0];
            for (data[1..]) |v| {
                if (v < min_val) min_val = v;
            }
            return min_val;
        },
        else => return 0.0,
    }
}

/// Index of maximum element (argmax)
pub fn argmax(a: *const Tensor) usize {
    switch (a.dtype) {
        .f32 => {
            const data = a.asConstSlice(f32);
            if (data.len == 0) return 0;
            var max_idx: usize = 0;
            var max_val: f32 = data[0];
            for (data[1..], 1..) |v, i| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            return max_idx;
        },
        .f64 => {
            const data = a.asConstSlice(f64);
            if (data.len == 0) return 0;
            var max_idx: usize = 0;
            var max_val: f64 = data[0];
            for (data[1..], 1..) |v, i| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            return max_idx;
        },
        else => return 0,
    }
}

/// Index of minimum element (argmin)
pub fn argmin(a: *const Tensor) usize {
    switch (a.dtype) {
        .f32 => {
            const data = a.asConstSlice(f32);
            if (data.len == 0) return 0;
            var min_idx: usize = 0;
            var min_val: f32 = data[0];
            for (data[1..], 1..) |v, i| {
                if (v < min_val) {
                    min_val = v;
                    min_idx = i;
                }
            }
            return min_idx;
        },
        .f64 => {
            const data = a.asConstSlice(f64);
            if (data.len == 0) return 0;
            var min_idx: usize = 0;
            var min_val: f64 = data[0];
            for (data[1..], 1..) |v, i| {
                if (v < min_val) {
                    min_val = v;
                    min_idx = i;
                }
            }
            return min_idx;
        },
        else => return 0,
    }
}

/// Dot product of two 1D tensors
pub fn dot(a: *const Tensor, b: *const Tensor) !f64 {
    if (a.shape.ndim != 1 or b.shape.ndim != 1) {
        return TensorError.DimensionMismatch;
    }
    if (!a.shape.eql(b.shape)) {
        return TensorError.ShapeMismatch;
    }

    switch (a.dtype) {
        .f32 => {
            const a_data = a.asConstSlice(f32);
            const b_data = b.asConstSlice(f32);
            var result: f64 = 0.0;
            for (a_data, b_data) |av, bv| {
                result += @as(f64, av) * @as(f64, bv);
            }
            return result;
        },
        .f64 => {
            const a_data = a.asConstSlice(f64);
            const b_data = b.asConstSlice(f64);
            var result: f64 = 0.0;
            for (a_data, b_data) |av, bv| {
                result += av * bv;
            }
            return result;
        },
        else => return TensorError.InvalidDataType,
    }
}

/// L2 norm (Euclidean norm) of tensor
pub fn norm(a: *const Tensor) f64 {
    switch (a.dtype) {
        .f32 => {
            var sum_sq: f64 = 0.0;
            for (a.asConstSlice(f32)) |v| {
                sum_sq += @as(f64, v) * @as(f64, v);
            }
            return @sqrt(sum_sq);
        },
        .f64 => {
            var sum_sq: f64 = 0.0;
            for (a.asConstSlice(f64)) |v| {
                sum_sq += v * v;
            }
            return @sqrt(sum_sq);
        },
        else => return 0.0,
    }
}

// ============================================================================
// Tests for new operations
// ============================================================================

test "sub" {
    const allocator = std.testing.allocator;

    const a_data = [_]f32{ 5, 7, 9 };
    const b_data = [_]f32{ 1, 2, 3 };

    var a = try Tensor.fromSlice(allocator, f32, &a_data, &.{3});
    defer a.deinit();

    var b = try Tensor.fromSlice(allocator, f32, &b_data, &.{3});
    defer b.deinit();

    var out = try Tensor.zeros(allocator, &.{3}, .f32);
    defer out.deinit();

    try sub(a, b, out);

    const result = out.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 4.0), result[0]);
    try std.testing.expectEqual(@as(f32, 5.0), result[1]);
    try std.testing.expectEqual(@as(f32, 6.0), result[2]);
}

test "mulScalar" {
    const allocator = std.testing.allocator;

    const a_data = [_]f32{ 1, 2, 3 };
    var a = try Tensor.fromSlice(allocator, f32, &a_data, &.{3});
    defer a.deinit();

    var out = try Tensor.zeros(allocator, &.{3}, .f32);
    defer out.deinit();

    try mulScalar(a, 2.0, out);

    const result = out.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 2.0), result[0]);
    try std.testing.expectEqual(@as(f32, 4.0), result[1]);
    try std.testing.expectEqual(@as(f32, 6.0), result[2]);
}

test "sum and mean" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 1, 2, 3, 4, 5 };
    var a = try Tensor.fromSlice(allocator, f32, &data, &.{5});
    defer a.deinit();

    const s = sum(a);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), s, 0.001);

    const m = mean(a);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), m, 0.001);
}

test "max and min" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a = try Tensor.fromSlice(allocator, f32, &data, &.{8});
    defer a.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 9.0), max(a), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), min(a), 0.001);
}

test "argmax and argmin" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a = try Tensor.fromSlice(allocator, f32, &data, &.{8});
    defer a.deinit();

    try std.testing.expectEqual(@as(usize, 5), argmax(a)); // 9 is at index 5
    try std.testing.expectEqual(@as(usize, 1), argmin(a)); // first 1 is at index 1
}

test "norm" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 3, 4 };
    var a = try Tensor.fromSlice(allocator, f32, &data, &.{2});
    defer a.deinit();

    // sqrt(3^2 + 4^2) = sqrt(25) = 5
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), norm(a), 0.001);
}
