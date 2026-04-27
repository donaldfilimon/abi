//! Tensor module for LLM operations.
//!
//! Provides multi-dimensional tensor abstraction with support for:
//! - Multiple data types (f32, f16, bf16, quantized)
//! - Zero-copy views and slicing
//! - Shape manipulation (reshape, transpose)
//! - Efficient memory layout for SIMD operations

const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const quantized = @import("quantized.zig");
pub const view = @import("view.zig");

// Re-exports
pub const Tensor = tensor.Tensor;
pub const DType = tensor.DType;
pub const Shape = tensor.Shape;
pub const TensorError = tensor.TensorError;

pub const Q4_0Block = quantized.Q4_0Block;
pub const Q4_1Block = quantized.Q4_1Block;
pub const Q8_0Block = quantized.Q8_0Block;
pub const QuantType = quantized.QuantType;
pub const dequantizeQ4_0 = quantized.dequantizeQ4_0;
pub const dequantizeQ4_1 = quantized.dequantizeQ4_1;
pub const dequantizeQ8_0 = quantized.dequantizeQ8_0;
pub const quantizeToQ4_1 = quantized.quantizeToQ4_1;
pub const quantizeToQ8_0 = quantized.quantizeToQ8_0;
pub const quantizedSize = quantized.quantizedSize;
pub const dequantizedSize = quantized.dequantizedSize;
pub const dotQ4_0F32 = quantized.dotQ4_0F32;
pub const dotQ4_1F32 = quantized.dotQ4_1F32;
pub const dotQ8_0F32 = quantized.dotQ8_0F32;

pub const TensorView = view.TensorView;

/// Create a tensor filled with zeros.
pub fn zeros(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
    return Tensor.zeros(allocator, shape, dtype);
}

/// Create a tensor filled with ones.
pub fn ones(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
    return Tensor.ones(allocator, shape, dtype);
}

/// Create a tensor from a slice of f32 values.
pub fn fromSlice(allocator: std.mem.Allocator, data: []const f32, shape: Shape) !Tensor {
    return Tensor.fromSlice(allocator, data, shape);
}

/// Create a 1D tensor (vector).
pub fn vector(allocator: std.mem.Allocator, data: []const f32) !Tensor {
    return fromSlice(allocator, data, .{ @intCast(data.len), 1, 1, 1 });
}

/// Create a 2D tensor (matrix).
pub fn matrix(allocator: std.mem.Allocator, data: []const f32, rows: u32, cols: u32) !Tensor {
    return fromSlice(allocator, data, .{ rows, cols, 1, 1 });
}

test "tensor module imports" {
    _ = tensor;
    _ = quantized;
    _ = view;
}

test {
    std.testing.refAllDecls(@This());
}
