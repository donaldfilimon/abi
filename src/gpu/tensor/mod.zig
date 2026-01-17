//! GPU tensor operations.
//!
//! Provides GPU-accelerated tensor operations for neural network inference
//! and general-purpose numerical computing.

const std = @import("std");
const build_options = @import("build_options");

pub const types = @import("types.zig");
pub const ops = @import("ops.zig");

// Type exports
pub const Tensor = types.Tensor;
pub const TensorShape = types.TensorShape;
pub const DataType = types.DataType;
pub const TensorLayout = types.TensorLayout;
pub const Device = types.Device;
pub const TensorError = types.TensorError;

// Operation exports
pub const matmul = ops.matmul;
pub const add = ops.add;
pub const mul = ops.mul;
pub const sub = ops.sub;
pub const div = ops.div;
pub const relu = ops.relu;
pub const sigmoid = ops.sigmoid;
pub const tanh = ops.tanh;
pub const softmax = ops.softmax;
pub const gelu = ops.gelu;
pub const layerNorm = ops.layerNorm;
pub const transpose = ops.transpose;
pub const reshape = ops.reshape;

// Scalar operations
pub const addScalar = ops.addScalar;
pub const mulScalar = ops.mulScalar;
pub const pow = ops.pow;
pub const sqrt = ops.sqrt;
pub const exp = ops.exp;
pub const log = ops.log;
pub const abs = ops.abs;
pub const neg = ops.neg;
pub const clamp = ops.clamp;

// Reduction operations
pub const sum = ops.sum;
pub const mean = ops.mean;
pub const variance = ops.variance;
pub const std_dev = ops.std_dev;
pub const max = ops.max;
pub const min = ops.min;
pub const argmax = ops.argmax;
pub const argmin = ops.argmin;
pub const dot = ops.dot;
pub const norm = ops.norm;

/// Tensor operation configuration.
pub const TensorConfig = struct {
    /// Default device for new tensors.
    default_device: Device = .cpu,
    /// Enable operation fusion.
    enable_fusion: bool = true,
    /// Enable automatic mixed precision.
    enable_amp: bool = false,
    /// Memory pool size (bytes).
    memory_pool_size: usize = 256 * 1024 * 1024,
};

/// Global tensor configuration.
var global_config: TensorConfig = .{};

/// Initialize tensor subsystem.
pub fn init(config: TensorConfig) void {
    global_config = config;
}

/// Get current configuration.
pub fn getConfig() TensorConfig {
    return global_config;
}

/// Create a tensor filled with zeros.
pub fn zeros(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
    return Tensor.zeros(allocator, shape, dtype);
}

/// Create a tensor filled with ones.
pub fn ones(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
    return Tensor.ones(allocator, shape, dtype);
}

/// Create a tensor from data.
pub fn fromSlice(allocator: std.mem.Allocator, comptime T: type, data: []const T, shape: []const usize) !*Tensor {
    return Tensor.fromSlice(allocator, T, data, shape);
}

/// Create a random tensor (uniform distribution).
pub fn rand(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
    return Tensor.rand(allocator, shape, dtype);
}

/// Create a random tensor (normal distribution).
pub fn randn(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
    return Tensor.randn(allocator, shape, dtype);
}

test "tensor creation" {
    const allocator = std.testing.allocator;

    var t = try zeros(allocator, &.{ 2, 3 }, .f32);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 2), t.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), t.shape.dims[1]);
}
