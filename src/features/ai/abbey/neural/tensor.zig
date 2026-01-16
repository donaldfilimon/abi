//! Abbey Tensor Operations
//!
//! Lightweight tensor library optimized for online learning.
//! Supports SIMD acceleration and efficient memory layouts.

const std = @import("std");
const types = @import("../core/types.zig");

// ============================================================================
// Tensor Type
// ============================================================================

/// N-dimensional tensor with dynamic shape
pub fn Tensor(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        data: []T,
        shape: []usize,
        strides: []usize,
        requires_grad: bool = false,
        grad: ?*Self = null,

        const Self = @This();

        /// Create a new tensor with given shape
        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
            const total_size = computeSize(shape);
            const data = try allocator.alloc(T, total_size);
            @memset(data, 0);

            const shape_copy = try allocator.dupe(usize, shape);
            const strides = try computeStrides(allocator, shape);

            return Self{
                .allocator = allocator,
                .data = data,
                .shape = shape_copy,
                .strides = strides,
            };
        }

        /// Create from existing data
        pub fn fromSlice(allocator: std.mem.Allocator, data: []const T, shape: []const usize) !Self {
            const total_size = computeSize(shape);
            if (data.len != total_size) return error.ShapeMismatch;

            const data_copy = try allocator.dupe(T, data);
            const shape_copy = try allocator.dupe(usize, shape);
            const strides = try computeStrides(allocator, shape);

            return Self{
                .allocator = allocator,
                .data = data_copy,
                .shape = shape_copy,
                .strides = strides,
            };
        }

        /// Create a tensor filled with a value
        pub fn fill(allocator: std.mem.Allocator, shape: []const usize, value: T) !Self {
            var tensor = try init(allocator, shape);
            @memset(tensor.data, value);
            return tensor;
        }

        /// Create a tensor with random values
        pub fn random(allocator: std.mem.Allocator, shape: []const usize, min_val: T, max_val: T) !Self {
            var tensor = try init(allocator, shape);
            var prng = std.Random.DefaultPrng.init(@intCast(types.getTimestampNs() & 0xFFFFFFFFFFFFFFFF));
            const rand = prng.random();

            for (tensor.data) |*val| {
                const t = rand.float(f32);
                val.* = min_val + @as(T, @floatCast(t)) * (max_val - min_val);
            }
            return tensor;
        }

        /// Create zeros
        pub fn zeros(allocator: std.mem.Allocator, shape: []const usize) !Self {
            return fill(allocator, shape, 0);
        }

        /// Create ones
        pub fn ones(allocator: std.mem.Allocator, shape: []const usize) !Self {
            return fill(allocator, shape, 1);
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            if (self.grad) |grad| {
                grad.deinit();
                self.allocator.destroy(grad);
            }
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        /// Get total number of elements
        pub fn size(self: *const Self) usize {
            return self.data.len;
        }

        /// Get number of dimensions
        pub fn ndim(self: *const Self) usize {
            return self.shape.len;
        }

        /// Get element at index
        pub fn get(self: *const Self, indices: []const usize) T {
            const flat_idx = self.flatIndex(indices);
            return self.data[flat_idx];
        }

        /// Set element at index
        pub fn set(self: *Self, indices: []const usize, value: T) void {
            const flat_idx = self.flatIndex(indices);
            self.data[flat_idx] = value;
        }

        /// Convert multi-dimensional index to flat index
        fn flatIndex(self: *const Self, indices: []const usize) usize {
            var flat: usize = 0;
            for (indices, 0..) |idx, i| {
                flat += idx * self.strides[i];
            }
            return flat;
        }

        // ====================================================================
        // Element-wise Operations
        // ====================================================================

        /// Add two tensors element-wise
        pub fn add(self: *const Self, other: *const Self) !Self {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }

            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = self.data[i] + other.data[i];
            }
            return result;
        }

        /// Subtract two tensors element-wise
        pub fn sub(self: *const Self, other: *const Self) !Self {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }

            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = self.data[i] - other.data[i];
            }
            return result;
        }

        /// Multiply two tensors element-wise (Hadamard product)
        pub fn mul(self: *const Self, other: *const Self) !Self {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }

            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = self.data[i] * other.data[i];
            }
            return result;
        }

        /// Divide two tensors element-wise
        pub fn div(self: *const Self, other: *const Self) !Self {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }

            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = self.data[i] / other.data[i];
            }
            return result;
        }

        /// Scale tensor by scalar
        pub fn scale(self: *const Self, scalar: T) !Self {
            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = self.data[i] * scalar;
            }
            return result;
        }

        /// In-place scale
        pub fn scaleInPlace(self: *Self, scalar: T) void {
            for (self.data) |*val| {
                val.* = val.* * scalar;
            }
        }

        /// In-place add
        pub fn addInPlace(self: *Self, other: *const Self) !void {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }
            for (self.data, 0..) |*val, i| {
                val.* += other.data[i];
            }
        }

        // ====================================================================
        // Reduction Operations
        // ====================================================================

        /// Sum all elements
        pub fn sum(self: *const Self) T {
            var total: T = 0;
            for (self.data) |val| {
                total += val;
            }
            return total;
        }

        /// Mean of all elements
        pub fn mean(self: *const Self) T {
            return self.sum() / @as(T, @floatFromInt(self.data.len));
        }

        /// Maximum element
        pub fn max(self: *const Self) T {
            var max_val = self.data[0];
            for (self.data[1..]) |val| {
                if (val > max_val) max_val = val;
            }
            return max_val;
        }

        /// Minimum element
        pub fn min(self: *const Self) T {
            var min_val = self.data[0];
            for (self.data[1..]) |val| {
                if (val < min_val) min_val = val;
            }
            return min_val;
        }

        /// L2 norm
        pub fn norm(self: *const Self) T {
            var sum_sq: T = 0;
            for (self.data) |val| {
                sum_sq += val * val;
            }
            return @sqrt(sum_sq);
        }

        // ====================================================================
        // Activation Functions
        // ====================================================================

        /// Apply ReLU activation
        pub fn relu(self: *const Self) !Self {
            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = @max(0, self.data[i]);
            }
            return result;
        }

        /// Apply sigmoid activation
        pub fn sigmoid(self: *const Self) !Self {
            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = 1.0 / (1.0 + @exp(-self.data[i]));
            }
            return result;
        }

        /// Apply tanh activation
        pub fn tanh(self: *const Self) !Self {
            var result = try Self.init(self.allocator, self.shape);
            for (result.data, 0..) |*val, i| {
                val.* = std.math.tanh(self.data[i]);
            }
            return result;
        }

        /// Apply softmax along last dimension
        pub fn softmax(self: *const Self) !Self {
            if (self.shape.len == 0) return error.InvalidShape;

            var result = try Self.init(self.allocator, self.shape);
            const last_dim = self.shape[self.shape.len - 1];
            const batch_size = self.data.len / last_dim;

            var batch: usize = 0;
            while (batch < batch_size) : (batch += 1) {
                const start = batch * last_dim;
                const end = start + last_dim;

                // Find max for numerical stability
                var max_val = self.data[start];
                for (self.data[start..end]) |val| {
                    if (val > max_val) max_val = val;
                }

                // Compute exp and sum
                var exp_sum: T = 0;
                for (result.data[start..end], 0..) |*val, i| {
                    val.* = @exp(self.data[start + i] - max_val);
                    exp_sum += val.*;
                }

                // Normalize
                for (result.data[start..end]) |*val| {
                    val.* /= exp_sum;
                }
            }

            return result;
        }

        // ====================================================================
        // Matrix Operations
        // ====================================================================

        /// Matrix multiplication (2D tensors)
        pub fn matmul(self: *const Self, other: *const Self) !Self {
            if (self.shape.len != 2 or other.shape.len != 2) {
                return error.InvalidShape;
            }
            if (self.shape[1] != other.shape[0]) {
                return error.ShapeMismatch;
            }

            const m = self.shape[0];
            const k = self.shape[1];
            const n = other.shape[1];

            var result = try Self.zeros(self.allocator, &.{ m, n });

            for (0..m) |i| {
                for (0..n) |j| {
                    var acc: T = 0;
                    for (0..k) |l| {
                        acc += self.data[i * k + l] * other.data[l * n + j];
                    }
                    result.data[i * n + j] = acc;
                }
            }

            return result;
        }

        /// Transpose (2D tensors)
        pub fn transpose(self: *const Self) !Self {
            if (self.shape.len != 2) return error.InvalidShape;

            const rows = self.shape[0];
            const cols = self.shape[1];
            var result = try Self.init(self.allocator, &.{ cols, rows });

            for (0..rows) |i| {
                for (0..cols) |j| {
                    result.data[j * rows + i] = self.data[i * cols + j];
                }
            }

            return result;
        }

        // ====================================================================
        // Gradient Operations
        // ====================================================================

        /// Enable gradient tracking
        pub fn requiresGrad(self: *Self, requires: bool) void {
            self.requires_grad = requires;
            if (requires and self.grad == null) {
                self.grad = self.allocator.create(Self) catch null;
                if (self.grad) |grad| {
                    grad.* = Self.zeros(self.allocator, self.shape) catch {
                        self.allocator.destroy(grad);
                        self.grad = null;
                        return;
                    };
                }
            }
        }

        /// Zero gradients
        pub fn zeroGrad(self: *Self) void {
            if (self.grad) |grad| {
                @memset(grad.data, 0);
            }
        }

        /// Accumulate gradient
        pub fn accumulateGrad(self: *Self, grad_tensor: *const Self) !void {
            if (self.grad) |grad| {
                try grad.addInPlace(grad_tensor);
            }
        }

        // ====================================================================
        // Utility Functions
        // ====================================================================

        /// Clone the tensor
        pub fn clone(self: *const Self) !Self {
            return fromSlice(self.allocator, self.data, self.shape);
        }

        /// Reshape tensor (must have same total size)
        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            const new_size = computeSize(new_shape);
            if (new_size != self.data.len) return error.ShapeMismatch;

            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            self.shape = try self.allocator.dupe(usize, new_shape);
            self.strides = try computeStrides(self.allocator, new_shape);
        }

        /// Print tensor for debugging
        pub fn print(self: *const Self, writer: anytype) !void {
            try writer.print("Tensor(shape=[", .{});
            for (self.shape, 0..) |dim, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{}", .{dim});
            }
            try writer.print("], data=[", .{});
            const max_print = @min(self.data.len, 10);
            for (self.data[0..max_print], 0..) |val, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{d:.4}", .{val});
            }
            if (self.data.len > max_print) {
                try writer.print(", ...", .{});
            }
            try writer.print("])\n", .{});
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

fn computeSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

fn computeStrides(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
    const strides = try allocator.alloc(usize, shape.len);
    var stride: usize = 1;
    var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// ============================================================================
// Type Aliases
// ============================================================================

pub const F32Tensor = Tensor(f32);
pub const F64Tensor = Tensor(f64);

// ============================================================================
// Tests
// ============================================================================

test "tensor creation" {
    const allocator = std.testing.allocator;

    var t = try F32Tensor.init(allocator, &.{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.size());
    try std.testing.expectEqual(@as(usize, 2), t.ndim());
}

test "tensor from slice" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try F32Tensor.fromSlice(allocator, &data, &.{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(@as(f32, 1), t.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 6), t.get(&.{ 1, 2 }));
}

test "tensor addition" {
    const allocator = std.testing.allocator;

    var a = try F32Tensor.fill(allocator, &.{ 2, 2 }, 1.0);
    defer a.deinit();
    var b = try F32Tensor.fill(allocator, &.{ 2, 2 }, 2.0);
    defer b.deinit();

    var c = try a.add(&b);
    defer c.deinit();

    try std.testing.expectEqual(@as(f32, 3.0), c.data[0]);
}

test "tensor matmul" {
    const allocator = std.testing.allocator;

    var a = try F32Tensor.fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try F32Tensor.fromSlice(allocator, &.{ 5, 6, 7, 8 }, &.{ 2, 2 });
    defer b.deinit();

    var c = try a.matmul(&b);
    defer c.deinit();

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    try std.testing.expectEqual(@as(f32, 19), c.data[0]);
    try std.testing.expectEqual(@as(f32, 22), c.data[1]);
    try std.testing.expectEqual(@as(f32, 43), c.data[2]);
    try std.testing.expectEqual(@as(f32, 50), c.data[3]);
}

test "tensor softmax" {
    const allocator = std.testing.allocator;

    var t = try F32Tensor.fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer t.deinit();

    var s = try t.softmax();
    defer s.deinit();

    // Sum should be 1
    const total = s.sum();
    try std.testing.expect(@abs(total - 1.0) < 0.0001);
}
