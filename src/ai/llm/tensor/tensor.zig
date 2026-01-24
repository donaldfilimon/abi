//! Core tensor abstraction for multi-dimensional arrays.
//!
//! Tensors are the fundamental data structure for LLM operations,
//! representing weights, activations, and intermediate results.

const std = @import("std");

/// Data type enum for tensor elements.
pub const DType = enum {
    f32,
    f16,
    bf16,
    i8,
    i16,
    i32,
    q4_0,
    q8_0,

    /// Get size in bytes for a single element.
    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8 => 1,
            .q4_0 => 0, // Block-based
            .q8_0 => 0, // Block-based
        };
    }

    /// Check if this is a quantized type.
    pub fn isQuantized(self: DType) bool {
        return switch (self) {
            .q4_0, .q8_0 => true,
            else => false,
        };
    }
};

/// Shape tuple for up to 4 dimensions.
/// Dimensions are stored as [dim0, dim1, dim2, dim3].
/// Unused dimensions should be 1.
pub const Shape = [4]u32;

pub const TensorError = error{
    ShapeMismatch,
    InvalidShape,
    DTypeMismatch,
    OutOfBounds,
    OutOfMemory,
    UnsupportedOperation,
};

/// Multi-dimensional tensor with owned or borrowed data.
pub const Tensor = struct {
    /// Raw data buffer (f32 for simplicity, convert on read/write for other types)
    data: []f32,
    /// Tensor shape [d0, d1, d2, d3]
    shape: Shape,
    /// Strides for each dimension (in elements)
    strides: [4]u32,
    /// Data type
    dtype: DType,
    /// Allocator used for data (null if view)
    allocator: ?std.mem.Allocator,
    /// Whether this tensor owns its data
    owns_data: bool,

    /// Create an uninitialized tensor.
    pub fn init(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        const total = elementCount(shape);
        const data = try allocator.alloc(f32, total);

        return Tensor{
            .data = data,
            .shape = shape,
            .strides = calculateStrides(shape),
            .dtype = dtype,
            .allocator = allocator,
            .owns_data = true,
        };
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        const t = try init(allocator, shape, dtype);
        @memset(t.data, 0);
        return t;
    }

    /// Create a tensor filled with ones.
    pub fn ones(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        const t = try init(allocator, shape, dtype);
        @memset(t.data, 1.0);
        return t;
    }

    /// Create a tensor from existing data.
    pub fn fromSlice(allocator: std.mem.Allocator, data: []const f32, shape: Shape) !Tensor {
        const total = elementCount(shape);
        if (data.len != total) return TensorError.ShapeMismatch;

        const owned = try allocator.alloc(f32, total);
        @memcpy(owned, data);

        return Tensor{
            .data = owned,
            .shape = shape,
            .strides = calculateStrides(shape),
            .dtype = .f32,
            .allocator = allocator,
            .owns_data = true,
        };
    }

    /// Create a tensor wrapping existing data (no copy, does not own).
    pub fn wrap(data: []f32, shape: Shape) TensorError!Tensor {
        const total = elementCount(shape);
        if (data.len != total) return TensorError.ShapeMismatch;

        return Tensor{
            .data = data,
            .shape = shape,
            .strides = calculateStrides(shape),
            .dtype = .f32,
            .allocator = null,
            .owns_data = false,
        };
    }

    /// Free tensor memory if owned.
    pub fn deinit(self: *Tensor) void {
        if (self.owns_data) {
            if (self.allocator) |alloc| {
                alloc.free(self.data);
            }
        }
        self.* = undefined;
    }

    /// Get total number of elements.
    pub fn len(self: *const Tensor) usize {
        return elementCount(self.shape);
    }

    /// Get number of dimensions (non-1 trailing dims).
    pub fn ndim(self: *const Tensor) u32 {
        var n: u32 = 4;
        while (n > 1 and self.shape[n - 1] == 1) {
            n -= 1;
        }
        return n;
    }

    /// Get element at indices.
    pub fn get(self: *const Tensor, idx0: u32, idx1: u32, idx2: u32, idx3: u32) TensorError!f32 {
        const idx = self.flatIndex(idx0, idx1, idx2, idx3) orelse return TensorError.OutOfBounds;
        return self.data[idx];
    }

    /// Set element at indices.
    pub fn set(self: *Tensor, idx0: u32, idx1: u32, idx2: u32, idx3: u32, value: f32) TensorError!void {
        const idx = self.flatIndex(idx0, idx1, idx2, idx3) orelse return TensorError.OutOfBounds;
        self.data[idx] = value;
    }

    /// Get flat index for multi-dimensional indices.
    pub fn flatIndex(self: *const Tensor, idx0: u32, idx1: u32, idx2: u32, idx3: u32) ?usize {
        if (idx0 >= self.shape[0] or idx1 >= self.shape[1] or
            idx2 >= self.shape[2] or idx3 >= self.shape[3])
        {
            return null;
        }
        return @as(usize, idx0) * self.strides[0] +
            @as(usize, idx1) * self.strides[1] +
            @as(usize, idx2) * self.strides[2] +
            @as(usize, idx3) * self.strides[3];
    }

    /// Get a slice view along the first dimension.
    pub fn sliceFirst(self: *const Tensor, start: u32, end: u32) TensorError!Tensor {
        if (start >= end or end > self.shape[0]) return TensorError.OutOfBounds;

        const offset = start * self.strides[0];
        const new_len = (end - start) * self.strides[0];

        return Tensor{
            .data = self.data[offset .. offset + new_len],
            .shape = .{ end - start, self.shape[1], self.shape[2], self.shape[3] },
            .strides = self.strides,
            .dtype = self.dtype,
            .allocator = null,
            .owns_data = false,
        };
    }

    /// Reshape tensor (must have same total elements).
    pub fn reshape(self: *Tensor, new_shape: Shape) TensorError!void {
        if (elementCount(new_shape) != elementCount(self.shape)) {
            return TensorError.ShapeMismatch;
        }
        self.shape = new_shape;
        self.strides = calculateStrides(new_shape);
    }

    /// Create a reshaped view (no data copy).
    pub fn view(self: *const Tensor, new_shape: Shape) TensorError!Tensor {
        if (elementCount(new_shape) != elementCount(self.shape)) {
            return TensorError.ShapeMismatch;
        }
        return Tensor{
            .data = self.data,
            .shape = new_shape,
            .strides = calculateStrides(new_shape),
            .dtype = self.dtype,
            .allocator = null,
            .owns_data = false,
        };
    }

    /// Clone tensor with new data allocation.
    pub fn clone(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        const data = try allocator.alloc(f32, self.data.len);
        @memcpy(data, self.data);

        return Tensor{
            .data = data,
            .shape = self.shape,
            .strides = self.strides,
            .dtype = self.dtype,
            .allocator = allocator,
            .owns_data = true,
        };
    }

    /// Fill tensor with a value.
    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    /// Apply a function to each element in-place.
    pub fn mapInPlace(self: *Tensor, f: *const fn (f32) f32) void {
        for (self.data) |*v| {
            v.* = f(v.*);
        }
    }

    /// Apply a function to each element, returning new tensor.
    pub fn map(self: *const Tensor, allocator: std.mem.Allocator, f: *const fn (f32) f32) !Tensor {
        var result = try self.clone(allocator);
        result.mapInPlace(f);
        return result;
    }

    /// Element-wise addition in-place.
    pub fn addInPlace(self: *Tensor, other: *const Tensor) TensorError!void {
        if (!std.meta.eql(self.shape, other.shape)) return TensorError.ShapeMismatch;
        for (self.data, other.data) |*a, b| {
            a.* += b;
        }
    }

    /// Element-wise multiplication in-place.
    pub fn mulInPlace(self: *Tensor, other: *const Tensor) TensorError!void {
        if (!std.meta.eql(self.shape, other.shape)) return TensorError.ShapeMismatch;
        for (self.data, other.data) |*a, b| {
            a.* *= b;
        }
    }

    /// Scale all elements by a factor in-place.
    pub fn scaleInPlace(self: *Tensor, factor: f32) void {
        for (self.data) |*v| {
            v.* *= factor;
        }
    }

    /// Compute sum of all elements.
    pub fn sum(self: *const Tensor) f32 {
        var total: f32 = 0;
        for (self.data) |v| {
            total += v;
        }
        return total;
    }

    /// Compute mean of all elements.
    pub fn mean(self: *const Tensor) f32 {
        return self.sum() / @as(f32, @floatFromInt(self.data.len));
    }

    /// Compute max element.
    pub fn max(self: *const Tensor) f32 {
        var m: f32 = -std.math.inf(f32);
        for (self.data) |v| {
            if (v > m) m = v;
        }
        return m;
    }

    /// Compute argmax (index of maximum element).
    pub fn argmax(self: *const Tensor) usize {
        var max_idx: usize = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (self.data, 0..) |v, i| {
            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
        }
        return max_idx;
    }

    /// Get underlying data as slice.
    pub fn asSlice(self: *const Tensor) []const f32 {
        return self.data;
    }

    /// Get mutable underlying data.
    pub fn asMutSlice(self: *Tensor) []f32 {
        return self.data;
    }

    /// Format for printing.
    pub fn format(
        self: Tensor,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("Tensor(shape=[{d}, {d}, {d}, {d}], dtype={t}, elements={d})", .{
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            self.dtype,
            self.len(),
        });
    }
};

/// Calculate total number of elements in a shape.
pub fn elementCount(shape: Shape) usize {
    return @as(usize, shape[0]) * shape[1] * shape[2] * shape[3];
}

/// Calculate strides for row-major layout.
pub fn calculateStrides(shape: Shape) [4]u32 {
    return .{
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1,
    };
}

/// Check if two shapes are broadcastable.
pub fn broadcastable(a: Shape, b: Shape) bool {
    for (0..4) |i| {
        if (a[i] != b[i] and a[i] != 1 and b[i] != 1) {
            return false;
        }
    }
    return true;
}

/// Compute broadcast result shape.
pub fn broadcastShape(a: Shape, b: Shape) ?Shape {
    if (!broadcastable(a, b)) return null;
    return .{
        @max(a[0], b[0]),
        @max(a[1], b[1]),
        @max(a[2], b[2]),
        @max(a[3], b[3]),
    };
}

test "tensor creation and basics" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, .{ 2, 3, 1, 1 }, .f32);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.len());
    try std.testing.expectEqual(@as(u32, 2), t.ndim());
    try std.testing.expectEqual(@as(f32, 0.0), try t.get(0, 0, 0, 0));
}

test "tensor indexing" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, .{ 2, 3, 1, 1 }, .f32);
    defer t.deinit();

    try t.set(1, 2, 0, 0, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), try t.get(1, 2, 0, 0));
}

test "tensor reshape" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, .{ 2, 3, 1, 1 }, .f32);
    defer t.deinit();

    try t.reshape(.{ 6, 1, 1, 1 });
    try std.testing.expectEqual(@as(usize, 6), t.len());
    try std.testing.expectEqual(@as(u32, 1), t.ndim());
}

test "tensor from slice" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 2, 3, 1, 1 });
    defer t.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), try t.get(0, 0, 0, 0));
    try std.testing.expectEqual(@as(f32, 6.0), try t.get(1, 2, 0, 0));
}

test "tensor operations" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 6, 1, 1, 1 });
    defer t.deinit();

    try std.testing.expectEqual(@as(f32, 21.0), t.sum());
    try std.testing.expectEqual(@as(f32, 3.5), t.mean());
    try std.testing.expectEqual(@as(f32, 6.0), t.max());
    try std.testing.expectEqual(@as(usize, 5), t.argmax());
}

test "element count and strides" {
    const shape: Shape = .{ 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 120), elementCount(shape));

    const strides = calculateStrides(shape);
    try std.testing.expectEqual(@as(u32, 60), strides[0]); // 3*4*5
    try std.testing.expectEqual(@as(u32, 20), strides[1]); // 4*5
    try std.testing.expectEqual(@as(u32, 5), strides[2]); // 5
    try std.testing.expectEqual(@as(u32, 1), strides[3]); // 1
}
