//! Zero-copy tensor views for efficient slicing and reshaping.
//!
//! Views reference underlying tensor data without copying, enabling
//! efficient operations on sub-tensors.

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
const Shape = tensor_mod.Shape;
const DType = tensor_mod.DType;
const TensorError = tensor_mod.TensorError;

/// A view into a tensor that doesn't own its data.
pub const TensorView = struct {
    /// Pointer to underlying data
    data: []const f32,
    /// View shape
    shape: Shape,
    /// Strides for each dimension
    strides: [4]u32,
    /// Offset from start of underlying data
    offset: usize,
    /// Data type
    dtype: DType,

    /// Create a view of an entire tensor.
    pub fn fromTensor(t: *const Tensor) TensorView {
        return .{
            .data = t.data,
            .shape = t.shape,
            .strides = t.strides,
            .offset = 0,
            .dtype = t.dtype,
        };
    }

    /// Create a view from raw data.
    pub fn fromSlice(data: []const f32, shape: Shape) TensorError!TensorView {
        const total = tensor_mod.elementCount(shape);
        if (data.len < total) return TensorError.ShapeMismatch;

        return .{
            .data = data,
            .shape = shape,
            .strides = tensor_mod.calculateStrides(shape),
            .offset = 0,
            .dtype = .f32,
        };
    }

    /// Get total number of elements in the view.
    pub fn len(self: *const TensorView) usize {
        return tensor_mod.elementCount(self.shape);
    }

    /// Get element at indices.
    pub fn get(self: *const TensorView, idx0: u32, idx1: u32, idx2: u32, idx3: u32) TensorError!f32 {
        const idx = self.flatIndex(idx0, idx1, idx2, idx3) orelse return TensorError.OutOfBounds;
        return self.data[self.offset + idx];
    }

    /// Get flat index for multi-dimensional indices.
    pub fn flatIndex(self: *const TensorView, idx0: u32, idx1: u32, idx2: u32, idx3: u32) ?usize {
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

    /// Create a sub-view slicing the first dimension.
    pub fn sliceFirst(self: *const TensorView, start: u32, end: u32) TensorError!TensorView {
        if (start >= end or end > self.shape[0]) return TensorError.OutOfBounds;

        return .{
            .data = self.data,
            .shape = .{ end - start, self.shape[1], self.shape[2], self.shape[3] },
            .strides = self.strides,
            .offset = self.offset + start * self.strides[0],
            .dtype = self.dtype,
        };
    }

    /// Create a sub-view selecting a single index in the first dimension.
    pub fn select(self: *const TensorView, dim: u32, index: u32) TensorError!TensorView {
        if (dim >= 4) return TensorError.OutOfBounds;
        if (index >= self.shape[dim]) return TensorError.OutOfBounds;

        var new_shape = self.shape;
        new_shape[dim] = 1;

        const new_offset = self.offset + index * self.strides[dim];

        return .{
            .data = self.data,
            .shape = new_shape,
            .strides = self.strides,
            .offset = new_offset,
            .dtype = self.dtype,
        };
    }

    /// Create a reshaped view.
    pub fn reshape(self: *const TensorView, new_shape: Shape) TensorError!TensorView {
        if (tensor_mod.elementCount(new_shape) != self.len()) {
            return TensorError.ShapeMismatch;
        }

        // Only valid for contiguous views
        if (!self.isContiguous()) {
            return TensorError.UnsupportedOperation;
        }

        return .{
            .data = self.data,
            .shape = new_shape,
            .strides = tensor_mod.calculateStrides(new_shape),
            .offset = self.offset,
            .dtype = self.dtype,
        };
    }

    /// Check if the view is contiguous in memory.
    pub fn isContiguous(self: *const TensorView) bool {
        const expected = tensor_mod.calculateStrides(self.shape);
        return std.meta.eql(self.strides, expected);
    }

    /// Copy view data to a new tensor.
    pub fn toTensor(self: *const TensorView, allocator: std.mem.Allocator) !Tensor {
        const total = self.len();
        const result = try Tensor.init(allocator, self.shape, self.dtype);

        if (self.isContiguous()) {
            // Fast path: direct copy
            @memcpy(result.data, self.data[self.offset .. self.offset + total]);
        } else {
            // Slow path: element-by-element
            var idx: usize = 0;
            var dim0: u32 = 0;
            while (dim0 < self.shape[0]) : (dim0 += 1) {
                var dim1: u32 = 0;
                while (dim1 < self.shape[1]) : (dim1 += 1) {
                    var dim2: u32 = 0;
                    while (dim2 < self.shape[2]) : (dim2 += 1) {
                        var dim3: u32 = 0;
                        while (dim3 < self.shape[3]) : (dim3 += 1) {
                            result.data[idx] = try self.get(dim0, dim1, dim2, dim3);
                            idx += 1;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// Get a contiguous slice of the underlying data.
    /// Only valid for contiguous views.
    pub fn asSlice(self: *const TensorView) ?[]const f32 {
        if (!self.isContiguous()) return null;
        return self.data[self.offset .. self.offset + self.len()];
    }

    /// Get a row (1D view) from a 2D view.
    pub fn row(self: *const TensorView, index: u32) TensorError!TensorView {
        if (self.shape[0] <= index) return TensorError.OutOfBounds;

        return .{
            .data = self.data,
            .shape = .{ self.shape[1], 1, 1, 1 },
            .strides = .{ 1, 1, 1, 1 },
            .offset = self.offset + index * self.strides[0],
            .dtype = self.dtype,
        };
    }

    /// Get a column (1D view) from a 2D view.
    /// Note: Column views are not contiguous.
    pub fn col(self: *const TensorView, index: u32) TensorError!TensorView {
        if (self.shape[1] <= index) return TensorError.OutOfBounds;

        return .{
            .data = self.data,
            .shape = .{ self.shape[0], 1, 1, 1 },
            .strides = .{ self.strides[0], 1, 1, 1 },
            .offset = self.offset + index * self.strides[1],
            .dtype = self.dtype,
        };
    }
};

/// Create a transposed view (swap dimensions 0 and 1).
pub fn transpose2D(v: *const TensorView) TensorView {
    return .{
        .data = v.data,
        .shape = .{ v.shape[1], v.shape[0], v.shape[2], v.shape[3] },
        .strides = .{ v.strides[1], v.strides[0], v.strides[2], v.strides[3] },
        .offset = v.offset,
        .dtype = v.dtype,
    };
}

/// Create a broadcasted view.
pub fn broadcast(v: *const TensorView, target_shape: Shape) TensorError!TensorView {
    if (!tensor_mod.broadcastable(v.shape, target_shape)) {
        return TensorError.ShapeMismatch;
    }

    var new_strides: [4]u32 = undefined;
    for (0..4) |i| {
        if (v.shape[i] == 1 and target_shape[i] > 1) {
            new_strides[i] = 0; // Broadcasting: stride of 0
        } else {
            new_strides[i] = v.strides[i];
        }
    }

    return .{
        .data = v.data,
        .shape = target_shape,
        .strides = new_strides,
        .offset = v.offset,
        .dtype = v.dtype,
    };
}

test "view from tensor" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 2, 3, 1, 1 });
    defer t.deinit();

    const view = TensorView.fromTensor(&t);
    try std.testing.expectEqual(@as(usize, 6), view.len());
    try std.testing.expectEqual(@as(f32, 1.0), try view.get(0, 0, 0, 0));
    try std.testing.expectEqual(@as(f32, 6.0), try view.get(1, 2, 0, 0));
}

test "view slicing" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 2, 3, 1, 1 });
    defer t.deinit();

    const view = TensorView.fromTensor(&t);
    const slice_view = try view.sliceFirst(0, 1);

    try std.testing.expectEqual(@as(u32, 1), slice_view.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), slice_view.len());
    try std.testing.expectEqual(@as(f32, 1.0), try slice_view.get(0, 0, 0, 0));
}

test "view row and column" {
    const allocator = std.testing.allocator;
    // 2x3 matrix:
    // 1 2 3
    // 4 5 6
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 2, 3, 1, 1 });
    defer t.deinit();

    const view = TensorView.fromTensor(&t);

    // Row 1 should be [4, 5, 6]
    const row_view = try view.row(1);
    try std.testing.expectEqual(@as(f32, 4.0), try row_view.get(0, 0, 0, 0));

    // Column 1 should be [2, 5]
    const col_view = try view.col(1);
    try std.testing.expectEqual(@as(f32, 2.0), try col_view.get(0, 0, 0, 0));
    try std.testing.expectEqual(@as(f32, 5.0), try col_view.get(1, 0, 0, 0));
}

test "transpose 2D" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, &data, .{ 2, 3, 1, 1 });
    defer t.deinit();

    const view = TensorView.fromTensor(&t);
    const transposed = transpose2D(&view);

    try std.testing.expectEqual(@as(u32, 3), transposed.shape[0]);
    try std.testing.expectEqual(@as(u32, 2), transposed.shape[1]);

    // Original [0,1] = 2, transposed [1,0] = 2
    try std.testing.expectEqual(@as(f32, 2.0), try transposed.get(1, 0, 0, 0));
}

test {
    std.testing.refAllDecls(@This());
}
