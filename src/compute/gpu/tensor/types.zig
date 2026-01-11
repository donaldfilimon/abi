//! Tensor type definitions.
//!
//! Core types for tensor representation and manipulation.

const std = @import("std");

/// Tensor data types.
pub const DataType = enum(u8) {
    f16 = 0,
    f32 = 1,
    f64 = 2,
    i8 = 3,
    i16 = 4,
    i32 = 5,
    i64 = 6,
    u8 = 7,
    u16 = 8,
    u32 = 9,
    u64 = 10,
    bf16 = 11,

    /// Get the size in bytes for this data type.
    pub fn size(self: DataType) usize {
        return switch (self) {
            .f16, .bf16, .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .i8, .u8 => 1,
        };
    }

    /// Check if this is a floating point type.
    pub fn isFloat(self: DataType) bool {
        return switch (self) {
            .f16, .f32, .f64, .bf16 => true,
            else => false,
        };
    }

    /// Check if this is a signed integer type.
    pub fn isSigned(self: DataType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }
};

/// Maximum tensor dimensions.
pub const MAX_DIMS = 8;

/// Tensor shape.
pub const TensorShape = struct {
    dims: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS,
    ndim: usize = 0,

    /// Create a shape from dimensions.
    pub fn init(dims: []const usize) TensorShape {
        var shape = TensorShape{};
        shape.ndim = @min(dims.len, MAX_DIMS);
        for (dims[0..shape.ndim], 0..) |d, i| {
            shape.dims[i] = d;
        }
        return shape;
    }

    /// Get total number of elements.
    pub fn numel(self: TensorShape) usize {
        if (self.ndim == 0) return 0;
        var total: usize = 1;
        for (self.dims[0..self.ndim]) |d| {
            total *= d;
        }
        return total;
    }

    /// Get dimension at index.
    pub fn dim(self: TensorShape, idx: usize) usize {
        if (idx >= self.ndim) return 1;
        return self.dims[idx];
    }

    /// Check if shapes are equal.
    pub fn eql(self: TensorShape, other: TensorShape) bool {
        if (self.ndim != other.ndim) return false;
        for (self.dims[0..self.ndim], other.dims[0..other.ndim]) |a, b| {
            if (a != b) return false;
        }
        return true;
    }

    /// Check if shapes are broadcastable.
    pub fn broadcastable(self: TensorShape, other: TensorShape) bool {
        const max_ndim = @max(self.ndim, other.ndim);
        var i: usize = 0;
        while (i < max_ndim) : (i += 1) {
            const a = if (i < self.ndim) self.dims[self.ndim - 1 - i] else 1;
            const b = if (i < other.ndim) other.dims[other.ndim - 1 - i] else 1;
            if (a != b and a != 1 and b != 1) return false;
        }
        return true;
    }

    /// Get broadcasted shape.
    pub fn broadcast(self: TensorShape, other: TensorShape) ?TensorShape {
        if (!self.broadcastable(other)) return null;

        var result = TensorShape{};
        result.ndim = @max(self.ndim, other.ndim);

        var i: usize = 0;
        while (i < result.ndim) : (i += 1) {
            const a = if (i < self.ndim) self.dims[self.ndim - 1 - i] else 1;
            const b = if (i < other.ndim) other.dims[other.ndim - 1 - i] else 1;
            result.dims[result.ndim - 1 - i] = @max(a, b);
        }
        return result;
    }

    /// Format shape as string for debugging.
    pub fn format(self: TensorShape, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.writeByte('(');
        for (self.dims[0..self.ndim], 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try std.fmt.format(writer, "{}", .{d});
        }
        try writer.writeByte(')');
    }
};

/// Tensor memory layout.
pub const TensorLayout = enum {
    /// Row-major (C-style).
    row_major,
    /// Column-major (Fortran-style).
    col_major,
    /// Custom strides.
    strided,
};

/// Compute device.
pub const Device = enum {
    cpu,
    cuda,
    vulkan,
    metal,
    opencl,
    webgpu,

    pub fn isGpu(self: Device) bool {
        return self != .cpu;
    }
};

/// Tensor errors.
pub const TensorError = error{
    ShapeMismatch,
    DimensionMismatch,
    InvalidShape,
    InvalidDataType,
    DeviceMismatch,
    OutOfMemory,
    NotContiguous,
    InvalidOperation,
    NotImplemented,
};

/// A multi-dimensional array (tensor).
pub const Tensor = struct {
    allocator: std.mem.Allocator,
    data: []u8,
    shape: TensorShape,
    strides: [MAX_DIMS]usize,
    dtype: DataType,
    device: Device,
    layout: TensorLayout,
    owns_data: bool,
    offset: usize,

    /// Create a new tensor with given shape and data type.
    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
        const tensor_shape = TensorShape.init(shape);
        const num_elements = tensor_shape.numel();
        const byte_size = num_elements * dtype.size();

        const data = try allocator.alloc(u8, byte_size);
        @memset(data, 0);

        const tensor = try allocator.create(Tensor);
        tensor.* = .{
            .allocator = allocator,
            .data = data,
            .shape = tensor_shape,
            .strides = computeStrides(tensor_shape, .row_major),
            .dtype = dtype,
            .device = .cpu,
            .layout = .row_major,
            .owns_data = true,
            .offset = 0,
        };
        return tensor;
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
        return init(allocator, shape, dtype);
    }

    /// Create a tensor filled with ones.
    pub fn ones(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
        const tensor = try init(allocator, shape, dtype);

        // Fill with ones based on dtype
        switch (dtype) {
            .f32 => {
                const slice = tensor.asSlice(f32);
                @memset(slice, 1.0);
            },
            .f64 => {
                const slice = tensor.asSlice(f64);
                @memset(slice, 1.0);
            },
            .i32 => {
                const slice = tensor.asSlice(i32);
                @memset(slice, 1);
            },
            else => {},
        }

        return tensor;
    }

    /// Create a tensor from a slice.
    pub fn fromSlice(allocator: std.mem.Allocator, comptime T: type, data: []const T, shape: []const usize) !*Tensor {
        const dtype = comptime typeToDataType(T);
        const tensor = try init(allocator, shape, dtype);

        const num_elements = tensor.shape.numel();
        if (data.len != num_elements) return TensorError.ShapeMismatch;

        const dest = tensor.asSlice(T);
        @memcpy(dest, data);

        return tensor;
    }

    /// Create a random tensor (uniform 0-1).
    pub fn rand(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
        const tensor = try init(allocator, shape, dtype);

        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        switch (dtype) {
            .f32 => {
                const slice = tensor.asSlice(f32);
                for (slice) |*v| {
                    v.* = random.float(f32);
                }
            },
            .f64 => {
                const slice = tensor.asSlice(f64);
                for (slice) |*v| {
                    v.* = random.float(f64);
                }
            },
            else => {},
        }

        return tensor;
    }

    /// Create a random tensor (normal distribution).
    pub fn randn(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*Tensor {
        const tensor = try init(allocator, shape, dtype);

        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        switch (dtype) {
            .f32 => {
                const slice = tensor.asSlice(f32);
                for (slice) |*v| {
                    // Box-Muller transform
                    const rand1 = random.float(f32);
                    const rand2 = random.float(f32);
                    v.* = @sqrt(-2.0 * @log(rand1 + 1e-10)) * @cos(2.0 * std.math.pi * rand2);
                }
            },
            .f64 => {
                const slice = tensor.asSlice(f64);
                for (slice) |*v| {
                    const rand1 = random.float(f64);
                    const rand2 = random.float(f64);
                    v.* = @sqrt(-2.0 * @log(rand1 + 1e-10)) * @cos(2.0 * std.math.pi * rand2);
                }
            },
            else => {},
        }

        return tensor;
    }

    /// Deinitialize tensor.
    pub fn deinit(self: *Tensor) void {
        if (self.owns_data) {
            self.allocator.free(self.data);
        }
        self.allocator.destroy(self);
    }

    /// Get data as typed slice.
    pub fn asSlice(self: *Tensor, comptime T: type) []T {
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0..self.shape.numel()];
    }

    /// Get data as const typed slice.
    pub fn asConstSlice(self: *const Tensor, comptime T: type) []const T {
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0..self.shape.numel()];
    }

    /// Check if tensor is contiguous.
    pub fn isContiguous(self: *const Tensor) bool {
        if (self.shape.ndim == 0) return true;

        var expected_stride: usize = 1;
        var i = self.shape.ndim;
        while (i > 0) : (i -= 1) {
            if (self.strides[i - 1] != expected_stride) return false;
            expected_stride *= self.shape.dims[i - 1];
        }
        return true;
    }

    /// Get number of elements.
    pub fn numel(self: *const Tensor) usize {
        return self.shape.numel();
    }

    /// Get number of dimensions.
    pub fn ndim(self: *const Tensor) usize {
        return self.shape.ndim;
    }

    /// Get size in bytes.
    pub fn byteSize(self: *const Tensor) usize {
        return self.shape.numel() * self.dtype.size();
    }

    /// Clone tensor.
    pub fn clone(self: *const Tensor) !*Tensor {
        const new_tensor = try Tensor.init(self.allocator, self.shape.dims[0..self.shape.ndim], self.dtype);
        @memcpy(new_tensor.data, self.data);
        return new_tensor;
    }

    /// Create a view with new shape.
    pub fn view(self: *Tensor, new_shape: []const usize) !*Tensor {
        const new_tensor_shape = TensorShape.init(new_shape);
        if (new_tensor_shape.numel() != self.shape.numel()) {
            return TensorError.ShapeMismatch;
        }
        if (!self.isContiguous()) {
            return TensorError.NotContiguous;
        }

        const new_tensor = try self.allocator.create(Tensor);
        new_tensor.* = self.*;
        new_tensor.shape = new_tensor_shape;
        new_tensor.strides = computeStrides(new_tensor_shape, self.layout);
        new_tensor.owns_data = false;

        return new_tensor;
    }

    /// Fill tensor with a value.
    pub fn fill(self: *Tensor, comptime T: type, value: T) void {
        const slice = self.asSlice(T);
        @memset(slice, value);
    }

    /// Get element at index.
    pub fn get(self: *const Tensor, comptime T: type, indices: []const usize) T {
        const offset = self.computeOffset(indices);
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[offset];
    }

    /// Set element at index.
    pub fn set(self: *Tensor, comptime T: type, indices: []const usize, value: T) void {
        const offset = self.computeOffset(indices);
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        ptr[offset] = value;
    }

    fn computeOffset(self: *const Tensor, indices: []const usize) usize {
        var offset: usize = self.offset;
        for (indices, 0..) |idx, i| {
            offset += idx * self.strides[i];
        }
        return offset;
    }
};

/// Compute strides for a shape.
fn computeStrides(shape: TensorShape, layout: TensorLayout) [MAX_DIMS]usize {
    var strides = [_]usize{0} ** MAX_DIMS;
    if (shape.ndim == 0) return strides;

    switch (layout) {
        .row_major => {
            strides[shape.ndim - 1] = 1;
            var i = shape.ndim - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape.dims[i];
            }
        },
        .col_major => {
            strides[0] = 1;
            for (1..shape.ndim) |i| {
                strides[i] = strides[i - 1] * shape.dims[i - 1];
            }
        },
        .strided => {},
    }
    return strides;
}

/// Convert Zig type to DataType.
fn typeToDataType(comptime T: type) DataType {
    return switch (T) {
        f16 => .f16,
        f32 => .f32,
        f64 => .f64,
        i8 => .i8,
        i16 => .i16,
        i32 => .i32,
        i64 => .i64,
        u8 => .u8,
        u16 => .u16,
        u32 => .u32,
        u64 => .u64,
        else => @compileError("Unsupported tensor type"),
    };
}

test "tensor shape" {
    const shape = TensorShape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), shape.ndim);
    try std.testing.expectEqual(@as(usize, 24), shape.numel());
    try std.testing.expectEqual(@as(usize, 2), shape.dim(0));
    try std.testing.expectEqual(@as(usize, 3), shape.dim(1));
    try std.testing.expectEqual(@as(usize, 4), shape.dim(2));
}

test "shape broadcasting" {
    const a = TensorShape.init(&.{ 3, 1 });
    const b = TensorShape.init(&.{ 1, 4 });
    try std.testing.expect(a.broadcastable(b));

    const result = a.broadcast(b).?;
    try std.testing.expectEqual(@as(usize, 2), result.ndim);
    try std.testing.expectEqual(@as(usize, 3), result.dim(0));
    try std.testing.expectEqual(@as(usize, 4), result.dim(1));
}

test "tensor creation" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, &.{ 2, 3 }, .f32);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.numel());
    try std.testing.expectEqual(@as(usize, 2), t.ndim());
    try std.testing.expect(t.isContiguous());
}

test "tensor from slice" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor.fromSlice(allocator, f32, &data, &.{ 2, 3 });
    defer t.deinit();

    const slice = t.asConstSlice(f32);
    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 6.0), slice[5]);
}

test "tensor get/set" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, &.{ 2, 3 }, .f32);
    defer t.deinit();

    t.set(f32, &.{ 0, 1 }, 42.0);
    const val = t.get(f32, &.{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 42.0), val);
}
