//! Tensor Operations
//!
//! High-level tensor operations that dispatch to the hardware driver.

const std = @import("std");

const driver = @import("driver.zig");

// Tensor structure
pub const Tensor = struct {
    data: []u8,
    shape: []const usize,
    data_type: DataType,
    driver: driver.Driver,

    pub const DataType = enum {
        float32,
        float16,
        int8,
        uint8,
    };

    pub fn init(allocator: std.mem.Allocator, drv: driver.Driver, shape: []const usize, dtype: DataType) !Tensor {
        // Calculate size based on shape and dtype
        var elements: usize = 1;
        for (shape) |dim| {
            elements *= dim;
        }

        const type_size = switch (dtype) {
            .float32 => 4,
            .float16 => 2,
            .int8, .uint8 => 1,
        };
        const size = elements * type_size;

        const data = try drv.allocate(size);
        const tensor_shape = try allocator.dupe(usize, shape);

        return Tensor{
            .data = data,
            .shape = tensor_shape,
            .data_type = dtype,
            .driver = drv,
        };
    }

    pub fn deinit(self: Tensor, allocator: std.mem.Allocator) void {
        self.driver.free(self.data);
        allocator.free(self.shape);
    }

    /// Perform matrix multiplication: C = A * B
    pub fn matmul(c: *Tensor, a: Tensor, b: Tensor) !void {
        // Basic validation
        if (a.data_type != b.data_type or a.data_type != c.data_type) return error.DataTypeMismatch;
        if (a.shape.len != 2 or b.shape.len != 2 or c.shape.len != 2) return error.InvalidShape;

        const m = a.shape[0];
        const k = a.shape[1];
        const n = b.shape[1];

        if (b.shape[0] != k) return error.DimensionMismatch;
        if (c.shape[0] != m or c.shape[1] != n) return error.DimensionMismatch;

        // Dispatch to driver implementation
        return a.driver.matmul(c.data, a.data, b.data, m, n, k);
    }

    /// Perform 2D convolution
    pub fn conv2d(output: *Tensor, input: Tensor, kernel: Tensor) !void {
        // Minimal validation for 2D conv: input [H, W, C], kernel [K, K, C, Out], output [H, W, Out]
        // In strict ML frameworks dimensions are 4D (N, C, H, W) or (N, H, W, C), simplified here

        // Dispatch to driver
        var input_dims: [3]usize = undefined;
        var kernel_dims: [4]usize = undefined;

        // Populate dimensions (assuming simplified 3D input / 4D kernel for this example)
        if (input.shape.len == 3) {
            input_dims[0] = input.shape[0];
            input_dims[1] = input.shape[1];
            input_dims[2] = input.shape[2];
        } else return error.InvalidShape;

        if (kernel.shape.len == 4) {
            kernel_dims[0] = kernel.shape[0];
            kernel_dims[1] = kernel.shape[1];
            kernel_dims[2] = kernel.shape[2];
            kernel_dims[3] = kernel.shape[3];
        } else return error.InvalidShape;

        return input.driver.conv2d(output.data, input.data, kernel.data, input_dims, kernel_dims);
    }
};
