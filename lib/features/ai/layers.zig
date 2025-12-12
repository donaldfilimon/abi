//! Neural Network Layer Abstraction
//!
//! Modular neural network layers with automatic differentiation support.

const std = @import("std");

const accelerator = @import("../gpu/accelerator.zig");
const Tensor = accelerator.Tensor;
const Accelerator = accelerator.Accelerator;
const TensorOps = accelerator.TensorOps;

/// Parameter storage with gradients
pub const Parameter = struct {
    data: Tensor,
    grad: ?Tensor,
    requires_grad: bool = true,

    pub fn init(allocator: std.mem.Allocator, accel: *Accelerator, shape: []const usize) !Parameter {
        const data = try Tensor.init(allocator, accel, shape, .f32);
        // Allocation of gradient is optional/deferred in some frameworks, but we do it eagerly here
        const grad = try Tensor.init(allocator, accel, shape, .f32);

        return .{
            .data = data,
            .grad = grad,
        };
    }

    pub fn deinit(self: *Parameter, accel: *Accelerator) void {
        self.data.deinit(accel);
        if (self.grad) |*g| g.deinit(accel);
    }

    pub fn elementCount(self: Parameter) usize {
        return self.data.elementCount();
    }
};

/// Base layer interface
pub const Layer = struct {
    allocator: std.mem.Allocator,

    // Virtual table equivalent could be here, or generic wrapper.
    // For now we use specific structs like Dense, Conv2D.
};

/// Dense/Linear layer: y = Wx + b
pub const Dense = struct {
    weight: Parameter,
    bias: Parameter,
    input_size: usize,
    output_size: usize,
    accel: *Accelerator,
    last_input: ?Tensor = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, accel: *Accelerator, input_size: usize, output_size: usize) !Dense {
        var weight = try Parameter.init(allocator, accel, &[_]usize{ output_size, input_size });
        var bias = try Parameter.init(allocator, accel, &[_]usize{output_size});

        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        try randomInit(allocator, accel, &weight.data, weight.elementCount(), scale);
        try zeroInit(allocator, accel, &bias.data, bias.elementCount());

        return Dense{
            .weight = weight,
            .bias = bias,
            .input_size = input_size,
            .output_size = output_size,
            .accel = accel,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Dense) void {
        self.weight.deinit(self.accel);
        self.bias.deinit(self.accel);
        if (self.last_input) |*inp| inp.deinit(self.accel);
    }

    pub fn forward(self: *Dense, input: Tensor, batch_size: usize) !Tensor {
        // Save input for backward pass
        if (self.last_input) |*old| old.deinit(self.accel);

        // Copy input for backward pass
        // Since we don't have deep copy tensor util yet, we re-alloc and copy
        var input_copy = try Tensor.init(self.allocator, self.accel, input.shape, input.data_type);
        // Hack: copy via host for now as we lack D2D copy
        // Actually, accelerator.zig only has copyToDevice (Host->Device) and copyFromDevice (Device->Host).
        // If pointers are accessible (CPU backend), we can memcpy.
        // For GPU, we need D2D. Assuming CPU fallback for now as per accelerator.zig.

        // For CPU fallback, input.data.ptr is a host pointer.
        // We can just use copyToDevice with the data from copyFromDevice?
        // Let's implement a clean copy if possible.
        // For now, implementing a simplistic copy using copyFromDevice to temp buffer.
        const temp_buf = try self.allocator.alloc(u8, input.data.size);
        defer self.allocator.free(temp_buf);
        try self.accel.copyFromDevice(temp_buf, input.data);
        try self.accel.copyToDevice(input_copy.data, temp_buf);
        self.last_input = input_copy;

        const output_shape = [_]usize{ batch_size, self.output_size };
        const output = try Tensor.init(self.allocator, self.accel, &output_shape, .f32);

        var ops = TensorOps.init(self.accel);
        ops.matmul(output, self.weight.data, input);

        // Bias add would go here (omitted for brevity)

        return output;
    }

    pub fn backward(self: *Dense, grad_output: Tensor, batch_size: usize) !Tensor {
        _ = batch_size;
        // Compute gradients (Stubbed logic matching previous implementation style)
        var ops = TensorOps.init(self.accel);
        if (self.weight.grad) |*w_grad| {
            if (self.last_input) |*inp| {
                // dW = grad_output * input^T (conceptually)
                // We'd need specific matmul variant or transpose.
                // Using stub matmul for connectivity.
                ops.matmul(w_grad.*, grad_output, inp.*);
            }
        }

        // grad_input = W^T * grad_output
        const input_shape = [_]usize{ grad_output.shape[0], self.input_size };
        const grad_input = try Tensor.init(self.allocator, self.accel, &input_shape, .f32);
        ops.matmul(grad_input, self.weight.data, grad_output);

        return grad_input;
    }
};

/// 2D Convolution Layer
pub const Conv2D = struct {
    weight: Parameter,
    bias: Parameter,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    accel: *Accelerator,
    last_input: ?Tensor = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, accel: *Accelerator, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) !Conv2D {
        const k_shape = [_]usize{ out_channels, in_channels, kernel_size, kernel_size };
        var weight = try Parameter.init(allocator, accel, &k_shape);
        const b_shape = [_]usize{out_channels};
        var bias = try Parameter.init(allocator, accel, &b_shape);

        // Kaiming/He initialization
        const fan_in = in_channels * kernel_size * kernel_size;
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));
        try randomInit(allocator, accel, &weight.data, weight.elementCount(), scale);
        try zeroInit(allocator, accel, &bias.data, bias.elementCount());

        return Conv2D{
            .weight = weight,
            .bias = bias,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .accel = accel,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Conv2D) void {
        self.weight.deinit(self.accel);
        self.bias.deinit(self.accel);
        if (self.last_input) |*inp| inp.deinit(self.accel);
    }

    pub fn forward(self: *Conv2D, input: Tensor) !Tensor {
        // Helper copy input
        if (self.last_input) |*old| old.deinit(self.accel);
        var input_copy = try Tensor.init(self.allocator, self.accel, input.shape, input.data_type);
        const temp_buf = try self.allocator.alloc(u8, input.data.size);
        defer self.allocator.free(temp_buf);
        try self.accel.copyFromDevice(temp_buf, input.data);
        try self.accel.copyToDevice(input_copy.data, temp_buf);
        self.last_input = input_copy;

        // Calculate output shape
        const N = input.shape[0];
        const H_in = input.shape[2];
        const W_in = input.shape[3];

        const H_out = (H_in + 2 * self.padding - self.kernel_size) / self.stride + 1;
        const W_out = (W_in + 2 * self.padding - self.kernel_size) / self.stride + 1;

        const out_shape = [_]usize{ N, self.out_channels, H_out, W_out };
        const output = try Tensor.init(self.allocator, self.accel, &out_shape, .f32);

        var ops = TensorOps.init(self.accel);
        ops.conv2d(output, input, self.weight.data, self.stride, self.padding);

        return output;
    }

    // Backward omitted for brevity (requires conv2d_grad in accelerator)
};

/// ReLU activation layer
pub const ReLU = struct {
    accel: *Accelerator,
    last_input: ?Tensor = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, accel: *Accelerator) ReLU {
        return .{
            .accel = accel,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReLU) void {
        if (self.last_input) |*inp| inp.deinit(self.accel);
    }

    pub fn forward(self: *ReLU, input: Tensor, size: usize) !Tensor {
        _ = size;
        const output = try Tensor.init(self.allocator, self.accel, input.shape, input.data_type);

        var ops = TensorOps.init(self.accel);
        ops.relu(output, input);

        // Save input
        if (self.last_input) |*old| old.deinit(self.accel);
        var input_copy = try Tensor.init(self.allocator, self.accel, input.shape, input.data_type);
        const temp_buf = try self.allocator.alloc(u8, input.data.size);
        defer self.allocator.free(temp_buf);
        try self.accel.copyFromDevice(temp_buf, input.data);
        try self.accel.copyToDevice(input_copy.data, temp_buf);
        self.last_input = input_copy;

        return output;
    }

    pub fn backward(self: *ReLU, grad_output: Tensor, size: usize) !Tensor {
        _ = size;
        var grad_input = try Tensor.init(self.allocator, self.accel, grad_output.shape, grad_output.data_type);

        // Should implement relu_backward in accelerator
        // For now stub copy
        const temp_buf = try self.allocator.alloc(u8, grad_output.data.size);
        defer self.allocator.free(temp_buf);
        try self.accel.copyFromDevice(temp_buf, grad_output.data);
        try self.accel.copyToDevice(grad_input.data, temp_buf);

        return grad_input;
    }
};

// Helper functions (same as before but updated signatures)
fn randomInit(allocator: std.mem.Allocator, accel: *Accelerator, data: *Tensor, count: usize, scale: f32) !void {
    const temp_data = try allocator.alloc(f32, count);
    defer allocator.free(temp_data);

    var prng = std.Random.DefaultPrng.init(std.crypto.random.int(u64));
    const random = prng.random();

    for (temp_data) |*d| {
        d.* = (random.float(f32) * 2.0 - 1.0) * scale;
    }

    try accel.copyToDevice(data.data, std.mem.sliceAsBytes(temp_data));
}

fn zeroInit(allocator: std.mem.Allocator, accel: *Accelerator, data: *Tensor, count: usize) !void {
    const temp_data = try allocator.alloc(f32, count);
    defer allocator.free(temp_data);
    @memset(temp_data, 0);
    try accel.copyToDevice(data.data, std.mem.sliceAsBytes(temp_data));
}
