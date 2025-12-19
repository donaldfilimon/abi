//! Neural Network Layer Abstraction
//!
//! Modular neural network layers with automatic differentiation support.

const std = @import("std");

const accelerator = @import("../../shared/platform/accelerator/accelerator.zig");
const ArrayList = std.array_list.Managed;

/// Parameter storage with gradients
pub const Parameter = struct {
    data: accelerator.DeviceMemory,
    grad: ?accelerator.DeviceMemory,
    shape: []const usize,
    requires_grad: bool = true,

    pub fn init(accel: *accelerator.Accelerator, shape: []const usize) !Parameter {
        const size = blk: {
            var s: usize = 1;
            for (shape) |dim| s *= dim;
            break :blk s * @sizeOf(f32);
        };

        const data = try accel.alloc(size);
        const grad = if (true) try accel.alloc(size) else null;

        return .{
            .data = data,
            .grad = grad,
            .shape = shape,
        };
    }

    pub fn deinit(self: *Parameter, accel: *accelerator.Accelerator) void {
        accel.free(&self.data);
        if (self.grad) |*g| accel.free(g);
    }

    pub fn elementCount(self: Parameter) usize {
        var count: usize = 1;
        for (self.shape) |dim| count *= dim;
        return count;
    }
};

/// Base layer interface
pub const Layer = struct {
    forward_fn: *const fn (*Layer, accelerator.DeviceMemory) accelerator.DeviceMemory,
    backward_fn: *const fn (*Layer, accelerator.DeviceMemory) void,
    params: ArrayList(Parameter),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Layer {
        return .{
            .forward_fn = undefined,
            .backward_fn = undefined,
            .params = ArrayList(Parameter).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Layer, accel: *accelerator.Accelerator) void {
        for (self.params.items) |*p| p.deinit(accel);
        self.params.deinit();
    }
};

/// Dense/Linear layer: y = Wx + b
pub const Dense = struct {
    layer: Layer,
    weight: Parameter,
    bias: Parameter,
    input_size: usize,
    output_size: usize,
    accel: *accelerator.Accelerator,
    last_input: ?accelerator.DeviceMemory = null,

    /// Initialize a dense layer.
    /// @param allocator: Memory allocator for internal structures
    /// @param accel: GPU accelerator for computations
    /// @param input_size: Number of input features
    /// @param output_size: Number of output features
    /// @return: Initialized Dense layer
    pub fn init(allocator: std.mem.Allocator, accel: *accelerator.Accelerator, input_size: usize, output_size: usize) !Dense {
        if (input_size == 0 or output_size == 0) return error.InvalidParameter;
        if (input_size == 0 or output_size == 0) return error.InvalidParameter;

        var weight = try Parameter.init(accel, &[_]usize{ output_size, input_size });
        errdefer weight.deinit(accel);
        var bias = try Parameter.init(accel, &[_]usize{output_size});
        errdefer bias.deinit(accel);

        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        randomInit(accel, &weight.data, weight.elementCount(), scale) catch |err| {
            weight.deinit(accel);
            bias.deinit(accel);
            return err;
        };
        zeroInit(accel, &bias.data, bias.elementCount()) catch |err| {
            weight.deinit(accel);
            bias.deinit(accel);
            return err;
        };

        return .{
            .layer = Layer.init(allocator),
            .weight = weight,
            .bias = bias,
            .input_size = input_size,
            .output_size = output_size,
            .accel = accel,
        };
    }

    pub fn deinit(self: *Dense) void {
        self.weight.deinit(self.accel);
        self.bias.deinit(self.accel);
        if (self.last_input) |*inp| self.accel.free(inp);
    }

    /// Forward pass through the dense layer.
    /// @param input: Input device memory
    /// @param batch_size: Number of samples in batch
    /// @return: Output device memory
    pub fn forward(self: *Dense, input: accelerator.DeviceMemory, batch_size: usize) !accelerator.DeviceMemory {
        // Save input for backward pass
        if (self.last_input) |*old| self.accel.free(old);
        const input_size = batch_size * self.input_size * @sizeOf(f32);
        self.last_input = try self.accel.alloc(input_size);
        errdefer if (self.last_input) |*li| self.accel.free(li);
        const src_slice: [*]const u8 = @ptrCast(input.ptr.?);
        try self.accel.copyToDevice(self.last_input.?, src_slice[0..input_size]);

        // Allocate output
        const output = try self.accel.alloc(batch_size * self.output_size * @sizeOf(f32));
        errdefer self.accel.free(&output);

        // Compute y = Wx + b
        const ops = accelerator.TensorOps.init(self.accel);
        ops.matmul(output, self.weight.data, input, self.output_size, batch_size, self.input_size);

        // Add bias (broadcasted)
        for (0..batch_size) |i| {
            const offset = i * self.output_size * @sizeOf(f32);
            const out_ptr: [*]u8 = @ptrCast(output.ptr.?);
            const bias_ptr: [*]const u8 = @ptrCast(self.bias.data.ptr.?);
            @memcpy(out_ptr[offset..][0 .. self.output_size * @sizeOf(f32)], bias_ptr[0 .. self.output_size * @sizeOf(f32)]);
        }

        return output;
    }

    /// Backward pass through the dense layer.
    /// @param grad_output: Gradient of output
    /// @param batch_size: Number of samples in batch
    /// @return: Gradient of input
    pub fn backward(self: *Dense, grad_output: accelerator.DeviceMemory, batch_size: usize) !accelerator.DeviceMemory {
        // Compute gradient w.r.t. weights: dW = grad_output^T @ input
        const ops = accelerator.TensorOps.init(self.accel);
        if (self.weight.grad) |weight_grad| {
            if (self.last_input) |input| {
                ops.matmul(weight_grad, grad_output, input, self.output_size, self.input_size, batch_size);
            }
        }

        // Compute gradient w.r.t. input: grad_input = W^T @ grad_output
        const grad_input = try self.accel.alloc(batch_size * self.input_size * @sizeOf(f32));
        errdefer self.accel.free(&grad_input);
        ops.matmul(grad_input, self.weight.data, grad_output, self.input_size, batch_size, self.output_size);

        return grad_input;
    }
};

/// ReLU activation layer
pub const ReLU = struct {
    accel: *accelerator.Accelerator,
    last_input: ?accelerator.DeviceMemory = null,

    pub fn init(accel: *accelerator.Accelerator) ReLU {
        return .{ .accel = accel };
    }

    pub fn deinit(self: *ReLU) void {
        if (self.last_input) |*inp| self.accel.free(inp);
    }

    pub fn forward(self: *ReLU, input: accelerator.DeviceMemory, size: usize) !accelerator.DeviceMemory {
        const output = try self.accel.alloc(size * @sizeOf(f32));
        const ops = accelerator.TensorOps.init(self.accel);
        ops.relu(output, input, size);

        // Save for backward
        if (self.last_input) |*old| self.accel.free(old);
        self.last_input = try self.accel.alloc(size * @sizeOf(f32));
        const src: [*]const u8 = @ptrCast(input.ptr.?);
        try self.accel.copyToDevice(self.last_input.?, src[0 .. size * @sizeOf(f32)]);

        return output;
    }

    pub fn backward(self: *ReLU, grad_output: accelerator.DeviceMemory, size: usize) !accelerator.DeviceMemory {
        const grad_input = try self.accel.alloc(size * @sizeOf(f32));

        // ReLU gradient: grad_input = grad_output * (input > 0)
        if (self.last_input) |input| {
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(input.ptr.?));
            const grad_out_ptr: [*]const f32 = @ptrCast(@alignCast(grad_output.ptr.?));
            const grad_in_ptr: [*]f32 = @ptrCast(@alignCast(grad_input.ptr.?));

            for (0..size) |i| {
                grad_in_ptr[i] = if (in_ptr[i] > 0) grad_out_ptr[i] else 0;
            }
        }

        return grad_input;
    }
};

/// Sequential model container
pub const Sequential = struct {
    layers: ArrayList(*anyopaque),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Sequential {
        return .{
            .layers = ArrayList(*anyopaque).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Sequential) void {
        self.layers.deinit();
    }
};

// Helper functions using proper memory management
fn randomInit(accel: *accelerator.Accelerator, mem: *accelerator.DeviceMemory, count: usize, scale: f32) !void {
    // Use arena allocator for temporary data generation
    var arena = std.heap.ArenaAllocator.init(accel.allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    const data = try temp_allocator.alloc(f32, count);

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp() / 1_000_000));
    const random = prng.random();

    for (data) |*d| {
        d.* = (random.float(f32) * 2.0 - 1.0) * scale;
    }

    try accel.copyToDevice(mem.*, std.mem.sliceAsBytes(data));
}

fn zeroInit(accel: *accelerator.Accelerator, mem: *accelerator.DeviceMemory, count: usize) !void {
    // Use arena allocator for temporary buffer
    var arena = std.heap.ArenaAllocator.init(accel.allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    const data = try temp_allocator.alloc(f32, count);
    @memset(data, 0);
    try accel.copyToDevice(mem.*, std.mem.sliceAsBytes(data));
}

test "dense layer forward" {
    const testing = std.testing;

    var accel = try accelerator.createBestAccelerator(testing.allocator);
    defer accel.deinit(&accel);
    var dense = try Dense.init(testing.allocator, &accel, 3, 2);
    defer dense.deinit();

    // Input: 1 sample, 3 features
    const input_mem = try accel.alloc(3 * @sizeOf(f32));
    defer accel.free(&input_mem);

    const input_data = [_]f32{ 1.0, 2.0, 3.0 };
    try accel.copyToDevice(input_mem, std.mem.sliceAsBytes(&input_data));

    const output = try dense.forward(input_mem, 1);
    defer accel.free(@constCast(&output));

    try testing.expect(output.isValid());
}
