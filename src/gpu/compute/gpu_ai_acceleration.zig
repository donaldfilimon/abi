//! GPU AI/ML Acceleration Module
//!
//! This module provides GPU-accelerated AI and machine learning operations:
//! - Tensor operations (matrix multiplication, convolution, element-wise)
//! - Neural network layer acceleration (dense, conv, pooling)
//! - Training acceleration (backpropagation, optimization)
//! - Memory-efficient data transfer between CPU/GPU
//! - Automatic fallback to CPU when GPU unavailable

const std = @import("std");
const kernels = @import("kernels.zig");
const gpu_renderer = @import("../core/gpu_renderer.zig");

/// Tensor data type for GPU operations
pub const Tensor = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    shape: []usize,
    gpu_buffer: ?u32 = null,
    is_on_gpu: bool = false,

    /// Create a new tensor
    pub fn create(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
        const total_size = calculateSize(shape);
        const data = try allocator.alloc(f32, total_size);

        const tensor = try allocator.create(Tensor);
        tensor.* = .{
            .allocator = allocator,
            .data = data,
            .shape = try allocator.dupe(usize, shape),
        };
        return tensor;
    }

    /// Initialize tensor with values
    pub fn initWithData(allocator: std.mem.Allocator, shape: []const usize, values: []const f32) !*Tensor {
        const tensor = try create(allocator, shape);
        std.debug.assert(values.len == tensor.data.len);
        @memcpy(tensor.data, values);
        return tensor;
    }

    /// Upload tensor to GPU
    pub fn uploadToGpu(self: *Tensor, renderer: *gpu_renderer.GPURenderer) !void {
        if (self.gpu_buffer == null) {
            self.gpu_buffer = try renderer.createBuffer(.{
                .size = @as(u64, @intCast(self.data.len * @sizeOf(f32))),
                .usage = .{ .storage = true, .copy_dst = true },
            });
        }

        // Upload data to GPU
        try renderer.writeBuffer(self.gpu_buffer.?, self.data);
        self.is_on_gpu = true;
    }

    /// Download tensor from GPU
    pub fn downloadFromGpu(self: *Tensor, renderer: *gpu_renderer.GPURenderer) !void {
        if (!self.is_on_gpu or self.gpu_buffer == null) return;

        try renderer.readBuffer(self.gpu_buffer.?, self.data);
        self.is_on_gpu = false;
    }

    /// Get tensor element count
    pub fn size(self: Tensor) usize {
        return self.data.len;
    }

    /// Cleanup tensor resources
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        if (self.gpu_buffer) |buffer| {
            // Note: GPU buffer cleanup would happen in renderer
            _ = buffer;
        }
        self.allocator.destroy(self);
    }

    fn calculateSize(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| total *= dim;
        return total;
    }
};

/// GPU-accelerated matrix operations
pub const MatrixOps = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) MatrixOps {
        return .{
            .allocator = allocator,
            .renderer = renderer,
        };
    }

    /// Matrix multiplication: C = A * B
    pub fn matmul(self: *MatrixOps, a: *Tensor, b: *Tensor, c: *Tensor) !void {
        std.debug.assert(a.shape.len == 2 and b.shape.len == 2 and c.shape.len == 2);
        std.debug.assert(a.shape[1] == b.shape[0]);
        std.debug.assert(c.shape[0] == a.shape[0] and c.shape[1] == b.shape[1]);

        // Ensure tensors are on GPU
        if (!a.is_on_gpu) try a.uploadToGpu(self.renderer);
        if (!b.is_on_gpu) try b.uploadToGpu(self.renderer);
        if (!c.is_on_gpu) {
            c.gpu_buffer = try self.renderer.createBuffer(.{
                .size = @as(u64, @intCast(c.data.len * @sizeOf(f32))),
                .usage = .{ .storage = true, .copy_src = true },
            });
            c.is_on_gpu = true;
        }

        // TODO: Implement GPU matrix multiplication kernel
        // For now, fall back to CPU implementation
        try self.matmulCpu(a, b, c);
    }

    /// CPU fallback for matrix multiplication
    fn matmulCpu(self: *MatrixOps, a: *Tensor, b: *Tensor, c: *Tensor) !void {
        const m = a.shape[0];
        const n = b.shape[1];
        const k = a.shape[1];

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..k) |l| {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                c.data[i * n + j] = sum;
            }
        }
    }

    /// Matrix transpose
    pub fn transpose(self: *MatrixOps, input: *Tensor, output: *Tensor) !void {
        std.debug.assert(input.shape.len == 2 and output.shape.len == 2);
        std.debug.assert(input.shape[0] == output.shape[1] and input.shape[1] == output.shape[0]);

        const rows = input.shape[0];
        const cols = input.shape[1];

        for (0..rows) |i| {
            for (0..cols) |j| {
                output.data[j * rows + i] = input.data[i * cols + j];
            }
        }
    }

    /// Element-wise operations
    pub fn elementWiseAdd(self: *MatrixOps, a: *Tensor, b: *Tensor, result: *Tensor) !void {
        std.debug.assert(a.size() == b.size() and b.size() == result.size());

        for (0..a.size()) |i| {
            result.data[i] = a.data[i] + b.data[i];
        }
    }

    pub fn elementWiseMultiply(self: *MatrixOps, a: *Tensor, b: *Tensor, result: *Tensor) !void {
        std.debug.assert(a.size() == b.size() and b.size() == result.size());

        for (0..a.size()) |i| {
            result.data[i] = a.data[i] * b.data[i];
        }
    }
};

/// GPU-accelerated neural network operations
pub const NeuralNetworkOps = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    matrix_ops: MatrixOps,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) NeuralNetworkOps {
        return .{
            .allocator = allocator,
            .renderer = renderer,
            .matrix_ops = MatrixOps.init(allocator, renderer),
        };
    }

    /// Dense layer forward pass: output = activation(input * weights + biases)
    pub fn denseForward(self: *NeuralNetworkOps, input: *Tensor, weights: *Tensor, biases: *Tensor, output: *Tensor, activation: kernels.ActivationType) !void {
        // Input validation
        std.debug.assert(input.shape.len == 2); // [batch_size, input_features]
        std.debug.assert(weights.shape.len == 2); // [input_features, output_features]
        std.debug.assert(biases.shape.len == 2); // [1, output_features]
        std.debug.assert(output.shape.len == 2); // [batch_size, output_features]
        std.debug.assert(input.shape[1] == weights.shape[0]);
        std.debug.assert(weights.shape[1] == biases.shape[1]);
        std.debug.assert(output.shape[1] == biases.shape[1]);

        const batch_size = input.shape[0];
        const output_features = weights.shape[1];

        // Create temporary tensor for linear transformation
        var linear_output = try Tensor.create(self.allocator, &[_]usize{ batch_size, output_features });
        defer linear_output.deinit();

        // Linear transformation: input * weights
        try self.matrix_ops.matmul(input, weights, linear_output);

        // Add biases: linear_output + biases (broadcasting)
        for (0..batch_size) |batch| {
            for (0..output_features) |feature| {
                const bias_idx = feature; // biases shape is [1, output_features]
                linear_output.data[batch * output_features + feature] += biases.data[bias_idx];
            }
        }

        // Apply activation function
        for (0..linear_output.size()) |i| {
            output.data[i] = self.applyActivation(linear_output.data[i], activation);
        }
    }

    /// Convolution operation (simplified 2D convolution)
    pub fn conv2dForward(self: *NeuralNetworkOps, input: *Tensor, kernels_tensor: *Tensor, biases: *Tensor, output: *Tensor, stride: usize, padding: usize) !void {
        // Input validation
        std.debug.assert(input.shape.len == 4); // [batch, channels, height, width]
        std.debug.assert(kernels_tensor.shape.len == 4); // [out_channels, in_channels, kernel_h, kernel_w]
        std.debug.assert(biases.shape.len == 1); // [out_channels]
        std.debug.assert(output.shape.len == 4); // [batch, out_channels, out_h, out_w]

        const batch_size = input.shape[0];
        const in_channels = input.shape[1];
        const in_height = input.shape[2];
        const in_width = input.shape[3];
        const out_channels = kernels_tensor.shape[0];
        const kernel_h = kernels_tensor.shape[2];
        const kernel_w = kernels_tensor.shape[3];

        const out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
        const out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

        std.debug.assert(output.shape[0] == batch_size);
        std.debug.assert(output.shape[1] == out_channels);
        std.debug.assert(output.shape[2] == out_height);
        std.debug.assert(output.shape[3] == out_width);

        // Simple CPU implementation (GPU acceleration would use compute shaders)
        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var sum: f32 = biases.data[oc];

                        for (0..in_channels) |ic| {
                            for (0..kernel_h) |kh| {
                                for (0..kernel_w) |kw| {
                                    const ih = oh * stride + kh - padding;
                                    const iw = ow * stride + kw - padding;

                                    if (ih >= 0 and ih < in_height and iw >= 0 and iw < in_width) {
                                        const input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                        const kernel_idx = (((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw);
                                        sum += input.data[input_idx] * kernels_tensor.data[kernel_idx];
                                    }
                                }
                            }
                        }

                        const output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        output.data[output_idx] = sum;
                    }
                }
            }
        }
    }

    /// Max pooling operation
    pub fn maxPool2d(self: *NeuralNetworkOps, input: *Tensor, output: *Tensor, kernel_size: usize, stride: usize) !void {
        std.debug.assert(input.shape.len == 4); // [batch, channels, height, width]
        std.debug.assert(output.shape.len == 4); // [batch, channels, out_h, out_w]

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const in_height = input.shape[2];
        const in_width = input.shape[3];

        const out_height = (in_height - kernel_size) / stride + 1;
        const out_width = (in_width - kernel_size) / stride + 1;

        for (0..batch_size) |b| {
            for (0..channels) |c| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var max_val: f32 = -std.math.inf(f32);

                        for (0..kernel_size) |kh| {
                            for (0..kernel_size) |kw| {
                                const ih = oh * stride + kh;
                                const iw = ow * stride + kw;

                                const input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                max_val = @max(max_val, input.data[input_idx]);
                            }
                        }

                        const output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output.data[output_idx] = max_val;
                    }
                }
            }
        }
    }

    /// Apply activation function
    fn applyActivation(self: NeuralNetworkOps, x: f32, activation: kernels.ActivationType) f32 {
        _ = self;
        return switch (activation) {
            .relu => if (x > 0) x else 0,
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .tanh => std.math.tanh(x),
            .softmax => x, // Softmax handled separately
            .leaky_relu => if (x > 0) x else 0.01 * x,
            .elu => if (x > 0) x else @exp(x) - 1,
            .swish => x / (1.0 + @exp(-x)),
        };
    }
};

/// Training acceleration for neural networks
pub const TrainingAcceleration = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    nn_ops: NeuralNetworkOps,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) TrainingAcceleration {
        return .{
            .allocator = allocator,
            .renderer = renderer,
            .nn_ops = NeuralNetworkOps.init(allocator, renderer),
        };
    }

    /// Backpropagation for dense layer
    pub fn denseBackward(self: *TrainingAcceleration, input: *Tensor, weights: *Tensor, output_grad: *Tensor, input_grad: *Tensor, weights_grad: *Tensor, biases_grad: *Tensor, activation: kernels.ActivationType) !void {
        // Input validation
        std.debug.assert(input.shape.len == 2);
        std.debug.assert(weights.shape.len == 2);
        std.debug.assert(output_grad.shape == input_grad.shape);
        std.debug.assert(weights_grad.shape == weights.shape);
        std.debug.assert(biases_grad.shape.len == 2 and biases_grad.shape[1] == weights.shape[1]);

        // Apply activation derivative
        var activation_grad = try Tensor.create(self.allocator, output_grad.shape);
        defer activation_grad.deinit();

        for (0..output_grad.size()) |i| {
            activation_grad.data[i] = output_grad.data[i] * self.activationDerivative(input_grad.data[i], activation);
        }

        // Gradient w.r.t. weights: weights_grad = input^T * activation_grad
        try self.matrix_ops.transpose(input, input_grad); // Reuse input_grad as temp
        try self.matrix_ops.matmul(input_grad, activation_grad, weights_grad);

        // Gradient w.r.t. biases: biases_grad = sum(activation_grad, axis=0)
        const batch_size = activation_grad.shape[0];
        const output_features = activation_grad.shape[1];

        for (0..output_features) |feature| {
            var sum: f32 = 0;
            for (0..batch_size) |batch| {
                const idx = batch * output_features + feature;
                sum += activation_grad.data[idx];
            }
            biases_grad.data[feature] = sum;
        }

        // Gradient w.r.t. input: input_grad = activation_grad * weights^T
        var weights_t = try Tensor.create(self.allocator, &[_]usize{ weights.shape[1], weights.shape[0] });
        defer weights_t.deinit();

        try self.matrix_ops.transpose(weights, weights_t);
        try self.matrix_ops.matmul(activation_grad, weights_t, input_grad);
    }

    /// Activation function derivative
    fn activationDerivative(self: TrainingAcceleration, x: f32, activation: kernels.ActivationType) f32 {
        _ = self;
        return switch (activation) {
            .relu => if (x > 0) 1 else 0,
            .sigmoid => {
                const s = 1.0 / (1.0 + @exp(-x));
                return s * (1 - s);
            },
            .tanh => {
                const t = std.math.tanh(x);
                return 1 - t * t;
            },
            .softmax => 1, // Simplified
            .leaky_relu => if (x > 0) 1 else 0.01,
            .elu => if (x > 0) 1 else x + 1,
            .swish => {
                const s = 1.0 / (1.0 + @exp(-x));
                return s + x * s * (1 - s);
            },
        };
    }

    /// SGD optimizer step
    pub fn sgdStep(self: *TrainingAcceleration, weights: *Tensor, biases: *Tensor, weights_grad: *Tensor, biases_grad: *Tensor, learning_rate: f32) void {
        _ = self;
        // Update weights: weights -= learning_rate * weights_grad
        for (0..weights.size()) |i| {
            weights.data[i] -= learning_rate * weights_grad.data[i];
        }

        // Update biases: biases -= learning_rate * biases_grad
        for (0..biases.size()) |i| {
            biases.data[i] -= learning_rate * biases_grad.data[i];
        }
    }
};

/// Main AI/ML Acceleration Manager
pub const AIMLAcceleration = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    matrix_ops: MatrixOps,
    nn_ops: NeuralNetworkOps,
    training_accel: TrainingAcceleration,
    tensors: std.ArrayList(*Tensor),

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*AIMLAcceleration {
        const self = try allocator.create(AIMLAcceleration);
        self.* = .{
            .allocator = allocator,
            .renderer = renderer,
            .matrix_ops = MatrixOps.init(allocator, renderer),
            .nn_ops = NeuralNetworkOps.init(allocator, renderer),
            .training_accel = TrainingAcceleration.init(allocator, renderer),
            .tensors = std.ArrayList(*Tensor).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *AIMLAcceleration) void {
        for (self.tensors.items) |tensor| {
            tensor.deinit();
        }
        self.tensors.deinit();
        self.allocator.destroy(self);
    }

    /// Create and track a tensor
    pub fn createTensor(self: *AIMLAcceleration, shape: []const usize) !*Tensor {
        const tensor = try Tensor.create(self.allocator, shape);
        try self.tensors.append(tensor);
        return tensor;
    }

    /// Create tensor with data and track it
    pub fn createTensorWithData(self: *AIMLAcceleration, shape: []const usize, data: []const f32) !*Tensor {
        const tensor = try Tensor.initWithData(self.allocator, shape, data);
        try self.tensors.append(tensor);
        return tensor;
    }

    /// Get performance statistics
    pub fn getStats(self: *AIMLAcceleration) struct {
        total_tensors: usize,
        gpu_tensors: usize,
        cpu_tensors: usize,
        total_memory_mb: f32,
    } {
        var gpu_count: usize = 0;
        var cpu_count: usize = 0;
        var total_memory: usize = 0;

        for (self.tensors.items) |tensor| {
            if (tensor.is_on_gpu) {
                gpu_count += 1;
            } else {
                cpu_count += 1;
            }
            total_memory += tensor.size() * @sizeOf(f32);
        }

        return .{
            .total_tensors = self.tensors.items.len,
            .gpu_tensors = gpu_count,
            .cpu_tensors = cpu_count,
            .total_memory_mb = @as(f32, @floatFromInt(total_memory)) / (1024 * 1024),
        };
    }
};

/// Example usage and demonstration
pub fn demo() !void {
    std.debug.print("🚀 AI/ML GPU Acceleration Demo\n", .{});
    std.debug.print("===============================\n\n", .{});

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create mock GPU renderer (in real implementation, this would be initialized properly)
    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    // Initialize AI/ML acceleration
    const accel = try AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    std.debug.print("✅ AI/ML Acceleration initialized\n", .{});

    // Create sample tensors for neural network
    const input = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &[_]f32{
        0.5, 0.8, 0.2,
        0.1, 0.9, 0.6,
    });

    const weights = try accel.createTensorWithData(&[_]usize{ 3, 2 }, &[_]f32{
        0.1, 0.4,
        0.2, 0.5,
        0.3, 0.6,
    });

    const biases = try accel.createTensorWithData(&[_]usize{ 1, 2 }, &[_]f32{ 0.1, 0.2 });
    const output = try accel.createTensor(&[_]usize{ 2, 2 });

    std.debug.print("✅ Sample tensors created\n", .{});

    // Perform dense layer forward pass
    try accel.nn_ops.denseForward(input, weights, biases, output, .relu);

    std.debug.print("✅ Dense layer forward pass completed\n", .{});
    std.debug.print("Input:\n", .{});
    printTensor(input);
    std.debug.print("Output:\n", .{});
    printTensor(output);

    // Show performance statistics
    const stats = accel.getStats();
    std.debug.print("\n📊 Performance Statistics:\n", .{});
    std.debug.print("├─ Total tensors: {}\n", .{stats.total_tensors});
    std.debug.print("├─ GPU tensors: {}\n", .{stats.gpu_tensors});
    std.debug.print("├─ CPU tensors: {}\n", .{stats.cpu_tensors});
    std.debug.print("└─ Memory usage: {d:.2} MB\n", .{stats.total_memory_mb});

    std.debug.print("\n🎉 AI/ML GPU Acceleration Demo Complete!\n", .{});
}

/// Helper function to print tensor contents
fn printTensor(tensor: *Tensor) void {
    if (tensor.shape.len == 2) {
        const rows = tensor.shape[0];
        const cols = tensor.shape[1];
        for (0..rows) |i| {
            std.debug.print("  ", .{});
            for (0..cols) |j| {
                std.debug.print("{d:.3} ", .{tensor.data[i * cols + j]});
            }
            std.debug.print("\n", .{});
        }
    }
}
