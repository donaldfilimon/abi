//! GPU AI/ML Acceleration Tests
//!
//! This module tests the GPU-accelerated AI/ML operations:
//! - Tensor operations and memory management
//! - Matrix operations (multiplication, element-wise)
//! - Neural network operations (dense layers, convolutions)
//! - Training acceleration (backpropagation, optimization)
//! - Memory efficiency and performance

const std = @import("std");
const testing = std.testing;
const gpu_accel = @import("../src/gpu/compute/gpu_ai_acceleration.zig");
const gpu_renderer = @import("../src/gpu/core/gpu_renderer.zig");

/// Test tensor creation and basic operations
test "GPU AI: Tensor creation and basic operations" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create mock renderer
    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    // Create acceleration module
    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Test tensor creation
    const tensor = try accel.createTensor(&[_]usize{ 2, 3 });
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 6), tensor.size());
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, tensor.shape);
    try testing.expect(!tensor.is_on_gpu);

    // Test tensor creation with data
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const tensor_with_data = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &data);
    defer tensor_with_data.deinit();

    try testing.expectEqualSlices(f32, &data, tensor_with_data.data);
}

/// Test matrix operations
test "GPU AI: Matrix operations" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create test matrices
    const a = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &[_]f32{
        1, 2, 3,
        4, 5, 6,
    });
    defer a.deinit();

    const b = try accel.createTensorWithData(&[_]usize{ 3, 2 }, &[_]f32{
        7, 8,
        9, 10,
        11, 12,
    });
    defer b.deinit();

    const result = try accel.createTensor(&[_]usize{ 2, 2 });
    defer result.deinit();

    // Test matrix multiplication
    try accel.matrix_ops.matmul(a, b, result);

    // Expected result: [1,2,3; 4,5,6] * [7,8; 9,10; 11,12] = [58,64; 139,154]
    try testing.expectEqual(@as(f32, 58), result.data[0]);
    try testing.expectEqual(@as(f32, 64), result.data[1]);
    try testing.expectEqual(@as(f32, 139), result.data[2]);
    try testing.expectEqual(@as(f32, 154), result.data[3]);
}

/// Test element-wise operations
test "GPU AI: Element-wise operations" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    const a = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();

    const b = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &[_]f32{ 2, 3, 4, 5, 6, 7 });
    defer b.deinit();

    const result = try accel.createTensor(&[_]usize{ 2, 3 });
    defer result.deinit();

    // Test element-wise addition
    try accel.matrix_ops.elementWiseAdd(a, b, result);
    try testing.expectEqual(@as(f32, 3), result.data[0]);  // 1 + 2
    try testing.expectEqual(@as(f32, 5), result.data[1]);  // 2 + 3
    try testing.expectEqual(@as(f32, 7), result.data[2]);  // 3 + 4

    // Test element-wise multiplication
    try accel.matrix_ops.elementWiseMultiply(a, b, result);
    try testing.expectEqual(@as(f32, 2), result.data[0]);  // 1 * 2
    try testing.expectEqual(@as(f32, 6), result.data[1]);  // 2 * 3
    try testing.expectEqual(@as(f32, 12), result.data[2]); // 3 * 4
}

/// Test neural network operations
test "GPU AI: Neural network dense layer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create input (batch_size=2, input_features=3)
    const input = try accel.createTensorWithData(&[_]usize{ 2, 3 }, &[_]f32{
        0.5, 0.8, 0.2,
        0.1, 0.9, 0.6,
    });
    defer input.deinit();

    // Create weights (input_features=3, output_features=2)
    const weights = try accel.createTensorWithData(&[_]usize{ 3, 2 }, &[_]f32{
        0.1, 0.4,
        0.2, 0.5,
        0.3, 0.6,
    });
    defer weights.deinit();

    // Create biases (1, output_features=2)
    const biases = try accel.createTensorWithData(&[_]usize{ 1, 2 }, &[_]f32{ 0.1, 0.2 });
    defer biases.deinit();

    const output = try accel.createTensor(&[_]usize{ 2, 2 });
    defer output.deinit();

    // Test dense layer forward pass with ReLU
    try accel.nn_ops.denseForward(input, weights, biases, output, .relu);

    // Verify output dimensions
    try testing.expectEqual(@as(usize, 2), output.shape[0]); // batch_size
    try testing.expectEqual(@as(usize, 2), output.shape[1]); // output_features

    // Verify ReLU activation (no negative values)
    for (output.data) |val| {
        try testing.expect(val >= 0);
    }
}

/// Test convolution operations
test "GPU AI: Convolution operations" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create input (batch=1, channels=1, height=4, width=4)
    const input = try accel.createTensorWithData(&[_]usize{ 1, 1, 4, 4 }, &[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
    defer input.deinit();

    // Create kernel (out_channels=1, in_channels=1, kernel_h=2, kernel_w=2)
    const kernel = try accel.createTensorWithData(&[_]usize{ 1, 1, 2, 2 }, &[_]f32{
        1, 0,
        0, 1,
    });
    defer kernel.deinit();

    // Create biases (out_channels=1)
    const biases = try accel.createTensorWithData(&[_]usize{ 1 }, &[_]f32{ 0 });
    defer biases.deinit();

    // Create output (batch=1, out_channels=1, out_h=3, out_w=3) - 2x2 kernel, stride=1, no padding
    const output = try accel.createTensor(&[_]usize{ 1, 1, 3, 3 });
    defer output.deinit();

    // Test convolution
    try accel.nn_ops.conv2dForward(input, kernel, biases, output, 1, 0);

    // Verify output dimensions
    try testing.expectEqual(@as(usize, 1), output.shape[0]); // batch
    try testing.expectEqual(@as(usize, 1), output.shape[1]); // channels
    try testing.expectEqual(@as(usize, 3), output.shape[2]); // height
    try testing.expectEqual(@as(usize, 3), output.shape[3]); // width

    // Check some expected values (manual calculation)
    try testing.expectEqual(@as(f32, 1 + 6), output.data[0]); // top-left: 1*1 + 6*1 + 0*5 + 0*6
    try testing.expectEqual(@as(f32, 2 + 7), output.data[1]); // top-middle
}

/// Test max pooling
test "GPU AI: Max pooling" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create input (batch=1, channels=1, height=4, width=4)
    const input = try accel.createTensorWithData(&[_]usize{ 1, 1, 4, 4 }, &[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
    defer input.deinit();

    const output = try accel.createTensor(&[_]usize{ 1, 1, 2, 2 });
    defer output.deinit();

    // Test 2x2 max pooling with stride 2
    try accel.nn_ops.maxPool2d(input, output, 2, 2);

    // Verify output dimensions
    try testing.expectEqual(@as(usize, 1), output.shape[0]); // batch
    try testing.expectEqual(@as(usize, 1), output.shape[1]); // channels
    try testing.expectEqual(@as(usize, 2), output.shape[2]); // height
    try testing.expectEqual(@as(usize, 2), output.shape[3]); // width

    // Check max pooling results
    try testing.expectEqual(@as(f32, 6), output.data[0]);  // max of [1,2,5,6]
    try testing.expectEqual(@as(f32, 8), output.data[1]);  // max of [3,4,7,8]
    try testing.expectEqual(@as(f32, 14), output.data[2]); // max of [9,10,13,14]
    try testing.expectEqual(@as(f32, 16), output.data[3]); // max of [11,12,15,16]
}

/// Test activation functions
test "GPU AI: Activation functions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    const nn_ops = accel.nn_ops;

    // Test ReLU
    try testing.expectEqual(@as(f32, 0), nn_ops.applyActivation(-1.0, .relu));
    try testing.expectEqual(@as(f32, 0), nn_ops.applyActivation(0.0, .relu));
    try testing.expectEqual(@as(f32, 2.5), nn_ops.applyActivation(2.5, .relu));

    // Test Sigmoid
    const sigmoid_val = nn_ops.applyActivation(0.0, .sigmoid);
    try testing.expect(sigmoid_val > 0.49 and sigmoid_val < 0.51); // Should be close to 0.5

    // Test Tanh
    const tanh_val = nn_ops.applyActivation(0.0, .tanh);
    try testing.expect(tanh_val > -0.01 and tanh_val < 0.01); // Should be close to 0
}

/// Test training acceleration
test "GPU AI: Training acceleration" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create simple dense layer parameters
    const input = try accel.createTensorWithData(&[_]usize{ 2, 2 }, &[_]f32{ 0.5, 0.8, 0.1, 0.9 });
    defer input.deinit();

    const weights = try accel.createTensorWithData(&[_]usize{ 2, 1 }, &[_]f32{ 0.1, 0.2 });
    defer weights.deinit();

    const output_grad = try accel.createTensorWithData(&[_]usize{ 2, 1 }, &[_]f32{ 0.1, -0.1 });
    defer output_grad.deinit();

    const input_grad = try accel.createTensor(&[_]usize{ 2, 2 });
    defer input_grad.deinit();

    const weights_grad = try accel.createTensor(&[_]usize{ 2, 1 });
    defer weights_grad.deinit();

    const biases_grad = try accel.createTensor(&[_]usize{ 1, 1 });
    defer biases_grad.deinit();

    // Test backpropagation
    try accel.training_accel.denseBackward(input, weights, output_grad, input_grad, weights_grad, biases_grad, .relu);

    // Verify gradients are computed (non-zero)
    var has_nonzero_grad = false;
    for (weights_grad.data) |grad| {
        if (grad != 0) has_nonzero_grad = true;
    }
    try testing.expect(has_nonzero_grad);
}

/// Test performance statistics
test "GPU AI: Performance statistics" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create some tensors
    const tensor1 = try accel.createTensor(&[_]usize{ 10, 10 });
    const tensor2 = try accel.createTensor(&[_]usize{ 5, 5 });
    const tensor3 = try accel.createTensor(&[_]usize{ 20, 20 });

    // Test statistics
    const stats = accel.getStats();
    try testing.expectEqual(@as(usize, 3), stats.total_tensors);
    try testing.expectEqual(@as(usize, 0), stats.gpu_tensors); // No tensors uploaded to GPU
    try testing.expectEqual(@as(usize, 3), stats.cpu_tensors);
    try testing.expect(stats.total_memory_mb > 0);

    // Cleanup
    tensor1.deinit();
    tensor2.deinit();
    tensor3.deinit();
}

/// Test SGD optimization
test "GPU AI: SGD optimization" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const renderer = try gpu_renderer.GPURenderer.init(allocator, .vulkan);
    defer renderer.deinit();

    const accel = try gpu_accel.AIMLAcceleration.init(allocator, renderer);
    defer accel.deinit();

    // Create weights and gradients
    const weights = try accel.createTensorWithData(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer weights.deinit();

    const biases = try accel.createTensorWithData(&[_]usize{ 1, 2 }, &[_]f32{ 0.5, 1.5 });
    defer biases.deinit();

    const weights_grad = try accel.createTensorWithData(&[_]usize{ 2, 2 }, &[_]f32{ 0.1, 0.2, 0.3, 0.4 });
    defer weights_grad.deinit();

    const biases_grad = try accel.createTensorWithData(&[_]usize{ 1, 2 }, &[_]f32{ 0.05, 0.15 });
    defer biases_grad.deinit();

    const original_weight_0 = weights.data[0];
    const original_bias_0 = biases.data[0];

    // Apply SGD step
    accel.training_accel.sgdStep(weights, biases, weights_grad, biases_grad, 0.1);

    // Verify weights were updated: w = w - lr * grad
    try testing.expect(weights.data[0] < original_weight_0); // Should decrease
    try testing.expect(biases.data[0] < original_bias_0);   // Should decrease

    // Check specific values
    try testing.expectEqual(original_weight_0 - 0.1 * 0.1, weights.data[0]);
    try testing.expectEqual(original_bias_0 - 0.1 * 0.05, biases.data[0]);
}
