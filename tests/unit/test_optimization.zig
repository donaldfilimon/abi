// Tests for Optimization Module

const std = @import("std");
const ai = @import("abi").ai;

test "SGD optimizer basic operations" {
    const testing = std.testing;
    var optimizer = try ai.optimization.SGD.init(testing.allocator, 0.01, 0.9, 0.0001, 3);
    defer optimizer.deinit(testing.allocator);

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };

    // Store original values
    const original = params;

    // Perform optimization step
    optimizer.step(&params, &grads);

    // Parameters should decrease (gradient descent)
    try testing.expect(params[0] < original[0]);
    try testing.expect(params[1] < original[1]);
    try testing.expect(params[2] < original[2]);
}

test "Adam optimizer basic operations" {
    const testing = std.testing;
    var optimizer = try ai.optimization.Adam.init(testing.allocator, 0.001, 0.9, 0.999, 1e-8, 3);
    defer optimizer.deinit();

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };

    // Store original values
    const original = params;

    // Perform multiple optimization steps
    for (0..5) |_| {
        optimizer.step(&params, &grads);
    }

    // Parameters should change
    try testing.expect(params[0] != original[0]);
    try testing.expect(params[1] != original[1]);
    try testing.expect(params[2] != original[2]);
}

test "learning rate scheduler" {
    const constant_lr = ai.optimization.LearningRateScheduler{ .constant = 0.01 };
    try std.testing.expectEqual(@as(f32, 0.01), constant_lr.getLearningRate(0));
    try std.testing.expectEqual(@as(f32, 0.01), constant_lr.getLearningRate(100));

    const exp_lr = ai.optimization.LearningRateScheduler{ .exponential = .{ .initial = 0.1, .decay = 0.9 } };
    try std.testing.expect(exp_lr.getLearningRate(0) > exp_lr.getLearningRate(1));
    try std.testing.expect(exp_lr.getLearningRate(1) > exp_lr.getLearningRate(2));

    const step_lr = ai.optimization.LearningRateScheduler{ .step = .{ .initial = 0.1, .step_size = 5, .gamma = 0.5 } };
    try std.testing.expectEqual(@as(f32, 0.1), step_lr.getLearningRate(0));
    try std.testing.expectEqual(@as(f32, 0.1), step_lr.getLearningRate(4));
    try std.testing.expectEqual(@as(f32, 0.05), step_lr.getLearningRate(5));
    try std.testing.expectEqual(@as(f32, 0.05), step_lr.getLearningRate(9));
    try std.testing.expectEqual(@as(f32, 0.025), step_lr.getLearningRate(10));
}

test "optimizer interface" {
    const testing = std.testing;

    // Test SGD through interface
    var sgd = ai.optimization.Optimizer{ .sgd = try ai.optimization.SGD.init(testing.allocator, 0.01, 0.0, 0.0, 2) };
    defer sgd.deinit(testing.allocator);

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.1, 0.1 };

    sgd.step(&params, &grads);
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);

    // Test Adam through interface
    var adam = ai.optimization.Optimizer{ .adam = try ai.optimization.Adam.init(testing.allocator, 0.01, 0.9, 0.999, 1e-8, 2) };
    defer adam.deinit();

    params = [_]f32{ 1.0, 2.0 };
    adam.step(&params, &grads);

    try testing.expect(params[0] != 1.0);
    try testing.expect(params[1] != 2.0);
}

test "optimizer zero gradients" {
    const testing = std.testing;
    var optimizer = try ai.optimization.SGD.init(testing.allocator, 0.01, 0.0, 0.0, 2);
    defer optimizer.deinit(testing.allocator);

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.0, 0.0 };

    const original = params;
    optimizer.step(&params, &grads);

    // Parameters should remain unchanged with zero gradients
    try testing.expectEqual(original[0], params[0]);
    try testing.expectEqual(original[1], params[1]);
}

test "optimizer parameter validation" {
    const testing = std.testing;

    // Test negative learning rate (should still work but may not converge)
    var optimizer = try ai.optimization.SGD.init(testing.allocator, -0.01, 0.0, 0.0, 2);
    defer optimizer.deinit(testing.allocator);

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.1, 0.1 };

    optimizer.step(&params, &grads);

    // Parameters should increase with negative learning rate
    try testing.expect(params[0] > 1.0);
    try testing.expect(params[1] > 2.0);
}
