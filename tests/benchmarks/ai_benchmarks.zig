const std = @import("std");
const testing = std.testing;

const ai = @import("../../mod.zig");

/// Benchmark GPU acceleration performance
pub fn benchmarkGPUAcceleration() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    const size = 1000;
    var data_a = try allocator.alloc(f32, size * size);
    defer allocator.free(data_a);
    var data_b = try allocator.alloc(f32, size * size);
    defer allocator.free(data_b);

    // Initialize test data
    for (data_a, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
    for (data_b, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i % 100)) * 0.01;

    // Benchmark matrix multiplication
    const start_time = std.time.nanoTimestamp();
    // Simulate GPU operation
    var result = try allocator.alloc(f32, size * size);
    defer allocator.free(result);
    for (0..size) |i| {
        for (0..size) |j| {
            var sum: f32 = 0;
            for (0..size) |k| {
                sum += data_a[i * size + k] * data_b[k * size + j];
            }
            result[i * size + j] = sum;
        }
    }
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("GPU matrix multiplication ({d}x{d}): {d:.2}ms", .{ size, size, duration_ms });
}

/// Benchmark database operations using vector database
pub fn benchmarkDatabaseOperations() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Create vector search instance
    const vec_search = try ai.gpu.vector_search_gpu.VectorSearchGPU.init(allocator, &accel, 128);
    defer vec_search.deinit();

    const num_vectors = 1000;
    const start_time = std.time.nanoTimestamp();

    // Insert vectors
    for (0..num_vectors) |i| {
        var embedding: [128]f32 = undefined;
        for (&embedding, 0..) |*e, j| {
            e.* = @sin(@as(f32, @floatFromInt(i * 128 + j))) * 0.1;
        }
        _ = try vec_search.insert(&embedding);
    }

    // Search operation
    const query: [128]f32 = [_]f32{0.1} ** 128;
    const results = try vec_search.search(&query, 10);
    defer allocator.free(results);

    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("Database vector search benchmark ({} vectors): {d:.2}ms", .{ num_vectors, duration_ms });
}

/// Benchmark AI inference using neural network layers
pub fn benchmarkAIInference() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Create a simple dense layer for benchmarking
    var layer = try ai.layers.Dense.init(allocator, &accel, 784, 128); // MNIST-like input
    defer layer.deinit();

    const batch_size = 32;
    const input_size = batch_size * 784;

    // Allocate input
    const input = try accel.alloc(input_size * @sizeOf(f32));
    defer accel.free(input);

    // Initialize with test data
    const data = try allocator.alloc(f32, input_size);
    defer allocator.free(data);
    for (data, 0..) |*d, i| {
        d.* = @sin(@as(f32, @floatFromInt(i))) * 0.1; // Some variation
    }
    try accel.copyToDevice(input, std.mem.sliceAsBytes(data));

    // Benchmark inference
    const start_time = std.time.nanoTimestamp();

    const output = try layer.forward(input, batch_size);
    defer accel.free(output);

    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("AI inference benchmark (Dense layer, batch_size={}): {d:.2}ms", .{ batch_size, duration_ms });
}

/// Benchmark optimization algorithms performance
pub fn benchmarkOptimizationAlgorithms() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = ai.optimization.OptimizerConfig{ .adam = .{
        .learning_rate = 0.001,
        .beta1 = 0.9,
        .beta2 = 0.999,
    } };

    var optimizer = try ai.optimization.createOptimizer(allocator, config, 1000);
    defer optimizer.deinit();

    // Create test parameters and gradients
    var params = try allocator.alloc(f32, 1000);
    defer allocator.free(params);
    var grads = try allocator.alloc(f32, 1000);
    defer allocator.free(grads);

    // Initialize with test data
    for (params, 0..) |*p, i| p.* = @sin(@as(f32, @floatFromInt(i))) * 0.1;
    for (grads, 0..) |*g, i| g.* = @cos(@as(f32, @floatFromInt(i))) * 0.01;

    const start_time = std.time.nanoTimestamp();

    // Run multiple optimization steps
    for (0..100) |_| {
        optimizer.update(params, grads);
    }

    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("Optimization benchmark (Adam, 1000 params, 100 steps): {d:.2}ms", .{duration_ms});
}

/// Benchmark federated learning aggregation
pub fn benchmarkFederatedLearning() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var coordinator = try ai.federated.Coordinator.init(allocator, 512);
    defer coordinator.deinit();

    // Register clients
    for (0..10) |i| {
        const client_id = try std.fmt.allocPrint(allocator, "client_{d}", .{i});
        try coordinator.registerClient(client_id);
        allocator.free(client_id);
    }

    const start_time = std.time.nanoTimestamp();

    // Simulate multiple rounds of federated learning
    for (0..5) |_| {
        // Generate random updates from clients
        var updates = try allocator.alloc([]const f32, 10);
        defer {
            for (updates) |update| allocator.free(update);
            allocator.free(updates);
        }

        for (0..10) |i| {
            updates[i] = try allocator.alloc(f32, 512);
            for (updates[i], 0..) |*val, j| {
                val.* = @sin(@as(f32, @floatFromInt(i * 512 + j))) * 0.1;
            }
        }

        try coordinator.aggregateUpdates(&updates);
    }

    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("Federated learning benchmark (10 clients, 512 params, 5 rounds): {d:.2}ms", .{duration_ms});
}

/// Benchmark reinforcement learning Q-learning
pub fn benchmarkReinforcementLearning() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var agent = try ai.reinforcement_learning.QlearningAgent.init(allocator, 100, 4, .{});
    defer agent.deinit();

    const start_time = std.time.nanoTimestamp();

    // Simulate learning episodes
    for (0..100) |_| {
        var state: usize = 0;
        var steps: usize = 0;

        while (steps < 50) { // Max steps per episode
            const action = agent.chooseAction(state);
            // Simulate environment (simplified)
            const reward = if (action == 0) @as(f32, 1.0) else -0.1;
            const next_state = (state + action) % 100;

            agent.learn(state, action, reward, next_state);

            state = next_state;
            steps += 1;

            if (state == 99) break; // Goal reached
        }
    }

    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("Reinforcement learning benchmark (Q-Learning, 100 episodes): {d:.2}ms", .{duration_ms});
}

test "performance benchmarks" {
    try benchmarkGPUAcceleration();
    try benchmarkDatabaseOperations();
    try benchmarkAIInference();
    try benchmarkOptimizationAlgorithms();
    try benchmarkFederatedLearning();
    try benchmarkReinforcementLearning();
}
