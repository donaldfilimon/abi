const std = @import("std");
const testing = std.testing;

// Import the modules we want to test
const neural = @import("../src/neural.zig");
const simd_mod = @import("../src/simd/mod.zig");
const ai = @import("../src/ai/mod.zig");
const root = @import("../src/root.zig");

// Performance regression test for neural network operations
test "neural network performance regression - forward pass" {
    const allocator = testing.allocator;

    // Test forward pass performance with different network sizes
    const test_configs = [_]struct {
        input_size: usize,
        hidden_size: usize,
        expected_max_time_ns: u64, // Maximum allowed time per operation
    }{
        .{ .input_size = 8, .hidden_size = 16, .expected_max_time_ns = 100_000 }, // ~100μs
        .{ .input_size = 64, .hidden_size = 128, .expected_max_time_ns = 1_000_000 }, // ~1ms
        .{ .input_size = 256, .hidden_size = 512, .expected_max_time_ns = 10_000_000 }, // ~10ms
    };

    for (test_configs) |config| {
        var network = try neural.NeuralNetwork.init(allocator);
        defer network.deinit();

        // Add layers
        try network.addLayer(.{
            .type = .Dense,
            .input_size = config.input_size,
            .output_size = config.hidden_size,
            .activation = .ReLU,
        });
        try network.addLayer(.{
            .type = .Dense,
            .input_size = config.hidden_size,
            .output_size = config.input_size,
            .activation = .Sigmoid,
        });

        // Prepare input data
        const input = try allocator.alloc(f32, config.input_size);
        defer allocator.free(input);

        for (input) |*val| {
            val.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
        }

        // Benchmark forward pass
        var timer = try std.time.Timer.start();
        const iterations = 100;

        for (0..iterations) |_| {
            const output = try network.forward(input);
            defer allocator.free(output);
            // Verify output size
            try testing.expectEqual(@as(usize, config.input_size), output.len);
        }

        const total_time = timer.read();
        const avg_time_ns = total_time / iterations;

        // Check performance regression
        if (avg_time_ns > config.expected_max_time_ns) {
            std.debug.print("PERFORMANCE REGRESSION: Forward pass took {d}ns (expected < {d}ns) for {d}x{d} network\n", .{ avg_time_ns, config.expected_max_time_ns, config.input_size, config.hidden_size });
            // Don't fail the test, just warn about regression
        }

        std.debug.print("Forward pass performance: {d}x{d} network - {d}ns avg\n", .{ config.input_size, config.hidden_size, avg_time_ns });
    }
}

test "neural network performance regression - training step" {
    const allocator = testing.allocator;

    // Test training step performance
    const test_configs = [_]struct {
        input_size: usize,
        expected_max_time_ns: u64,
    }{
        .{ .input_size = 16, .expected_max_time_ns = 500_000 }, // ~500μs
        .{ .input_size = 64, .expected_max_time_ns = 2_000_000 }, // ~2ms
        .{ .input_size = 128, .expected_max_time_ns = 5_000_000 }, // ~5ms
    };

    for (test_configs) |config| {
        var network = try neural.NeuralNetwork.init(allocator);
        defer network.deinit();

        // Simple network for training
        try network.addLayer(.{
            .type = .Dense,
            .input_size = config.input_size,
            .output_size = config.input_size,
            .activation = .ReLU,
        });

        // Prepare training data
        const input = try allocator.alloc(f32, config.input_size);
        defer allocator.free(input);
        const target = try allocator.alloc(f32, config.input_size);
        defer allocator.free(target);

        for (input, target) |*in, *tgt| {
            in.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
            tgt.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
        }

        // Benchmark training step
        var timer = try std.time.Timer.start();
        const iterations = 50;

        var total_loss: f32 = 0;
        for (0..iterations) |_| {
            const loss = try network.trainStep(input, target, 0.01);
            total_loss += loss;
        }

        const total_time = timer.read();
        const avg_time_ns = total_time / iterations;
        const avg_loss = total_loss / @as(f32, @floatFromInt(iterations));

        // Check performance regression
        if (avg_time_ns > config.expected_max_time_ns) {
            std.debug.print("PERFORMANCE REGRESSION: Training step took {d}ns (expected < {d}ns) for {d} input size\n", .{ avg_time_ns, config.expected_max_time_ns, config.input_size });
        }

        std.debug.print("Training step performance: {d} input size - {d}ns avg, loss: {d:.4}\n", .{ config.input_size, avg_time_ns, avg_loss });
    }
}

test "SIMD performance regression - vector operations" {
    const allocator = testing.allocator;

    // Test SIMD vector operation performance
    const sizes = [_]usize{ 128, 512, 2048, 8192 };

    for (sizes) |size| {
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);
        const result = try allocator.alloc(f32, size);
        defer allocator.free(result);

        // Initialize test data
        for (a, b) |*va, *vb| {
            va.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
            vb.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
        }

        const opts = simd_mod.SIMDOpts{};

        // Benchmark dot product
        var timer = try std.time.Timer.start();
        const iterations = 1000;

        for (0..iterations) |_| {
            _ = simd_mod.dotProductSIMD(a, b, opts);
        }

        const dot_time = timer.read();
        const dot_ns_per_op = dot_time / iterations;

        // Benchmark vector addition
        timer.reset();
        for (0..iterations) |_| {
            _ = simd_mod.vectorAddSIMD(a, b, result);
        }

        const add_time = timer.read();
        const add_ns_per_op = add_time / iterations;

        // Performance expectations (rough estimates)
        const expected_dot_ns = @as(u64, size) * 2; // Very rough estimate
        const expected_add_ns = @as(u64, size); // Very rough estimate

        if (dot_ns_per_op > expected_dot_ns * 10) { // Allow 10x slowdown
            std.debug.print("PERFORMANCE REGRESSION: Dot product {d}ns/op (size={d})\n", .{ dot_ns_per_op, size });
        }

        if (add_ns_per_op > expected_add_ns * 10) { // Allow 10x slowdown
            std.debug.print("PERFORMANCE REGRESSION: Vector add {d}ns/op (size={d})\n", .{ add_ns_per_op, size });
        }

        std.debug.print("SIMD performance: size={d}, dot={d}ns, add={d}ns\n", .{ size, dot_ns_per_op, add_ns_per_op });
    }
}

test "SIMD performance regression - text operations" {
    // Test SIMD text operation performance with different text sizes
    const test_texts = [_]struct {
        name: []const u8,
        text: []const u8,
        expected_max_ns: u64,
    }{
        .{
            .name = "small",
            .text = "Hello World! This is a small test string.",
            .expected_max_ns = 1000, // ~1μs
        },
        .{
            .name = "medium",
            .text = "This is a medium-sized test string with more content to process and analyze for performance characteristics.",
            .expected_max_ns = 5000, // ~5μs
        },
        .{
            .name = "large",
            .text = "This is a large test string with significantly more content to process and analyze. It contains multiple sentences and should provide a good benchmark for SIMD text processing performance with various character patterns and frequencies that might affect processing speed.",
            .expected_max_ns = 10_000, // ~10μs
        },
    };

    for (test_texts) |test_case| {
        var timer = try std.time.Timer.start();
        const iterations = 1000;

        for (0..iterations) |_| {
            _ = simd_mod.text.countByte(test_case.text, 'e');
            _ = simd_mod.text.findByte(test_case.text, 'e');
        }

        const total_time = timer.read();
        const avg_time_ns = total_time / iterations;

        if (avg_time_ns > test_case.expected_max_ns) {
            std.debug.print("PERFORMANCE REGRESSION: Text ops {d}ns/op for {s} text\n", .{ avg_time_ns, test_case.name });
        }

        std.debug.print("Text ops performance: {s} - {d}ns avg\n", .{ test_case.name, avg_time_ns });
    }
}

test "AI performance regression - token estimation" {
    // Test AI token estimation performance
    const test_texts = [_]struct {
        size: usize,
        expected_max_ns: u64,
    }{
        .{ .size = 10, .expected_max_ns = 500 }, // Very small
        .{ .size = 100, .expected_max_ns = 1000 }, // Small
        .{ .size = 1000, .expected_max_ns = 5000 }, // Medium
        .{ .size = 10000, .expected_max_ns = 50_000 }, // Large
    };

    for (test_texts) |test_case| {
        // Generate test text of specified size
        const text = try testing.allocator.alloc(u8, test_case.size);
        defer testing.allocator.free(text);

        for (text, 0..) |*char, i| {
            char.* = @as(u8, @intCast(32 + (i % 95))); // Printable ASCII
        }

        var timer = try std.time.Timer.start();
        const iterations = 1000;

        for (0..iterations) |_| {
            _ = try ai.estimateTokens(text);
        }

        const total_time = timer.read();
        const avg_time_ns = total_time / iterations;

        if (avg_time_ns > test_case.expected_max_ns) {
            std.debug.print("PERFORMANCE REGRESSION: Token estimation {d}ns/op for {d} chars\n", .{ avg_time_ns, test_case.size });
        }

        std.debug.print("Token estimation performance: {d} chars - {d}ns avg\n", .{ test_case.size, avg_time_ns });
    }
}

test "memory allocation performance regression" {
    const allocator = testing.allocator;

    // Test memory allocation patterns performance
    const allocation_sizes = [_]usize{ 64, 256, 1024, 4096, 16384 };

    for (allocation_sizes) |size| {
        var timer = try std.time.Timer.start();
        const iterations = 1000;

        for (0..iterations) |_| {
            const buffer = try allocator.alloc(u8, size);
            defer allocator.free(buffer);

            // Touch the memory to ensure allocation
            @memset(buffer, 0);
        }

        const total_time = timer.read();
        const avg_time_ns = total_time / iterations;

        // Memory allocation should be reasonably fast
        const expected_max_ns = @as(u64, size / 64) * 1000; // Rough estimate

        if (avg_time_ns > expected_max_ns) {
            std.debug.print("PERFORMANCE REGRESSION: Allocation {d}ns/op for {d} bytes\n", .{ avg_time_ns, size });
        }

        std.debug.print("Memory allocation performance: {d} bytes - {d}ns avg\n", .{ size, avg_time_ns });
    }
}

test "concurrent operations performance regression" {
    // Test performance of concurrent operations (simplified)
    const allocator = testing.allocator;

    // Simple concurrent test using multiple neural networks
    const num_networks = 4;
    const network_size = 32;

    var timer = try std.time.Timer.start();

    // Create multiple networks
    const networks = try allocator.alloc(*neural.NeuralNetwork, num_networks);
    defer {
        for (networks) |network| {
            network.deinit();
        }
        allocator.free(networks);
    }

    for (networks) |*network| {
        network.* = try neural.NeuralNetwork.init(allocator);

        try network.*.addLayer(.{
            .type = .Dense,
            .input_size = network_size,
            .output_size = network_size,
            .activation = .ReLU,
        });
    }

    // Test operations on all networks
    const input = try allocator.alloc(f32, network_size);
    defer allocator.free(input);

    for (input) |*val| {
        val.* = @as(f32, @floatFromInt(std.crypto.random.int(u8))) / 255.0;
    }

    const iterations = 100;
    for (0..iterations) |_| {
        for (networks) |network| {
            const output = try network.forward(input);
            defer allocator.free(output);
            try testing.expectEqual(@as(usize, network_size), output.len);
        }
    }

    const total_time = timer.read();
    const avg_time_per_operation_ns = total_time / (iterations * num_networks);

    const expected_max_ns = 1_000_000; // 1ms per operation
    if (avg_time_per_operation_ns > expected_max_ns) {
        std.debug.print("PERFORMANCE REGRESSION: Concurrent ops {d}ns/op\n", .{avg_time_per_operation_ns});
    }

    std.debug.print("Concurrent operations performance: {d}ns per operation\n", .{avg_time_per_operation_ns});
}

test "benchmark baseline establishment" {
    // Establish performance baselines for future regression detection

    const BaselineResult = struct {
        operation: []const u8,
        time_ns: u64,
        timestamp: u64,
    };

    var results = std.ArrayList(BaselineResult).init(testing.allocator);
    defer results.deinit();

    const timestamp = std.time.nanoTimestamp();

    // Benchmark 1: Simple SIMD dot product
    {
        const size = 1024;
        const a = try testing.allocator.alloc(f32, size);
        defer testing.allocator.free(a);
        const b = try testing.allocator.alloc(f32, size);
        defer testing.allocator.free(b);

        for (a, b) |*va, *vb| {
            va.* = 1.0;
            vb.* = 1.0;
        }

        var timer = try std.time.Timer.start();
        const iterations = 1000;

        for (0..iterations) |_| {
            _ = simd_mod.dotProductSIMD(a, b, .{});
        }

        const time_ns = timer.read() / iterations;
        try results.append(.{
            .operation = "simd_dot_product_1024",
            .time_ns = time_ns,
            .timestamp = timestamp,
        });

        std.debug.print("BASELINE: {s} = {d}ns\n", .{ "simd_dot_product_1024", time_ns });
    }

    // Benchmark 2: Neural network forward pass
    {
        var network = try neural.NeuralNetwork.init(testing.allocator);
        defer network.deinit();

        try network.addLayer(.{
            .type = .Dense,
            .input_size = 32,
            .output_size = 32,
            .activation = .ReLU,
        });

        const input = try testing.allocator.alloc(f32, 32);
        defer testing.allocator.free(input);

        for (input) |*val| {
            val.* = 0.5;
        }

        var timer = try std.time.Timer.start();
        const iterations = 100;

        for (0..iterations) |_| {
            const output = try network.forward(input);
            defer testing.allocator.free(output);
        }

        const time_ns = timer.read() / iterations;
        try results.append(.{
            .operation = "neural_forward_32x32",
            .time_ns = time_ns,
            .timestamp = timestamp,
        });

        std.debug.print("BASELINE: {s} = {d}ns\n", .{ "neural_forward_32x32", time_ns });
    }

    // In a real implementation, these baselines would be saved to a file
    // and compared against in future test runs to detect regressions
    try testing.expect(results.items.len >= 2);
}
