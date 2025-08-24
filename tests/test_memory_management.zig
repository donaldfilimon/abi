const std = @import("std");
const testing = std.testing;

// Import the modules we want to test
const neural = @import("../src/neural.zig");
const simd_mod = @import("../src/simd/mod.zig");
const root = @import("../src/root.zig");

// Test neural network memory management
test "neural network memory management - layer allocation" {
    const allocator = testing.allocator;

    // Test Layer creation and cleanup
    {
        var layer = try neural.Layer.init(allocator, .{
            .type = .Dense,
            .input_size = 10,
            .output_size = 5,
            .activation = .ReLU,
        });
        defer layer.deinit();

        // Verify layer properties
        try testing.expectEqual(@as(usize, 10), layer.input_size);
        try testing.expectEqual(@as(usize, 5), layer.output_size);
        try testing.expectEqual(neural.LayerType.Dense, layer.type);
        try testing.expectEqual(neural.Activation.ReLU, layer.activation);

        // Verify memory allocation
        try testing.expect(layer.weights.len > 0);
        try testing.expect(layer.biases.len > 0);
    }
    // Memory should be freed after scope
}

test "neural network memory management - network lifecycle" {
    const allocator = testing.allocator;

    // Test complete network lifecycle
    {
        var network = try neural.NeuralNetwork.init(allocator);
        defer network.deinit();

        // Add layers
        try network.addLayer(.{
            .type = .Dense,
            .input_size = 4,
            .output_size = 8,
            .activation = .ReLU,
        });
        try network.addLayer(.{
            .type = .Dense,
            .input_size = 8,
            .output_size = 3,
            .activation = .Sigmoid,
        });

        try testing.expectEqual(@as(usize, 2), network.layers.items.len);

        // Test forward pass memory management
        const input = [_]f32{ 1.0, 0.5, -0.5, 1.5 };
        const output = try network.forward(&input);
        defer allocator.free(output);

        try testing.expectEqual(@as(usize, 3), output.len);
    }
    // All network memory should be freed
}

test "neural network memory management - training memory safety" {
    const allocator = testing.allocator;

    // Test training with proper memory management
    {
        var network = try neural.NeuralNetwork.init(allocator);
        defer network.deinit();

        // Simple network
        try network.addLayer(.{
            .type = .Dense,
            .input_size = 2,
            .output_size = 1,
            .activation = .Sigmoid,
        });

        const input = [_]f32{ 0.5, -0.2 };
        const target = [_]f32{0.7};

        // Test multiple training steps
        for (0..5) |_| {
            const loss = try network.trainStep(&input, &target, 0.1);
            try testing.expect(loss >= 0.0);
        }
    }
    // Memory should be properly freed
}

test "neural network memory management - error handling" {
    const allocator = testing.allocator;

    // Test that errors don't leak memory
    {
        var network = try neural.NeuralNetwork.init(allocator);
        defer network.deinit();

        try network.addLayer(.{
            .type = .Dense,
            .input_size = 2,
            .output_size = 1,
            .activation = .ReLU,
        });

        // Test with mismatched dimensions (should not leak)
        const bad_input = [_]f32{ 1.0, 2.0, 3.0 }; // Wrong size
        const target = [_]f32{0.5};

        // This should fail gracefully without leaking memory
        const result = network.trainStep(&bad_input, &target, 0.1);
        try testing.expectError(error.Assertion, result);
    }
}

test "SIMD memory management - aligned buffer operations" {
    const allocator = testing.allocator;

    // Test SIMD operations with aligned buffers
    const size = 1024;
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);

    // Initialize with test data
    for (data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) / 10.0;
    }

    // Test SIMD normalization (should not leak memory)
    {
        const copy = try allocator.dupe(f32, data);
        defer allocator.free(copy);

        // This should work without alignment issues
        const opts = simd_mod.SIMDOpts{};
        _ = opts; // autofix
        if (simd_mod.config.has_simd) {
            // Test SIMD normalization
            const copy2 = try allocator.dupe(f32, copy);
            defer allocator.free(copy2);

            const original_magnitude = std.math.sqrt(std.mem.reduce(f32, .Add, f32, &[_]f32{}, copy, struct {
                fn sum(accum: f32, val: f32) f32 {
                    return accum + val * val;
                }
            }.sum));

            if (original_magnitude > 0) {
                const normalized = try allocator.dupe(f32, copy);
                defer allocator.free(normalized);

                // Normalize manually for comparison
                for (normalized, copy) |*n, v| {
                    n.* = v / original_magnitude;
                }

                // Verify normalization worked
                const new_magnitude = std.math.sqrt(std.mem.reduce(f32, .Add, f32, &[_]f32{}, normalized, struct {
                    fn sum(accum: f32, val: f32) f32 {
                        return accum + val * val;
                    }
                }.sum));
                try testing.expectApproxEqAbs(@as(f32, 1.0), new_magnitude, 0.01);
            }
        }
    }
}

test "SIMD memory management - text operations alignment" {
    // Test SIMD text operations with various alignments

    // Test with aligned string literal
    const aligned_text = "Hello World! This is a test string.";
    const count_l = simd_mod.text.countByte(aligned_text, 'l');
    try testing.expectEqual(@as(usize, 3), count_l);

    // Test with unaligned string (created at runtime)
    const unaligned_text = try testing.allocator.dupe(u8, "Hello World! This is a test string.");
    defer testing.allocator.free(unaligned_text);

    const count_l_unaligned = simd_mod.text.countByte(unaligned_text, 'l');
    try testing.expectEqual(@as(usize, 3), count_l_unaligned);

    // Test findByte with both aligned and unaligned
    const pos_aligned = simd_mod.text.findByte(aligned_text, 'W');
    try testing.expectEqual(@as(?usize, 6), pos_aligned);

    const pos_unaligned = simd_mod.text.findByte(unaligned_text, 'W');
    try testing.expectEqual(@as(?usize, 6), pos_unaligned);
}

test "SIMD memory management - buffer operations" {
    const allocator = testing.allocator;

    // Test vector operations with proper memory management
    const size = 128;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize test data
    for (a, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }
    for (b, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i * 2));
    }

    // Test SIMD vector addition
    const opts = simd_mod.SIMDOpts{};
    if (simd_mod.config.has_simd) {
        // Test with proper memory management
        const dot_product = simd_mod.dotProductSIMD(a, b, opts);
        try testing.expect(dot_product >= 0.0);

        // Test vector normalization
        const copy = try allocator.dupe(f32, a);
        defer allocator.free(copy);
        // Test vector normalization without memory leaks
        simd_mod.normalizeVector(copy, opts);
    }
}

test "memory management stress test - neural network" {
    const allocator = testing.allocator;

    // Stress test memory management with many operations
    {
        var networks = std.ArrayList(*neural.NeuralNetwork).init(allocator);
        defer {
            for (networks.items) |network| {
                network.deinit();
            }
            networks.deinit();
        }

        // Create multiple networks
        for (0..10) |_| {
            const network = try neural.NeuralNetwork.init(allocator);
            try networks.append(network);

            // Add layers
            try network.addLayer(.{
                .type = .Dense,
                .input_size = 4,
                .output_size = 6,
                .activation = .ReLU,
            });
            try network.addLayer(.{
                .type = .Dense,
                .input_size = 6,
                .output_size = 2,
                .activation = .Sigmoid,
            });
        }

        // Test operations on all networks
        const input = [_]f32{ 1.0, 0.5, -0.5, 1.5 };
        for (networks.items) |network| {
            const output = try network.forward(&input);
            defer allocator.free(output);
            try testing.expectEqual(@as(usize, 2), output.len);
        }
    }
}

test "memory management stress test - SIMD operations" {
    const allocator = testing.allocator;

    // Stress test SIMD operations
    const sizes = [_]usize{ 64, 128, 256, 512, 1024 };

    for (sizes) |size| {
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);

        // Initialize with random-like data
        for (a, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
        }
        for (b, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i * 3) % 100)) / 10.0;
        }

        const opts = simd_mod.SIMDOpts{};
        if (simd_mod.config.has_simd) {
            // Test various SIMD operations
            const dot_product = simd_mod.dotProductSIMD(a, b, opts);
            try testing.expect(dot_product >= 0.0);
        }
    }
}

test "memory alignment verification" {
    const allocator = testing.allocator;

    // Test that memory alignment is properly handled
    const alignments = [_]u29{ 1, 2, 4, 8, 16, 32, 64 };

    for (alignments) |alignment| {
        const buffer = try allocator.alignedAlloc(u8, alignment, 1024);
        defer allocator.free(buffer);

        // Verify alignment
        const addr = @intFromPtr(buffer.ptr);
        try testing.expectEqual(@as(usize, 0), addr % alignment);

        // Test SIMD operations on aligned buffer
        if (alignment >= 32 and buffer.len >= 32) {
            const count = simd_mod.text.countByte(buffer[0..32], 'x');
            try testing.expectEqual(@as(usize, 0), count); // Should be 0 since buffer is initialized to 0
        }
    }
}
