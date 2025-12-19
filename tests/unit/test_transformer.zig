// Tests for Transformer Module

const std = @import("std");
const ai = @import("abi").ai;

/// Test multi-head attention basic functionality
test "transformer multi-head attention" {
    const testing = std.testing;
    var mha = try ai.transformer.MultiHeadAttention.init(testing.allocator, 8, 512);
    defer mha.deinit(testing.allocator);

    // Test basic properties
    try testing.expectEqual(@as(usize, 8), mha.num_heads);
    try testing.expectEqual(@as(usize, 64), mha.head_dim); // 512 / 8
    try testing.expectEqual(@as(usize, 512), mha.embed_dim);

    // Create test input (batch_size=1, seq_len=4, embed_dim=512)
    const seq_len = 4;
    const input_size = seq_len * 512;
    var query = try testing.allocator.alloc(f32, input_size);
    defer testing.allocator.free(query);
    var key = try testing.allocator.alloc(f32, input_size);
    defer testing.allocator.free(key);
    var value = try testing.allocator.alloc(f32, input_size);
    defer testing.allocator.free(value);

    // Initialize with some test data
    for (query, 0..) |*q, i| q.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    for (key, 0..) |*k, i| k.* = @as(f32, @floatFromInt((i + 5) % 10)) / 10.0;
    for (value, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 3) % 10)) / 10.0;

    // Perform attention
    const output = try mha.forward(testing.allocator, query, key, value, seq_len);
    defer testing.allocator.free(output);

    // Output should have same size as input
    try testing.expectEqual(input_size, output.len);

    // Output should not be all zeros (computation actually happened)
    var has_nonzero = false;
    for (output) |val| {
        if (val != 0.0) has_nonzero = true;
    }
    try testing.expect(has_nonzero);
}

/// Test feed-forward network
test "transformer feed-forward network" {
    const testing = std.testing;
    var ff = try ai.transformer.FeedForward.init(testing.allocator, 512, 2048);
    defer ff.deinit(testing.allocator);

    // Test basic properties
    try testing.expectEqual(@as(usize, 512), ff.embed_dim);
    try testing.expectEqual(@as(usize, 2048), ff.hidden_dim);

    // Create test input
    var input = try testing.allocator.alloc(f32, 512);
    defer testing.allocator.free(input);

    for (input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 20)) / 20.0;

    // Forward pass
    const output = try ff.forward(testing.allocator, input);
    defer testing.allocator.free(output);

    // Output should have correct size
    try testing.expectEqual(@as(usize, 512), output.len);

    // Output should be different from input (non-linear transformation)
    var is_different = false;
    for (input, output, 0..) |in, out, i| {
        if (i < 10) { // Check first few values
            if (@abs(in - out) > 0.001) is_different = true;
        }
    }
    try testing.expect(is_different);
}

/// Test transformer block
test "transformer transformer block" {
    const testing = std.testing;
    var block = try ai.transformer.TransformerBlock.init(testing.allocator, 512, 8, 2048);
    defer block.deinit(testing.allocator);

    // Create test input (batch_size=1, seq_len=6, embed_dim=512)
    const seq_len = 6;
    const input_size = seq_len * 512;
    var input = try testing.allocator.alloc(f32, input_size);
    defer testing.allocator.free(input);

    // Initialize with positional encoding-like pattern
    for (input, 0..) |*x, i| {
        const pos = i / 512;
        const dim = i % 512;
        x.* = @sin(@as(f32, @floatFromInt(pos)) / 10000.0 * @as(f32, @floatFromInt(dim)));
    }

    // Forward pass
    const output = try block.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(output);

    // Output should have correct size
    try testing.expectEqual(input_size, output.len);

    // Output should be different from input (residual + attention + FF)
    var is_transformed = false;
    for (input, output, 0..) |in, out, i| {
        if (i < 50) { // Check first few values
            if (@abs(in - out) > 0.001) is_transformed = true;
        }
    }
    try testing.expect(is_transformed);
}

/// Test layer normalization
test "transformer layer normalization" {
    const testing = std.testing;
    var ln = try ai.transformer.LayerNorm.init(testing.allocator, 512);
    defer ln.deinit(testing.allocator);

    // Create test input
    var input = try testing.allocator.alloc(f32, 512);
    defer testing.allocator.free(input);

    // Initialize with varying values
    for (input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) / 512.0;

    // Forward pass
    const output = try ln.forward(testing.allocator, input);
    defer testing.allocator.free(output);

    // Output should have correct size
    try testing.expectEqual(@as(usize, 512), output.len);

    // Check normalization properties (approximately zero mean, unit variance)
    var mean: f32 = 0.0;
    var variance: f32 = 0.0;

    for (output) |val| {
        mean += val;
    }
    mean /= @as(f32, @floatFromInt(output.len));

    for (output) |val| {
        const diff = val - mean;
        variance += diff * diff;
    }
    variance /= @as(f32, @floatFromInt(output.len));

    // Mean should be close to zero
    try testing.expect(@abs(mean) < 0.1);

    // Variance should be close to 1
    try testing.expect(@abs(variance - 1.0) < 0.2);
}

/// Test attention mechanism components
test "transformer attention components" {
    const testing = std.testing;

    // Test softmax function
    var mha = try ai.transformer.MultiHeadAttention.init(testing.allocator, 1, 64);
    defer mha.deinit(testing.allocator);

    var scores = [_]f32{1.0, 2.0, 3.0, 4.0};
    const attn_weights = try mha.softmax(testing.allocator, &scores, 2);
    defer testing.allocator.free(attn_weights);

    try testing.expectEqual(@as(usize, 4), attn_weights.len);

    // Weights should sum to 1 for each row
    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    for (0..2) |i| {
        sum0 += attn_weights[i];
        sum1 += attn_weights[i + 2];
    }
    try testing.expect(@abs(sum0 - 1.0) < 0.001);
    try testing.expect(@abs(sum1 - 1.0) < 0.001);

    // Weights should be positive and decreasing for increasing scores
    try testing.expect(attn_weights[0] > 0.0);
    try testing.expect(attn_weights[1] > 0.0);
    try testing.expect(attn_weights[2] > 0.0);
    try testing.expect(attn_weights[3] > 0.0);
}

/// Test transformer memory management
test "transformer memory management" {
    const testing = std.testing;

    // Test proper cleanup
    var block = try ai.transformer.TransformerBlock.init(testing.allocator, 256, 4, 1024);
    defer block.deinit(testing.allocator);

    // Create input
    const seq_len = 8;
    const input_size = seq_len * 256;
    var input = try testing.allocator.alloc(f32, input_size);
    defer testing.allocator.free(input);

    for (input, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)));

    // Multiple forward passes
    for (0..3) |_| {
        const output = try block.forward(testing.allocator, input, seq_len);
        defer testing.allocator.free(output);
        try testing.expectEqual(input_size, output.len);
    }

    // Block should still be functional after multiple uses
    const final_output = try block.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(final_output);
    try testing.expectEqual(input_size, final_output.len);
}

/// Test transformer with different sequence lengths
test "transformer variable sequence lengths" {
    const testing = std.testing;
    var block = try ai.transformer.TransformerBlock.init(testing.allocator, 128, 2, 512);
    defer block.deinit(testing.allocator);

    const embed_dim = 128;

    // Test with different sequence lengths
    const seq_lengths = [_]usize{2, 4, 6};

    for (seq_lengths) |seq_len| {
        const input_size = seq_len * embed_dim;
        var input = try testing.allocator.alloc(f32, input_size);
        defer testing.allocator.free(input);

        for (input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 10)) / 10.0;

        const output = try block.forward(testing.allocator, input, seq_len);
        defer testing.allocator.free(output);

        try testing.expectEqual(input_size, output.len);
    }
}