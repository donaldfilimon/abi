//! Unit tests for the AI component.

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const ai = abi.ai;

test "Neural network dense forward" {
    const allocator = testing.allocator;
    var net = try ai.NeuralNetwork.init(allocator, &[_]usize{4}, &[_]usize{2});
    defer net.deinit();

    try net.addDenseLayer(8, .relu);
    try net.addDenseLayer(2, .softmax);
    try net.compile();

    const input = [_]f32{ 1.0, 0.5, -0.25, 2.0 };
    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);
    try net.forward(&input, output);

    try testing.expectEqual(@as(usize, 2), output.len);
    try testing.expect(std.math.isFinite(output[0]) and std.math.isFinite(output[1]));
}

// Activation helpers are covered via network forward with activations

test "Embedding generator end-to-end" {
    const allocator = testing.allocator;
    var gen = try ai.EmbeddingGenerator.init(allocator, 8, 4);
    defer gen.deinit();

    const input = [_]f32{1.0} ** 8;
    const embedding = try allocator.alloc(f32, 4);
    defer allocator.free(embedding);

    try gen.generateEmbedding(&input, embedding);
    try testing.expectEqual(@as(usize, 4), embedding.len);
}
