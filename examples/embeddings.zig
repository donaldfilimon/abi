//! Embeddings Example
//!
//! Demonstrates SIMD-accelerated vector operations for embeddings,
//! including dot products, L2 norms, and cosine similarity.

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("ABI Embeddings Example - SIMD Vector Operations\n", .{});
    std.debug.print("================================================\n\n", .{});

    // Create sample embedding vectors (e.g., 128-dimensional embeddings)
    const dim = 128;
    var vec_a = try allocator.alloc(f32, dim);
    defer allocator.free(vec_a);
    var vec_b = try allocator.alloc(f32, dim);
    defer allocator.free(vec_b);
    const result = try allocator.alloc(f32, dim);
    defer allocator.free(result);

    // Initialize vectors with sample data
    for (0..dim) |i| {
        vec_a[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim));
        vec_b[i] = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim));
    }

    // Vector addition using SIMD
    abi.shared.simd.vectorAdd(vec_a, vec_b, result);
    std.debug.print("Vector Addition (first 5 elements):\n", .{});
    std.debug.print("  a[0..5] = [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        vec_a[0], vec_a[1], vec_a[2], vec_a[3], vec_a[4],
    });
    std.debug.print("  b[0..5] = [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        vec_b[0], vec_b[1], vec_b[2], vec_b[3], vec_b[4],
    });
    std.debug.print("  sum[0..5] = [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}]\n\n", .{
        result[0], result[1], result[2], result[3], result[4],
    });

    // Dot product using SIMD
    const dot = abi.shared.simd.vectorDot(vec_a, vec_b);
    std.debug.print("Dot Product: {d:.6}\n", .{dot});

    // L2 norms using SIMD
    const norm_a = abi.shared.simd.vectorL2Norm(vec_a);
    const norm_b = abi.shared.simd.vectorL2Norm(vec_b);
    std.debug.print("L2 Norm of vec_a: {d:.6}\n", .{norm_a});
    std.debug.print("L2 Norm of vec_b: {d:.6}\n\n", .{norm_b});

    // Cosine similarity using SIMD
    const similarity = abi.shared.simd.cosineSimilarity(vec_a, vec_b);
    std.debug.print("Cosine Similarity: {d:.6}\n", .{similarity});
    std.debug.print("  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)\n\n", .{});

    // Batch cosine similarity for multiple queries
    std.debug.print("Batch Similarity Demo:\n", .{});
    const query = vec_a;

    // Create a small "database" of vectors
    const num_vectors = 4;
    var vectors = try allocator.alloc([]const f32, num_vectors);
    defer allocator.free(vectors);

    var vector_storage = try allocator.alloc(f32, dim * num_vectors);
    defer allocator.free(vector_storage);

    for (0..num_vectors) |i| {
        const offset = i * dim;
        vectors[i] = vector_storage[offset .. offset + dim];
        // Initialize each vector differently
        for (0..dim) |j| {
            const factor = @as(f32, @floatFromInt(i + 1)) * 0.25;
            vector_storage[offset + j] = factor * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(dim));
        }
    }

    // Compute similarities
    for (0..num_vectors) |i| {
        const sim = abi.shared.simd.cosineSimilarity(query, vectors[i]);
        std.debug.print("  Similarity with vector {d}: {d:.6}\n", .{ i, sim });
    }

    std.debug.print("\nEmbeddings example completed successfully!\n", .{});
}
