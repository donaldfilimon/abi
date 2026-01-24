//! Built-in GPU Kernel Definitions
//!
//! Pre-defined kernel IR using the DSL for common GPU operations.
//! These kernels are compiled to backend-specific code on demand.
//!
//! ## Supported Operations
//!
//! - **Element-wise**: vector_add, vector_sub, vector_mul, vector_div, vector_scale
//! - **Reductions**: reduce_sum, reduce_max, reduce_min, reduce_product
//! - **Matrix**: matrix_multiply, matrix_transpose
//! - **Neural Network**: softmax, relu, sigmoid, tanh, gelu, silu, swiglu
//! - **Normalization**: layer_norm, rms_norm, batch_norm
//! - **Linear Algebra**: dot_product, normalize, saxpy
//! - **Vision (CNN)**: conv2d, max_pool2d, avg_pool2d, batch_norm2d, im2col, col2im
//!
//! ## Usage
//!
//! ```zig
//! const kernel = @import("builtin_kernels.zig");
//!
//! var ir = try kernel.buildKernelIR(allocator, .vector_add);
//! // Use ir with dsl.compile() to generate backend code
//! ```
//!
//! ## Module Organization
//!
//! This file re-exports from the `kernels/` subdirectory for backward compatibility.
//! The implementation is split into logical modules:
//! - `kernels/elementwise.zig` - Element-wise operations
//! - `kernels/matrix.zig` - Matrix operations
//! - `kernels/reduction.zig` - Reduction operations
//! - `kernels/activation.zig` - Neural network activations
//! - `kernels/normalization.zig` - Normalization layers
//! - `kernels/linalg.zig` - Linear algebra operations
//! - `kernels/batch.zig` - Batch operations
//! - `kernels/vision.zig` - Vision/CNN operations

const std = @import("std");
const dsl = @import("dsl/mod.zig");

// Import the modular implementation
const kernels = @import("kernels/mod.zig");

// Re-export types for backward compatibility
pub const KernelIR = kernels.KernelIR;
pub const KernelBuilder = kernels.KernelBuilder;
pub const Type = kernels.Type;
pub const AccessMode = kernels.AccessMode;
pub const BuiltinKernel = kernels.BuiltinKernel;

// Re-export sub-modules
pub const elementwise = kernels.elementwise;
pub const matrix = kernels.matrix;
pub const reduction = kernels.reduction;
pub const activation = kernels.activation;
pub const normalization = kernels.normalization;
pub const linalg = kernels.linalg;
pub const batch = kernels.batch;
pub const vision = kernels.vision;

/// Build kernel IR for a given builtin kernel type.
pub const buildKernelIR = kernels.buildKernelIR;

// ============================================================================
// Re-export individual kernel builders for backward compatibility
// ============================================================================

// Element-wise operations
pub const buildVectorAddKernel = kernels.elementwise.buildVectorAddKernel;
pub const buildVectorSubKernel = kernels.elementwise.buildVectorSubKernel;
pub const buildVectorMulKernel = kernels.elementwise.buildVectorMulKernel;
pub const buildVectorDivKernel = kernels.elementwise.buildVectorDivKernel;
pub const buildVectorScaleKernel = kernels.elementwise.buildVectorScaleKernel;

// Matrix operations
pub const buildMatrixMultiplyKernel = kernels.matrix.buildMatrixMultiplyKernel;
pub const buildMatrixTransposeKernel = kernels.matrix.buildMatrixTransposeKernel;

// Reduction operations
pub const buildReduceSumKernel = kernels.reduction.buildReduceSumKernel;
pub const buildReduceMaxKernel = kernels.reduction.buildReduceMaxKernel;
pub const buildReduceMinKernel = kernels.reduction.buildReduceMinKernel;
pub const buildReduceProductKernel = kernels.reduction.buildReduceProductKernel;

// Basic activations
pub const buildSoftmaxKernel = kernels.activation.buildSoftmaxKernel;
pub const buildReluKernel = kernels.activation.buildReluKernel;
pub const buildSigmoidKernel = kernels.activation.buildSigmoidKernel;
pub const buildTanhKernel = kernels.activation.buildTanhKernel;

// Advanced activations
pub const buildGeluKernel = kernels.activation.buildGeluKernel;
pub const buildGeluFastKernel = kernels.activation.buildGeluFastKernel;
pub const buildSiluKernel = kernels.activation.buildSiluKernel;
pub const buildSwigluKernel = kernels.activation.buildSwigluKernel;

// Linear algebra
pub const buildDotProductKernel = kernels.linalg.buildDotProductKernel;
pub const buildNormalizeKernel = kernels.linalg.buildNormalizeKernel;
pub const buildSaxpyKernel = kernels.linalg.buildSaxpyKernel;
pub const buildCopyKernel = kernels.linalg.buildCopyKernel;
pub const buildFillKernel = kernels.linalg.buildFillKernel;

// Normalization layers
pub const buildLayerNormKernel = kernels.normalization.buildLayerNormKernel;
pub const buildRmsNormKernel = kernels.normalization.buildRmsNormKernel;
pub const buildBatchNormKernel = kernels.normalization.buildBatchNormKernel;

// Fused operations
pub const buildFusedAddNormKernel = kernels.normalization.buildFusedAddNormKernel;
pub const buildFusedLinearGeluKernel = kernels.normalization.buildFusedLinearGeluKernel;

// Batch operations
pub const buildBatchMatmulKernel = kernels.batch.buildBatchMatmulKernel;
pub const buildBatchCosineSimilarityKernel = kernels.batch.buildBatchCosineSimilarityKernel;

// Vision operations
pub const buildConv2dKernel = kernels.vision.buildConv2dKernel;
pub const buildMaxPool2dKernel = kernels.vision.buildMaxPool2dKernel;
pub const buildAvgPool2dKernel = kernels.vision.buildAvgPool2dKernel;
pub const buildBatchNorm2dKernel = kernels.vision.buildBatchNorm2dKernel;
pub const buildIm2colKernel = kernels.vision.buildIm2colKernel;
pub const buildCol2imKernel = kernels.vision.buildCol2imKernel;

// ============================================================================
// Tests
// ============================================================================

test "buildVectorAddKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildVectorAddKernel(allocator);
    try std.testing.expectEqualStrings("vector_add", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len);
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildKernelIR for all types" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test a few kernel types to ensure they build
    const kernel_types = [_]BuiltinKernel{
        .vector_add,
        .vector_sub,
        .reduce_sum,
        .softmax,
        .relu,
        .dot_product,
        .copy,
        // NN kernels
        .gelu,
        .gelu_fast,
        .silu,
        .swiglu,
        .layer_norm,
        .rms_norm,
        .batch_matmul,
        .batch_cosine_similarity,
        // Vision kernels
        .conv2d,
        .max_pool2d,
        .avg_pool2d,
        .batch_norm2d,
        .im2col,
        .col2im,
    };

    for (kernel_types) |kernel_type| {
        const ir = try buildKernelIR(allocator, kernel_type);
        try std.testing.expect(ir.name.len > 0);
        try std.testing.expect(ir.buffers.len > 0);
    }
}

test "buildGeluKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildGeluKernel(allocator);
    try std.testing.expectEqualStrings("gelu", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len); // n
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildLayerNormKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildLayerNormKernel(allocator);
    try std.testing.expectEqualStrings("layer_norm", ir.name);
    try std.testing.expectEqual(@as(usize, 4), ir.buffers.len); // input, gamma, beta, output
    try std.testing.expectEqual(@as(usize, 4), ir.uniforms.len); // mean, variance, epsilon, n
}

test "buildBatchCosineSimilarityKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildBatchCosineSimilarityKernel(allocator);
    try std.testing.expectEqualStrings("batch_cosine_similarity", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len); // query, vectors, output
    try std.testing.expectEqual(@as(usize, 3), ir.uniforms.len); // query_norm, num_vectors, dim
}

test "buildMatrixMultiplyKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildMatrixMultiplyKernel(allocator);
    try std.testing.expectEqualStrings("matrix_multiply", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 3), ir.uniforms.len); // m, n, k
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildConv2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildConv2dKernel(allocator);
    try std.testing.expectEqualStrings("conv2d", ir.name);
    try std.testing.expectEqual(@as(usize, 4), ir.buffers.len); // input, weights, bias, output
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildMaxPool2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildMaxPool2dKernel(allocator);
    try std.testing.expectEqualStrings("max_pool2d", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len); // input, output, indices
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildAvgPool2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildAvgPool2dKernel(allocator);
    try std.testing.expectEqualStrings("avg_pool2d", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildBatchNorm2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildBatchNorm2dKernel(allocator);
    try std.testing.expectEqualStrings("batch_norm2d", ir.name);
    try std.testing.expectEqual(@as(usize, 6), ir.buffers.len); // input, gamma, beta, mean, var, output
    try std.testing.expectEqual(@as(usize, 5), ir.uniforms.len); // batch, channels, height, width, epsilon
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildIm2colKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildIm2colKernel(allocator);
    try std.testing.expectEqualStrings("im2col", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildCol2imKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildCol2imKernel(allocator);
    try std.testing.expectEqualStrings("col2im", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // col_input, output
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildKernelIR for vision kernels" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test all vision kernel types
    const vision_kernels = [_]BuiltinKernel{
        .conv2d,
        .max_pool2d,
        .avg_pool2d,
        .batch_norm2d,
        .im2col,
        .col2im,
    };

    for (vision_kernels) |kernel_type| {
        const ir = try buildKernelIR(allocator, kernel_type);
        try std.testing.expect(ir.name.len > 0);
        try std.testing.expect(ir.buffers.len > 0);
    }
}
