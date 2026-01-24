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
//! const kernels = @import("gpu/kernels/mod.zig");
//!
//! var ir = try kernels.buildKernelIR(allocator, .vector_add);
//! // Use ir with dsl.compile() to generate backend code
//! ```

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

// Re-export sub-modules
pub const elementwise = @import("elementwise.zig");
pub const matrix = @import("matrix.zig");
pub const reduction = @import("reduction.zig");
pub const activation = @import("activation.zig");
pub const normalization = @import("normalization.zig");
pub const linalg = @import("linalg.zig");
pub const batch = @import("batch.zig");
pub const vision = @import("vision.zig");

// Re-export common types
pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;
pub const BuiltinKernel = dsl.BuiltinKernel;

/// Build kernel IR for a given builtin kernel type.
pub fn buildKernelIR(allocator: std.mem.Allocator, kernel_type: BuiltinKernel) !*const KernelIR {
    return switch (kernel_type) {
        // Element-wise operations
        .vector_add => elementwise.buildVectorAddKernel(allocator),
        .vector_sub => elementwise.buildVectorSubKernel(allocator),
        .vector_mul => elementwise.buildVectorMulKernel(allocator),
        .vector_div => elementwise.buildVectorDivKernel(allocator),
        .vector_scale => elementwise.buildVectorScaleKernel(allocator),
        // Matrix operations
        .matrix_multiply => matrix.buildMatrixMultiplyKernel(allocator),
        .matrix_transpose => matrix.buildMatrixTransposeKernel(allocator),
        // Reductions
        .reduce_sum => reduction.buildReduceSumKernel(allocator),
        .reduce_max => reduction.buildReduceMaxKernel(allocator),
        .reduce_min => reduction.buildReduceMinKernel(allocator),
        .reduce_product => reduction.buildReduceProductKernel(allocator),
        // Basic activations
        .softmax => activation.buildSoftmaxKernel(allocator),
        .relu => activation.buildReluKernel(allocator),
        .sigmoid => activation.buildSigmoidKernel(allocator),
        .tanh => activation.buildTanhKernel(allocator),
        // Linear algebra
        .dot_product => linalg.buildDotProductKernel(allocator),
        .normalize => linalg.buildNormalizeKernel(allocator),
        .saxpy => linalg.buildSaxpyKernel(allocator),
        .copy => linalg.buildCopyKernel(allocator),
        .fill => linalg.buildFillKernel(allocator),
        // Neural network activations
        .gelu => activation.buildGeluKernel(allocator),
        .gelu_fast => activation.buildGeluFastKernel(allocator),
        .silu => activation.buildSiluKernel(allocator),
        .swiglu => activation.buildSwigluKernel(allocator),
        // Normalization layers
        .layer_norm => normalization.buildLayerNormKernel(allocator),
        .rms_norm => normalization.buildRmsNormKernel(allocator),
        .batch_norm => normalization.buildBatchNormKernel(allocator),
        // Fused operations
        .fused_add_norm => normalization.buildFusedAddNormKernel(allocator),
        .fused_linear_gelu => normalization.buildFusedLinearGeluKernel(allocator),
        // Batch operations
        .batch_matmul => batch.buildBatchMatmulKernel(allocator),
        .batch_cosine_similarity => batch.buildBatchCosineSimilarityKernel(allocator),
        // Vision operations
        .conv2d => vision.buildConv2dKernel(allocator),
        .max_pool2d => vision.buildMaxPool2dKernel(allocator),
        .avg_pool2d => vision.buildAvgPool2dKernel(allocator),
        .batch_norm2d => vision.buildBatchNorm2dKernel(allocator),
        .im2col => vision.buildIm2colKernel(allocator),
        .col2im => vision.buildCol2imKernel(allocator),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "buildKernelIR for all types" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test a few kernel types to ensure they build
    const kernels = [_]BuiltinKernel{
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

    for (kernels) |kernel_type| {
        const ir = try buildKernelIR(allocator, kernel_type);
        try std.testing.expect(ir.name.len > 0);
        try std.testing.expect(ir.buffers.len > 0);
    }
}
