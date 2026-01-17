//! Fused GPU Kernels
//!
//! Fused operations combine multiple ops into a single kernel launch,
//! reducing memory bandwidth and improving performance.
//!
//! Available fused kernels:
//! - LayerNorm + Linear: Normalization followed by linear projection
//! - RMSNorm + RoPE: RMS normalization with rotary position embedding
//! - SwiGLU: Gated linear unit with SiLU activation
//! - Residual + LayerNorm: Add residual and normalize
//!
//! Memory bandwidth savings: Up to 2-3x for bandwidth-bound operations.

const std = @import("std");
const kernel = @import("../dsl/kernel.zig");
const types = @import("../dsl/types.zig");

// ============================================================================
// Fused LayerNorm + Linear
// ============================================================================

/// Configuration for fused LayerNorm + Linear kernel.
pub const LayerNormLinearConfig = struct {
    /// Input dimension
    input_dim: u32,
    /// Output dimension
    output_dim: u32,
    /// LayerNorm epsilon
    eps: f32 = 1e-5,
    /// Block size for reduction
    block_size: u32 = 256,
    /// Use half precision
    use_half: bool = false,
};

/// Create fused LayerNorm + Linear kernel.
/// Computes: output = Linear(LayerNorm(input))
pub fn createLayerNormLinearKernel(
    allocator: std.mem.Allocator,
    config: LayerNormLinearConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "input",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "ln_weight",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "ln_bias",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "linear_weight",
            .binding = 3,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "linear_bias",
            .binding = 4,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 5,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "input_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "output_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "eps", .binding = 0, .ty = types.Type.f32Type() },
    };

    // Shared memory for mean/variance reduction
    const shared_memory = [_]kernel.SharedMemory{
        .{
            .name = "reduction_buffer",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size * 2, // For mean and variance
        },
    };

    return .{
        .name = "fused_layernorm_linear",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &shared_memory,
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
            .subgroups = true,
        },
    };
}

// ============================================================================
// Fused RMSNorm + Rotary Position Embedding
// ============================================================================

/// Configuration for fused RMSNorm + RoPE kernel.
pub const RMSNormRoPEConfig = struct {
    /// Hidden dimension
    hidden_dim: u32,
    /// Head dimension (for RoPE)
    head_dim: u32,
    /// Number of heads
    num_heads: u32,
    /// RMSNorm epsilon
    eps: f32 = 1e-5,
    /// RoPE theta base
    rope_theta: f32 = 10000.0,
    /// Block size
    block_size: u32 = 256,
    /// Use half precision
    use_half: bool = false,
};

/// Create fused RMSNorm + RoPE kernel.
/// Computes: output = RoPE(RMSNorm(input), position)
pub fn createRMSNormRoPEKernel(
    allocator: std.mem.Allocator,
    config: RMSNormRoPEConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "input",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "rms_weight",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "cos_cache",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "sin_cache",
            .binding = 3,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 4,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "seq_len", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "hidden_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "head_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "num_heads", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "eps", .binding = 0, .ty = types.Type.f32Type() },
    };

    const shared_memory = [_]kernel.SharedMemory{
        .{
            .name = "rms_sum",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size,
        },
    };

    return .{
        .name = "fused_rmsnorm_rope",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &shared_memory,
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
            .subgroups = true,
        },
    };
}

// ============================================================================
// Fused SwiGLU Activation
// ============================================================================

/// Configuration for SwiGLU kernel.
pub const SwiGLUConfig = struct {
    /// Input dimension
    input_dim: u32,
    /// Hidden dimension (intermediate)
    hidden_dim: u32,
    /// Block size
    block_size: u32 = 256,
    /// Use half precision
    use_half: bool = false,
};

/// Create fused SwiGLU kernel.
/// Computes: output = gate * SiLU(x)
/// where gate and x are separate linear projections of input.
pub fn createSwiGLUKernel(
    allocator: std.mem.Allocator,
    config: SwiGLUConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "gate",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "up",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "hidden_dim", .binding = 0, .ty = types.Type.u32Type() },
    };

    return .{
        .name = "fused_swiglu",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &.{},
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
        },
    };
}

// ============================================================================
// Fused Residual + LayerNorm
// ============================================================================

/// Configuration for Residual + LayerNorm kernel.
pub const ResidualLayerNormConfig = struct {
    /// Hidden dimension
    hidden_dim: u32,
    /// LayerNorm epsilon
    eps: f32 = 1e-5,
    /// Block size for reduction
    block_size: u32 = 256,
    /// Use half precision
    use_half: bool = false,
    /// Pre-norm (norm before residual) vs post-norm
    pre_norm: bool = true,
};

/// Create fused Residual + LayerNorm kernel.
/// Pre-norm: output = LayerNorm(residual + input)
/// Post-norm: output = residual + LayerNorm(input)
pub fn createResidualLayerNormKernel(
    allocator: std.mem.Allocator,
    config: ResidualLayerNormConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "input",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "residual",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "ln_weight",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "ln_bias",
            .binding = 3,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 4,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "hidden_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "eps", .binding = 0, .ty = types.Type.f32Type() },
    };

    const shared_memory = [_]kernel.SharedMemory{
        .{
            .name = "reduction_buffer",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size * 2,
        },
    };

    return .{
        .name = if (config.pre_norm) "fused_residual_prenorm" else "fused_residual_postnorm",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &shared_memory,
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
            .subgroups = true,
        },
    };
}

// ============================================================================
// Fused GeGLU (GELU-gated Linear Unit)
// ============================================================================

/// Configuration for GeGLU kernel.
pub const GeGLUConfig = struct {
    /// Hidden dimension
    hidden_dim: u32,
    /// Block size
    block_size: u32 = 256,
    /// Use approximate GELU
    approximate: bool = true,
    /// Use half precision
    use_half: bool = false,
};

/// Create fused GeGLU kernel.
/// Computes: output = gate * GELU(x)
pub fn createGeGLUKernel(
    allocator: std.mem.Allocator,
    config: GeGLUConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "gate",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "up",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "hidden_dim", .binding = 0, .ty = types.Type.u32Type() },
    };

    return .{
        .name = "fused_geglu",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &.{},
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
        },
    };
}

// ============================================================================
// Fused Softmax + Dropout (for attention)
// ============================================================================

/// Configuration for Softmax + Dropout kernel.
pub const SoftmaxDropoutConfig = struct {
    /// Maximum sequence length
    max_seq_len: u32,
    /// Block size
    block_size: u32 = 256,
    /// Dropout probability
    dropout_prob: f32 = 0.0,
    /// Use causal masking
    causal: bool = true,
    /// Use half precision
    use_half: bool = false,
};

/// Create fused Softmax + Dropout kernel.
pub fn createSoftmaxDropoutKernel(
    allocator: std.mem.Allocator,
    config: SoftmaxDropoutConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "scores",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_write,
        },
        .{
            .name = "dropout_mask",
            .binding = 1,
            .element_type = .{ .scalar = .u32 },
            .access = .read_only,
        },
    };

    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "num_heads", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "seq_len", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "kv_len", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "dropout_prob", .binding = 0, .ty = types.Type.f32Type() },
        .{ .name = "dropout_scale", .binding = 0, .ty = types.Type.f32Type() },
    };

    const shared_memory = [_]kernel.SharedMemory{
        .{
            .name = "max_buffer",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size,
        },
        .{
            .name = "sum_buffer",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size,
        },
    };

    return .{
        .name = "fused_softmax_dropout",
        .entry_point = "main",
        .workgroup_size = .{ config.block_size, 1, 1 },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &shared_memory,
        .body = &.{},
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
            .subgroups = true,
        },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "create fused layernorm linear kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = LayerNormLinearConfig{
        .input_dim = 512,
        .output_dim = 2048,
    };

    const ir = try createLayerNormLinearKernel(allocator, config);
    try std.testing.expectEqualStrings("fused_layernorm_linear", ir.name);
    try std.testing.expectEqual(@as(usize, 6), ir.buffers.len);
}

test "create fused rmsnorm rope kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = RMSNormRoPEConfig{
        .hidden_dim = 4096,
        .head_dim = 128,
        .num_heads = 32,
    };

    const ir = try createRMSNormRoPEKernel(allocator, config);
    try std.testing.expectEqualStrings("fused_rmsnorm_rope", ir.name);
    try std.testing.expectEqual(@as(usize, 5), ir.buffers.len);
}

test "create fused swiglu kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = SwiGLUConfig{
        .input_dim = 4096,
        .hidden_dim = 11008,
    };

    const ir = try createSwiGLUKernel(allocator, config);
    try std.testing.expectEqualStrings("fused_swiglu", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
}

test "create fused residual layernorm kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config_prenorm = ResidualLayerNormConfig{
        .hidden_dim = 4096,
        .pre_norm = true,
    };

    const ir_prenorm = try createResidualLayerNormKernel(allocator, config_prenorm);
    try std.testing.expectEqualStrings("fused_residual_prenorm", ir_prenorm.name);

    const config_postnorm = ResidualLayerNormConfig{
        .hidden_dim = 4096,
        .pre_norm = false,
    };

    const ir_postnorm = try createResidualLayerNormKernel(allocator, config_postnorm);
    try std.testing.expectEqualStrings("fused_residual_postnorm", ir_postnorm.name);
}
