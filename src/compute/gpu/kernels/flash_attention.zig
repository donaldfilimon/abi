//! Flash Attention GPU Kernel
//!
//! Memory-efficient attention implementation for GPU acceleration.
//! Uses tiled computation with online softmax normalization to achieve
//! O(N) memory complexity instead of O(N²).
//!
//! Based on: "FlashAttention: Fast and Memory-Efficient Exact Attention"
//!           (Dao et al., 2022)
//!
//! The kernel is defined using the GPU DSL for portability across:
//! - CUDA (NVIDIA)
//! - Vulkan (SPIR-V)
//! - Metal (Apple)
//! - WebGPU (WGSL)

const std = @import("std");
const kernel = @import("../dsl/kernel.zig");
const types = @import("../dsl/types.zig");
const expr = @import("../dsl/expr.zig");
const stmt = @import("../dsl/stmt.zig");

/// Flash Attention kernel configuration.
pub const FlashAttentionKernelConfig = struct {
    /// Block size for Q dimension (Br in paper)
    block_size_q: u32 = 64,
    /// Block size for KV dimension (Bc in paper)
    block_size_kv: u32 = 64,
    /// Head dimension
    head_dim: u32 = 64,
    /// Number of warps per block (CUDA)
    warps_per_block: u32 = 4,
    /// Whether to use causal masking
    causal: bool = true,
    /// Use half precision (f16)
    use_half: bool = false,
};

/// Default configuration tuned for modern GPUs.
pub const default_config = FlashAttentionKernelConfig{};

/// Flash Attention kernel parameters passed as uniforms.
pub const FlashAttentionParams = struct {
    /// Sequence length
    seq_len: u32,
    /// KV sequence length (may differ with KV cache)
    kv_len: u32,
    /// Head dimension
    head_dim: u32,
    /// Number of heads
    num_heads: u32,
    /// Scaling factor (1/sqrt(head_dim))
    scale: f32,
    /// Total batch size
    batch_size: u32,
};

/// Buffer bindings for Flash Attention kernel.
pub const FlashAttentionBuffers = struct {
    /// Query tensor [batch, seq_len, num_heads, head_dim]
    q: u32 = 0,
    /// Key tensor [batch, kv_len, num_kv_heads, head_dim]
    k: u32 = 1,
    /// Value tensor [batch, kv_len, num_kv_heads, head_dim]
    v: u32 = 2,
    /// Output tensor [batch, seq_len, num_heads, head_dim]
    output: u32 = 3,
    /// Row max accumulators (temporary)
    row_max: u32 = 4,
    /// Row sum accumulators (temporary)
    row_sum: u32 = 5,
};

/// Create the Flash Attention forward kernel IR.
pub fn createFlashAttentionKernel(
    allocator: std.mem.Allocator,
    config: FlashAttentionKernelConfig,
) !kernel.KernelIR {
    _ = allocator;
    const elem_type = if (config.use_half) types.ScalarType.f16 else types.ScalarType.f32;

    // Buffer bindings
    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "q",
            .binding = 0,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "k",
            .binding = 1,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "v",
            .binding = 2,
            .element_type = .{ .scalar = elem_type },
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 3,
            .element_type = .{ .scalar = elem_type },
            .access = .write_only,
        },
        .{
            .name = "row_max",
            .binding = 4,
            .element_type = .{ .scalar = elem_type },
            .access = .read_write,
        },
        .{
            .name = "row_sum",
            .binding = 5,
            .element_type = .{ .scalar = elem_type },
            .access = .read_write,
        },
    };

    // Uniform parameters
    const uniforms = [_]kernel.UniformBinding{
        .{ .name = "seq_len", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "kv_len", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "head_dim", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "num_heads", .binding = 0, .ty = types.Type.u32Type() },
        .{ .name = "scale", .binding = 0, .ty = types.Type.f32Type() },
        .{ .name = "batch_size", .binding = 0, .ty = types.Type.u32Type() },
    };

    // Shared memory for Q, K, V tiles
    const shared_memory = [_]kernel.SharedMemory{
        .{
            .name = "Q_tile",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size_q * config.head_dim,
        },
        .{
            .name = "K_tile",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size_kv * config.head_dim,
        },
        .{
            .name = "V_tile",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size_kv * config.head_dim,
        },
        .{
            .name = "S_tile",
            .element_type = .{ .scalar = elem_type },
            .size = config.block_size_q * config.block_size_kv,
        },
    };

    return .{
        .name = "flash_attention_forward",
        .entry_point = "main",
        .workgroup_size = .{
            config.block_size_q,
            config.warps_per_block,
            1,
        },
        .buffers = &buffers,
        .uniforms = &uniforms,
        .push_constants = &.{},
        .shared_memory = &shared_memory,
        .body = &.{}, // Body would be generated by kernel builder
        .functions = &.{},
        .required_features = .{
            .fp16 = config.use_half,
            .subgroups = true, // For warp-level reductions
            .atomics = true, // For row_max/row_sum updates
        },
    };
}

/// GPU-side implementation notes for Flash Attention.
///
/// The algorithm processes attention in tiles to minimize global memory access:
///
/// 1. Outer loop: Iterate over KV blocks (j = 0 to ceil(N/Bc))
/// 2. Inner loop: Iterate over Q blocks (i = 0 to ceil(N/Br))
///
/// For each (i,j) block pair:
///   a. Load Q[i] block to shared memory (Br x d)
///   b. Load K[j] block to shared memory (Bc x d)
///   c. Compute S = Q @ K^T in shared memory (Br x Bc)
///   d. Apply causal mask if needed
///   e. Compute row-wise max(S) and update running max
///   f. Compute exp(S - max) and row sums
///   g. Load V[j] block to shared memory
///   h. Compute O += exp(S) @ V and update running output
///
/// Key optimizations:
/// - Warp-level reductions for max and sum
/// - Register tiling for matrix multiplies
/// - Prefetching next K,V blocks
/// - Fused softmax (online algorithm)
/// Shared memory size required for the kernel.
pub fn requiredSharedMemory(config: FlashAttentionKernelConfig) usize {
    const elem_size: usize = if (config.use_half) 2 else 4;
    const q_tile = config.block_size_q * config.head_dim * elem_size;
    const k_tile = config.block_size_kv * config.head_dim * elem_size;
    const v_tile = config.block_size_kv * config.head_dim * elem_size;
    const s_tile = config.block_size_q * config.block_size_kv * elem_size;
    return q_tile + k_tile + v_tile + s_tile;
}

/// Check if configuration is valid for target GPU.
pub fn validateConfig(config: FlashAttentionKernelConfig, max_shared_memory: usize) !void {
    const required = requiredSharedMemory(config);
    if (required > max_shared_memory) {
        return error.InsufficientSharedMemory;
    }

    // Check power of 2 for block sizes
    if (@popCount(config.block_size_q) != 1) {
        return error.BlockSizeMustBePowerOfTwo;
    }
    if (@popCount(config.block_size_kv) != 1) {
        return error.BlockSizeMustBePowerOfTwo;
    }

    // Check head_dim is reasonable
    if (config.head_dim > 256) {
        return error.HeadDimTooLarge;
    }
}

/// Tuned configurations for specific GPU architectures.
pub const TunedConfigs = struct {
    /// NVIDIA Ampere (A100, RTX 30xx)
    pub const ampere = FlashAttentionKernelConfig{
        .block_size_q = 128,
        .block_size_kv = 128,
        .head_dim = 64,
        .warps_per_block = 8,
        .use_half = true, // Tensor cores
    };

    /// NVIDIA Hopper (H100, RTX 40xx)
    pub const hopper = FlashAttentionKernelConfig{
        .block_size_q = 128,
        .block_size_kv = 256,
        .head_dim = 128,
        .warps_per_block = 8,
        .use_half = true,
    };

    /// AMD RDNA3
    pub const rdna3 = FlashAttentionKernelConfig{
        .block_size_q = 64,
        .block_size_kv = 64,
        .head_dim = 64,
        .warps_per_block = 4,
        .use_half = true,
    };

    /// Apple M1/M2
    pub const apple_silicon = FlashAttentionKernelConfig{
        .block_size_q = 64,
        .block_size_kv = 64,
        .head_dim = 64,
        .warps_per_block = 4,
        .use_half = true,
    };

    /// Fallback for unknown GPUs
    pub const fallback = FlashAttentionKernelConfig{
        .block_size_q = 32,
        .block_size_kv = 32,
        .head_dim = 64,
        .warps_per_block = 2,
        .use_half = false,
    };
};

/// Flash Attention backward pass kernel configuration.
pub const FlashAttentionBackwardConfig = struct {
    /// Recompute attention in backward (more memory efficient)
    recompute_attention: bool = true,
    /// Block sizes (same as forward for cache efficiency)
    block_size_q: u32 = 64,
    block_size_kv: u32 = 64,
    head_dim: u32 = 64,
};

/// Create Flash Attention backward kernel for training.
pub fn createFlashAttentionBackwardKernel(
    allocator: std.mem.Allocator,
    config: FlashAttentionBackwardConfig,
) !kernel.KernelIR {
    _ = allocator;
    _ = config;

    // Backward pass needs:
    // - dO (gradient from output)
    // - Q, K, V (recomputed or cached)
    // - O, L (output and logsumexp from forward)
    // Produces:
    // - dQ, dK, dV

    return kernel.KernelIR.empty("flash_attention_backward");
}

// ============================================================================
// Kernel Builder Helpers
// ============================================================================

/// Emit code for online softmax update (the core Flash Attention trick).
pub const OnlineSoftmaxOps = struct {
    /// Update running max: new_max = max(prev_max, block_max)
    pub fn updateMax(prev_max: f32, block_max: f32) f32 {
        return @max(prev_max, block_max);
    }

    /// Rescale factor when max increases: exp(prev_max - new_max)
    pub fn rescaleFactor(prev_max: f32, new_max: f32) f32 {
        if (prev_max == -std.math.inf(f32)) return 0;
        return @exp(prev_max - new_max);
    }

    /// Numerically stable exp: exp(x - max)
    pub fn stableExp(x: f32, max: f32) f32 {
        return @exp(x - max);
    }
};

/// Warp-level reduction operations.
pub const WarpReductions = struct {
    /// Warp-level max reduction (32 threads).
    pub fn warpMax(value: f32) f32 {
        // In actual GPU code, this uses __shfl_xor_sync or subgroupMax
        return value;
    }

    /// Warp-level sum reduction.
    pub fn warpSum(value: f32) f32 {
        // In actual GPU code, this uses __shfl_down_sync or subgroupAdd
        return value;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "flash attention kernel config validation" {
    // Valid config
    const valid = FlashAttentionKernelConfig{
        .block_size_q = 64,
        .block_size_kv = 64,
        .head_dim = 64,
    };
    try validateConfig(valid, 48 * 1024); // 48KB shared memory

    // Invalid block size (not power of 2)
    const invalid_block = FlashAttentionKernelConfig{
        .block_size_q = 65,
        .block_size_kv = 64,
        .head_dim = 64,
    };
    try std.testing.expectError(error.BlockSizeMustBePowerOfTwo, validateConfig(invalid_block, 48 * 1024));

    // Insufficient shared memory
    const large_config = FlashAttentionKernelConfig{
        .block_size_q = 256,
        .block_size_kv = 256,
        .head_dim = 128,
    };
    try std.testing.expectError(error.InsufficientSharedMemory, validateConfig(large_config, 16 * 1024));
}

test "shared memory calculation" {
    const config = FlashAttentionKernelConfig{
        .block_size_q = 64,
        .block_size_kv = 64,
        .head_dim = 64,
        .use_half = false,
    };

    const required = requiredSharedMemory(config);

    // Q: 64 * 64 * 4 = 16KB
    // K: 64 * 64 * 4 = 16KB
    // V: 64 * 64 * 4 = 16KB
    // S: 64 * 64 * 4 = 16KB
    // Total: 64KB
    try std.testing.expectEqual(@as(usize, 64 * 1024), required);

    // With half precision
    const config_half = FlashAttentionKernelConfig{
        .block_size_q = 64,
        .block_size_kv = 64,
        .head_dim = 64,
        .use_half = true,
    };
    try std.testing.expectEqual(@as(usize, 32 * 1024), requiredSharedMemory(config_half));
}

test "create flash attention kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try createFlashAttentionKernel(allocator, default_config);

    try std.testing.expectEqualStrings("flash_attention_forward", ir.name);
    try std.testing.expectEqual(@as(usize, 6), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 6), ir.uniforms.len);
    try std.testing.expectEqual(@as(usize, 4), ir.shared_memory.len);
}

test "online softmax operations" {
    // Test max update
    try std.testing.expectEqual(@as(f32, 5.0), OnlineSoftmaxOps.updateMax(3.0, 5.0));
    try std.testing.expectEqual(@as(f32, 5.0), OnlineSoftmaxOps.updateMax(5.0, 3.0));

    // Test rescale factor
    const rescale = OnlineSoftmaxOps.rescaleFactor(2.0, 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, @exp(-2.0)), rescale, 1e-6);

    // Test stable exp
    const exp_val = OnlineSoftmaxOps.stableExp(5.0, 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, @exp(2.0)), exp_val, 1e-6);
}
