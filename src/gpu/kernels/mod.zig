//! GPU Kernel Library
//!
//! Pre-built kernel definitions using the GPU DSL for common operations.
//! These kernels can be compiled to any supported backend (CUDA, Vulkan, Metal, WebGPU).
//!
//! ## Available Kernels
//!
//! ### Attention
//! - `flash_attention` - Memory-efficient attention (FlashAttention algorithm)
//!
//! ### Fused Operations
//! - `fused_ops` - Fused kernels for reduced memory bandwidth
//!
//! ## Usage
//!
//! ```zig
//! const kernels = @import("kernels");
//! const flash = kernels.flash_attention;
//!
//! // Create kernel IR
//! const ir = try flash.createFlashAttentionKernel(allocator, flash.default_config);
//!
//! // Compile to target backend
//! const spirv = try spirv_gen.generate(&ir);
//! ```

const std = @import("std");

/// Flash Attention kernel implementation.
pub const flash_attention = @import("flash_attention.zig");

/// Fused operations kernels (LayerNorm+Linear, RMSNorm+RoPE, etc.).
pub const fused_ops = @import("fused_ops.zig");

/// Get recommended kernel configuration for GPU.
pub fn getRecommendedConfig(gpu_vendor: GpuVendor, gpu_arch: ?[]const u8) flash_attention.FlashAttentionKernelConfig {
    _ = gpu_arch;
    return switch (gpu_vendor) {
        .nvidia => flash_attention.TunedConfigs.ampere,
        .amd => flash_attention.TunedConfigs.rdna3,
        .apple => flash_attention.TunedConfigs.apple_silicon,
        .intel => flash_attention.TunedConfigs.fallback,
        .unknown => flash_attention.TunedConfigs.fallback,
    };
}

/// GPU vendor identification.
pub const GpuVendor = enum {
    nvidia,
    amd,
    apple,
    intel,
    unknown,
};

test {
    std.testing.refAllDecls(@This());
}
