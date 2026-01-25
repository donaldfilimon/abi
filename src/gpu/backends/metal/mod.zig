//! Metal Backend Module
//!
//! Provides GPU acceleration on Apple Silicon using Metal Shading Language.
//!
//! ## Features
//! - Q4/Q8 quantized matrix-vector multiplication
//! - Fused SwiGLU, RMSNorm, Softmax, SiLU kernels
//! - Optimized for Apple Silicon unified memory
//!
//! ## Usage
//! ```zig
//! const metal = @import("metal");
//! const quant = metal.quantized_kernels;
//!
//! var module = try quant.QuantizedKernelModule.init(allocator);
//! defer module.deinit();
//!
//! if (module.isAvailable()) {
//!     try module.q4Matmul(a_buffer, x_buffer, y_buffer, m, k);
//! }
//! ```

pub const quantized_kernels = @import("quantized_kernels.zig");

/// Re-export key types for convenience
pub const QuantizedKernelModule = quantized_kernels.QuantizedKernelModule;
pub const QuantConfig = quantized_kernels.QuantConfig;
pub const QuantKernelError = quantized_kernels.QuantKernelError;

/// Quantization block sizes
pub const Q4_BLOCK_SIZE = quantized_kernels.Q4_BLOCK_SIZE;
pub const Q4_BLOCK_BYTES = quantized_kernels.Q4_BLOCK_BYTES;
pub const Q8_BLOCK_SIZE = quantized_kernels.Q8_BLOCK_SIZE;
pub const Q8_BLOCK_BYTES = quantized_kernels.Q8_BLOCK_BYTES;

/// Check if Metal quantized kernels are available on this system.
pub fn isAvailable() bool {
    return quantized_kernels.isAvailable();
}

test {
    _ = quantized_kernels;
}
