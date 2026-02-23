//! WebGPU Backend Module
//!
//! Provides GPU acceleration using WebGPU/WGSL for cross-platform web and native compute.
//!
//! ## Features
//! - Q4/Q8 quantized matrix-vector multiplication (WGSL)
//! - Fused SwiGLU, RMSNorm, Softmax, SiLU kernels
//! - WASM-compatible for browser deployment
//! - Native support via wgpu-native
//!
//! ## Usage
//!
//! The WebGPU module provides WGSL shader sources that can be compiled
//! by the JavaScript host (in browsers) or wgpu-native (on desktop):
//!
//! ```zig
//! const webgpu = @import("webgpu");
//! const quant = webgpu.quantized_kernels;
//!
//! var module = try quant.QuantizedKernelModule.init(allocator);
//! defer module.deinit();
//!
//! if (module.isAvailable()) {
//!     // Get shader source for external compilation
//!     const q4_source = module.getQ4ShaderSource();
//!     // Compile via WebGPU API...
//! }
//! ```
//!
//! ## Browser Integration
//!
//! When targeting WASM, export the shader sources to JavaScript:
//!
//! ```javascript
//! // JavaScript side
//! const device = await navigator.gpu.requestDevice();
//! const q4Module = device.createShaderModule({
//!     code: wasmModule.exports.getQ4ShaderSource()
//! });
//! ```
const std = @import("std");

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

/// Check if WebGPU quantized kernels are available on this system.
pub fn isAvailable() bool {
    return quantized_kernels.isAvailable();
}

test {
    _ = quantized_kernels;
}

test {
    std.testing.refAllDecls(@This());
}
