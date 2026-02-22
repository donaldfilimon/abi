//! Metal Backend Module
//!
//! Provides GPU acceleration on Apple Silicon using Metal Shading Language
//! and Apple's Accelerate framework for CPU-side optimizations.
//!
//! ## Features
//! - Q4/Q8 quantized matrix-vector multiplication (Metal GPU)
//! - Fused SwiGLU, RMSNorm, Softmax, SiLU kernels (Metal GPU)
//! - vBLAS/vDSP integration via Accelerate framework (AMX-accelerated)
//! - Unified memory architecture support for zero-copy operations
//! - Neural network primitives optimized for Apple Silicon
//! - GPU Family detection (Apple1-9, Mac1-2)
//! - Metal Performance Shaders (MPS) for linear algebra and neural networks
//! - CoreML integration for model inference via Neural Engine
//! - Mesh Shaders (Metal 3+ / Apple7+)
//! - Ray Tracing with acceleration structures (Metal 3+ / Apple7+)
//!
//! ## Usage
//! ```zig
//! const metal = @import("metal");
//!
//! // GPU kernels via Metal
//! var quant_module = try metal.QuantizedKernelModule.init(allocator);
//! defer quant_module.deinit();
//!
//! // CPU operations via Accelerate (AMX-accelerated on Apple Silicon)
//! if (metal.accelerate.is_available) {
//!     try metal.accelerate.sgemm(.no_trans, .no_trans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
//! }
//!
//! // Check unified memory for zero-copy operations
//! if (metal.accelerate.hasUnifiedMemory()) {
//!     // Data can be shared between CPU and GPU without explicit copies
//! }
//! ```

// Core submodules
pub const quantized_kernels = @import("quantized_kernels.zig");
pub const accelerate = @import("accelerate.zig");
pub const unified_memory = @import("unified_memory.zig");

// Metal 3+ feature submodules
pub const gpu_family = @import("gpu_family.zig");
pub const mps = @import("mps.zig");
pub const coreml = @import("coreml.zig");
pub const mesh_shaders = @import("mesh_shaders.zig");
pub const ray_tracing = @import("ray_tracing.zig");

// Unified acceleration pipeline
pub const macos_accelerator = @import("macos_accelerator.zig");

/// Re-export key types for convenience
pub const QuantizedKernelModule = quantized_kernels.QuantizedKernelModule;
pub const QuantConfig = quantized_kernels.QuantConfig;
pub const QuantKernelError = quantized_kernels.QuantKernelError;

/// Accelerate framework types
pub const AccelerateError = accelerate.AccelerateError;
pub const CblasTranspose = accelerate.CblasTranspose;

/// Unified memory types
pub const UnifiedMemoryManager = unified_memory.UnifiedMemoryManager;
pub const UnifiedMemoryConfig = unified_memory.UnifiedMemoryConfig;
pub const UnifiedTensor = unified_memory.UnifiedTensor;
pub const StorageMode = unified_memory.StorageMode;
pub const MemoryStats = unified_memory.MemoryStats;

/// GPU Family types
pub const MetalGpuFamily = gpu_family.MetalGpuFamily;
pub const MetalFeatureSet = gpu_family.MetalFeatureSet;

/// MPS types
pub const MpsMatMul = mps.MpsMatMul;
pub const MpsConvolution = mps.MpsConvolution;
pub const MpsGraph = mps.MpsGraph;

/// CoreML types
pub const CoreMlModel = coreml.CoreMlModel;
pub const ComputeUnit = coreml.ComputeUnit;

/// Mesh shader types
pub const MeshPipeline = mesh_shaders.MeshPipeline;
pub const MeshPipelineConfig = mesh_shaders.MeshPipelineConfig;

/// Ray tracing types
pub const AccelerationStructure = ray_tracing.AccelerationStructure;
pub const TriangleGeometry = ray_tracing.TriangleGeometry;
pub const InstanceDescriptor = ray_tracing.InstanceDescriptor;

/// macOS Accelerator types
pub const MacOSAccelerator = macos_accelerator.MacOSAccelerator;
pub const AcceleratorBackend = macos_accelerator.AcceleratorBackend;
pub const AcceleratorConfig = macos_accelerator.AcceleratorConfig;

/// Quantization block sizes
pub const Q4_BLOCK_SIZE = quantized_kernels.Q4_BLOCK_SIZE;
pub const Q4_BLOCK_BYTES = quantized_kernels.Q4_BLOCK_BYTES;
pub const Q8_BLOCK_SIZE = quantized_kernels.Q8_BLOCK_SIZE;
pub const Q8_BLOCK_BYTES = quantized_kernels.Q8_BLOCK_BYTES;

/// Check if Metal quantized kernels are available on this system.
pub fn isAvailable() bool {
    return quantized_kernels.isAvailable();
}

/// Check if Accelerate framework is available (macOS/iOS/tvOS).
pub fn hasAccelerate() bool {
    return accelerate.is_available;
}

/// Check if unified memory is available (Apple Silicon).
pub fn hasUnifiedMemory() bool {
    return accelerate.hasUnifiedMemory();
}

/// Get recommended memory alignment for unified memory operations.
pub fn unifiedMemoryAlignment() usize {
    return accelerate.unifiedMemoryAlignment();
}

/// Check if MPS framework is available on this system.
pub fn hasMps() bool {
    return mps.isAvailable();
}

/// Check if CoreML framework is available on this system.
pub fn hasCoreml() bool {
    return coreml.isAvailable();
}

/// Check if the unified macOS accelerator is available (Accelerate + MPS + CoreML).
pub fn hasAccelerator() bool {
    return accelerate.is_available;
}

test {
    _ = quantized_kernels;
    _ = accelerate;
    _ = unified_memory;
    _ = gpu_family;
    _ = mps;
    _ = coreml;
    _ = mesh_shaders;
    _ = ray_tracing;
    _ = macos_accelerator;
}
