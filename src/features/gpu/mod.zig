//! GPU Feature Module
//!
//! Cross-platform GPU acceleration and compute functionality

const std = @import("std");

// Core GPU components
pub const gpu_renderer = @import("gpu_renderer.zig");
pub const unified_memory = @import("unified_memory.zig");
pub const hardware_detection = @import("hardware_detection.zig");

// GPU backends and compute
pub const backends = @import("backends/mod.zig");
pub const compute = @import("compute/mod.zig");
pub const core = @import("core/mod.zig");

// Specialized GPU features
pub const wasm_support = @import("wasm_support.zig");
pub const cross_compilation = @import("cross_compilation.zig");

// Memory management
pub const memory = @import("memory/mod.zig");

// Testing and benchmarking
pub const testing = @import("testing/mod.zig");
pub const benchmark = @import("benchmark/mod.zig");

// Mobile and optimizations
pub const mobile = @import("mobile/mod.zig");
pub const optimizations = @import("optimizations/mod.zig");

// Libraries and demos (available in submodules)
pub const libraries = @import("libraries/mod.zig");
// Note: demo modules are available via direct import from demo/ directory

// Legacy compatibility - examples module
pub const gpu_examples = @import("gpu_examples.zig");

// --------------------------------------------------------------------------
// Convenience exports for example code
// --------------------------------------------------------------------------
// Expose the GPUContext type from the core GPU renderer.
pub const GPUContext = core.gpu_renderer.GPUContext;
