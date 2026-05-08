//! Generic GPU Code Generator Facade
//!
//! A comptime-generic code generator template that uses backend configuration
//! to produce output for any target shading language. This eliminates code
//! duplication across GLSL, WGSL, MSL, CUDA, and other backends.
//!
//! This module has been decomposed into smaller components inside the `generic/` directory.

const std = @import("std");

/// The core CodeGenerator factory function
pub const CodeGenerator = @import("generic/core.zig").CodeGenerator;

const instances = @import("generic/instances.zig");

/// GLSL code generator using generic template.
pub const GlslGenerator = instances.GlslGenerator;

/// WGSL code generator using generic template.
pub const WgslGenerator = instances.WgslGenerator;

/// MSL code generator using generic template.
pub const MslGenerator = instances.MslGenerator;

/// CUDA code generator using generic template.
pub const CudaGenerator = instances.CudaGenerator;

test {
    _ = @import("generic/instances.zig");
    _ = @import("generic/tests.zig");
    std.testing.refAllDecls(@This());
}
