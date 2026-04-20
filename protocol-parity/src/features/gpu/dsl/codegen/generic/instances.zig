//! Pre-instantiated Code Generators
//!
//! Concrete backend instantiations of the generic CodeGenerator template.

const CodeGenerator = @import("../generic.zig").CodeGenerator;

/// GLSL code generator using generic template.
pub const GlslGenerator = CodeGenerator(@import("../configs/glsl_config.zig"));

/// WGSL code generator using generic template.
pub const WgslGenerator = CodeGenerator(@import("../configs/wgsl_config.zig"));

/// MSL code generator using generic template.
pub const MslGenerator = CodeGenerator(@import("../configs/msl_config.zig"));

/// CUDA code generator using generic template.
pub const CudaGenerator = CodeGenerator(@import("../configs/cuda_config.zig"));
