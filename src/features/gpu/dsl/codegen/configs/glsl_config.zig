//! GLSL Backend Configuration
//!
//! Comptime configuration for GLSL code generation targeting Vulkan,
//! OpenGL, and OpenGL ES.

const mod = @import("mod.zig");

/// GLSL target variants.
pub const Target = enum {
    vulkan, // GLSL 450 with Vulkan extensions
    opengl, // GLSL 430 compute shaders
    opengles, // GLSL ES 310+ compute
};

/// Get version directive for target.
pub fn getVersionDirective(target: Target) []const u8 {
    return switch (target) {
        .vulkan => "#version 450",
        .opengl => "#version 430",
        .opengles => "#version 310 es",
    };
}

/// Get header extensions for target.
pub fn getExtensions(target: Target) []const []const u8 {
    return switch (target) {
        .vulkan => &[_][]const u8{"#extension GL_ARB_separate_shader_objects : enable"},
        .opengl => &[_][]const u8{},
        .opengles => &[_][]const u8{},
    };
}

/// Whether target needs precision qualifiers.
pub fn needsPrecisionQualifiers(target: Target) bool {
    return target == .opengles;
}

/// GLSL type names.
pub const type_names = mod.TypeNames{
    .bool_ = "bool",
    .i8_ = "int",
    .i16_ = "int",
    .i32_ = "int",
    .i64_ = "int64_t",
    .u8_ = "uint",
    .u16_ = "uint",
    .u32_ = "uint",
    .u64_ = "uint64_t",
    .f16_ = "float16_t",
    .f32_ = "float",
    .f64_ = "double",
    .void_ = "void",
};

/// GLSL vector naming (vec4, ivec4, uvec4).
pub const vector_naming = mod.VectorNaming{
    .style = .prefix,
    .float_prefix = "vec",
    .int_prefix = "ivec",
    .uint_prefix = "uvec",
    .bool_prefix = "bvec",
};

/// GLSL literal formatting.
pub const literal_format = mod.LiteralFormat{
    .bool_true = "true",
    .bool_false = "false",
    .i32_suffix = "",
    .i64_suffix = "L",
    .u32_suffix = "u",
    .u64_suffix = "UL",
    .f32_suffix = "",
    .f64_suffix = "lf",
    .f32_decimal_suffix = ".0",
    .f64_decimal_suffix = ".0lf",
};

/// GLSL atomic operations.
pub const atomics = mod.AtomicSyntax{
    .add_fn = "atomicAdd",
    .sub_fn = "atomicAdd", // Use with negative value
    .and_fn = "atomicAnd",
    .or_fn = "atomicOr",
    .xor_fn = "atomicXor",
    .min_fn = "atomicMin",
    .max_fn = "atomicMax",
    .exchange_fn = "atomicExchange",
    .compare_exchange_fn = "atomicCompSwap",
    .ptr_prefix = "",
    .suffix = ")",
};

/// GLSL barriers.
pub const barriers = mod.BarrierSyntax{
    .barrier = "barrier()",
    .memory_barrier = "memoryBarrier()",
    .memory_barrier_buffer = "memoryBarrierBuffer()",
    .memory_barrier_shared = "memoryBarrierShared()",
};

/// GLSL built-in variables.
pub const builtins = mod.BuiltinVars{
    .global_invocation_id = "gl_GlobalInvocationID",
    .local_invocation_id = "gl_LocalInvocationID",
    .workgroup_id = "gl_WorkGroupID",
    .local_invocation_index = "gl_LocalInvocationIndex",
    .num_workgroups = "gl_NumWorkGroups",
};

/// Complete GLSL backend configuration.
pub const config = mod.BackendConfig{
    .language = .glsl,
    .type_names = type_names,
    .vector_naming = vector_naming,
    .literal_format = literal_format,
    .atomics = atomics,
    .barriers = barriers,
    .builtins = builtins,
    .var_keyword = "",
    .const_keyword = "const ",
    .discard_stmt = "discard;",
    .select_reversed = false,
    .while_style = .native,
};
