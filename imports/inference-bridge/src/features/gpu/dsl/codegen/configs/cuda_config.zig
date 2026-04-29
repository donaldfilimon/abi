//! CUDA Backend Configuration
//!
//! Comptime configuration for CUDA C code generation targeting NVIDIA GPUs.

const mod = @import("mod.zig");
const std = @import("std");

/// CUDA type names.
pub const type_names = mod.TypeNames{
    .bool_ = "bool",
    .i8_ = "int8_t",
    .i16_ = "int16_t",
    .i32_ = "int",
    .i64_ = "int64_t",
    .u8_ = "uint8_t",
    .u16_ = "uint16_t",
    .u32_ = "unsigned int",
    .u64_ = "uint64_t",
    .f16_ = "half",
    .f32_ = "float",
    .f64_ = "double",
    .void_ = "void",
};

/// CUDA vector naming (float4, int4).
pub const vector_naming = mod.VectorNaming{
    .style = .type_suffix,
    .float_prefix = "float",
    .int_prefix = "int",
    .uint_prefix = "uint",
    .bool_prefix = "bool",
};

/// CUDA literal formatting.
pub const literal_format = mod.LiteralFormat{
    .bool_true = "true",
    .bool_false = "false",
    .i32_suffix = "",
    .i64_suffix = "LL",
    .u32_suffix = "u",
    .u64_suffix = "ULL",
    .f32_suffix = "f",
    .f64_suffix = "",
    .f32_decimal_suffix = ".0f",
    .f64_decimal_suffix = ".0",
};

/// CUDA atomic operations.
pub const atomics = mod.AtomicSyntax{
    .add_fn = "atomicAdd",
    .sub_fn = "atomicSub",
    .and_fn = "atomicAnd",
    .or_fn = "atomicOr",
    .xor_fn = "atomicXor",
    .min_fn = "atomicMin",
    .max_fn = "atomicMax",
    .exchange_fn = "atomicExch",
    .compare_exchange_fn = "atomicCAS",
    .ptr_prefix = "",
    .suffix = ")",
};

/// CUDA barriers.
pub const barriers = mod.BarrierSyntax{
    .barrier = "__syncthreads()",
    .memory_barrier = "__threadfence()",
    .memory_barrier_buffer = "__threadfence()",
    .memory_barrier_shared = "__syncthreads()",
};

/// CUDA built-in variables.
pub const builtins = mod.BuiltinVars{
    .global_invocation_id = "globalInvocationId",
    .local_invocation_id = "localInvocationId",
    .workgroup_id = "workgroupId",
    .local_invocation_index = "localInvocationIndex",
    .num_workgroups = "gridDim",
    .workgroup_size = "workgroupSize",
};

/// CUDA unary functions (some have f suffix).
pub const unary_fns = mod.UnaryFunctions{
    .abs = "fabsf",
    .sqrt = "sqrtf",
    .sin = "sinf",
    .cos = "cosf",
    .tan = "tanf",
    .asin = "asinf",
    .acos = "acosf",
    .atan = "atanf",
    .sinh = "sinhf",
    .cosh = "coshf",
    .tanh = "tanhf",
    .exp = "expf",
    .exp2 = "exp2f",
    .log = "logf",
    .log2 = "log2f",
    .log10 = "log10f",
    .floor = "floorf",
    .ceil = "ceilf",
    .round = "roundf",
    .trunc = "truncf",
    .fract = "__fract_helper",
    .sign = "__sign_helper",
    .normalize = "normalize",
    .length = "length",
};

/// CUDA binary functions.
pub const binary_fns = mod.BinaryFunctions{
    .min = "fminf",
    .max = "fmaxf",
    .pow = "powf",
    .atan2 = "atan2f",
    .dot = "dot",
    .cross = "cross",
    .distance = "distance",
    .step = "step",
    .reflect = "reflect",
};

/// CUDA built-in functions.
pub const builtin_fns = mod.BuiltinFunctions{
    .clamp = "clamp",
    .mix = "lerp", // CUDA uses lerp
    .smoothstep = "smoothstep",
    .fma = "fmaf",
    .select = "select",
    .all = "all",
    .any = "any",
};

/// Complete CUDA backend configuration.
pub const config = mod.BackendConfig{
    .language = .cuda,
    .type_names = type_names,
    .vector_naming = vector_naming,
    .literal_format = literal_format,
    .atomics = atomics,
    .barriers = barriers,
    .builtins = builtins,
    .unary_fns = unary_fns,
    .binary_fns = binary_fns,
    .builtin_fns = builtin_fns,
    .var_keyword = "",
    .const_keyword = "const ",
    .discard_stmt = "", // No discard in compute
    .select_reversed = false,
    .while_style = .native,
};

test {
    std.testing.refAllDecls(@This());
}
