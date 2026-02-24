//! WGSL Backend Configuration
//!
//! Comptime configuration for WGSL code generation targeting WebGPU.

const mod = @import("mod.zig");
const std = @import("std");

/// WGSL type names.
pub const type_names = mod.TypeNames{
    .bool_ = "bool",
    .i8_ = "i32",
    .i16_ = "i32",
    .i32_ = "i32",
    .i64_ = "i64",
    .u8_ = "u32",
    .u16_ = "u32",
    .u32_ = "u32",
    .u64_ = "u64",
    .f16_ = "f16",
    .f32_ = "f32",
    .f64_ = "f64",
    .void_ = "void",
};

/// WGSL vector naming (vec4<f32>).
pub const vector_naming = mod.VectorNaming{
    .style = .generic,
    .float_prefix = "f32",
    .int_prefix = "i32",
    .uint_prefix = "u32",
    .bool_prefix = "bool",
    .generic_format = true,
};

/// WGSL literal formatting.
pub const literal_format = mod.LiteralFormat{
    .bool_true = "true",
    .bool_false = "false",
    .i32_suffix = "i",
    .i64_suffix = "i",
    .u32_suffix = "u",
    .u64_suffix = "u",
    .f32_suffix = "",
    .f64_suffix = "",
    .f32_decimal_suffix = ".0",
    .f64_decimal_suffix = ".0",
};

/// WGSL atomic operations.
pub const atomics = mod.AtomicSyntax{
    .add_fn = "atomicAdd",
    .sub_fn = "atomicSub",
    .and_fn = "atomicAnd",
    .or_fn = "atomicOr",
    .xor_fn = "atomicXor",
    .min_fn = "atomicMin",
    .max_fn = "atomicMax",
    .exchange_fn = "atomicExchange",
    .compare_exchange_fn = "atomicCompareExchangeWeak",
    .load_fn = "atomicLoad",
    .store_fn = "atomicStore",
    .ptr_prefix = "&",
    .suffix = ")",
};

/// WGSL barriers.
pub const barriers = mod.BarrierSyntax{
    .barrier = "workgroupBarrier()",
    .memory_barrier = "storageBarrier()",
    .memory_barrier_buffer = "storageBarrier()",
    .memory_barrier_shared = "workgroupBarrier()",
};

/// WGSL built-in variables.
pub const builtins = mod.BuiltinVars{
    .global_invocation_id = "globalInvocationId",
    .local_invocation_id = "localInvocationId",
    .workgroup_id = "workgroupId",
    .local_invocation_index = "localInvocationIndex",
    .num_workgroups = "numWorkgroups",
};

/// WGSL built-in functions (some differ from defaults).
pub const builtin_fns = mod.BuiltinFunctions{
    .clamp = "clamp",
    .mix = "mix",
    .smoothstep = "smoothstep",
    .fma = "fma",
    .select = "select",
    .all = "all",
    .any = "any",
};

/// Complete WGSL backend configuration.
pub const config = mod.BackendConfig{
    .language = .wgsl,
    .type_names = type_names,
    .vector_naming = vector_naming,
    .literal_format = literal_format,
    .atomics = atomics,
    .barriers = barriers,
    .builtins = builtins,
    .builtin_fns = builtin_fns,
    .var_keyword = "var ",
    .const_keyword = "let ",
    .discard_stmt = "discard;",
    .select_reversed = true, // WGSL select is (false, true, cond)
    .while_style = .loop_break, // WGSL uses loop { if (!cond) break; }
};

test {
    std.testing.refAllDecls(@This());
}
