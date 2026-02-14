//! Metal Shading Language (MSL) Backend Configuration
//!
//! Comptime configuration for MSL code generation targeting Apple Metal.

const mod = @import("mod.zig");

/// MSL type names.
pub const type_names = mod.TypeNames{
    .bool_ = "bool",
    .i8_ = "int8_t",
    .i16_ = "int16_t",
    .i32_ = "int",
    .i64_ = "int64_t",
    .u8_ = "uint8_t",
    .u16_ = "uint16_t",
    .u32_ = "uint",
    .u64_ = "uint64_t",
    .f16_ = "half",
    .f32_ = "float",
    .f64_ = "double",
    .void_ = "void",
};

/// MSL vector naming (float4, int4, uint4).
pub const vector_naming = mod.VectorNaming{
    .style = .type_suffix,
    .float_prefix = "float",
    .int_prefix = "int",
    .uint_prefix = "uint",
    .bool_prefix = "bool",
};

/// MSL-specific vector element type names.
pub const vector_element_names = struct {
    pub fn get(comptime element: @import("../../types.zig").ScalarType) []const u8 {
        return switch (element) {
            .bool_ => "bool",
            .i8 => "char",
            .i16 => "short",
            .i32 => "int",
            .i64 => "long",
            .u8 => "uchar",
            .u16 => "ushort",
            .u32 => "uint",
            .u64 => "ulong",
            .f16 => "half",
            .f32 => "float",
            .f64 => "double",
        };
    }
};

/// MSL literal formatting.
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

/// MSL atomic operations.
pub const atomics = mod.AtomicSyntax{
    .add_fn = "atomic_fetch_add_explicit",
    .sub_fn = "atomic_fetch_sub_explicit",
    .and_fn = "atomic_fetch_and_explicit",
    .or_fn = "atomic_fetch_or_explicit",
    .xor_fn = "atomic_fetch_xor_explicit",
    .min_fn = "atomic_fetch_min_explicit",
    .max_fn = "atomic_fetch_max_explicit",
    .exchange_fn = "atomic_exchange_explicit",
    .compare_exchange_fn = "atomic_compare_exchange_weak_explicit",
    .load_fn = "atomic_load_explicit",
    .store_fn = "atomic_store_explicit",
    .ptr_prefix = "",
    .suffix = ", memory_order_relaxed)",
    .needs_cast = true,
    .cast_template = "(device atomic_uint*)&",
};

/// MSL barriers.
pub const barriers = mod.BarrierSyntax{
    .barrier = "threadgroup_barrier(mem_flags::mem_threadgroup)",
    .memory_barrier = "threadgroup_barrier(mem_flags::mem_device)",
    .memory_barrier_buffer = "threadgroup_barrier(mem_flags::mem_device)",
    .memory_barrier_shared = "threadgroup_barrier(mem_flags::mem_threadgroup)",
};

/// MSL built-in variables (via attributes).
pub const builtins = mod.BuiltinVars{
    .global_invocation_id = "globalInvocationId",
    .local_invocation_id = "localInvocationId",
    .workgroup_id = "workgroupId",
    .local_invocation_index = "localInvocationIndex",
    .num_workgroups = "numWorkgroups",
};

/// MSL built-in attributes for kernel parameters.
pub const builtin_attributes = struct {
    pub const global_invocation_id = "[[thread_position_in_grid]]";
    pub const local_invocation_id = "[[thread_position_in_threadgroup]]";
    pub const workgroup_id = "[[threadgroup_position_in_grid]]";
    pub const local_invocation_index = "[[thread_index_in_threadgroup]]";
    pub const num_workgroups = "[[threadgroups_per_grid]]";
};

/// MSL built-in functions.
pub const builtin_fns = mod.BuiltinFunctions{
    .clamp = "clamp",
    .mix = "mix",
    .smoothstep = "smoothstep",
    .fma = "fma",
    .select = "select",
    .all = "all",
    .any = "any",
};

/// MSL mesh shader attributes (Metal 3+ / Apple7+).
pub const mesh_attributes = struct {
    pub const object_payload = "[[payload]]";
    pub const mesh_grid_properties = "[[mesh_grid_properties]]";
    pub const object_thread_position_in_grid = "[[thread_position_in_grid]]";
    pub const object_threadgroup_position_in_grid = "[[threadgroup_position_in_grid]]";
    pub const object_threads_per_grid = "[[threads_per_grid]]";
};

/// MSL ray tracing attributes (Metal 3+ / Apple7+).
pub const ray_tracing_attributes = struct {
    pub const ray_data = "[[ray_data]]";
    pub const intersection_result = "[[intersection_result]]";
    pub const primitive_id = "[[primitive_id]]";
    pub const instance_id = "[[instance_id]]";
    pub const geometry_id = "[[geometry_id]]";
    pub const world_space_origin = "[[world_space_origin]]";
    pub const world_space_direction = "[[world_space_direction]]";
    pub const ray_flags = "[[ray_flags]]";
};

/// Complete MSL backend configuration.
pub const config = mod.BackendConfig{
    .language = .msl,
    .type_names = type_names,
    .vector_naming = vector_naming,
    .literal_format = literal_format,
    .atomics = atomics,
    .barriers = barriers,
    .builtins = builtins,
    .builtin_fns = builtin_fns,
    .var_keyword = "",
    .const_keyword = "const ",
    .discard_stmt = "discard_fragment();",
    .select_reversed = true, // MSL select is (false, true, cond)
    .while_style = .native,
};
