//! GPU Codegen Backend Configurations
//!
//! Comptime configuration types and backend-specific configurations
//! used by the generic code generator to produce backend-specific output.

const std = @import("std");
const types = @import("../../types.zig");
const expr = @import("../../expr.zig");

// Re-export all backend configs
pub const glsl = @import("glsl_config.zig");
pub const wgsl = @import("wgsl_config.zig");
pub const msl = @import("msl_config.zig");
pub const cuda = @import("cuda_config.zig");

/// Shading language enumeration.
pub const ShadingLanguage = enum {
    glsl,
    wgsl,
    msl,
    cuda,
    spirv,
    hlsl,
};

/// Backend target enumeration.
pub const BackendTarget = enum {
    vulkan,
    opengl,
    opengles,
    webgpu,
    metal,
    cuda,
};

/// Literal formatting configuration.
pub const LiteralFormat = struct {
    bool_true: []const u8 = "true",
    bool_false: []const u8 = "false",
    i32_suffix: []const u8 = "",
    i64_suffix: []const u8 = "L",
    u32_suffix: []const u8 = "u",
    u64_suffix: []const u8 = "UL",
    f32_suffix: []const u8 = "",
    f64_suffix: []const u8 = "",
    f32_decimal_suffix: []const u8 = ".0",
    f64_decimal_suffix: []const u8 = ".0",
};

/// Atomic operation syntax configuration.
pub const AtomicSyntax = struct {
    add_fn: []const u8,
    sub_fn: []const u8,
    and_fn: []const u8,
    or_fn: []const u8,
    xor_fn: []const u8,
    min_fn: []const u8,
    max_fn: []const u8,
    exchange_fn: []const u8,
    compare_exchange_fn: []const u8,
    load_fn: ?[]const u8 = null,
    store_fn: ?[]const u8 = null,
    /// Prefix for atomic pointer argument (e.g., "&" for WGSL, "" for others)
    ptr_prefix: []const u8 = "",
    /// Suffix for atomic operations (e.g., ", memory_order_relaxed)" for Metal)
    suffix: []const u8 = ")",
    /// Whether atomics need explicit pointer cast (Metal style)
    needs_cast: bool = false,
    cast_template: []const u8 = "",
};

/// Barrier/synchronization syntax.
pub const BarrierSyntax = struct {
    barrier: []const u8,
    memory_barrier: []const u8,
    memory_barrier_buffer: []const u8,
    memory_barrier_shared: []const u8,
};

/// Built-in variable names for compute shaders.
pub const BuiltinVars = struct {
    global_invocation_id: []const u8,
    local_invocation_id: []const u8,
    workgroup_id: []const u8,
    local_invocation_index: []const u8,
    num_workgroups: []const u8,
    workgroup_size: ?[]const u8 = null,
};

/// Unary function name mappings.
pub const UnaryFunctions = struct {
    abs: []const u8 = "abs",
    sqrt: []const u8 = "sqrt",
    sin: []const u8 = "sin",
    cos: []const u8 = "cos",
    tan: []const u8 = "tan",
    asin: []const u8 = "asin",
    acos: []const u8 = "acos",
    atan: []const u8 = "atan",
    sinh: []const u8 = "sinh",
    cosh: []const u8 = "cosh",
    tanh: []const u8 = "tanh",
    exp: []const u8 = "exp",
    exp2: []const u8 = "exp2",
    log: []const u8 = "log",
    log2: []const u8 = "log2",
    log10: []const u8 = "log10",
    floor: []const u8 = "floor",
    ceil: []const u8 = "ceil",
    round: []const u8 = "round",
    trunc: []const u8 = "trunc",
    fract: []const u8 = "fract",
    sign: []const u8 = "sign",
    normalize: []const u8 = "normalize",
    length: []const u8 = "length",

    /// Number of UnaryOp enum values.
    pub const unary_op_count = @typeInfo(expr.UnaryOp).@"enum".fields.len;

    /// Comptime-generated lookup table type.
    pub const LookupTable = [unary_op_count][]const u8;

    /// Build a lookup table at comptime from a UnaryFunctions config.
    /// Call this at comptime with a known config value.
    pub fn buildTable(comptime self: UnaryFunctions) LookupTable {
        comptime var table: LookupTable = undefined;
        // Initialize all entries to "unknown" for ops we don't have mappings for
        for (&table) |*entry| {
            entry.* = "unknown";
        }
        // Map supported operations
        table[@intFromEnum(expr.UnaryOp.abs)] = self.abs;
        table[@intFromEnum(expr.UnaryOp.sqrt)] = self.sqrt;
        table[@intFromEnum(expr.UnaryOp.sin)] = self.sin;
        table[@intFromEnum(expr.UnaryOp.cos)] = self.cos;
        table[@intFromEnum(expr.UnaryOp.tan)] = self.tan;
        table[@intFromEnum(expr.UnaryOp.asin)] = self.asin;
        table[@intFromEnum(expr.UnaryOp.acos)] = self.acos;
        table[@intFromEnum(expr.UnaryOp.atan)] = self.atan;
        table[@intFromEnum(expr.UnaryOp.sinh)] = self.sinh;
        table[@intFromEnum(expr.UnaryOp.cosh)] = self.cosh;
        table[@intFromEnum(expr.UnaryOp.tanh)] = self.tanh;
        table[@intFromEnum(expr.UnaryOp.exp)] = self.exp;
        table[@intFromEnum(expr.UnaryOp.exp2)] = self.exp2;
        table[@intFromEnum(expr.UnaryOp.log)] = self.log;
        table[@intFromEnum(expr.UnaryOp.log2)] = self.log2;
        table[@intFromEnum(expr.UnaryOp.log10)] = self.log10;
        table[@intFromEnum(expr.UnaryOp.floor)] = self.floor;
        table[@intFromEnum(expr.UnaryOp.ceil)] = self.ceil;
        table[@intFromEnum(expr.UnaryOp.round)] = self.round;
        table[@intFromEnum(expr.UnaryOp.trunc)] = self.trunc;
        table[@intFromEnum(expr.UnaryOp.fract)] = self.fract;
        table[@intFromEnum(expr.UnaryOp.sign)] = self.sign;
        table[@intFromEnum(expr.UnaryOp.normalize)] = self.normalize;
        table[@intFromEnum(expr.UnaryOp.length)] = self.length;
        return table;
    }

    /// O(1) lookup using comptime-generated table.
    pub fn get(comptime self: UnaryFunctions, op: expr.UnaryOp) []const u8 {
        const table = comptime self.buildTable();
        return table[@intFromEnum(op)];
    }
};

/// Binary function name mappings.
pub const BinaryFunctions = struct {
    min: []const u8 = "min",
    max: []const u8 = "max",
    pow: []const u8 = "pow",
    atan2: []const u8 = "atan2",
    dot: []const u8 = "dot",
    cross: []const u8 = "cross",
    distance: []const u8 = "distance",
    step: []const u8 = "step",
    reflect: []const u8 = "reflect",

    /// Number of BinaryOp enum values.
    pub const binary_op_count = @typeInfo(expr.BinaryOp).@"enum".fields.len;

    /// Comptime-generated lookup table type.
    pub const LookupTable = [binary_op_count][]const u8;

    /// Build a lookup table at comptime from a BinaryFunctions config.
    pub fn buildTable(comptime self: BinaryFunctions) LookupTable {
        comptime var table: LookupTable = undefined;
        // Initialize all entries to "unknown" for ops we don't have mappings for
        for (&table) |*entry| {
            entry.* = "unknown";
        }
        // Map supported operations
        table[@intFromEnum(expr.BinaryOp.min)] = self.min;
        table[@intFromEnum(expr.BinaryOp.max)] = self.max;
        table[@intFromEnum(expr.BinaryOp.pow)] = self.pow;
        table[@intFromEnum(expr.BinaryOp.atan2)] = self.atan2;
        table[@intFromEnum(expr.BinaryOp.dot)] = self.dot;
        table[@intFromEnum(expr.BinaryOp.cross)] = self.cross;
        table[@intFromEnum(expr.BinaryOp.distance)] = self.distance;
        table[@intFromEnum(expr.BinaryOp.step)] = self.step;
        table[@intFromEnum(expr.BinaryOp.reflect)] = self.reflect;
        return table;
    }

    /// O(1) lookup using comptime-generated table.
    pub fn get(comptime self: BinaryFunctions, op: expr.BinaryOp) []const u8 {
        const table = comptime self.buildTable();
        return table[@intFromEnum(op)];
    }
};

/// Common built-in function names.
pub const BuiltinFunctions = struct {
    clamp: []const u8 = "clamp",
    mix: []const u8 = "mix",
    smoothstep: []const u8 = "smoothstep",
    fma: []const u8 = "fma",
    select: []const u8 = "select",
    all: []const u8 = "all",
    any: []const u8 = "any",
};

/// Type name configuration for a backend.
pub const TypeNames = struct {
    bool_: []const u8,
    i8_: []const u8,
    i16_: []const u8,
    i32_: []const u8,
    i64_: []const u8,
    u8_: []const u8,
    u16_: []const u8,
    u32_: []const u8,
    u64_: []const u8,
    f16_: []const u8,
    f32_: []const u8,
    f64_: []const u8,
    void_: []const u8,

    /// Number of ScalarType enum values.
    pub const scalar_type_count = @typeInfo(types.ScalarType).@"enum".fields.len;

    /// Comptime-generated lookup table type.
    pub const LookupTable = [scalar_type_count][]const u8;

    /// Build a lookup table at comptime from a TypeNames config.
    pub fn buildTable(comptime self: TypeNames) LookupTable {
        comptime var table: LookupTable = undefined;
        // Map all scalar types
        table[@intFromEnum(types.ScalarType.bool_)] = self.bool_;
        table[@intFromEnum(types.ScalarType.i8)] = self.i8_;
        table[@intFromEnum(types.ScalarType.i16)] = self.i16_;
        table[@intFromEnum(types.ScalarType.i32)] = self.i32_;
        table[@intFromEnum(types.ScalarType.i64)] = self.i64_;
        table[@intFromEnum(types.ScalarType.u8)] = self.u8_;
        table[@intFromEnum(types.ScalarType.u16)] = self.u16_;
        table[@intFromEnum(types.ScalarType.u32)] = self.u32_;
        table[@intFromEnum(types.ScalarType.u64)] = self.u64_;
        table[@intFromEnum(types.ScalarType.f16)] = self.f16_;
        table[@intFromEnum(types.ScalarType.f32)] = self.f32_;
        table[@intFromEnum(types.ScalarType.f64)] = self.f64_;
        return table;
    }

    /// O(1) lookup using comptime-generated table.
    pub fn getScalar(comptime self: TypeNames, scalar: types.ScalarType) []const u8 {
        const table = comptime self.buildTable();
        return table[@intFromEnum(scalar)];
    }
};

/// Vector type naming configuration.
pub const VectorNaming = struct {
    /// Format style: "prefix" = vec4, ivec4 | "type_suffix" = float4, int4
    style: enum { prefix, type_suffix, generic } = .prefix,
    /// Prefix for float vectors (vec, float)
    float_prefix: []const u8 = "vec",
    /// Prefix for signed int vectors (ivec, int)
    int_prefix: []const u8 = "ivec",
    /// Prefix for unsigned int vectors (uvec, uint)
    uint_prefix: []const u8 = "uvec",
    /// Prefix for bool vectors (bvec, bool)
    bool_prefix: []const u8 = "bvec",
    /// For generic style (WGSL): format like vec{N}<{T}>
    generic_format: bool = false,

    /// Number of ScalarType enum values.
    pub const scalar_type_count = @typeInfo(types.ScalarType).@"enum".fields.len;

    /// Comptime-generated lookup table type.
    pub const LookupTable = [scalar_type_count][]const u8;

    /// Build a lookup table at comptime from a VectorNaming config.
    pub fn buildTable(comptime self: VectorNaming) LookupTable {
        comptime var table: LookupTable = undefined;
        // Map scalar types to their vector prefixes
        table[@intFromEnum(types.ScalarType.f16)] = self.float_prefix;
        table[@intFromEnum(types.ScalarType.f32)] = self.float_prefix;
        table[@intFromEnum(types.ScalarType.f64)] = self.float_prefix;
        table[@intFromEnum(types.ScalarType.i8)] = self.int_prefix;
        table[@intFromEnum(types.ScalarType.i16)] = self.int_prefix;
        table[@intFromEnum(types.ScalarType.i32)] = self.int_prefix;
        table[@intFromEnum(types.ScalarType.i64)] = self.int_prefix;
        table[@intFromEnum(types.ScalarType.u8)] = self.uint_prefix;
        table[@intFromEnum(types.ScalarType.u16)] = self.uint_prefix;
        table[@intFromEnum(types.ScalarType.u32)] = self.uint_prefix;
        table[@intFromEnum(types.ScalarType.u64)] = self.uint_prefix;
        table[@intFromEnum(types.ScalarType.bool_)] = self.bool_prefix;
        return table;
    }

    /// O(1) lookup using comptime-generated table.
    pub fn getPrefix(comptime self: VectorNaming, element: types.ScalarType) []const u8 {
        const table = comptime self.buildTable();
        return table[@intFromEnum(element)];
    }
};

/// Complete backend configuration.
pub const BackendConfig = struct {
    /// Language identifier
    language: ShadingLanguage,
    /// Type names
    type_names: TypeNames,
    /// Vector naming convention
    vector_naming: VectorNaming,
    /// Literal formatting
    literal_format: LiteralFormat,
    /// Atomic operation syntax
    atomics: AtomicSyntax,
    /// Barrier syntax
    barriers: BarrierSyntax,
    /// Built-in variable names
    builtins: BuiltinVars,
    /// Unary function names
    unary_fns: UnaryFunctions = .{},
    /// Binary function names
    binary_fns: BinaryFunctions = .{},
    /// Common built-in functions
    builtin_fns: BuiltinFunctions = .{},

    // Syntax features
    /// Variable declaration keyword for mutable vars
    var_keyword: []const u8 = "",
    /// Variable declaration keyword for constants
    const_keyword: []const u8 = "const ",
    /// Discard statement
    discard_stmt: []const u8 = "discard;",
    /// Whether select uses (false, true, cond) order (WGSL/MSL) vs ternary
    select_reversed: bool = false,
    /// While loop style: "native" | "loop_break"
    while_style: enum { native, loop_break } = .native,
};
