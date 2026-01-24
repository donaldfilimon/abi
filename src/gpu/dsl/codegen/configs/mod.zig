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

    pub fn get(self: UnaryFunctions, op: expr.UnaryOp) []const u8 {
        return switch (op) {
            .abs => self.abs,
            .sqrt => self.sqrt,
            .sin => self.sin,
            .cos => self.cos,
            .tan => self.tan,
            .asin => self.asin,
            .acos => self.acos,
            .atan => self.atan,
            .sinh => self.sinh,
            .cosh => self.cosh,
            .tanh => self.tanh,
            .exp => self.exp,
            .exp2 => self.exp2,
            .log => self.log,
            .log2 => self.log2,
            .log10 => self.log10,
            .floor => self.floor,
            .ceil => self.ceil,
            .round => self.round,
            .trunc => self.trunc,
            .fract => self.fract,
            .sign => self.sign,
            .normalize => self.normalize,
            .length => self.length,
            else => "unknown",
        };
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

    pub fn get(self: BinaryFunctions, op: expr.BinaryOp) []const u8 {
        return switch (op) {
            .min => self.min,
            .max => self.max,
            .pow => self.pow,
            .atan2 => self.atan2,
            .dot => self.dot,
            .cross => self.cross,
            .distance => self.distance,
            .step => self.step,
            .reflect => self.reflect,
            else => "unknown",
        };
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

    pub fn getScalar(self: TypeNames, scalar: types.ScalarType) []const u8 {
        return switch (scalar) {
            .bool_ => self.bool_,
            .i8 => self.i8_,
            .i16 => self.i16_,
            .i32 => self.i32_,
            .i64 => self.i64_,
            .u8 => self.u8_,
            .u16 => self.u16_,
            .u32 => self.u32_,
            .u64 => self.u64_,
            .f16 => self.f16_,
            .f32 => self.f32_,
            .f64 => self.f64_,
        };
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

    pub fn getPrefix(self: VectorNaming, element: types.ScalarType) []const u8 {
        return switch (element) {
            .f16, .f32, .f64 => self.float_prefix,
            .i8, .i16, .i32, .i64 => self.int_prefix,
            .u8, .u16, .u32, .u64 => self.uint_prefix,
            .bool_ => self.bool_prefix,
        };
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
