//! GPU Kernel DSL Expression AST
//!
//! Defines the expression nodes for the kernel IR abstract syntax tree.
//! Expressions represent computations that produce values.

const std = @import("std");
const types = @import("types.zig");

/// Reference to a value (variable, parameter, built-in, etc.).
pub const ValueRef = struct {
    id: u32,
    ty: types.Type,
    name: ?[]const u8 = null,
};

/// Unary operations.
pub const UnaryOp = enum {
    // Arithmetic
    neg, // -x

    // Logical
    not, // !x (logical not)

    // Bitwise
    bit_not, // ~x

    // Math functions
    abs,
    sqrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    exp,
    exp2,
    log,
    log2,
    log10,
    floor,
    ceil,
    round,
    trunc,
    fract,
    sign,

    // Vector operations
    normalize,
    length,

    /// Returns the name of this operation.
    pub fn name(self: UnaryOp) []const u8 {
        return switch (self) {
            .neg => "-",
            .not => "!",
            .bit_not => "~",
            .abs => "abs",
            .sqrt => "sqrt",
            .sin => "sin",
            .cos => "cos",
            .tan => "tan",
            .asin => "asin",
            .acos => "acos",
            .atan => "atan",
            .sinh => "sinh",
            .cosh => "cosh",
            .tanh => "tanh",
            .exp => "exp",
            .exp2 => "exp2",
            .log => "log",
            .log2 => "log2",
            .log10 => "log10",
            .floor => "floor",
            .ceil => "ceil",
            .round => "round",
            .trunc => "trunc",
            .fract => "fract",
            .sign => "sign",
            .normalize => "normalize",
            .length => "length",
        };
    }

    /// Returns true if this is a prefix operator.
    pub fn isPrefix(self: UnaryOp) bool {
        return switch (self) {
            .neg, .not, .bit_not => true,
            else => false,
        };
    }
};

/// Binary operations.
pub const BinaryOp = enum {
    // Arithmetic
    add,
    sub,
    mul,
    div,
    mod,

    // Comparison
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    // Logical
    and_,
    or_,
    xor,

    // Bitwise
    bit_and,
    bit_or,
    bit_xor,
    shl,
    shr,

    // Vector/Math
    dot,
    cross,
    min,
    max,
    pow,
    atan2,
    step,
    reflect,
    distance,

    /// Returns the operator string for this operation.
    pub fn symbol(self: BinaryOp) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .mod => "%",
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
            .and_ => "&&",
            .or_ => "||",
            .xor => "^^",
            .bit_and => "&",
            .bit_or => "|",
            .bit_xor => "^",
            .shl => "<<",
            .shr => ">>",
            .dot => "dot",
            .cross => "cross",
            .min => "min",
            .max => "max",
            .pow => "pow",
            .atan2 => "atan2",
            .step => "step",
            .reflect => "reflect",
            .distance => "distance",
        };
    }

    /// Returns true if this is an infix operator.
    pub fn isInfix(self: BinaryOp) bool {
        return switch (self) {
            .add, .sub, .mul, .div, .mod => true,
            .eq, .ne, .lt, .le, .gt, .ge => true,
            .and_, .or_, .xor => true,
            .bit_and, .bit_or, .bit_xor, .shl, .shr => true,
            else => false,
        };
    }

    /// Returns true if this produces a boolean result.
    pub fn isComparison(self: BinaryOp) bool {
        return switch (self) {
            .eq, .ne, .lt, .le, .gt, .ge => true,
            else => false,
        };
    }

    /// Returns the precedence of this operator (higher = binds tighter).
    pub fn precedence(self: BinaryOp) u8 {
        return switch (self) {
            .or_ => 1,
            .xor => 2,
            .and_ => 3,
            .bit_or => 4,
            .bit_xor => 5,
            .bit_and => 6,
            .eq, .ne => 7,
            .lt, .le, .gt, .ge => 8,
            .shl, .shr => 9,
            .add, .sub => 10,
            .mul, .div, .mod => 11,
            else => 12, // function-like ops
        };
    }
};

/// Built-in functions.
pub const BuiltinFn = enum {
    // Synchronization
    barrier, // workgroup barrier
    memory_barrier, // memory fence
    memory_barrier_buffer, // buffer-specific fence
    memory_barrier_shared, // shared memory fence

    // Atomics
    atomic_add,
    atomic_sub,
    atomic_and,
    atomic_or,
    atomic_xor,
    atomic_min,
    atomic_max,
    atomic_exchange,
    atomic_compare_exchange,
    atomic_load,
    atomic_store,

    // Math
    clamp,
    mix, // lerp
    smoothstep,
    fma, // fused multiply-add

    // Utility
    select, // ternary select
    all, // all components true
    any, // any component true

    /// Returns the name of this built-in function.
    pub fn name(self: BuiltinFn) []const u8 {
        return switch (self) {
            .barrier => "barrier",
            .memory_barrier => "memoryBarrier",
            .memory_barrier_buffer => "memoryBarrierBuffer",
            .memory_barrier_shared => "memoryBarrierShared",
            .atomic_add => "atomicAdd",
            .atomic_sub => "atomicSub",
            .atomic_and => "atomicAnd",
            .atomic_or => "atomicOr",
            .atomic_xor => "atomicXor",
            .atomic_min => "atomicMin",
            .atomic_max => "atomicMax",
            .atomic_exchange => "atomicExchange",
            .atomic_compare_exchange => "atomicCompareExchange",
            .atomic_load => "atomicLoad",
            .atomic_store => "atomicStore",
            .clamp => "clamp",
            .mix => "mix",
            .smoothstep => "smoothstep",
            .fma => "fma",
            .select => "select",
            .all => "all",
            .any => "any",
        };
    }

    /// Returns the number of arguments expected.
    pub fn argCount(self: BuiltinFn) ?u8 {
        return switch (self) {
            .barrier, .memory_barrier, .memory_barrier_buffer, .memory_barrier_shared => 0,
            .atomic_load => 1,
            .atomic_add, .atomic_sub, .atomic_and, .atomic_or, .atomic_xor => 2,
            .atomic_min, .atomic_max, .atomic_exchange, .atomic_store => 2,
            .clamp, .mix, .smoothstep, .fma, .select => 3,
            .atomic_compare_exchange => 3,
            .all, .any => 1,
        };
    }
};

/// Literal value.
pub const Literal = union(enum) {
    bool_: bool,
    i32_: i32,
    i64_: i64,
    u32_: u32,
    u64_: u64,
    f32_: f32,
    f64_: f64,

    /// Returns the type of this literal.
    pub fn getType(self: Literal) types.Type {
        return switch (self) {
            .bool_ => types.Type.boolType(),
            .i32_ => types.Type.i32Type(),
            .i64_ => .{ .scalar = .i64 },
            .u32_ => types.Type.u32Type(),
            .u64_ => .{ .scalar = .u64 },
            .f32_ => types.Type.f32Type(),
            .f64_ => .{ .scalar = .f64 },
        };
    }
};

/// Expression node.
pub const Expr = union(enum) {
    /// Literal value.
    literal: Literal,

    /// Variable/parameter reference.
    ref: ValueRef,

    /// Unary operation.
    unary: UnaryExpr,

    /// Binary operation.
    binary: BinaryExpr,

    /// Function call (built-in).
    call: CallExpr,

    /// Vector construction (vec2, vec3, vec4).
    vector_construct: VectorConstruct,

    /// Array/vector element access (base[index]).
    index: IndexExpr,

    /// Struct/vector field access (base.field).
    field: FieldExpr,

    /// Type cast.
    cast: CastExpr,

    /// Ternary select (condition ? true_val : false_val).
    select: SelectExpr,

    /// Swizzle operation (e.g., v.xyz, v.xxyy).
    swizzle: SwizzleExpr,

    pub const UnaryExpr = struct {
        op: UnaryOp,
        operand: *const Expr,
    };

    pub const BinaryExpr = struct {
        op: BinaryOp,
        left: *const Expr,
        right: *const Expr,
    };

    pub const CallExpr = struct {
        function: BuiltinFn,
        args: []const *const Expr,
    };

    pub const VectorConstruct = struct {
        element_type: types.ScalarType,
        size: u8,
        components: []const *const Expr,
    };

    pub const IndexExpr = struct {
        base: *const Expr,
        index: *const Expr,
    };

    pub const FieldExpr = struct {
        base: *const Expr,
        field: []const u8,
    };

    pub const CastExpr = struct {
        target_type: types.Type,
        operand: *const Expr,
    };

    pub const SelectExpr = struct {
        condition: *const Expr,
        true_value: *const Expr,
        false_value: *const Expr,
    };

    pub const SwizzleExpr = struct {
        base: *const Expr,
        /// Swizzle mask using indices 0-3 for x,y,z,w.
        /// Length determines result vector size.
        components: []const u8,
    };
};

/// Built-in variables available in compute shaders.
pub const BuiltinVar = enum {
    /// Global invocation ID (vec3<u32>).
    global_invocation_id,
    /// Local invocation ID within workgroup (vec3<u32>).
    local_invocation_id,
    /// Workgroup ID (vec3<u32>).
    workgroup_id,
    /// Workgroup size (vec3<u32>, compile-time constant).
    workgroup_size,
    /// Number of workgroups (vec3<u32>).
    num_workgroups,
    /// Linear local invocation index (u32).
    local_invocation_index,
    /// Subgroup size (u32).
    subgroup_size,
    /// Subgroup invocation ID (u32).
    subgroup_invocation_id,

    /// Returns the type of this built-in variable.
    pub fn getType(self: BuiltinVar) types.Type {
        return switch (self) {
            .global_invocation_id,
            .local_invocation_id,
            .workgroup_id,
            .workgroup_size,
            .num_workgroups,
            => types.Type.vec3u32(),
            .local_invocation_index,
            .subgroup_size,
            .subgroup_invocation_id,
            => types.Type.u32Type(),
        };
    }

    /// Returns the name of this built-in variable.
    pub fn name(self: BuiltinVar) []const u8 {
        return switch (self) {
            .global_invocation_id => "global_invocation_id",
            .local_invocation_id => "local_invocation_id",
            .workgroup_id => "workgroup_id",
            .workgroup_size => "workgroup_size",
            .num_workgroups => "num_workgroups",
            .local_invocation_index => "local_invocation_index",
            .subgroup_size => "subgroup_size",
            .subgroup_invocation_id => "subgroup_invocation_id",
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "UnaryOp.isPrefix" {
    try std.testing.expect(UnaryOp.neg.isPrefix());
    try std.testing.expect(UnaryOp.not.isPrefix());
    try std.testing.expect(!UnaryOp.sqrt.isPrefix());
    try std.testing.expect(!UnaryOp.sin.isPrefix());
}

test "BinaryOp.isComparison" {
    try std.testing.expect(BinaryOp.eq.isComparison());
    try std.testing.expect(BinaryOp.lt.isComparison());
    try std.testing.expect(!BinaryOp.add.isComparison());
    try std.testing.expect(!BinaryOp.mul.isComparison());
}

test "BinaryOp.precedence" {
    // Multiplication should bind tighter than addition
    try std.testing.expect(BinaryOp.mul.precedence() > BinaryOp.add.precedence());
    // Addition should bind tighter than comparison
    try std.testing.expect(BinaryOp.add.precedence() > BinaryOp.lt.precedence());
    // Comparison should bind tighter than logical and
    try std.testing.expect(BinaryOp.lt.precedence() > BinaryOp.and_.precedence());
}

test "Literal.getType" {
    const lit_f32: Literal = .{ .f32_ = 1.0 };
    try std.testing.expectEqual(types.Type.f32Type(), lit_f32.getType());

    const lit_i32: Literal = .{ .i32_ = 42 };
    try std.testing.expectEqual(types.Type.i32Type(), lit_i32.getType());

    const lit_bool: Literal = .{ .bool_ = true };
    try std.testing.expectEqual(types.Type.boolType(), lit_bool.getType());
}

test "BuiltinVar.getType" {
    try std.testing.expectEqual(types.Type.vec3u32(), BuiltinVar.global_invocation_id.getType());
    try std.testing.expectEqual(types.Type.u32Type(), BuiltinVar.local_invocation_index.getType());
}
