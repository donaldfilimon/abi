//! CEL Abstract Syntax Tree
//!
//! AST node definitions for the CEL programming language.
//! Extends the GPU DSL type system with general-purpose language constructs.

const std = @import("std");
const token = @import("token");

pub const Loc = token.Loc;

// =========================================================================
// Top-Level Declarations
// =========================================================================

/// A complete CEL source file.
pub const File = struct {
    decls: []const Decl,
    eof_loc: Loc,
};

/// Top-level declaration.
pub const Decl = struct {
    kind: Kind,
    loc: Loc,

    pub const Kind = union(enum) {
        fn_decl: FnDecl,
        var_decl: VarDecl,
        struct_decl: StructDecl,
        enum_decl: EnumDecl,
        union_decl: UnionDecl,
        test_decl: TestDecl,
        comptime_block: *Block,
        using: UsingDecl,
    };
};

/// Function declaration.
pub const FnDecl = struct {
    name: token.Loc,
    params: []const Param,
    return_type: ?*TypeExpr,
    body: ?*Block,
    is_pub: bool,
    is_inline: bool,
    is_export: bool,
    is_extern: bool,
    doc_comment: ?token.Loc,
};

/// Function parameter.
pub const Param = struct {
    name: token.Loc,
    type_expr: *TypeExpr,
    is_comptime: bool,
    is_noalias: bool,
};

/// Variable / constant declaration.
pub const VarDecl = struct {
    name: token.Loc,
    type_expr: ?*TypeExpr,
    init_expr: ?*Expr,
    is_const: bool,
    is_comptime: bool,
    is_pub: bool,
    is_threadlocal: bool,
    is_extern: bool,
    doc_comment: ?token.Loc,
};

/// Struct declaration.
pub const StructDecl = struct {
    name: token.Loc,
    fields: []const FieldDecl,
    decls: []const Decl,
    is_pub: bool,
    is_packed: bool,
    is_extern_struct: bool,
    doc_comment: ?token.Loc,
};

/// Struct field.
pub const FieldDecl = struct {
    name: token.Loc,
    type_expr: *TypeExpr,
    default_value: ?*Expr,
    alignment: ?*Expr,
    doc_comment: ?token.Loc,
};

/// Enum declaration.
pub const EnumDecl = struct {
    name: token.Loc,
    tag_type: ?*TypeExpr,
    fields: []const EnumField,
    decls: []const Decl,
    is_pub: bool,
    doc_comment: ?token.Loc,
};

/// Enum field.
pub const EnumField = struct {
    name: token.Loc,
    value: ?*Expr,
    doc_comment: ?token.Loc,
};

/// Union declaration.
pub const UnionDecl = struct {
    name: token.Loc,
    tag_type: ?*TypeExpr,
    fields: []const UnionField,
    decls: []const Decl,
    is_pub: bool,
    doc_comment: ?token.Loc,
};

/// Union field.
pub const UnionField = struct {
    name: token.Loc,
    type_expr: ?*TypeExpr,
    doc_comment: ?token.Loc,
};

/// Test declaration.
pub const TestDecl = struct {
    name: ?token.Loc,
    body: *Block,
};

/// Using declaration (import re-export).
pub const UsingDecl = struct {
    path: *Expr,
    is_pub: bool,
};

// =========================================================================
// Type Expressions
// =========================================================================

/// A type expression in CEL source code.
pub const TypeExpr = struct {
    kind: Kind,
    loc: Loc,

    pub const Kind = union(enum) {
        /// Named type: `i32`, `MyStruct`, `void`
        named: token.Loc,
        /// Pointer: `*T`, `*const T`
        pointer: Pointer,
        /// Optional: `?T`
        optional: *TypeExpr,
        /// Array: `[N]T`
        array: Array,
        /// Slice: `[]T`, `[]const T`
        slice: Slice,
        /// Multi-pointer: `[*]T`
        multi_pointer: *TypeExpr,
        /// Sentinel-terminated pointer: `[*:0]T`
        sentinel: Sentinel,
        /// Error union: `E!T`
        error_union: ErrorUnion,
        /// Function type: `fn(i32, i32) i32`
        function: FnType,
        /// Builtin call: `@Vector(4, f32)`, `@Type(...)`
        builtin_call: BuiltinCall,
        /// typeof: `@TypeOf(expr)`
        typeof: *Expr,
        /// Inferred: `_` or anytype
        inferred,
    };
};

pub const Pointer = struct {
    child: *TypeExpr,
    is_const: bool,
    alignment: ?*Expr,
    is_volatile: bool,
    is_allowzero: bool,
    sentinel: ?*Expr,
};

pub const Array = struct {
    len: *Expr,
    child: *TypeExpr,
    sentinel: ?*Expr,
};

pub const Slice = struct {
    child: *TypeExpr,
    is_const: bool,
    sentinel: ?*Expr,
};

pub const Sentinel = struct {
    child: *TypeExpr,
    sentinel_val: *Expr,
};

pub const ErrorUnion = struct {
    error_type: ?*TypeExpr,
    payload: *TypeExpr,
};

pub const FnType = struct {
    params: []const *TypeExpr,
    return_type: *TypeExpr,
};

pub const BuiltinCall = struct {
    name: token.Loc,
    args: []const *TypeExpr,
};

// =========================================================================
// Expressions
// =========================================================================

/// Expression node.
pub const Expr = struct {
    kind: Kind,
    loc: Loc,

    pub const Kind = union(enum) {
        // Literals
        integer_literal: token.Loc,
        float_literal: token.Loc,
        string_literal: token.Loc,
        char_literal: token.Loc,
        bool_literal: bool,
        null_literal,
        undefined_literal,
        unreachable_literal,

        // References
        identifier: token.Loc,
        /// @import, @intCast, etc.
        builtin_call: BuiltinCallExpr,

        // Operators
        unary: UnaryExpr,
        binary: BinaryExpr,
        /// a orelse b
        orelse_expr: BinaryPair,
        /// a catch |e| b
        catch_expr: CatchExpr,

        // Access
        field_access: FieldAccess,
        index: IndexExpr,
        /// ptr.*
        deref,
        /// opt.?
        unwrap_optional,
        /// a[b..c]
        slice_expr: SliceExpr,

        // Calls
        fn_call: FnCallExpr,

        // Control flow expressions
        if_expr: IfExpr,
        switch_expr: SwitchExpr,
        block_expr: *Block,

        // Try
        try_expr: *Expr,

        // Compound literals
        struct_init: StructInit,
        array_init: ArrayInit,

        // Comptime
        comptime_expr: *Expr,

        // Error
        error_value: token.Loc,
        error_union: ErrorUnionExpr,

        // Grouping
        grouped: *Expr,
    };
};

pub const UnaryExpr = struct {
    op: UnaryOp,
    operand: *Expr,
};

pub const UnaryOp = enum {
    negate, // -x
    bool_not, // !x
    bit_not, // ~x
    address_of, // &x
    ptr_deref, // x.*

    pub fn symbol(self: UnaryOp) []const u8 {
        return switch (self) {
            .negate => "-",
            .bool_not => "!",
            .bit_not => "~",
            .address_of => "&",
            .ptr_deref => ".*",
        };
    }
};

pub const BinaryExpr = struct {
    op: BinaryOp,
    lhs: *Expr,
    rhs: *Expr,
};

pub const BinaryOp = enum {
    // Arithmetic (precedence 10-11)
    add,
    sub,
    mul,
    div,
    mod,

    // Comparison (precedence 7-8)
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    // Logical (precedence 1-3)
    bool_and,
    bool_or,

    // Bitwise (precedence 4-6, 9)
    bit_and,
    bit_or,
    bit_xor,
    shl,
    shr,

    // Compound assignment
    add_assign,
    sub_assign,
    mul_assign,
    div_assign,
    mod_assign,
    bit_and_assign,
    bit_or_assign,
    bit_xor_assign,
    shl_assign,
    shr_assign,

    // Assignment
    assign,

    /// Operator precedence for Pratt parsing (higher = tighter binding).
    pub fn precedence(self: BinaryOp) u8 {
        return switch (self) {
            .assign, .add_assign, .sub_assign, .mul_assign, .div_assign, .mod_assign, .bit_and_assign, .bit_or_assign, .bit_xor_assign, .shl_assign, .shr_assign => 1,
            .bool_or => 2,
            .bool_and => 3,
            .bit_or => 4,
            .bit_xor => 5,
            .bit_and => 6,
            .eq, .ne => 7,
            .lt, .le, .gt, .ge => 8,
            .shl, .shr => 9,
            .add, .sub => 10,
            .mul, .div, .mod => 11,
        };
    }

    /// Whether this operator is right-associative.
    pub fn isRightAssociative(self: BinaryOp) bool {
        return switch (self) {
            .assign, .add_assign, .sub_assign, .mul_assign, .div_assign, .mod_assign, .bit_and_assign, .bit_or_assign, .bit_xor_assign, .shl_assign, .shr_assign => true,
            else => false,
        };
    }

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
            .bool_and => "and",
            .bool_or => "or",
            .bit_and => "&",
            .bit_or => "|",
            .bit_xor => "^",
            .shl => "<<",
            .shr => ">>",
            .assign => "=",
            .add_assign => "+=",
            .sub_assign => "-=",
            .mul_assign => "*=",
            .div_assign => "/=",
            .mod_assign => "%=",
            .bit_and_assign => "&=",
            .bit_or_assign => "|=",
            .bit_xor_assign => "^=",
            .shl_assign => "<<=",
            .shr_assign => ">>=",
        };
    }
};

pub const BinaryPair = struct {
    lhs: *Expr,
    rhs: *Expr,
};

pub const CatchExpr = struct {
    lhs: *Expr,
    capture: ?token.Loc,
    rhs: *Expr,
};

pub const FieldAccess = struct {
    object: *Expr,
    field: token.Loc,
};

pub const IndexExpr = struct {
    object: *Expr,
    index: *Expr,
};

pub const SliceExpr = struct {
    object: *Expr,
    start: ?*Expr,
    end: ?*Expr,
    sentinel: ?*Expr,
};

pub const FnCallExpr = struct {
    callee: *Expr,
    args: []const *Expr,
};

pub const BuiltinCallExpr = struct {
    name: token.Loc,
    args: []const *Expr,
};

pub const IfExpr = struct {
    condition: *Expr,
    capture: ?token.Loc,
    then_expr: *Expr,
    else_expr: ?*Expr,
};

pub const SwitchExpr = struct {
    operand: *Expr,
    cases: []const SwitchCase,
};

pub const SwitchCase = struct {
    values: []const *Expr, // empty = else prong
    capture: ?token.Loc,
    body: *Expr,
    is_inline: bool,
};

pub const StructInit = struct {
    type_expr: ?*TypeExpr,
    fields: []const StructInitField,
};

pub const StructInitField = struct {
    name: token.Loc,
    value: *Expr,
};

pub const ArrayInit = struct {
    type_expr: ?*TypeExpr,
    elements: []const *Expr,
};

pub const ErrorUnionExpr = struct {
    error_set: *Expr,
    payload: *Expr,
};

// =========================================================================
// Statements
// =========================================================================

/// Statement node.
pub const Stmt = struct {
    kind: Kind,
    loc: Loc,

    pub const Kind = union(enum) {
        var_decl: VarDecl,
        assign: AssignStmt,
        expr_stmt: *Expr,
        return_stmt: ?*Expr,
        break_stmt: BreakStmt,
        continue_stmt: ?token.Loc, // optional label
        if_stmt: IfStmt,
        while_stmt: WhileStmt,
        for_stmt: ForStmt,
        switch_stmt: SwitchStmt,
        block: *Block,
        defer_stmt: *Stmt,
        errdefer_stmt: ErrdeferStmt,
        comptime_stmt: *Stmt,
        nosuspend_stmt: *Stmt,
        labeled: LabeledStmt,
    };
};

pub const AssignStmt = struct {
    target: *Expr,
    op: BinaryOp,
    value: *Expr,
};

pub const BreakStmt = struct {
    label: ?token.Loc,
    value: ?*Expr,
};

pub const IfStmt = struct {
    condition: *Expr,
    capture: ?token.Loc,
    then_body: *Block,
    else_body: ?ElseBody,

    pub const ElseBody = union(enum) {
        else_block: *Block,
        else_if: *Stmt, // points to another if_stmt
    };
};

pub const WhileStmt = struct {
    condition: *Expr,
    capture: ?token.Loc,
    body: *Block,
    else_body: ?*Block,
    continue_expr: ?*Expr,
};

pub const ForStmt = struct {
    iterables: []const ForIterable,
    body: *Block,
    else_body: ?*Block,
};

pub const ForIterable = struct {
    range_or_slice: *Expr,
    capture: ?token.Loc,
};

pub const SwitchStmt = struct {
    operand: *Expr,
    cases: []const SwitchCase,
};

pub const ErrdeferStmt = struct {
    capture: ?token.Loc,
    body: *Stmt,
};

pub const LabeledStmt = struct {
    label: token.Loc,
    stmt: *Stmt,
};

/// A block of statements.
pub const Block = struct {
    stmts: []const Stmt,
    loc: Loc,
};

// =========================================================================
// Tests
// =========================================================================

test "BinaryOp precedence ordering" {
    const testing = std.testing;
    // Multiplication binds tighter than addition
    try testing.expect(BinaryOp.mul.precedence() > BinaryOp.add.precedence());
    // Addition binds tighter than comparison
    try testing.expect(BinaryOp.add.precedence() > BinaryOp.eq.precedence());
    // Comparison binds tighter than logical and
    try testing.expect(BinaryOp.eq.precedence() > BinaryOp.bool_and.precedence());
    // Logical and binds tighter than logical or
    try testing.expect(BinaryOp.bool_and.precedence() > BinaryOp.bool_or.precedence());
    // Assignment is lowest
    try testing.expect(BinaryOp.bool_or.precedence() > BinaryOp.assign.precedence());
}

test "BinaryOp assignment is right-associative" {
    const testing = std.testing;
    try testing.expect(BinaryOp.assign.isRightAssociative());
    try testing.expect(BinaryOp.add_assign.isRightAssociative());
    try testing.expect(!BinaryOp.add.isRightAssociative());
    try testing.expect(!BinaryOp.eq.isRightAssociative());
}

test "UnaryOp symbols" {
    const testing = std.testing;
    try testing.expectEqualStrings("-", UnaryOp.negate.symbol());
    try testing.expectEqualStrings("!", UnaryOp.bool_not.symbol());
    try testing.expectEqualStrings("~", UnaryOp.bit_not.symbol());
}

test "BinaryOp symbols" {
    const testing = std.testing;
    try testing.expectEqualStrings("+", BinaryOp.add.symbol());
    try testing.expectEqualStrings("==", BinaryOp.eq.symbol());
    try testing.expectEqualStrings("<<=", BinaryOp.shl_assign.symbol());
}
