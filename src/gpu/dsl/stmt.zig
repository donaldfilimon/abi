//! GPU Kernel DSL Statement AST
//!
//! Defines the statement nodes for the kernel IR abstract syntax tree.
//! Statements represent actions that don't produce values.

const std = @import("std");
const types = @import("types.zig");
const expr = @import("expr.zig");

/// Statement node.
pub const Stmt = union(enum) {
    /// Variable declaration.
    var_decl: VarDecl,

    /// Assignment statement.
    assign: Assignment,

    /// Compound assignment (+=, -=, *=, etc.).
    compound_assign: CompoundAssign,

    /// If statement.
    if_: IfStmt,

    /// For loop.
    for_: ForStmt,

    /// While loop.
    while_: WhileStmt,

    /// Do-while loop.
    do_while: DoWhileStmt,

    /// Return statement.
    return_: Return,

    /// Break statement.
    break_: void,

    /// Continue statement.
    continue_: void,

    /// Discard statement (for fragment shaders, no-op in compute).
    discard: void,

    /// Expression statement (for side effects like atomics, barriers).
    expr_stmt: *const expr.Expr,

    /// Block (sequence of statements).
    block: Block,

    /// Switch statement.
    switch_: SwitchStmt,

    pub const VarDecl = struct {
        /// Variable name.
        name: []const u8,
        /// Variable type.
        ty: types.Type,
        /// Initial value (optional).
        init: ?*const expr.Expr,
        /// Address space (private by default).
        address_space: types.AddressSpace,
        /// Whether the variable is const.
        is_const: bool = false,
    };

    pub const Assignment = struct {
        /// Target lvalue expression.
        target: *const expr.Expr,
        /// Value to assign.
        value: *const expr.Expr,
    };

    pub const CompoundAssign = struct {
        /// Target lvalue expression.
        target: *const expr.Expr,
        /// Operator (add, sub, mul, div, mod, bit_and, bit_or, bit_xor, shl, shr).
        op: expr.BinaryOp,
        /// Value for the operation.
        value: *const expr.Expr,
    };

    pub const IfStmt = struct {
        /// Condition expression (must be bool).
        condition: *const expr.Expr,
        /// Statements to execute if condition is true.
        then_body: []const *const Stmt,
        /// Statements to execute if condition is false (optional).
        else_body: ?[]const *const Stmt,
    };

    pub const ForStmt = struct {
        /// Initialization statement (optional).
        init: ?*const Stmt,
        /// Loop condition (optional, defaults to true).
        condition: ?*const expr.Expr,
        /// Update statement executed at end of each iteration (optional).
        update: ?*const Stmt,
        /// Loop body statements.
        body: []const *const Stmt,
    };

    pub const WhileStmt = struct {
        /// Loop condition.
        condition: *const expr.Expr,
        /// Loop body statements.
        body: []const *const Stmt,
    };

    pub const DoWhileStmt = struct {
        /// Loop body statements.
        body: []const *const Stmt,
        /// Loop condition (checked after body execution).
        condition: *const expr.Expr,
    };

    pub const Return = struct {
        /// Return value (optional for void functions).
        value: ?*const expr.Expr,
    };

    pub const Block = struct {
        /// Statements in the block.
        statements: []const *const Stmt,
    };

    pub const SwitchStmt = struct {
        /// Expression to switch on.
        selector: *const expr.Expr,
        /// Switch cases.
        cases: []const SwitchCase,
        /// Default case body (optional).
        default: ?[]const *const Stmt,
    };

    pub const SwitchCase = struct {
        /// Case value (must be compile-time constant).
        value: expr.Literal,
        /// Case body statements.
        body: []const *const Stmt,
        /// Whether to fall through to next case.
        fallthrough: bool = false,
    };
};

/// Create a variable declaration statement.
pub fn varDecl(
    allocator: std.mem.Allocator,
    var_name: []const u8,
    ty: types.Type,
    init_expr: ?*const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .var_decl = .{
            .name = var_name,
            .ty = ty,
            .init = init_expr,
            .address_space = .private,
            .is_const = false,
        },
    };
    return s;
}

/// Create a const variable declaration statement.
pub fn constDecl(
    allocator: std.mem.Allocator,
    var_name: []const u8,
    ty: types.Type,
    init_expr: *const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .var_decl = .{
            .name = var_name,
            .ty = ty,
            .init = init_expr,
            .address_space = .private,
            .is_const = true,
        },
    };
    return s;
}

/// Create an assignment statement.
pub fn assign(
    allocator: std.mem.Allocator,
    target: *const expr.Expr,
    value: *const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .assign = .{
            .target = target,
            .value = value,
        },
    };
    return s;
}

/// Create a compound assignment statement (+=, -=, etc.).
pub fn compoundAssign(
    allocator: std.mem.Allocator,
    target: *const expr.Expr,
    op: expr.BinaryOp,
    value: *const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .compound_assign = .{
            .target = target,
            .op = op,
            .value = value,
        },
    };
    return s;
}

/// Create an if statement.
pub fn ifStmt(
    allocator: std.mem.Allocator,
    condition: *const expr.Expr,
    then_body: []const *const Stmt,
    else_body: ?[]const *const Stmt,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .if_ = .{
            .condition = condition,
            .then_body = then_body,
            .else_body = else_body,
        },
    };
    return s;
}

/// Create a for loop statement.
pub fn forLoop(
    allocator: std.mem.Allocator,
    init_stmt: ?*const Stmt,
    condition: ?*const expr.Expr,
    update: ?*const Stmt,
    body: []const *const Stmt,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .for_ = .{
            .init = init_stmt,
            .condition = condition,
            .update = update,
            .body = body,
        },
    };
    return s;
}

/// Create a while loop statement.
pub fn whileLoop(
    allocator: std.mem.Allocator,
    condition: *const expr.Expr,
    body: []const *const Stmt,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .while_ = .{
            .condition = condition,
            .body = body,
        },
    };
    return s;
}

/// Create a return statement.
pub fn returnStmt(
    allocator: std.mem.Allocator,
    value: ?*const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .return_ = .{
            .value = value,
        },
    };
    return s;
}

/// Create a break statement.
pub fn breakStmt(allocator: std.mem.Allocator) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{ .break_ = {} };
    return s;
}

/// Create a continue statement.
pub fn continueStmt(allocator: std.mem.Allocator) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{ .continue_ = {} };
    return s;
}

/// Create an expression statement (for side effects).
pub fn exprStmt(
    allocator: std.mem.Allocator,
    expression: *const expr.Expr,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{ .expr_stmt = expression };
    return s;
}

/// Create a block statement.
pub fn block(
    allocator: std.mem.Allocator,
    statements: []const *const Stmt,
) !*const Stmt {
    const s = try allocator.create(Stmt);
    s.* = .{
        .block = .{
            .statements = statements,
        },
    };
    return s;
}

// ============================================================================
// Tests
// ============================================================================

test "varDecl creates variable declaration" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const decl = try varDecl(allocator, "x", types.Type.f32Type(), null);
    try std.testing.expect(decl.* == .var_decl);
    try std.testing.expectEqualStrings("x", decl.var_decl.name);
    try std.testing.expectEqual(types.Type.f32Type(), decl.var_decl.ty);
    try std.testing.expect(!decl.var_decl.is_const);
}

test "constDecl creates const declaration" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const lit = try allocator.create(expr.Expr);
    lit.* = .{ .literal = .{ .f32_ = 1.0 } };

    const decl = try constDecl(allocator, "PI", types.Type.f32Type(), lit);
    try std.testing.expect(decl.* == .var_decl);
    try std.testing.expect(decl.var_decl.is_const);
    try std.testing.expect(decl.var_decl.init != null);
}

test "ifStmt creates if statement" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const cond = try allocator.create(expr.Expr);
    cond.* = .{ .literal = .{ .bool_ = true } };

    const empty_body: []const *const Stmt = &.{};
    const stmt = try ifStmt(allocator, cond, empty_body, null);

    try std.testing.expect(stmt.* == .if_);
    try std.testing.expect(stmt.if_.else_body == null);
}
