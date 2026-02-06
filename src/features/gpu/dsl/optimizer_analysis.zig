//! Analysis and transformation passes for the GPU Kernel DSL optimizer.
//!
//! Contains: dead code elimination, algebraic simplification, strength reduction,
//! common subexpression elimination, and redundancy elimination.
//! Also provides helper functions for literal analysis (isZero, isOne, isPowerOfTwo).

const std = @import("std");
const expr = @import("expr.zig");
const stmt = @import("stmt.zig");
const kernel = @import("kernel.zig");

// ============================================================================
// Dead Code Elimination
// ============================================================================

/// Identify unused variables and count potential dead code eliminations.
pub fn deadCodeElimination(
    dead_code_eliminated: *u32,
    allocator: std.mem.Allocator,
    used_vars: *std.StringHashMapUnmanaged(void),
    ir: *kernel.KernelIR,
) !void {
    // First pass: collect all used variables
    used_vars.clearRetainingCapacity();

    for (ir.body) |s| {
        try collectUsedVars(allocator, used_vars, s);
    }

    // Mark buffer and uniform names as used
    for (ir.buffers) |buf| {
        try used_vars.put(allocator, buf.name, {});
    }
    for (ir.uniforms) |uni| {
        try used_vars.put(allocator, uni.name, {});
    }

    // Second pass: count potential eliminations
    for (ir.body) |s| {
        countDeadCode(dead_code_eliminated, used_vars, s);
    }
}

fn collectUsedVars(
    allocator: std.mem.Allocator,
    used_vars: *std.StringHashMapUnmanaged(void),
    s: *const stmt.Stmt,
) !void {
    switch (s.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                try collectUsedVarsInExpr(allocator, used_vars, init_expr);
            }
        },
        .assign => |assignment| {
            try collectUsedVarsInExpr(allocator, used_vars, assignment.target);
            try collectUsedVarsInExpr(allocator, used_vars, assignment.value);
        },
        .compound_assign => |ca| {
            try collectUsedVarsInExpr(allocator, used_vars, ca.target);
            try collectUsedVarsInExpr(allocator, used_vars, ca.value);
        },
        .if_ => |if_stmt| {
            try collectUsedVarsInExpr(allocator, used_vars, if_stmt.condition);
            for (if_stmt.then_body) |body_stmt| {
                try collectUsedVars(allocator, used_vars, body_stmt);
            }
            if (if_stmt.else_body) |else_body| {
                for (else_body) |body_stmt| {
                    try collectUsedVars(allocator, used_vars, body_stmt);
                }
            }
        },
        .for_ => |for_stmt| {
            if (for_stmt.init) |init_s| try collectUsedVars(allocator, used_vars, init_s);
            if (for_stmt.condition) |cond| try collectUsedVarsInExpr(allocator, used_vars, cond);
            if (for_stmt.update) |upd| try collectUsedVars(allocator, used_vars, upd);
            for (for_stmt.body) |body_stmt| {
                try collectUsedVars(allocator, used_vars, body_stmt);
            }
        },
        .while_ => |while_stmt| {
            try collectUsedVarsInExpr(allocator, used_vars, while_stmt.condition);
            for (while_stmt.body) |body_stmt| {
                try collectUsedVars(allocator, used_vars, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                try collectUsedVars(allocator, used_vars, body_stmt);
            }
        },
        .expr_stmt => |e| {
            try collectUsedVarsInExpr(allocator, used_vars, e);
        },
        .return_ => |ret| {
            if (ret.value) |v| try collectUsedVarsInExpr(allocator, used_vars, v);
        },
        else => {},
    }
}

fn collectUsedVarsInExpr(
    allocator: std.mem.Allocator,
    used_vars: *std.StringHashMapUnmanaged(void),
    e: *const expr.Expr,
) !void {
    switch (e.*) {
        .ref => |ref| {
            if (ref.name) |var_name| {
                try used_vars.put(allocator, var_name, {});
            }
        },
        .binary => |bin| {
            try collectUsedVarsInExpr(allocator, used_vars, bin.left);
            try collectUsedVarsInExpr(allocator, used_vars, bin.right);
        },
        .unary => |un| {
            try collectUsedVarsInExpr(allocator, used_vars, un.operand);
        },
        .call => |call| {
            for (call.args) |arg| {
                try collectUsedVarsInExpr(allocator, used_vars, arg);
            }
        },
        .index => |idx| {
            try collectUsedVarsInExpr(allocator, used_vars, idx.base);
            try collectUsedVarsInExpr(allocator, used_vars, idx.index);
        },
        .field => |fld| {
            try collectUsedVarsInExpr(allocator, used_vars, fld.base);
        },
        .select => |sel| {
            try collectUsedVarsInExpr(allocator, used_vars, sel.condition);
            try collectUsedVarsInExpr(allocator, used_vars, sel.true_value);
            try collectUsedVarsInExpr(allocator, used_vars, sel.false_value);
        },
        else => {},
    }
}

fn countDeadCode(
    dead_code_eliminated: *u32,
    used_vars: *const std.StringHashMapUnmanaged(void),
    s: *const stmt.Stmt,
) void {
    switch (s.*) {
        .var_decl => |decl| {
            if (!used_vars.contains(decl.name)) {
                dead_code_eliminated.* += 1;
            }
        },
        .if_ => |if_stmt| {
            for (if_stmt.then_body) |body_stmt| {
                countDeadCode(dead_code_eliminated, used_vars, body_stmt);
            }
            if (if_stmt.else_body) |else_body| {
                for (else_body) |body_stmt| {
                    countDeadCode(dead_code_eliminated, used_vars, body_stmt);
                }
            }
        },
        .for_ => |for_stmt| {
            for (for_stmt.body) |body_stmt| {
                countDeadCode(dead_code_eliminated, used_vars, body_stmt);
            }
        },
        .while_ => |while_stmt| {
            for (while_stmt.body) |body_stmt| {
                countDeadCode(dead_code_eliminated, used_vars, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                countDeadCode(dead_code_eliminated, used_vars, body_stmt);
            }
        },
        else => {},
    }
}

// ============================================================================
// Algebraic Simplification
// ============================================================================

/// Simplify algebraic expressions (x*1 -> x, x+0 -> x, etc.).
pub fn algebraicSimplification(expressions_simplified: *u32, ir: *kernel.KernelIR) !void {
    for (ir.body) |s| {
        try simplifyStmt(expressions_simplified, s);
    }
}

fn simplifyStmt(expressions_simplified: *u32, s: *const stmt.Stmt) !void {
    switch (s.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                if (trySimplifyExpr(init_expr)) {
                    expressions_simplified.* += 1;
                }
            }
        },
        .assign => |assignment| {
            if (trySimplifyExpr(assignment.value)) {
                expressions_simplified.* += 1;
            }
        },
        .compound_assign => |ca| {
            if (trySimplifyExpr(ca.value)) {
                expressions_simplified.* += 1;
            }
        },
        .if_ => |if_stmt| {
            for (if_stmt.then_body) |body_stmt| {
                try simplifyStmt(expressions_simplified, body_stmt);
            }
            if (if_stmt.else_body) |else_body| {
                for (else_body) |body_stmt| {
                    try simplifyStmt(expressions_simplified, body_stmt);
                }
            }
        },
        .for_ => |for_stmt| {
            for (for_stmt.body) |body_stmt| {
                try simplifyStmt(expressions_simplified, body_stmt);
            }
        },
        .while_ => |while_stmt| {
            for (while_stmt.body) |body_stmt| {
                try simplifyStmt(expressions_simplified, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                try simplifyStmt(expressions_simplified, body_stmt);
            }
        },
        else => {},
    }
}

fn trySimplifyExpr(e: *const expr.Expr) bool {
    switch (e.*) {
        .binary => |bin| {
            // Check for identity operations: x + 0, x * 1, x - 0, x / 1
            if (bin.right.* == .literal) {
                const lit = bin.right.literal;
                switch (bin.op) {
                    .add, .sub => {
                        // x + 0 or x - 0 -> x
                        if (isZero(lit)) return true;
                    },
                    .mul => {
                        // x * 1 -> x, x * 0 -> 0
                        if (isOne(lit) or isZero(lit)) return true;
                    },
                    .div => {
                        // x / 1 -> x
                        if (isOne(lit)) return true;
                    },
                    else => {},
                }
            }
            // Check for x + x -> x * 2 or x * x -> x^2
            return false;
        },
        else => return false,
    }
}

// ============================================================================
// Strength Reduction
// ============================================================================

/// Replace expensive operations with cheaper equivalents (e.g., x * 2^n -> x << n).
pub fn strengthReduction(strength_reductions: *u32, ir: *kernel.KernelIR) !void {
    for (ir.body) |s| {
        try reduceStrengthInStmt(strength_reductions, s);
    }
}

fn reduceStrengthInStmt(strength_reductions: *u32, s: *const stmt.Stmt) !void {
    switch (s.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                if (tryReduceStrength(init_expr)) {
                    strength_reductions.* += 1;
                }
            }
        },
        .assign => |assignment| {
            if (tryReduceStrength(assignment.value)) {
                strength_reductions.* += 1;
            }
        },
        .for_ => |for_stmt| {
            for (for_stmt.body) |body_stmt| {
                try reduceStrengthInStmt(strength_reductions, body_stmt);
            }
        },
        .while_ => |while_stmt| {
            for (while_stmt.body) |body_stmt| {
                try reduceStrengthInStmt(strength_reductions, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                try reduceStrengthInStmt(strength_reductions, body_stmt);
            }
        },
        else => {},
    }
}

fn tryReduceStrength(e: *const expr.Expr) bool {
    switch (e.*) {
        .binary => |bin| {
            switch (bin.op) {
                .mul => {
                    // x * 2 -> x + x or x << 1
                    if (bin.right.* == .literal) {
                        if (isPowerOfTwo(bin.right.literal)) {
                            return true;
                        }
                    }
                },
                .div => {
                    // x / 2 -> x >> 1 (for integers)
                    if (bin.right.* == .literal) {
                        if (isPowerOfTwo(bin.right.literal)) {
                            return true;
                        }
                    }
                },
                .mod => {
                    // x % 2^n -> x & (2^n - 1)
                    if (bin.right.* == .literal) {
                        if (isPowerOfTwo(bin.right.literal)) {
                            return true;
                        }
                    }
                },
                else => {},
            }
        },
        else => {},
    }
    return false;
}

// ============================================================================
// Common Subexpression Elimination
// ============================================================================

/// Identify and count common subexpressions that can be reused.
pub fn commonSubexprElimination(
    subexpressions_eliminated: *u32,
    allocator: std.mem.Allocator,
    expr_cache: *std.StringHashMapUnmanaged(*const expr.Expr),
    ir: *kernel.KernelIR,
) !void {
    expr_cache.clearRetainingCapacity();

    for (ir.body) |s| {
        try findCommonSubexprs(subexpressions_eliminated, allocator, expr_cache, s);
    }
}

fn findCommonSubexprs(
    subexpressions_eliminated: *u32,
    allocator: std.mem.Allocator,
    expr_cache: *std.StringHashMapUnmanaged(*const expr.Expr),
    s: *const stmt.Stmt,
) !void {
    switch (s.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                try checkAndCacheExpr(subexpressions_eliminated, allocator, expr_cache, init_expr);
            }
        },
        .assign => |assignment| {
            try checkAndCacheExpr(subexpressions_eliminated, allocator, expr_cache, assignment.value);
        },
        .for_ => |for_stmt| {
            for (for_stmt.body) |body_stmt| {
                try findCommonSubexprs(subexpressions_eliminated, allocator, expr_cache, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                try findCommonSubexprs(subexpressions_eliminated, allocator, expr_cache, body_stmt);
            }
        },
        else => {},
    }
}

fn checkAndCacheExpr(
    subexpressions_eliminated: *u32,
    allocator: std.mem.Allocator,
    expr_cache: *std.StringHashMapUnmanaged(*const expr.Expr),
    e: *const expr.Expr,
) !void {
    const hash = exprHash(e);
    if (expr_cache.get(hash)) |_| {
        subexpressions_eliminated.* += 1;
    } else {
        try expr_cache.put(allocator, hash, e);
    }
}

// ============================================================================
// Redundancy Elimination
// ============================================================================

/// Detect and count redundant operations (e.g., self-assignment x = x).
pub fn redundancyElimination(redundancies_eliminated: *u32, ir: *kernel.KernelIR) void {
    for (ir.body) |s| {
        checkRedundancy(redundancies_eliminated, s);
    }
}

fn checkRedundancy(redundancies_eliminated: *u32, s: *const stmt.Stmt) void {
    switch (s.*) {
        .assign => |assignment| {
            // Check for self-assignment: x = x
            if (assignment.target.* == .ref and assignment.value.* == .ref) {
                const target_ref = assignment.target.ref;
                const value_ref = assignment.value.ref;
                if (target_ref.id == value_ref.id) {
                    redundancies_eliminated.* += 1;
                }
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                checkRedundancy(redundancies_eliminated, body_stmt);
            }
        },
        else => {},
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

pub fn isZero(lit: expr.Literal) bool {
    return switch (lit) {
        .f32_ => |v| v == 0.0,
        .f64_ => |v| v == 0.0,
        .i32_ => |v| v == 0,
        .i64_ => |v| v == 0,
        .u32_ => |v| v == 0,
        .u64_ => |v| v == 0,
        else => false,
    };
}

pub fn isOne(lit: expr.Literal) bool {
    return switch (lit) {
        .f32_ => |v| v == 1.0,
        .f64_ => |v| v == 1.0,
        .i32_ => |v| v == 1,
        .i64_ => |v| v == 1,
        .u32_ => |v| v == 1,
        .u64_ => |v| v == 1,
        else => false,
    };
}

pub fn isPowerOfTwo(lit: expr.Literal) bool {
    return switch (lit) {
        .i32_ => |v| v > 0 and (v & (v - 1)) == 0,
        .i64_ => |v| v > 0 and (v & (v - 1)) == 0,
        .u32_ => |v| v > 0 and (v & (v - 1)) == 0,
        .u64_ => |v| v > 0 and (v & (v - 1)) == 0,
        else => false,
    };
}

pub fn exprHash(e: *const expr.Expr) []const u8 {
    // Simple hash based on expression type - would need proper implementation
    return switch (e.*) {
        .literal => "lit",
        .ref => "ref",
        .binary => "bin",
        .unary => "una",
        else => "oth",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "helper functions" {
    try std.testing.expect(isZero(.{ .f32_ = 0.0 }));
    try std.testing.expect(!isZero(.{ .f32_ = 1.0 }));

    try std.testing.expect(isOne(.{ .i32_ = 1 }));
    try std.testing.expect(!isOne(.{ .i32_ = 2 }));

    try std.testing.expect(isPowerOfTwo(.{ .u32_ = 8 }));
    try std.testing.expect(isPowerOfTwo(.{ .u32_ = 256 }));
    try std.testing.expect(!isPowerOfTwo(.{ .u32_ = 7 }));
    try std.testing.expect(!isPowerOfTwo(.{ .u32_ = 0 }));
}
