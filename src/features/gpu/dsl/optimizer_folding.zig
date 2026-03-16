//! Constant folding optimization pass for the GPU Kernel DSL.
//!
//! Evaluates compile-time constant expressions, folding binary and unary
//! operations on literal values into their results.

const std = @import("std");
const expr = @import("expr.zig");
const stmt = @import("stmt.zig");
const kernel = @import("kernel.zig");

/// Run constant folding on kernel IR.
/// Increments `constants_folded` for each folded expression.
pub fn run(constants_folded: *u32, ir: *kernel.KernelIR) !void {
    for (ir.body) |s| {
        try foldConstantsInStmt(constants_folded, s);
    }

    for (ir.functions) |func| {
        for (func.body) |s| {
            try foldConstantsInStmt(constants_folded, s);
        }
    }
}

fn foldConstantsInStmt(constants_folded: *u32, s: *const stmt.Stmt) !void {
    switch (s.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                _ = try tryFoldExpr(constants_folded, init_expr);
            }
        },
        .assign => |assignment| {
            _ = try tryFoldExpr(constants_folded, assignment.value);
        },
        .compound_assign => |ca| {
            _ = try tryFoldExpr(constants_folded, ca.value);
        },
        .if_ => |if_stmt| {
            _ = try tryFoldExpr(constants_folded, if_stmt.condition);
            for (if_stmt.then_body) |body_stmt| {
                try foldConstantsInStmt(constants_folded, body_stmt);
            }
            if (if_stmt.else_body) |else_body| {
                for (else_body) |body_stmt| {
                    try foldConstantsInStmt(constants_folded, body_stmt);
                }
            }
        },
        .for_ => |for_stmt| {
            if (for_stmt.init) |init_s| {
                try foldConstantsInStmt(constants_folded, init_s);
            }
            if (for_stmt.condition) |cond| {
                _ = try tryFoldExpr(constants_folded, cond);
            }
            for (for_stmt.body) |body_stmt| {
                try foldConstantsInStmt(constants_folded, body_stmt);
            }
        },
        .while_ => |while_stmt| {
            _ = try tryFoldExpr(constants_folded, while_stmt.condition);
            for (while_stmt.body) |body_stmt| {
                try foldConstantsInStmt(constants_folded, body_stmt);
            }
        },
        .block => |blk| {
            for (blk.statements) |body_stmt| {
                try foldConstantsInStmt(constants_folded, body_stmt);
            }
        },
        .expr_stmt => |e| {
            _ = try tryFoldExpr(constants_folded, e);
        },
        .return_ => |ret| {
            if (ret.value) |v| {
                _ = try tryFoldExpr(constants_folded, v);
            }
        },
        else => {},
    }
}

fn tryFoldExpr(constants_folded: *u32, e: *const expr.Expr) !?expr.Literal {
    switch (e.*) {
        .literal => |lit| return lit,
        .binary => |bin| {
            const left_lit = try tryFoldExpr(constants_folded, bin.left);
            const right_lit = try tryFoldExpr(constants_folded, bin.right);

            if (left_lit != null and right_lit != null) {
                if (try foldBinaryOp(bin.op, left_lit.?, right_lit.?)) |result| {
                    constants_folded.* += 1;
                    return result;
                }
            }
            return null;
        },
        .unary => |un| {
            const operand_lit = try tryFoldExpr(constants_folded, un.operand);
            if (operand_lit) |lit| {
                if (try foldUnaryOp(un.op, lit)) |result| {
                    constants_folded.* += 1;
                    return result;
                }
            }
            return null;
        },
        else => return null,
    }
}

/// Fold a binary operation on two literal values.
pub fn foldBinaryOp(op: expr.BinaryOp, left: expr.Literal, right: expr.Literal) !?expr.Literal {
    // f32 operations
    if (left == .f32_ and right == .f32_) {
        const l = left.f32_;
        const r = right.f32_;
        return switch (op) {
            .add => expr.Literal{ .f32_ = l + r },
            .sub => expr.Literal{ .f32_ = l - r },
            .mul => expr.Literal{ .f32_ = l * r },
            .div => if (r != 0) expr.Literal{ .f32_ = l / r } else null,
            .min => expr.Literal{ .f32_ = @min(l, r) },
            .max => expr.Literal{ .f32_ = @max(l, r) },
            else => null,
        };
    }

    // i32 operations
    if (left == .i32_ and right == .i32_) {
        const l = left.i32_;
        const r = right.i32_;
        return switch (op) {
            .add => expr.Literal{ .i32_ = l +% r },
            .sub => expr.Literal{ .i32_ = l -% r },
            .mul => expr.Literal{ .i32_ = l *% r },
            .div => if (r != 0) expr.Literal{ .i32_ = @divTrunc(l, r) } else null,
            .mod => if (r != 0) expr.Literal{ .i32_ = @rem(l, r) } else null,
            .eq => expr.Literal{ .bool_ = l == r },
            .ne => expr.Literal{ .bool_ = l != r },
            .lt => expr.Literal{ .bool_ = l < r },
            .le => expr.Literal{ .bool_ = l <= r },
            .gt => expr.Literal{ .bool_ = l > r },
            .ge => expr.Literal{ .bool_ = l >= r },
            .bit_and => expr.Literal{ .i32_ = l & r },
            .bit_or => expr.Literal{ .i32_ = l | r },
            .bit_xor => expr.Literal{ .i32_ = l ^ r },
            else => null,
        };
    }

    // u32 operations
    if (left == .u32_ and right == .u32_) {
        const l = left.u32_;
        const r = right.u32_;
        return switch (op) {
            .add => expr.Literal{ .u32_ = l +% r },
            .sub => expr.Literal{ .u32_ = l -% r },
            .mul => expr.Literal{ .u32_ = l *% r },
            .div => if (r != 0) expr.Literal{ .u32_ = l / r } else null,
            .mod => if (r != 0) expr.Literal{ .u32_ = l % r } else null,
            .eq => expr.Literal{ .bool_ = l == r },
            .ne => expr.Literal{ .bool_ = l != r },
            .lt => expr.Literal{ .bool_ = l < r },
            .le => expr.Literal{ .bool_ = l <= r },
            .gt => expr.Literal{ .bool_ = l > r },
            .ge => expr.Literal{ .bool_ = l >= r },
            .bit_and => expr.Literal{ .u32_ = l & r },
            .bit_or => expr.Literal{ .u32_ = l | r },
            .bit_xor => expr.Literal{ .u32_ = l ^ r },
            else => null,
        };
    }

    // Boolean operations
    if (left == .bool_ and right == .bool_) {
        const l = left.bool_;
        const r = right.bool_;
        return switch (op) {
            .and_ => expr.Literal{ .bool_ = l and r },
            .or_ => expr.Literal{ .bool_ = l or r },
            .xor => expr.Literal{ .bool_ = l != r },
            .eq => expr.Literal{ .bool_ = l == r },
            .ne => expr.Literal{ .bool_ = l != r },
            else => null,
        };
    }

    return null;
}

/// Fold a unary operation on a literal value.
pub fn foldUnaryOp(op: expr.UnaryOp, operand: expr.Literal) !?expr.Literal {
    switch (operand) {
        .f32_ => |v| {
            return switch (op) {
                .neg => expr.Literal{ .f32_ = -v },
                .abs => expr.Literal{ .f32_ = @abs(v) },
                .sqrt => if (v >= 0) expr.Literal{ .f32_ = @sqrt(v) } else null,
                .floor => expr.Literal{ .f32_ = @floor(v) },
                .ceil => expr.Literal{ .f32_ = @ceil(v) },
                .round => expr.Literal{ .f32_ = @round(v) },
                .trunc => expr.Literal{ .f32_ = @trunc(v) },
                .sign => expr.Literal{ .f32_ = if (v > 0) 1.0 else if (v < 0) -1.0 else 0.0 },
                else => null,
            };
        },
        .i32_ => |v| {
            return switch (op) {
                .neg => expr.Literal{ .i32_ = -%v },
                .abs => expr.Literal{ .i32_ = if (v < 0) -v else v },
                .bit_not => expr.Literal{ .i32_ = ~v },
                else => null,
            };
        },
        .u32_ => |v| {
            return switch (op) {
                .bit_not => expr.Literal{ .u32_ = ~v },
                else => null,
            };
        },
        .bool_ => |v| {
            return switch (op) {
                .not => expr.Literal{ .bool_ = !v },
                else => null,
            };
        },
        else => return null,
    }
}

// ============================================================================
// Tests
// ============================================================================

test "constant folding binary ops" {
    // f32 addition
    const result1 = try foldBinaryOp(.add, .{ .f32_ = 1.5 }, .{ .f32_ = 2.5 });
    try std.testing.expect(result1 != null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result1.?.f32_, 0.001);

    // i32 multiplication
    const result2 = try foldBinaryOp(.mul, .{ .i32_ = 6 }, .{ .i32_ = 7 });
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(i32, 42), result2.?.i32_);

    // Boolean and
    const result3 = try foldBinaryOp(.and_, .{ .bool_ = true }, .{ .bool_ = false });
    try std.testing.expect(result3 != null);
    try std.testing.expectEqual(false, result3.?.bool_);
}

test "constant folding unary ops" {
    // f32 negation
    const result1 = try foldUnaryOp(.neg, .{ .f32_ = 3.14 });
    try std.testing.expect(result1 != null);
    try std.testing.expectApproxEqAbs(@as(f32, -3.14), result1.?.f32_, 0.001);

    // i32 absolute value
    const result2 = try foldUnaryOp(.abs, .{ .i32_ = -42 });
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(i32, 42), result2.?.i32_);

    // Boolean not
    const result3 = try foldUnaryOp(.not, .{ .bool_ = true });
    try std.testing.expect(result3 != null);
    try std.testing.expectEqual(false, result3.?.bool_);
}

test {
    std.testing.refAllDecls(@This());
}
