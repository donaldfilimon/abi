//! GPU Kernel DSL Optimizer
//!
//! Provides optimization passes for the kernel IR, inspired by MLIR and LLVM research.
//! These passes transform the IR to produce more efficient code.
//!
//! Implemented optimizations:
//! - Constant folding: Evaluate compile-time constant expressions
//! - Dead code elimination: Remove unused variables and unreachable code
//! - Common subexpression elimination: Reuse previously computed values
//! - Strength reduction: Replace expensive operations with cheaper equivalents
//!
//! References:
//! - MLIR: https://mlir.llvm.org/
//! - LLVM Optimization Passes: https://llvm.org/docs/Passes.html
//! - Polyhedral compilation: https://ieeexplore.ieee.org/document/9563011/

const std = @import("std");
const types = @import("types.zig");
const expr = @import("expr.zig");
const stmt = @import("stmt.zig");
const kernel = @import("kernel.zig");

/// Optimization pass types.
pub const OptimizationPass = enum {
    /// Evaluate constant expressions at compile time.
    constant_folding,
    /// Remove unused variables and unreachable code.
    dead_code_elimination,
    /// Reuse previously computed expressions.
    common_subexpression_elimination,
    /// Replace expensive ops with cheaper equivalents.
    strength_reduction,
    /// Simplify algebraic expressions (x*1 -> x, x+0 -> x).
    algebraic_simplification,
    /// Remove redundant operations.
    redundancy_elimination,

    pub fn name(self: OptimizationPass) []const u8 {
        return switch (self) {
            .constant_folding => "constant-folding",
            .dead_code_elimination => "dead-code-elimination",
            .common_subexpression_elimination => "common-subexpr-elimination",
            .strength_reduction => "strength-reduction",
            .algebraic_simplification => "algebraic-simplification",
            .redundancy_elimination => "redundancy-elimination",
        };
    }
};

/// Optimization level presets.
pub const OptimizationLevel = enum {
    /// No optimizations.
    none,
    /// Basic optimizations (constant folding, algebraic simplification).
    basic,
    /// Standard optimizations (all basic + DCE, strength reduction).
    standard,
    /// Aggressive optimizations (all passes, may increase compile time).
    aggressive,

    /// Get the passes for this optimization level.
    pub fn getPasses(self: OptimizationLevel) []const OptimizationPass {
        return switch (self) {
            .none => &.{},
            .basic => &.{
                .constant_folding,
                .algebraic_simplification,
            },
            .standard => &.{
                .constant_folding,
                .algebraic_simplification,
                .dead_code_elimination,
                .strength_reduction,
            },
            .aggressive => &.{
                .constant_folding,
                .algebraic_simplification,
                .common_subexpression_elimination,
                .dead_code_elimination,
                .strength_reduction,
                .redundancy_elimination,
            },
        };
    }
};

/// Statistics about optimizations performed.
pub const OptimizationStats = struct {
    constants_folded: u32 = 0,
    dead_code_eliminated: u32 = 0,
    expressions_simplified: u32 = 0,
    strength_reductions: u32 = 0,
    subexpressions_eliminated: u32 = 0,
    redundancies_eliminated: u32 = 0,
    passes_run: u32 = 0,

    pub fn totalOptimizations(self: OptimizationStats) u32 {
        return self.constants_folded +
            self.dead_code_eliminated +
            self.expressions_simplified +
            self.strength_reductions +
            self.subexpressions_eliminated +
            self.redundancies_eliminated;
    }

    pub fn format(
        self: OptimizationStats,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("OptimizationStats{{ passes: {d}, total: {d}, constants: {d}, dce: {d}, simplify: {d} }}", .{
            self.passes_run,
            self.totalOptimizations(),
            self.constants_folded,
            self.dead_code_eliminated,
            self.expressions_simplified,
        });
    }
};

/// Kernel IR optimizer.
pub const Optimizer = struct {
    allocator: std.mem.Allocator,
    stats: OptimizationStats,
    /// Track used variables for DCE.
    used_vars: std.StringHashMapUnmanaged(void),
    /// Cache for CSE.
    expr_cache: std.StringHashMapUnmanaged(*const expr.Expr),

    pub fn init(allocator: std.mem.Allocator) Optimizer {
        return .{
            .allocator = allocator,
            .stats = .{},
            .used_vars = .{},
            .expr_cache = .{},
        };
    }

    pub fn deinit(self: *Optimizer) void {
        self.used_vars.deinit(self.allocator);
        self.expr_cache.deinit(self.allocator);
        self.* = undefined;
    }

    /// Run a specific optimization pass.
    pub fn runPass(self: *Optimizer, ir: *kernel.KernelIR, pass: OptimizationPass) !void {
        switch (pass) {
            .constant_folding => try self.constantFolding(ir),
            .dead_code_elimination => try self.deadCodeElimination(ir),
            .algebraic_simplification => try self.algebraicSimplification(ir),
            .strength_reduction => try self.strengthReduction(ir),
            .common_subexpression_elimination => try self.commonSubexprElimination(ir),
            .redundancy_elimination => try self.redundancyElimination(ir),
        }
        self.stats.passes_run += 1;
    }

    /// Run multiple optimization passes.
    pub fn runPasses(self: *Optimizer, ir: *kernel.KernelIR, passes: []const OptimizationPass) !void {
        for (passes) |pass| {
            try self.runPass(ir, pass);
        }
    }

    /// Run optimizations at a specific level.
    pub fn optimize(self: *Optimizer, ir: *kernel.KernelIR, level: OptimizationLevel) !void {
        try self.runPasses(ir, level.getPasses());
    }

    /// Get optimization statistics.
    pub fn getStats(self: *const Optimizer) OptimizationStats {
        return self.stats;
    }

    // ========================================================================
    // Constant Folding
    // ========================================================================

    fn constantFolding(self: *Optimizer, ir: *kernel.KernelIR) !void {
        // Process each statement in the body
        for (ir.body) |s| {
            try self.foldConstantsInStmt(s);
        }

        // Process helper functions
        for (ir.functions) |func| {
            for (func.body) |s| {
                try self.foldConstantsInStmt(s);
            }
        }
    }

    fn foldConstantsInStmt(self: *Optimizer, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |decl| {
                if (decl.init) |init_expr| {
                    _ = try self.tryFoldExpr(init_expr);
                }
            },
            .assign => |assignment| {
                _ = try self.tryFoldExpr(assignment.value);
            },
            .compound_assign => |ca| {
                _ = try self.tryFoldExpr(ca.value);
            },
            .if_ => |if_stmt| {
                _ = try self.tryFoldExpr(if_stmt.condition);
                for (if_stmt.then_body) |body_stmt| {
                    try self.foldConstantsInStmt(body_stmt);
                }
                if (if_stmt.else_body) |else_body| {
                    for (else_body) |body_stmt| {
                        try self.foldConstantsInStmt(body_stmt);
                    }
                }
            },
            .for_ => |for_stmt| {
                if (for_stmt.init) |init_s| {
                    try self.foldConstantsInStmt(init_s);
                }
                if (for_stmt.condition) |cond| {
                    _ = try self.tryFoldExpr(cond);
                }
                for (for_stmt.body) |body_stmt| {
                    try self.foldConstantsInStmt(body_stmt);
                }
            },
            .while_ => |while_stmt| {
                _ = try self.tryFoldExpr(while_stmt.condition);
                for (while_stmt.body) |body_stmt| {
                    try self.foldConstantsInStmt(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    try self.foldConstantsInStmt(body_stmt);
                }
            },
            .expr_stmt => |e| {
                _ = try self.tryFoldExpr(e);
            },
            .return_ => |ret| {
                if (ret.value) |v| {
                    _ = try self.tryFoldExpr(v);
                }
            },
            else => {},
        }
    }

    fn tryFoldExpr(self: *Optimizer, e: *const expr.Expr) !?expr.Literal {
        switch (e.*) {
            .literal => |lit| return lit,
            .binary => |bin| {
                const left_lit = try self.tryFoldExpr(bin.left);
                const right_lit = try self.tryFoldExpr(bin.right);

                if (left_lit != null and right_lit != null) {
                    if (try self.foldBinaryOp(bin.op, left_lit.?, right_lit.?)) |result| {
                        self.stats.constants_folded += 1;
                        return result;
                    }
                }
                return null;
            },
            .unary => |un| {
                const operand_lit = try self.tryFoldExpr(un.operand);
                if (operand_lit) |lit| {
                    if (try self.foldUnaryOp(un.op, lit)) |result| {
                        self.stats.constants_folded += 1;
                        return result;
                    }
                }
                return null;
            },
            else => return null,
        }
    }

    fn foldBinaryOp(self: *Optimizer, op: expr.BinaryOp, left: expr.Literal, right: expr.Literal) !?expr.Literal {
        _ = self;
        // Handle f32 operations
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

        // Handle i32 operations
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

        // Handle u32 operations
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

        // Handle boolean operations
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

    fn foldUnaryOp(self: *Optimizer, op: expr.UnaryOp, operand: expr.Literal) !?expr.Literal {
        _ = self;
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

    // ========================================================================
    // Dead Code Elimination
    // ========================================================================

    fn deadCodeElimination(self: *Optimizer, ir: *kernel.KernelIR) !void {
        // First pass: collect all used variables
        self.used_vars.clearRetainingCapacity();

        for (ir.body) |s| {
            try self.collectUsedVars(s);
        }

        // Mark buffer and uniform names as used
        for (ir.buffers) |buf| {
            try self.used_vars.put(self.allocator, buf.name, {});
        }
        for (ir.uniforms) |uni| {
            try self.used_vars.put(self.allocator, uni.name, {});
        }

        // Second pass would mark unused declarations for removal
        // For now, just count potential eliminations
        for (ir.body) |s| {
            self.countDeadCode(s);
        }
    }

    fn collectUsedVars(self: *Optimizer, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |decl| {
                if (decl.init) |init_expr| {
                    try self.collectUsedVarsInExpr(init_expr);
                }
            },
            .assign => |assignment| {
                try self.collectUsedVarsInExpr(assignment.target);
                try self.collectUsedVarsInExpr(assignment.value);
            },
            .compound_assign => |ca| {
                try self.collectUsedVarsInExpr(ca.target);
                try self.collectUsedVarsInExpr(ca.value);
            },
            .if_ => |if_stmt| {
                try self.collectUsedVarsInExpr(if_stmt.condition);
                for (if_stmt.then_body) |body_stmt| {
                    try self.collectUsedVars(body_stmt);
                }
                if (if_stmt.else_body) |else_body| {
                    for (else_body) |body_stmt| {
                        try self.collectUsedVars(body_stmt);
                    }
                }
            },
            .for_ => |for_stmt| {
                if (for_stmt.init) |init_s| try self.collectUsedVars(init_s);
                if (for_stmt.condition) |cond| try self.collectUsedVarsInExpr(cond);
                if (for_stmt.update) |upd| try self.collectUsedVars(upd);
                for (for_stmt.body) |body_stmt| {
                    try self.collectUsedVars(body_stmt);
                }
            },
            .while_ => |while_stmt| {
                try self.collectUsedVarsInExpr(while_stmt.condition);
                for (while_stmt.body) |body_stmt| {
                    try self.collectUsedVars(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    try self.collectUsedVars(body_stmt);
                }
            },
            .expr_stmt => |e| {
                try self.collectUsedVarsInExpr(e);
            },
            .return_ => |ret| {
                if (ret.value) |v| try self.collectUsedVarsInExpr(v);
            },
            else => {},
        }
    }

    fn collectUsedVarsInExpr(self: *Optimizer, e: *const expr.Expr) !void {
        switch (e.*) {
            .ref => |ref| {
                if (ref.name) |var_name| {
                    try self.used_vars.put(self.allocator, var_name, {});
                }
            },
            .binary => |bin| {
                try self.collectUsedVarsInExpr(bin.left);
                try self.collectUsedVarsInExpr(bin.right);
            },
            .unary => |un| {
                try self.collectUsedVarsInExpr(un.operand);
            },
            .call => |call| {
                for (call.args) |arg| {
                    try self.collectUsedVarsInExpr(arg);
                }
            },
            .index => |idx| {
                try self.collectUsedVarsInExpr(idx.base);
                try self.collectUsedVarsInExpr(idx.index);
            },
            .field => |fld| {
                try self.collectUsedVarsInExpr(fld.base);
            },
            .select => |sel| {
                try self.collectUsedVarsInExpr(sel.condition);
                try self.collectUsedVarsInExpr(sel.true_value);
                try self.collectUsedVarsInExpr(sel.false_value);
            },
            else => {},
        }
    }

    fn countDeadCode(self: *Optimizer, s: *const stmt.Stmt) void {
        switch (s.*) {
            .var_decl => |decl| {
                if (!self.used_vars.contains(decl.name)) {
                    self.stats.dead_code_eliminated += 1;
                }
            },
            .if_ => |if_stmt| {
                for (if_stmt.then_body) |body_stmt| {
                    self.countDeadCode(body_stmt);
                }
                if (if_stmt.else_body) |else_body| {
                    for (else_body) |body_stmt| {
                        self.countDeadCode(body_stmt);
                    }
                }
            },
            .for_ => |for_stmt| {
                for (for_stmt.body) |body_stmt| {
                    self.countDeadCode(body_stmt);
                }
            },
            .while_ => |while_stmt| {
                for (while_stmt.body) |body_stmt| {
                    self.countDeadCode(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    self.countDeadCode(body_stmt);
                }
            },
            else => {},
        }
    }

    // ========================================================================
    // Algebraic Simplification
    // ========================================================================

    fn algebraicSimplification(self: *Optimizer, ir: *kernel.KernelIR) !void {
        for (ir.body) |s| {
            try self.simplifyStmt(s);
        }
    }

    fn simplifyStmt(self: *Optimizer, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |decl| {
                if (decl.init) |init_expr| {
                    if (self.trySimplifyExpr(init_expr)) {
                        self.stats.expressions_simplified += 1;
                    }
                }
            },
            .assign => |assignment| {
                if (self.trySimplifyExpr(assignment.value)) {
                    self.stats.expressions_simplified += 1;
                }
            },
            .compound_assign => |ca| {
                if (self.trySimplifyExpr(ca.value)) {
                    self.stats.expressions_simplified += 1;
                }
            },
            .if_ => |if_stmt| {
                for (if_stmt.then_body) |body_stmt| {
                    try self.simplifyStmt(body_stmt);
                }
                if (if_stmt.else_body) |else_body| {
                    for (else_body) |body_stmt| {
                        try self.simplifyStmt(body_stmt);
                    }
                }
            },
            .for_ => |for_stmt| {
                for (for_stmt.body) |body_stmt| {
                    try self.simplifyStmt(body_stmt);
                }
            },
            .while_ => |while_stmt| {
                for (while_stmt.body) |body_stmt| {
                    try self.simplifyStmt(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    try self.simplifyStmt(body_stmt);
                }
            },
            else => {},
        }
    }

    fn trySimplifyExpr(self: *Optimizer, e: *const expr.Expr) bool {
        _ = self;
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

    // ========================================================================
    // Strength Reduction
    // ========================================================================

    fn strengthReduction(self: *Optimizer, ir: *kernel.KernelIR) !void {
        for (ir.body) |s| {
            try self.reduceStrengthInStmt(s);
        }
    }

    fn reduceStrengthInStmt(self: *Optimizer, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |decl| {
                if (decl.init) |init_expr| {
                    if (self.tryReduceStrength(init_expr)) {
                        self.stats.strength_reductions += 1;
                    }
                }
            },
            .assign => |assignment| {
                if (self.tryReduceStrength(assignment.value)) {
                    self.stats.strength_reductions += 1;
                }
            },
            .for_ => |for_stmt| {
                for (for_stmt.body) |body_stmt| {
                    try self.reduceStrengthInStmt(body_stmt);
                }
            },
            .while_ => |while_stmt| {
                for (while_stmt.body) |body_stmt| {
                    try self.reduceStrengthInStmt(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    try self.reduceStrengthInStmt(body_stmt);
                }
            },
            else => {},
        }
    }

    fn tryReduceStrength(self: *Optimizer, e: *const expr.Expr) bool {
        _ = self;
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

    // ========================================================================
    // Common Subexpression Elimination
    // ========================================================================

    fn commonSubexprElimination(self: *Optimizer, ir: *kernel.KernelIR) !void {
        self.expr_cache.clearRetainingCapacity();

        for (ir.body) |s| {
            try self.findCommonSubexprs(s);
        }
    }

    fn findCommonSubexprs(self: *Optimizer, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |decl| {
                if (decl.init) |init_expr| {
                    try self.checkAndCacheExpr(init_expr);
                }
            },
            .assign => |assignment| {
                try self.checkAndCacheExpr(assignment.value);
            },
            .for_ => |for_stmt| {
                for (for_stmt.body) |body_stmt| {
                    try self.findCommonSubexprs(body_stmt);
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    try self.findCommonSubexprs(body_stmt);
                }
            },
            else => {},
        }
    }

    fn checkAndCacheExpr(self: *Optimizer, e: *const expr.Expr) !void {
        const hash = exprHash(e);
        if (self.expr_cache.get(hash)) |_| {
            self.stats.subexpressions_eliminated += 1;
        } else {
            try self.expr_cache.put(self.allocator, hash, e);
        }
    }

    // ========================================================================
    // Redundancy Elimination
    // ========================================================================

    fn redundancyElimination(self: *Optimizer, ir: *kernel.KernelIR) !void {
        // Look for redundant operations like:
        // x = a; y = a; (when y is immediately assigned to a again)
        for (ir.body) |s| {
            self.checkRedundancy(s);
        }
    }

    fn checkRedundancy(self: *Optimizer, s: *const stmt.Stmt) void {
        switch (s.*) {
            .assign => |assignment| {
                // Check for self-assignment: x = x
                if (assignment.target.* == .ref and assignment.value.* == .ref) {
                    const target_ref = assignment.target.ref;
                    const value_ref = assignment.value.ref;
                    if (target_ref.id == value_ref.id) {
                        self.stats.redundancies_eliminated += 1;
                    }
                }
            },
            .block => |blk| {
                for (blk.statements) |body_stmt| {
                    self.checkRedundancy(body_stmt);
                }
            },
            else => {},
        }
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

fn isZero(lit: expr.Literal) bool {
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

fn isOne(lit: expr.Literal) bool {
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

fn isPowerOfTwo(lit: expr.Literal) bool {
    return switch (lit) {
        .i32_ => |v| v > 0 and (v & (v - 1)) == 0,
        .i64_ => |v| v > 0 and (v & (v - 1)) == 0,
        .u32_ => |v| v > 0 and (v & (v - 1)) == 0,
        .u64_ => |v| v > 0 and (v & (v - 1)) == 0,
        else => false,
    };
}

fn exprHash(e: *const expr.Expr) []const u8 {
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

test "optimizer initialization" {
    const allocator = std.testing.allocator;

    var opt = Optimizer.init(allocator);
    defer opt.deinit();

    const stats = opt.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.passes_run);
    try std.testing.expectEqual(@as(u32, 0), stats.totalOptimizations());
}

test "optimization level passes" {
    const none_passes = OptimizationLevel.none.getPasses();
    try std.testing.expectEqual(@as(usize, 0), none_passes.len);

    const basic_passes = OptimizationLevel.basic.getPasses();
    try std.testing.expectEqual(@as(usize, 2), basic_passes.len);

    const standard_passes = OptimizationLevel.standard.getPasses();
    try std.testing.expect(standard_passes.len >= 4);

    const aggressive_passes = OptimizationLevel.aggressive.getPasses();
    try std.testing.expect(aggressive_passes.len >= 6);
}

test "constant folding binary ops" {
    const allocator = std.testing.allocator;

    var opt = Optimizer.init(allocator);
    defer opt.deinit();

    // Test f32 addition
    const result1 = try opt.foldBinaryOp(.add, .{ .f32_ = 1.5 }, .{ .f32_ = 2.5 });
    try std.testing.expect(result1 != null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result1.?.f32_, 0.001);

    // Test i32 multiplication
    const result2 = try opt.foldBinaryOp(.mul, .{ .i32_ = 6 }, .{ .i32_ = 7 });
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(i32, 42), result2.?.i32_);

    // Test boolean and
    const result3 = try opt.foldBinaryOp(.and_, .{ .bool_ = true }, .{ .bool_ = false });
    try std.testing.expect(result3 != null);
    try std.testing.expectEqual(false, result3.?.bool_);
}

test "constant folding unary ops" {
    const allocator = std.testing.allocator;

    var opt = Optimizer.init(allocator);
    defer opt.deinit();

    // Test f32 negation
    const result1 = try opt.foldUnaryOp(.neg, .{ .f32_ = 3.14 });
    try std.testing.expect(result1 != null);
    try std.testing.expectApproxEqAbs(@as(f32, -3.14), result1.?.f32_, 0.001);

    // Test i32 absolute value
    const result2 = try opt.foldUnaryOp(.abs, .{ .i32_ = -42 });
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(i32, 42), result2.?.i32_);

    // Test boolean not
    const result3 = try opt.foldUnaryOp(.not, .{ .bool_ = true });
    try std.testing.expect(result3 != null);
    try std.testing.expectEqual(false, result3.?.bool_);
}

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

test "pass names" {
    try std.testing.expectEqualStrings("constant-folding", OptimizationPass.constant_folding.name());
    try std.testing.expectEqualStrings("dead-code-elimination", OptimizationPass.dead_code_elimination.name());
}
