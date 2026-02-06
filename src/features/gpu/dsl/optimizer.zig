//! GPU Kernel DSL Optimizer
//!
//! Provides optimization passes for the kernel IR, inspired by MLIR and LLVM research.
//! Pass implementations are split across:
//!   - optimizer_folding.zig: Constant folding
//!   - optimizer_analysis.zig: DCE, algebraic simplification, strength reduction, CSE, redundancy
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

// Pass implementations (split for maintainability)
const folding = @import("optimizer_folding.zig");
const analysis = @import("optimizer_analysis.zig");

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
            .constant_folding => try folding.run(&self.stats.constants_folded, ir),
            .dead_code_elimination => try analysis.deadCodeElimination(
                &self.stats.dead_code_eliminated,
                self.allocator,
                &self.used_vars,
                ir,
            ),
            .algebraic_simplification => try analysis.algebraicSimplification(
                &self.stats.expressions_simplified,
                ir,
            ),
            .strength_reduction => try analysis.strengthReduction(
                &self.stats.strength_reductions,
                ir,
            ),
            .common_subexpression_elimination => try analysis.commonSubexprElimination(
                &self.stats.subexpressions_eliminated,
                self.allocator,
                &self.expr_cache,
                ir,
            ),
            .redundancy_elimination => analysis.redundancyElimination(
                &self.stats.redundancies_eliminated,
                ir,
            ),
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
};

// ============================================================================
// Tests
// ============================================================================

// Ensure pass file tests are discovered
test {
    _ = folding;
    _ = analysis;
}

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

test "pass names" {
    try std.testing.expectEqualStrings("constant-folding", OptimizationPass.constant_folding.name());
    try std.testing.expectEqualStrings("dead-code-elimination", OptimizationPass.dead_code_elimination.name());
}
