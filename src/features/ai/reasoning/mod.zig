//! Reasoning Sub-module
//!
//! Provides structured chain-of-thought reasoning with step tracking,
//! confidence calibration, and research triggers.
//!
//! Architecture:
//! - **ReasoningChain**: State management for a single query's thought process.
//! - **ReasoningStep**: Individual steps (assessment, retrieval, synthesis, etc.).
//! - **Confidence**: Integration with core confidence scoring.

const std = @import("std");
const ai_config = @import("../../../core/config/ai.zig");
pub const engine = @import("engine.zig");

// Re-export core types
pub const ReasoningChain = engine.ReasoningChain;
pub const ReasoningStep = engine.ReasoningStep;
pub const StepType = engine.StepType;
pub const ConfidenceLevel = engine.ConfidenceLevel;
pub const Confidence = engine.Confidence;

/// Configuration for reasoning engine
pub const ReasoningConfig = ai_config.AiConfig.ReasoningConfig;

/// Reasoning context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: ReasoningConfig,

    pub fn init(allocator: std.mem.Allocator, cfg: ReasoningConfig) !*Context {
        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }

    /// Create a new reasoning chain for a query.
    pub fn createChain(self: *Context, query: []const u8) ReasoningChain {
        return ReasoningChain.init(self.allocator, query);
    }
};

/// Whether the reasoning feature is enabled in this build.
pub fn isEnabled() bool {
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "reasoning context init" {
    const allocator = std.testing.allocator;
    const ctx = try Context.init(allocator, .{});
    defer ctx.deinit();

    var chain = ctx.createChain("test");
    defer chain.deinit();
    try std.testing.expectEqual(@as(usize, 0), chain.stepCount());
}

test {
    _ = engine;
}
