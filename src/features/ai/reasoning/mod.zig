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
const build_options = @import("build_options");
const config_module = @import("../../../core/config/mod.zig");
pub const engine = @import("engine.zig");

// Re-export core types
pub const ReasoningChain = engine.ReasoningChain;
pub const ReasoningStep = engine.ReasoningStep;
pub const StepType = engine.StepType;

/// Configuration for reasoning engine
pub const ReasoningConfig = struct {
    /// Enable research triggers when confidence is low
    enable_research_triggers: bool = true,
    /// Confidence threshold for triggering research (0.0 - 1.0)
    research_threshold: f32 = 0.5,
    /// Maximum number of reasoning steps per query
    max_steps: u32 = 20,
    /// Enable detailed JSON logging of reasoning chains
    log_chains: bool = false,
};

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
