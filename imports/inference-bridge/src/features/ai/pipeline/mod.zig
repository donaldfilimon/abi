//! Abbey Dynamic Model — Prompt Pipeline DSL
//!
//! A composable pipeline where each step is a typed operation backed by
//! WDBX block-chained storage. Each step execution is recorded as a
//! ConversationBlock with cryptographic integrity.
//!
//! Usage:
//!   const pipeline_mod = abi.ai.pipeline;
//!   var builder = pipeline_mod.chain(allocator, "session-123");
//!   defer builder.deinit();
//!   var p = builder
//!       .withChain(&wdbx_chain)
//!       .retrieve(.wdbx, .{ .k = 5 })
//!       .template("Given {context}, respond to: {input}")
//!       .route(.adaptive)
//!       .modulate()
//!       .generate(.{})
//!       .validate(.constitution)
//!       .store(.wdbx)
//!       .build();
//!   defer p.deinit();
//!   const result = try p.run("Hello Abbey!");

const std = @import("std");

pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const builder = @import("builder.zig");
pub const executor = @import("executor.zig");
pub const persistence = @import("persistence.zig");

// Re-export primary types for convenience
pub const PipelineBuilder = builder.PipelineBuilder;
pub const Pipeline = executor.Pipeline;
pub const PipelineContext = context.PipelineContext;
pub const PipelineResult = types.PipelineResult;
pub const PipelineError = types.PipelineError;
pub const StepKind = types.StepKind;
pub const StepConfig = types.StepConfig;
pub const Step = types.Step;
pub const RetrieveConfig = types.RetrieveConfig;
pub const TemplateConfig = types.TemplateConfig;
pub const RouteConfig = types.RouteConfig;
pub const GenerateConfig = types.GenerateConfig;
pub const ValidateConfig = types.ValidateConfig;
pub const StoreConfig = types.StoreConfig;
pub const ModulateConfig = types.ModulateConfig;
pub const ReasonConfig = types.ReasonConfig;
pub const TransformConfig = types.TransformConfig;
pub const FilterConfig = types.FilterConfig;
pub const RetrieveSource = types.RetrieveSource;
pub const RouteStrategy = types.RouteStrategy;
pub const GenerateMode = types.GenerateMode;
pub const ValidateTarget = types.ValidateTarget;
pub const StoreTarget = types.StoreTarget;

// Step implementations
pub const steps = struct {
    pub const retrieve = @import("steps/retrieve.zig");
    pub const template = @import("steps/template.zig");
    pub const store = @import("steps/store.zig");
    pub const route = @import("steps/route.zig");
    pub const generate = @import("steps/generate.zig");
    pub const validate = @import("steps/validate.zig");
    pub const modulate = @import("steps/modulate.zig");
    pub const reason = @import("steps/reason.zig");
    pub const transform = @import("steps/transform.zig");
    pub const filter = @import("steps/filter.zig");
};

/// Entry point: create a new pipeline builder for a session.
pub fn chain(allocator: std.mem.Allocator, session_id: []const u8) PipelineBuilder {
    return PipelineBuilder.init(allocator, session_id);
}

test "pipeline mod compiles" {
    _ = types;
    _ = context;
    _ = builder;
    _ = executor;
    _ = persistence;
    _ = steps;
}

test {
    std.testing.refAllDecls(@This());
}
