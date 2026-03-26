//! Pipeline DSL Stub
//!
//! API-compatible no-ops for when feat_reasoning is disabled.
//! All builder methods chain correctly; run() returns FeatureDisabled.

const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const builder = struct {};
pub const executor = struct {};
pub const persistence = struct {};
pub const steps = struct {};

pub const StepKind = types.StepKind;
pub const StepConfig = types.StepConfig;
pub const Step = types.Step;
pub const PipelineResult = types.PipelineResult;
pub const PipelineError = types.PipelineError;
pub const PipelineContext = context.PipelineContext;
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

pub const PipelineBuilder = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, _: []const u8) Self {
        return .{ .allocator = allocator };
    }

    pub fn withChain(self: *Self, _: anytype) *Self {
        return self;
    }

    pub fn retrieve(self: *Self, _: types.RetrieveSource, _: RetrieveConfig) *Self {
        return self;
    }

    pub fn template(self: *Self, _: []const u8) *Self {
        return self;
    }

    pub fn route(self: *Self, _: RouteStrategy) *Self {
        return self;
    }

    pub fn modulate(self: *Self) *Self {
        return self;
    }

    pub fn generate(self: *Self, _: GenerateConfig) *Self {
        return self;
    }

    pub fn validate(self: *Self, _: ValidateTarget) *Self {
        return self;
    }

    pub fn store(self: *Self, _: StoreTarget) *Self {
        return self;
    }

    pub fn transform(self: *Self, _: types.TransformFn) *Self {
        return self;
    }

    pub fn filter(self: *Self, _: types.FilterFn, _: bool) *Self {
        return self;
    }

    pub fn reason(self: *Self, _: ReasonConfig) *Self {
        return self;
    }

    pub fn build(self: *Self) Pipeline {
        return .{ .allocator = self.allocator };
    }

    pub fn deinit(_: *Self) void {}
};

pub const Pipeline = struct {
    allocator: std.mem.Allocator,

    pub fn run(_: *Pipeline, _: []const u8) PipelineError!PipelineResult {
        return PipelineError.FeatureDisabled;
    }

    pub fn deinit(_: *Pipeline) void {}
};

pub fn chain(allocator: std.mem.Allocator, session_id: []const u8) PipelineBuilder {
    return PipelineBuilder.init(allocator, session_id);
}

test "pipeline stub compiles" {
    const allocator = std.testing.allocator;
    var pb = chain(allocator, "test-session");
    defer pb.deinit();

    var p = pb
        .retrieve(.wdbx, .{ .k = 5 })
        .template("Hello {context}")
        .route(.adaptive)
        .modulate()
        .generate(.{})
        .validate(.constitution)
        .store(.wdbx)
        .build();
    defer p.deinit();

    const result = p.run("test input");
    try std.testing.expectError(PipelineError.FeatureDisabled, result);
}
