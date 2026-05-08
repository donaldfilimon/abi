//! Federated learning stub — active when AI feature is disabled.

const std = @import("std");
const types = @import("types.zig");

pub const NodeInfo = types.NodeInfo;
pub const CoordinatorError = types.CoordinatorError;
pub const AggregationStrategy = types.AggregationStrategy;
pub const ModelUpdateView = types.ModelUpdateView;
pub const ModelUpdate = types.ModelUpdate;
pub const CoordinatorConfig = types.CoordinatorConfig;

pub const Registry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .empty,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *Registry) void {}
    pub fn touch(_: *Registry, _: []const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn remove(_: *Registry, _: []const u8) bool {
        return false;
    }
    pub fn list(_: *const Registry) []const NodeInfo {
        return &.{};
    }
    pub fn count(_: *const Registry) usize {
        return 0;
    }
    pub fn prune(_: *Registry, _: i64) usize {
        return 0;
    }
};

pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    registry: Registry,
    updates: std.ArrayListUnmanaged(ModelUpdate) = .empty,
    global_weights: []f32 = &.{},
    scratch: []f32 = &.{},
    config: CoordinatorConfig = .{},
    current_step: u64 = 0,

    pub fn init(_: std.mem.Allocator, _: CoordinatorConfig, _: usize) error{FeatureDisabled}!Coordinator {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Coordinator) void {}
    pub fn registerNode(_: *Coordinator, _: []const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn submitUpdate(_: *Coordinator, _: ModelUpdateView) !void {
        return error.FeatureDisabled;
    }
    pub fn aggregate(_: *Coordinator) error{FeatureDisabled}![]const f32 {
        return error.FeatureDisabled;
    }
    pub fn globalWeights(_: *const Coordinator) []const f32 {
        return &.{};
    }
    pub fn step(_: *const Coordinator) u64 {
        return 0;
    }
};

test {
    std.testing.refAllDecls(@This());
}
