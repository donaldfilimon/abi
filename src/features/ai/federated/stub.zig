//! Federated learning stub — active when AI feature is disabled.

const std = @import("std");

pub const NodeInfo = struct {
    id: []const u8 = "",
    last_update: i64 = 0,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .{},

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

pub const CoordinatorError = error{
    InsufficientUpdates,
    InvalidUpdate,
    FeatureDisabled,
};

pub const AggregationStrategy = enum {
    mean,
    weighted_mean,
};

pub const ModelUpdateView = struct {
    node_id: []const u8 = "",
    step: u64 = 0,
    weights: []const f32 = &.{},
    sample_count: u32 = 1,
};

pub const ModelUpdate = struct {
    node_id: []const u8 = "",
    step: u64 = 0,
    timestamp: u64 = 0,
    weights: []f32 = &.{},
    sample_count: u32 = 0,

    pub fn deinit(_: *ModelUpdate, _: std.mem.Allocator) void {}
};

pub const CoordinatorConfig = struct {
    min_updates: usize = 1,
    max_updates: usize = 64,
    max_staleness_seconds: u64 = 300,
    strategy: AggregationStrategy = .mean,
};

pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    registry: Registry,
    updates: std.ArrayListUnmanaged(ModelUpdate) = .{},
    global_weights: []f32 = &.{},
    scratch: []f32 = &.{},
    config: CoordinatorConfig = .{},
    current_step: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        _: CoordinatorConfig,
        _: usize,
    ) CoordinatorError!Coordinator {
        _ = allocator;
        return CoordinatorError.FeatureDisabled;
    }

    pub fn deinit(_: *Coordinator) void {}

    pub fn registerNode(_: *Coordinator, _: []const u8) !void {
        return error.FeatureDisabled;
    }

    pub fn submitUpdate(_: *Coordinator, _: ModelUpdateView) !void {
        return error.FeatureDisabled;
    }

    pub fn aggregate(_: *Coordinator) CoordinatorError![]const f32 {
        return CoordinatorError.FeatureDisabled;
    }

    pub fn globalWeights(_: *const Coordinator) []const f32 {
        return &.{};
    }

    pub fn step(_: *const Coordinator) u64 {
        return 0;
    }
};
