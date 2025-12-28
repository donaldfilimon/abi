//! Network feature module for distributed compute coordination.
//!
//! Provides node registry, task/result serialization protocols, and cluster state
//! management for distributed computing scenarios.

const std = @import("std");
const build_options = @import("build_options");

const registry = @import("registry.zig");
const protocol = @import("protocol.zig");
const scheduler = @import("scheduler.zig");
const ha = @import("ha.zig");

pub const NodeRegistry = registry.NodeRegistry;
pub const NodeInfo = registry.NodeInfo;
pub const NodeStatus = registry.NodeStatus;

pub const TaskEnvelope = protocol.TaskEnvelope;
pub const ResultEnvelope = protocol.ResultEnvelope;
pub const ResultStatus = protocol.ResultStatus;
pub const encodeTask = protocol.encodeTask;
pub const decodeTask = protocol.decodeTask;
pub const encodeResult = protocol.encodeResult;
pub const decodeResult = protocol.decodeResult;

pub const TaskScheduler = scheduler.TaskScheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const SchedulerError = scheduler.SchedulerError;
pub const TaskPriority = scheduler.TaskPriority;
pub const TaskState = scheduler.TaskState;
pub const ComputeNode = scheduler.ComputeNode;
pub const LoadBalancingStrategy = scheduler.LoadBalancingStrategy;
pub const SchedulerStats = scheduler.SchedulerStats;

pub const HealthCheck = ha.HealthCheck;
pub const ClusterConfig = ha.ClusterConfig;
pub const HaError = ha.HaError;
pub const NodeHealth = ha.NodeHealth;
pub const ClusterState = ha.ClusterState;
pub const HealthCheckResult = ha.HealthCheckResult;
pub const FailoverPolicy = ha.FailoverPolicy;

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

const DEFAULT_CLUSTER_ID = "default";
const DEFAULT_HEARTBEAT_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MAX_NODES: usize = 256;

pub const NetworkConfig = struct {
    cluster_id: []const u8 = DEFAULT_CLUSTER_ID,
    heartbeat_timeout_ms: u64 = DEFAULT_HEARTBEAT_TIMEOUT_MS,
    max_nodes: usize = DEFAULT_MAX_NODES,
};

pub const NetworkState = struct {
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    registry: NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: NetworkConfig) !NetworkState {
        const cluster_id = try allocator.dupe(u8, config.cluster_id);
        return .{
            .allocator = allocator,
            .config = .{
                .cluster_id = cluster_id,
                .heartbeat_timeout_ms = config.heartbeat_timeout_ms,
                .max_nodes = config.max_nodes,
            },
            .registry = NodeRegistry.init(allocator),
        };
    }

    pub fn deinit(self: *NetworkState) void {
        self.registry.deinit();
        self.allocator.free(self.config.cluster_id);
        self.* = undefined;
    }
};

var state_mutex = std.Thread.Mutex{};
var default_state: ?NetworkState = null;
var initialized: bool = false;

pub fn isEnabled() bool {
    return build_options.enable_network;
}

pub fn isInitialized() bool {
    state_mutex.lock();
    defer state_mutex.unlock();
    return initialized;
}

pub fn init(allocator: std.mem.Allocator) !void {
    return initWithConfig(allocator, .{});
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) !void {
    if (!isEnabled()) return NetworkError.NetworkDisabled;

    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state == null) {
        default_state = try NetworkState.init(allocator, config);
    }
    initialized = true;
}

pub fn deinit() void {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        state.deinit();
        default_state = null;
    }
    initialized = false;
}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        return &state.registry;
    }
    return NetworkError.NotInitialized;
}

pub fn defaultConfig() ?NetworkConfig {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |state| {
        return .{
            .cluster_id = state.config.cluster_id,
            .heartbeat_timeout_ms = state.config.heartbeat_timeout_ms,
            .max_nodes = state.config.max_nodes,
        };
    }
    return null;
}

test "network state tracks nodes" {
    var state = try NetworkState.init(std.testing.allocator, .{ .cluster_id = "test" });
    defer state.deinit();

    try state.registry.register("node-a", "127.0.0.1:9000");
    try state.registry.register("node-b", "127.0.0.1:9001");
    try std.testing.expectEqual(@as(usize, 2), state.registry.list().len);
}

test "network default state" {
    if (!isEnabled()) return;

    try initWithConfig(std.testing.allocator, .{ .cluster_id = "cluster-a" });
    defer deinit();

    const registry_ptr = try defaultRegistry();
    try registry_ptr.register("node-a", "127.0.0.1:9000");

    const config = defaultConfig() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("cluster-a", config.cluster_id);
}
