const std = @import("std");

pub const HealthCheck = struct {
    pub fn init(_: std.mem.Allocator, _: ClusterConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ClusterConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_interval_ms: u64 = 5000,
    failure_threshold: u32 = 3,
};

pub const HaError = error{
    NetworkDisabled,
    ClusterUnavailable,
    FailoverFailed,
};

pub const NodeHealth = struct {
    node_id: []const u8 = "",
    healthy: bool = false,
    last_heartbeat_ms: i64 = 0,
};

pub const ClusterState = enum { forming, stable, degraded, failed };

pub const HealthCheckResult = struct {
    healthy: bool = false,
    nodes_checked: usize = 0,
    nodes_healthy: usize = 0,
};

pub const FailoverPolicy = enum { automatic, manual, consensus };
