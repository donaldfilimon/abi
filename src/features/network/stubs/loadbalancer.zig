const std = @import("std");

pub const LoadBalancer = struct {
    pub fn init(_: std.mem.Allocator, _: LoadBalancerConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const LoadBalancerConfig = struct {
    strategy: LoadBalancerStrategy = .round_robin,
    health_check_interval_ms: u64 = 5000,
};

pub const LoadBalancerStrategy = enum { round_robin, least_connections, weighted, random };

pub const LoadBalancerError = error{
    NetworkDisabled,
    NoHealthyNodes,
    NodeNotFound,
};

pub const NodeState = enum { active, draining, inactive };

pub const NodeStats = struct {
    active_connections: usize = 0,
    total_requests: u64 = 0,
    error_count: u64 = 0,
    avg_latency_ms: f64 = 0,
};
