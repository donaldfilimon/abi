const std = @import("std");
const core_config = @import("../../core/config/gateway.zig");
const radix = @import("../../services/shared/utils/radix_tree.zig");

const RouteTree = radix.RadixTree(u32);

pub const GatewayConfig = core_config.GatewayConfig;
pub const RateLimitConfig = core_config.RateLimitConfig;
pub const RateLimitAlgorithm = core_config.RateLimitAlgorithm;
pub const CircuitBreakerConfig = core_config.CircuitBreakerConfig;
pub const CircuitBreakerState = core_config.CircuitBreakerState;

pub const HttpMethod = enum { GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS };

pub const MiddlewareType = enum {
    auth,
    rate_limit,
    circuit_breaker,
    access_log,
    cors,
    response_transform,
    request_id,
};

pub const Route = struct {
    path: []const u8 = "/",
    method: HttpMethod = .GET,
    upstream: []const u8 = "",
    timeout_ms: u64 = 30_000,
    rate_limit: ?RateLimitConfig = null,
    middlewares: []const MiddlewareType = &.{},
};

pub const GatewayStats = struct {
    total_requests: u64 = 0,
    active_routes: u32 = 0,
    rate_limited_count: u64 = 0,
    circuit_breaker_trips: u64 = 0,
    upstream_errors: u64 = 0,
    avg_latency_ms: u64 = 0,
};

pub const MatchResult = struct {
    route: Route,
    params: [RouteTree.max_params]Param = [_]Param{.{}} ** RouteTree.max_params,
    param_count: u8 = 0,
    matched_route_idx: ?u32 = null, // internal: radix tree match result

    pub const Param = RouteTree.Param;

    pub fn getParam(self: *const MatchResult, name: []const u8) ?[]const u8 {
        for (self.params[0..self.param_count]) |p| {
            if (std.mem.eql(u8, p.name, name)) return p.value;
        }
        return null;
    }
};

pub const RateLimitResult = struct {
    allowed: bool = true,
    remaining: u32 = 0,
    reset_after_ms: u64 = 0,
};
