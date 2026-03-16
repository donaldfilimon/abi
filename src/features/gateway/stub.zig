//! Gateway Stub Module
//!
//! API-compatible no-op implementations when gateway is disabled.

const std = @import("std");
const core_config = @import("../../core/config/gateway.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const GatewayConfig = core_config.GatewayConfig;
pub const RateLimitConfig = core_config.RateLimitConfig;
pub const RateLimitAlgorithm = core_config.RateLimitAlgorithm;
pub const CircuitBreakerConfig = core_config.CircuitBreakerConfig;
pub const CircuitBreakerState = core_config.CircuitBreakerState;

pub const GatewayError = error{
    FeatureDisabled,
    RouteNotFound,
    RateLimitExceeded,
    CircuitOpen,
    UpstreamTimeout,
    InvalidRoute,
    TooManyRoutes,
    MiddlewareError,
    OutOfMemory,
};

pub const HttpMethod = enum { GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS };

pub const Route = struct {
    path: []const u8 = "/",
    method: HttpMethod = .GET,
    upstream: []const u8 = "",
    timeout_ms: u64 = 30_000,
    rate_limit: ?RateLimitConfig = null,
    middlewares: []const MiddlewareType = &.{},
};

pub const MiddlewareType = enum {
    auth,
    rate_limit,
    circuit_breaker,
    access_log,
    cors,
    response_transform,
    request_id,
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
    params: [8]Param = [_]Param{.{}} ** 8,
    param_count: u8 = 0,
    matched_route_idx: ?u32 = null,

    pub const Param = struct {
        name: []const u8 = "",
        value: []const u8 = "",
    };

    pub fn getParam(self: *const MatchResult, name: []const u8) ?[]const u8 {
        _ = self;
        _ = name;
        return null;
    }
};

pub const RateLimitResult = struct {
    allowed: bool = true,
    remaining: u32 = 0,
    reset_after_ms: u64 = 0,
};

pub const Context = stub_context.StubContextWithConfig(GatewayConfig);

pub fn init(_: std.mem.Allocator, _: GatewayConfig) GatewayError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn addRoute(_: Route) GatewayError!void {
    return error.FeatureDisabled;
}
pub fn removeRoute(_: []const u8) GatewayError!bool {
    return error.FeatureDisabled;
}
pub fn getRoutes() []const Route {
    return &.{};
}
pub fn getRouteCount() usize {
    return 0;
}
pub fn matchRoute(_: []const u8, _: HttpMethod) GatewayError!?MatchResult {
    return error.FeatureDisabled;
}
pub fn checkRateLimit(_: []const u8) RateLimitResult {
    return .{};
}
pub fn recordUpstreamResult(_: []const u8, _: bool) void {}

pub fn stats() GatewayStats {
    return .{};
}

pub fn getCircuitState(_: []const u8) CircuitBreakerState {
    return .closed;
}
pub fn resetCircuit(_: []const u8) void {}

test {
    std.testing.refAllDecls(@This());
}
