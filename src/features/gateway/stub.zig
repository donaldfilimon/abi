//! Gateway Stub Module
//!
//! API-compatible no-op implementations when gateway is disabled.
//! Types are shared with the real module via types.zig.

const std = @import("std");
const core_config = @import("../../core/config/gateway.zig");
const stub_context = @import("../../core/stub_helpers.zig");
pub const gateway_types = @import("types.zig");

// Config re-exports (from shared types.zig — same source as mod.zig)
pub const GatewayConfig = gateway_types.GatewayConfig;
pub const RateLimitConfig = gateway_types.RateLimitConfig;
pub const RateLimitAlgorithm = gateway_types.RateLimitAlgorithm;
pub const CircuitBreakerConfig = gateway_types.CircuitBreakerConfig;
pub const CircuitBreakerState = gateway_types.CircuitBreakerState;

// Type re-exports (shared with mod.zig via types.zig)
pub const GatewayError = gateway_types.GatewayError;
pub const Error = GatewayError;
pub const HttpMethod = gateway_types.HttpMethod;
pub const Route = gateway_types.Route;
pub const MiddlewareType = gateway_types.MiddlewareType;
pub const GatewayStats = gateway_types.GatewayStats;
pub const MatchResult = gateway_types.MatchResult;
pub const RateLimitResult = gateway_types.RateLimitResult;

const feature = stub_context.StubFeature(GatewayConfig, GatewayError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

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

pub const HttpStatus = enum(u16) {
    ok = 200,
    too_many_requests = 429,
    service_unavailable = 503,
    bad_gateway = 502,
    not_found = 404,
};

pub const RequestResult = struct {
    status: HttpStatus = .ok,
    match: ?MatchResult = null,
    rate_limit: ?RateLimitResult = null,
    latency_ns: u64 = 0,
};

pub fn dispatchRequest(
    _: []const u8,
    _: HttpMethod,
    _: ?*const fn (route: Route) bool,
) GatewayError!RequestResult {
    return error.FeatureDisabled;
}

pub fn checkCircuitBreaker(_: []const u8) bool {
    return true;
}

pub fn recordLatency(_: u64) void {}

test {
    std.testing.refAllDecls(@This());
}
