//! Gateway Stub Module
//!
//! API-compatible no-op implementations when gateway is disabled.
//! Types are shared with the real module via types.zig.

const std = @import("std");
const core_config = @import("../../core/config/gateway.zig");
const stub_context = @import("../../core/stub_context.zig");
const gateway_types = @import("types.zig");

// Config re-exports (from core config)
pub const GatewayConfig = core_config.GatewayConfig;
pub const RateLimitConfig = core_config.RateLimitConfig;
pub const RateLimitAlgorithm = core_config.RateLimitAlgorithm;
pub const CircuitBreakerConfig = core_config.CircuitBreakerConfig;
pub const CircuitBreakerState = core_config.CircuitBreakerState;

// Type re-exports (shared with mod.zig via types.zig)
pub const GatewayError = gateway_types.GatewayError;
pub const HttpMethod = gateway_types.HttpMethod;
pub const Route = gateway_types.Route;
pub const MiddlewareType = gateway_types.MiddlewareType;
pub const GatewayStats = gateway_types.GatewayStats;
pub const MatchResult = gateway_types.MatchResult;
pub const RateLimitResult = gateway_types.RateLimitResult;

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
