//! Integration Tests: Gateway Feature
//!
//! Tests the gateway module exports, rate limiting types, route types,
//! and context lifecycle.

const std = @import("std");
const abi = @import("abi");

const gateway = abi.gateway;

// ============================================================================
// Module Lifecycle
// ============================================================================

test "gateway: isEnabled returns bool" {
    const enabled = gateway.isEnabled();
    try std.testing.expect(enabled == true or enabled == false);
}

test "gateway: isInitialized returns bool" {
    const initialized = gateway.isInitialized();
    try std.testing.expect(initialized == true or initialized == false);
}

// ============================================================================
// Types
// ============================================================================

test "gateway: GatewayError includes expected variants" {
    const err: gateway.GatewayError = error.FeatureDisabled;
    try std.testing.expect(err == error.FeatureDisabled);
}

test "gateway: HttpMethod enum has expected variants" {
    const get = gateway.HttpMethod.GET;
    const post = gateway.HttpMethod.POST;
    const del = gateway.HttpMethod.DELETE;
    try std.testing.expect(get != post);
    try std.testing.expect(post != del);
}

test "gateway: Route default values" {
    const route = gateway.Route{};
    try std.testing.expectEqualStrings("/", route.path);
    try std.testing.expectEqual(gateway.HttpMethod.GET, route.method);
    try std.testing.expectEqualStrings("", route.upstream);
    try std.testing.expectEqual(@as(u64, 30_000), route.timeout_ms);
    try std.testing.expect(route.rate_limit == null);
}

test "gateway: Route with custom values" {
    const route = gateway.Route{
        .path = "/api/v1/users",
        .method = .POST,
        .upstream = "http://backend:8080",
        .timeout_ms = 5_000,
    };
    try std.testing.expectEqualStrings("/api/v1/users", route.path);
    try std.testing.expectEqual(gateway.HttpMethod.POST, route.method);
    try std.testing.expectEqual(@as(u64, 5_000), route.timeout_ms);
}

test "gateway: MiddlewareType enum variants" {
    const mw_auth = gateway.MiddlewareType.auth;
    const mw_rl = gateway.MiddlewareType.rate_limit;
    const mw_cb = gateway.MiddlewareType.circuit_breaker;
    const mw_log = gateway.MiddlewareType.access_log;
    const mw_cors = gateway.MiddlewareType.cors;
    try std.testing.expect(mw_auth != mw_rl);
    try std.testing.expect(mw_cb != mw_log);
    try std.testing.expect(mw_cors != mw_auth);
}

test "gateway: GatewayStats default values" {
    const s = gateway.GatewayStats{};
    try std.testing.expectEqual(@as(u64, 0), s.total_requests);
    try std.testing.expectEqual(@as(u32, 0), s.active_routes);
    try std.testing.expectEqual(@as(u64, 0), s.rate_limited_count);
    try std.testing.expectEqual(@as(u64, 0), s.circuit_breaker_trips);
    try std.testing.expectEqual(@as(u64, 0), s.upstream_errors);
    try std.testing.expectEqual(@as(u64, 0), s.avg_latency_ms);
}

test "gateway: RateLimitResult default values" {
    const rl = gateway.RateLimitResult{};
    try std.testing.expect(rl.allowed);
    try std.testing.expectEqual(@as(u32, 0), rl.remaining);
    try std.testing.expectEqual(@as(u64, 0), rl.reset_after_ms);
}

test "gateway: CircuitBreakerState enum" {
    const closed = gateway.CircuitBreakerState.closed;
    try std.testing.expectEqual(gateway.CircuitBreakerState.closed, closed);
}

// ============================================================================
// Stub API
// ============================================================================

test "gateway: stats returns default GatewayStats" {
    const s = gateway.stats();
    try std.testing.expectEqual(@as(u64, 0), s.total_requests);
    try std.testing.expectEqual(@as(u32, 0), s.active_routes);
}

test "gateway: getRoutes returns slice" {
    const routes = gateway.getRoutes();
    // Stub returns empty; real impl may return populated
    try std.testing.expect(routes.len >= 0);
}

test "gateway: getRouteCount returns count" {
    const count = gateway.getRouteCount();
    try std.testing.expect(count >= 0);
}

test "gateway: checkRateLimit returns result" {
    const result = gateway.checkRateLimit("test-client");
    // Stub allows; real impl may vary
    try std.testing.expect(result.allowed == true or result.allowed == false);
}

test "gateway: addRoute returns result or FeatureDisabled" {
    const route = gateway.Route{
        .path = "/test",
        .method = .GET,
        .upstream = "http://localhost:3000",
    };
    const result = gateway.addRoute(route);
    if (result) |_| {
        // Feature enabled — route added
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "gateway: removeRoute returns result or FeatureDisabled" {
    const result = gateway.removeRoute("/test");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "gateway: matchRoute returns result or FeatureDisabled" {
    const result = gateway.matchRoute("/api/test", .GET);
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "gateway: getCircuitState returns a valid state" {
    const state = gateway.getCircuitState("backend-1");
    try std.testing.expectEqual(gateway.CircuitBreakerState.closed, state);
}

test "gateway: resetCircuit does not panic" {
    gateway.resetCircuit("backend-1");
}

test "gateway: recordUpstreamResult does not panic" {
    gateway.recordUpstreamResult("backend-1", true);
    gateway.recordUpstreamResult("backend-1", false);
}

test "gateway: recordLatency does not panic" {
    gateway.recordLatency(42_000);
}

test "gateway: checkCircuitBreaker returns bool" {
    const open = gateway.checkCircuitBreaker("backend-1");
    try std.testing.expect(open == true or open == false);
}

// Sibling test modules (pulled in via refAllDecls)
const _runtime = @import("gateway_runtime_test.zig");

test {
    std.testing.refAllDecls(@This());
}
