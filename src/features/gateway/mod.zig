//! Gateway Module
//!
//! API gateway with radix-tree route matching, 3 rate limiting algorithms
//! (token bucket, sliding window, fixed window), circuit breaker state
//! machine, middleware chain, and latency tracking.
//!
//! Architecture:
//! - Radix tree for O(path_segments) route matching with params and wildcards
//! - Per-route rate limiters (token bucket, sliding/fixed window)
//! - Per-upstream circuit breakers (closed → open → half_open → closed)
//! - Latency histogram with 7 buckets for p50/p99 estimation
//! - RwLock for concurrent route lookups

const std = @import("std");
pub const gateway_types = @import("types.zig");
const middleware = @import("middleware.zig");
const gateway_pipeline = @import("pipeline.zig");
const gateway_routes = @import("routes.zig");
const gateway_state = @import("state.zig");

pub const GatewayConfig = gateway_types.GatewayConfig;
pub const RateLimitConfig = gateway_types.RateLimitConfig;
pub const RateLimitAlgorithm = gateway_types.RateLimitAlgorithm;
pub const CircuitBreakerConfig = gateway_types.CircuitBreakerConfig;
pub const CircuitBreakerState = gateway_types.CircuitBreakerState;

pub const GatewayError = gateway_types.GatewayError;
pub const Error = GatewayError;
pub const HttpMethod = gateway_types.HttpMethod;
pub const Route = gateway_types.Route;
pub const MiddlewareType = middleware.MiddlewareType;
pub const GatewayStats = gateway_types.GatewayStats;
pub const MatchResult = gateway_types.MatchResult;
pub const RateLimitResult = gateway_types.RateLimitResult;
pub const Context = gateway_state.Context;
pub const HttpStatus = gateway_pipeline.HttpStatus;
pub const RequestResult = gateway_pipeline.RequestResult;

pub const init = gateway_state.init;
pub const deinit = gateway_state.deinit;
pub const isInitialized = gateway_state.isInitialized;

pub fn isEnabled() bool {
    return true;
}

pub fn addRoute(route: Route) GatewayError!void {
    return gateway_routes.addRoute(route);
}

pub fn removeRoute(path: []const u8) GatewayError!bool {
    return gateway_routes.removeRoute(path);
}

pub fn getRoutes() []const Route {
    return gateway_routes.getRoutes();
}

pub fn getRouteCount() usize {
    return gateway_routes.getRouteCount();
}

pub fn matchRoute(path: []const u8, method: HttpMethod) GatewayError!?MatchResult {
    return gateway_routes.matchRoute(path, method);
}

pub fn checkRateLimit(path: []const u8) RateLimitResult {
    return gateway_pipeline.checkRateLimit(path);
}

pub fn recordUpstreamResult(upstream: []const u8, success: bool) void {
    gateway_pipeline.recordUpstreamResult(upstream, success);
}

pub fn stats() GatewayStats {
    return gateway_pipeline.stats();
}

pub fn getCircuitState(upstream: []const u8) CircuitBreakerState {
    return gateway_pipeline.getCircuitState(upstream);
}

pub fn resetCircuit(upstream: []const u8) void {
    gateway_pipeline.resetCircuit(upstream);
}

pub fn dispatchRequest(
    path: []const u8,
    method: HttpMethod,
    handler: ?*const fn (route: Route) bool,
) GatewayError!RequestResult {
    return gateway_pipeline.dispatchRequest(path, method, handler);
}

pub fn checkCircuitBreaker(upstream: []const u8) bool {
    return gateway_pipeline.checkCircuitBreaker(upstream);
}

pub fn recordLatency(latency_ns: u64) void {
    gateway_pipeline.recordLatency(latency_ns);
}

// ── Tests ──────────────────────────────────────────────────────────────

test "gateway basic route add/stats" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/users", .method = .GET, .upstream = "http://users:8080" });
    try addRoute(.{ .path = "/api/orders", .method = .POST, .upstream = "http://orders:8080" });

    const s = stats();
    try std.testing.expectEqual(@as(u32, 2), s.active_routes);
}

test "gateway route removal" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/test", .method = .GET, .upstream = "http://test:80" });
    try std.testing.expectEqual(@as(u32, 1), stats().active_routes);

    const removed = try removeRoute("/api/test");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(u32, 0), stats().active_routes);
}

test "gateway token bucket rate limiting" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{
        .path = "/api/limited",
        .method = .GET,
        .upstream = "http://backend:80",
        .rate_limit = .{
            .requests_per_second = 2,
            .burst_size = 2,
            .algorithm = .token_bucket,
        },
    });

    // First 2 requests should pass (burst)
    const r1 = checkRateLimit("/api/limited");
    try std.testing.expect(r1.allowed);
    const r2 = checkRateLimit("/api/limited");
    try std.testing.expect(r2.allowed);

    // Third should be rate limited
    const r3 = checkRateLimit("/api/limited");
    try std.testing.expect(!r3.allowed);
}

test "gateway circuit breaker state transitions" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 3,
            .reset_timeout_ms = 100,
            .half_open_max_requests = 1,
        },
    });
    defer deinit();

    // Initially closed
    try std.testing.expectEqual(CircuitBreakerState.closed, getCircuitState("backend"));

    // 3 failures should trip it
    recordUpstreamResult("backend", false);
    recordUpstreamResult("backend", false);
    recordUpstreamResult("backend", false);
    try std.testing.expectEqual(CircuitBreakerState.open, getCircuitState("backend"));

    // Reset
    resetCircuit("backend");
    try std.testing.expectEqual(CircuitBreakerState.closed, getCircuitState("backend"));
}

test "gateway too many routes" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_routes = 2 });
    defer deinit();

    try addRoute(.{ .path = "/a", .upstream = "http://a" });
    try addRoute(.{ .path = "/b", .upstream = "http://b" });

    const result = addRoute(.{ .path = "/c", .upstream = "http://c" });
    try std.testing.expectError(error.TooManyRoutes, result);
}

test "gateway matchRoute returns correct route" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/users", .method = .GET, .upstream = "http://users:8080" });
    try addRoute(.{ .path = "/api/orders", .method = .POST, .upstream = "http://orders:8080" });

    const match1 = try matchRoute("/api/users", .GET);
    try std.testing.expect(match1 != null);
    try std.testing.expectEqualStrings("/api/users", match1.?.route.path);
    try std.testing.expectEqual(HttpMethod.GET, match1.?.route.method);

    const match2 = try matchRoute("/api/orders", .POST);
    try std.testing.expect(match2 != null);
    try std.testing.expectEqualStrings("/api/orders", match2.?.route.path);
}

test "gateway matchRoute extracts path parameters" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/users/{id}", .method = .GET, .upstream = "http://users:8080" });

    const match = try matchRoute("/users/42", .GET);
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(u8, 1), match.?.param_count);
    const id_val = match.?.getParam("id");
    try std.testing.expect(id_val != null);
    try std.testing.expectEqualStrings("42", id_val.?);
}

test "gateway matchRoute handles wildcard routes" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/*", .method = .GET, .upstream = "http://fallback:80" });

    const match = try matchRoute("/api/anything", .GET);
    try std.testing.expect(match != null);
    try std.testing.expectEqualStrings("/api/*", match.?.route.path);
}

test "gateway matchRoute returns null for method mismatch" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/data", .method = .POST, .upstream = "http://data:80" });

    const match = try matchRoute("/api/data", .GET);
    try std.testing.expect(match == null);
}

test "gateway matchRoute returns null for unregistered path" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/known", .method = .GET, .upstream = "http://known:80" });

    const match = try matchRoute("/api/unknown", .GET);
    try std.testing.expect(match == null);
}

test "gateway sliding window rate limiting" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{
        .path = "/api/sw",
        .method = .GET,
        .upstream = "http://backend:80",
        .rate_limit = .{
            .requests_per_second = 2,
            .burst_size = 2,
            .algorithm = .sliding_window,
        },
    });

    const r1 = checkRateLimit("/api/sw");
    try std.testing.expect(r1.allowed);
    const r2 = checkRateLimit("/api/sw");
    try std.testing.expect(r2.allowed);
    const r3 = checkRateLimit("/api/sw");
    try std.testing.expect(!r3.allowed);
}

test "gateway fixed window rate limiting" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{
        .path = "/api/fw",
        .method = .GET,
        .upstream = "http://backend:80",
        .rate_limit = .{
            .requests_per_second = 3,
            .burst_size = 3,
            .algorithm = .fixed_window,
        },
    });

    const r1 = checkRateLimit("/api/fw");
    try std.testing.expect(r1.allowed);
    const r2 = checkRateLimit("/api/fw");
    try std.testing.expect(r2.allowed);
    const r3 = checkRateLimit("/api/fw");
    try std.testing.expect(r3.allowed);
    const r4 = checkRateLimit("/api/fw");
    try std.testing.expect(!r4.allowed);
}

test "gateway circuit breaker half-open to closed" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 2,
            .reset_timeout_ms = 1, // 1ms timeout for test
            .half_open_max_requests = 1,
        },
    });
    defer deinit();

    // Trip the circuit breaker
    recordUpstreamResult("svc", false);
    recordUpstreamResult("svc", false);
    try std.testing.expectEqual(CircuitBreakerState.open, getCircuitState("svc"));

    // Force transition to half_open for testing
    const s = gateway_state.gw_state.?;
    s.rw_lock.lock();
    if (s.circuit_breakers.get("svc")) |cb| {
        cb.forceHalfOpen();
    }
    s.rw_lock.unlock();

    try std.testing.expectEqual(CircuitBreakerState.half_open, getCircuitState("svc"));

    // Success in half_open → closed
    recordUpstreamResult("svc", true);
    try std.testing.expectEqual(CircuitBreakerState.closed, getCircuitState("svc"));
}

test "gateway circuit breaker half-open to open on failure" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 2,
            .reset_timeout_ms = 1,
            .half_open_max_requests = 2,
        },
    });
    defer deinit();

    // Trip the circuit breaker
    recordUpstreamResult("svc2", false);
    recordUpstreamResult("svc2", false);
    try std.testing.expectEqual(CircuitBreakerState.open, getCircuitState("svc2"));

    // Force half_open
    const s = gateway_state.gw_state.?;
    s.rw_lock.lock();
    if (s.circuit_breakers.get("svc2")) |cb| {
        cb.forceHalfOpen();
    }
    s.rw_lock.unlock();
    try std.testing.expectEqual(CircuitBreakerState.half_open, getCircuitState("svc2"));

    // Failure in half_open → back to open
    recordUpstreamResult("svc2", false);
    try std.testing.expectEqual(CircuitBreakerState.open, getCircuitState("svc2"));
}

test "gateway getParam on match result" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/items/{category}/{id}", .method = .GET, .upstream = "http://items:80" });

    const match = try matchRoute("/items/books/123", .GET);
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(u8, 2), match.?.param_count);

    // Params are collected in reverse order (recursive descent)
    const cat = match.?.getParam("category");
    try std.testing.expect(cat != null);
    const id = match.?.getParam("id");
    try std.testing.expect(id != null);
}

test "gateway stats accuracy after operations" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/stats", .method = .GET, .upstream = "http://s:80" });
    _ = try matchRoute("/api/stats", .GET);
    _ = try matchRoute("/api/stats", .GET);
    _ = try matchRoute("/api/missing", .GET);

    const s = stats();
    try std.testing.expectEqual(@as(u64, 3), s.total_requests);
    try std.testing.expectEqual(@as(u32, 1), s.active_routes);
}

test "gateway route with rate limit config" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{
        .path = "/api/ratelimited",
        .method = .GET,
        .upstream = "http://rl:80",
        .rate_limit = .{
            .requests_per_second = 10,
            .burst_size = 10,
            .algorithm = .token_bucket,
        },
    });

    // Verify rate limiter was created for the route
    const result = checkRateLimit("/api/ratelimited");
    try std.testing.expect(result.allowed);
}

test "gateway re-initialization guard" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    // Second init should be a no-op
    try init(allocator, .{ .max_routes = 1 });
    // Should still use original config (default max_routes, not 1)
    try addRoute(.{ .path = "/a", .upstream = "http://a" });
    try addRoute(.{ .path = "/b", .upstream = "http://b" });
    // If second init had taken effect, this would fail with TooManyRoutes
}

test "gateway remove then match" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/temp", .method = .GET, .upstream = "http://temp:80" });
    const before = try matchRoute("/api/temp", .GET);
    try std.testing.expect(before != null);

    const removed = try removeRoute("/api/temp");
    try std.testing.expect(removed);

    const after = try matchRoute("/api/temp", .GET);
    try std.testing.expect(after == null);
}

test "gateway resetCircuit on already closed is no-op" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    // resetCircuit on a circuit that was never opened — should not crash
    resetCircuit("http://backend:80");

    try addRoute(.{ .path = "/test", .method = .GET, .upstream = "http://backend:80" });
    resetCircuit("http://backend:80");

    // Route still works
    const result = try matchRoute("/test", .GET);
    try std.testing.expect(result != null);
}

test "gateway root path matches" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/", .method = .GET, .upstream = "http://root:80" });
    const result = try matchRoute("/", .GET);
    try std.testing.expect(result != null);
}

test "gateway getRoutes returns all routes" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/a", .method = .GET, .upstream = "http://a:80" });
    try addRoute(.{ .path = "/b", .method = .POST, .upstream = "http://b:80" });
    try addRoute(.{ .path = "/c", .method = .PUT, .upstream = "http://c:80" });

    const routes = getRoutes();
    try std.testing.expectEqual(@as(usize, 3), routes.len);
}

test "dispatchRequest returns 404 for unknown route" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    const result = try dispatchRequest("/nonexistent", .GET, null);
    try std.testing.expectEqual(HttpStatus.not_found, result.status);
    try std.testing.expect(result.match == null);
}

test "dispatchRequest succeeds for valid route" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/dispatch", .method = .GET, .upstream = "http://backend:80" });

    const result = try dispatchRequest("/api/dispatch", .GET, null);
    try std.testing.expectEqual(HttpStatus.ok, result.status);
    try std.testing.expect(result.match != null);
    try std.testing.expectEqualStrings("/api/dispatch", result.match.?.route.path);
}

test "dispatchRequest returns 429 when rate limited" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{
        .path = "/api/rl",
        .method = .GET,
        .upstream = "http://backend:80",
        .rate_limit = .{
            .requests_per_second = 1,
            .burst_size = 1,
            .algorithm = .token_bucket,
        },
    });

    // First request should succeed
    const r1 = try dispatchRequest("/api/rl", .GET, null);
    try std.testing.expectEqual(HttpStatus.ok, r1.status);

    // Second request should be rate limited
    const r2 = try dispatchRequest("/api/rl", .GET, null);
    try std.testing.expectEqual(HttpStatus.too_many_requests, r2.status);
    try std.testing.expect(r2.rate_limit != null);
    try std.testing.expect(!r2.rate_limit.?.allowed);
}

test "dispatchRequest returns 503 when circuit breaker is open" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 2,
            .reset_timeout_ms = 60_000, // long timeout so it stays open
            .half_open_max_requests = 1,
        },
    });
    defer deinit();

    try addRoute(.{ .path = "/api/cb", .method = .GET, .upstream = "http://flaky:80" });

    // Trip the circuit breaker
    recordUpstreamResult("http://flaky:80", false);
    recordUpstreamResult("http://flaky:80", false);
    try std.testing.expectEqual(CircuitBreakerState.open, getCircuitState("http://flaky:80"));

    // Request should be short-circuited with 503
    const result = try dispatchRequest("/api/cb", .GET, null);
    try std.testing.expectEqual(HttpStatus.service_unavailable, result.status);
}

test "dispatchRequest returns 502 on handler failure and records upstream error" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/fail", .method = .POST, .upstream = "http://fail:80" });

    const failing_handler: *const fn (Route) bool = &struct {
        fn handler(_: Route) bool {
            return false;
        }
    }.handler;

    const result = try dispatchRequest("/api/fail", .POST, failing_handler);
    try std.testing.expectEqual(HttpStatus.bad_gateway, result.status);
    // Upstream error should have been recorded
    try std.testing.expect(stats().upstream_errors > 0);
}

test "dispatchRequest records latency in histogram" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    try addRoute(.{ .path = "/api/latency", .method = .GET, .upstream = "http://lat:80" });

    _ = try dispatchRequest("/api/latency", .GET, null);

    // After dispatch, the histogram should have at least 1 entry
    const s = gateway_state.gw_state.?;
    s.rw_lock.lockShared();
    const count = s.latency.count;
    s.rw_lock.unlockShared();
    try std.testing.expect(count > 0);
}

test "checkCircuitBreaker allows when closed" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    // No circuit breaker created yet — should default to allowed
    try std.testing.expect(checkCircuitBreaker("http://unknown:80"));
}

test "checkCircuitBreaker blocks when open" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 2,
            .reset_timeout_ms = 60_000,
            .half_open_max_requests = 1,
        },
    });
    defer deinit();

    recordUpstreamResult("http://down:80", false);
    recordUpstreamResult("http://down:80", false);

    try std.testing.expect(!checkCircuitBreaker("http://down:80"));
}

test "recordLatency updates histogram" {
    const allocator = std.testing.allocator;
    try init(allocator, GatewayConfig.defaults());
    defer deinit();

    recordLatency(5 * std.time.ns_per_ms); // 5ms
    recordLatency(50 * std.time.ns_per_ms); // 50ms

    const s = gateway_state.gw_state.?;
    s.rw_lock.lockShared();
    const count = s.latency.count;
    s.rw_lock.unlockShared();
    try std.testing.expectEqual(@as(u64, 2), count);

    const st = stats();
    try std.testing.expect(st.avg_latency_ms > 0);
}

test "dispatchRequest pipeline: rate limit then circuit breaker ordering" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 1,
            .reset_timeout_ms = 60_000,
            .half_open_max_requests = 1,
        },
    });
    defer deinit();

    try addRoute(.{
        .path = "/api/order",
        .method = .GET,
        .upstream = "http://order:80",
        .rate_limit = .{
            .requests_per_second = 1,
            .burst_size = 1,
            .algorithm = .token_bucket,
        },
    });

    // First request succeeds, consuming rate limit token
    const r1 = try dispatchRequest("/api/order", .GET, null);
    try std.testing.expectEqual(HttpStatus.ok, r1.status);

    // Second request hits rate limiter (before circuit breaker check)
    const r2 = try dispatchRequest("/api/order", .GET, null);
    try std.testing.expectEqual(HttpStatus.too_many_requests, r2.status);
}

test {
    std.testing.refAllDecls(@This());
}
