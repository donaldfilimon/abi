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
const sync = @import("../../foundation/mod.zig").sync;
const time_mod = @import("../../foundation/mod.zig").time;
pub const gateway_types = @import("types.zig");
const routing = @import("routing.zig");
const rate_limit_mod = @import("rate_limit.zig");
const circuit_breaker_mod = @import("circuit_breaker.zig");
const gateway_stats = @import("stats.zig");
const middleware = @import("middleware.zig");

/// Shared radix tree instantiated for route indices.
const RouteTree = routing.RouteTree;

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

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: GatewayConfig,

    pub fn init(allocator: std.mem.Allocator, config: GatewayConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Internal Types ─────────────────────────────────────────────────────

const RouteEntry = struct {
    route: Route,
    path_owned: []u8,
    upstream_owned: []u8,
};

/// Radix tree node — shared implementation from `foundation/utils/radix_tree.zig`.
const RadixNode = RouteTree.Node;
const RateLimiter = rate_limit_mod.RateLimiter;
const CircuitBreaker = circuit_breaker_mod.CircuitBreaker;
const LatencyHistogram = gateway_stats.LatencyHistogram;

// ── Module State ───────────────────────────────────────────────────────

var gw_state: ?*GatewayState = null;

const GatewayState = struct {
    allocator: std.mem.Allocator,
    config: GatewayConfig,
    routes: std.ArrayListUnmanaged(RouteEntry),
    radix_root: *RadixNode,
    route_limiters: std.StringHashMapUnmanaged(*RateLimiter),
    circuit_breakers: std.StringHashMapUnmanaged(*CircuitBreaker),
    latency: LatencyHistogram,
    rw_lock: sync.RwLock,

    // Stats
    stat_total_requests: std.atomic.Value(u64),
    stat_rate_limited: std.atomic.Value(u64),
    stat_cb_trips: std.atomic.Value(u64),
    stat_upstream_errors: std.atomic.Value(u64),

    fn create(allocator: std.mem.Allocator, config: GatewayConfig) !*GatewayState {
        const root = try allocator.create(RadixNode);
        root.* = .{};

        const s = try allocator.create(GatewayState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .routes = .empty,
            .radix_root = root,
            .route_limiters = .empty,
            .circuit_breakers = .empty,
            .latency = .{},
            .rw_lock = sync.RwLock.init(),
            .stat_total_requests = std.atomic.Value(u64).init(0),
            .stat_rate_limited = std.atomic.Value(u64).init(0),
            .stat_cb_trips = std.atomic.Value(u64).init(0),
            .stat_upstream_errors = std.atomic.Value(u64).init(0),
        };
        return s;
    }

    fn destroy(self: *GatewayState) void {
        const allocator = self.allocator;

        // Free route entries
        for (self.routes.items) |entry| {
            allocator.free(entry.path_owned);
            allocator.free(entry.upstream_owned);
        }
        self.routes.deinit(allocator);

        // Free radix tree
        self.radix_root.deinitRecursive(allocator);
        allocator.destroy(self.radix_root);

        // Free rate limiters
        var rl_iter = self.route_limiters.iterator();
        while (rl_iter.next()) |entry| {
            allocator.destroy(entry.value_ptr.*);
        }
        self.route_limiters.deinit(allocator);

        // Free circuit breakers
        var cb_iter = self.circuit_breakers.iterator();
        while (cb_iter.next()) |entry| {
            allocator.destroy(entry.value_ptr.*);
        }
        self.circuit_breakers.deinit(allocator);

        allocator.destroy(self);
    }

    fn insertRadixRoute(
        self: *GatewayState,
        path: []const u8,
        route_idx: u32,
    ) !void {
        try RouteTree.insert(self.radix_root, self.allocator, path, route_idx);
    }
};

fn nowNs() u128 {
    return (time_mod.Instant.now() catch return 0).nanos;
}

// ── Public API ─────────────────────────────────────────────────────────

/// Initialize the API gateway singleton with routing, rate limiting,
/// and circuit breaker configuration.
pub fn init(allocator: std.mem.Allocator, config: GatewayConfig) GatewayError!void {
    if (gw_state != null) return;
    gw_state = GatewayState.create(allocator, config) catch return error.OutOfMemory;
}

/// Tear down the gateway, freeing all routes and internal state.
pub fn deinit() void {
    if (gw_state) |s| {
        s.destroy();
        gw_state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return gw_state != null;
}

/// Register an API route (method + path pattern). Supports path parameters
/// (`{id}`) and wildcards (`*`).
pub fn addRoute(route: Route) GatewayError!void {
    const s = gw_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.routes.items.len >= s.config.max_routes) return error.TooManyRoutes;
    if (route.path.len == 0) return error.InvalidRoute;

    // Pre-create rate limiter if configured (before adding route to avoid
    // orphan routes if limiter allocation fails)
    var limiter: ?*RateLimiter = null;
    if (route.rate_limit) |rl_config| {
        s.route_limiters.ensureUnusedCapacity(s.allocator, 1) catch
            return error.OutOfMemory;
        limiter = s.allocator.create(RateLimiter) catch return error.OutOfMemory;
        limiter.?.* = RateLimiter.init(rl_config, nowNs());
    }
    errdefer if (limiter) |l| s.allocator.destroy(l);

    const route_idx: u32 = @intCast(s.routes.items.len);

    // Copy owned data — explicit cleanup on each failure (no errdefer since ownership
    // transfers to s.routes on successful append)
    const path_owned = s.allocator.dupe(u8, route.path) catch return error.OutOfMemory;
    const upstream_owned = s.allocator.dupe(u8, route.upstream) catch {
        s.allocator.free(path_owned);
        return error.OutOfMemory;
    };

    s.routes.append(s.allocator, .{
        .route = .{
            .path = path_owned,
            .method = route.method,
            .upstream = upstream_owned,
            .timeout_ms = route.timeout_ms,
            .rate_limit = route.rate_limit,
            .middlewares = route.middlewares,
        },
        .path_owned = path_owned,
        .upstream_owned = upstream_owned,
    }) catch {
        s.allocator.free(path_owned);
        s.allocator.free(upstream_owned);
        return error.OutOfMemory;
    };

    // Insert into radix tree — roll back route entry on failure
    s.insertRadixRoute(path_owned, route_idx) catch {
        _ = s.routes.pop();
        s.allocator.free(path_owned);
        s.allocator.free(upstream_owned);
        return error.OutOfMemory;
    };

    // Register pre-created rate limiter — infallible due to ensureUnusedCapacity
    if (limiter) |l| {
        s.route_limiters.putAssumeCapacity(path_owned, l);
    }
}

/// Remove all routes registered under a given path. Returns `true` if any were removed.
pub fn removeRoute(path: []const u8) GatewayError!bool {
    const s = gw_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    for (s.routes.items, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.path_owned, path)) {
            // Remove rate limiter
            if (s.route_limiters.fetchRemove(entry.path_owned)) |kv| {
                s.allocator.destroy(kv.value);
            }

            s.allocator.free(entry.path_owned);
            s.allocator.free(entry.upstream_owned);
            _ = s.routes.orderedRemove(i);

            // Rebuild radix tree with corrected indices (orderedRemove shifts
            // all subsequent entries, invalidating stored route_idx values)
            s.radix_root.deinitRecursive(s.allocator);
            s.radix_root.* = .{};
            for (s.routes.items, 0..) |remaining, new_idx| {
                s.insertRadixRoute(remaining.path_owned, @intCast(new_idx)) catch |err| {
                    std.log.err("gateway: radix rebuild failed after removeRoute: {t}", .{err});
                    return error.OutOfMemory;
                };
            }
            return true;
        }
    }
    return false;
}

/// Module-level buffer for `getRoutes()` introspection results.
/// Bounded to 256 routes — well above typical gateway configs.
var route_view_buf: [256]Route = undefined;

pub fn getRoutes() []const Route {
    const s = gw_state orelse return &.{};
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();
    const count = @min(s.routes.items.len, route_view_buf.len);
    for (s.routes.items[0..count], 0..) |entry, i| {
        route_view_buf[i] = entry.route;
    }
    return route_view_buf[0..count];
}

pub fn getRouteCount() usize {
    const s = gw_state orelse return 0;
    return s.routes.items.len;
}

/// Match an incoming request path and method against the radix tree.
/// Returns the matching route and extracted path parameters, or `null`.
pub fn matchRoute(path: []const u8, method: HttpMethod) GatewayError!?MatchResult {
    const s = gw_state orelse return error.FeatureDisabled;
    _ = s.stat_total_requests.fetchAdd(1, .monotonic);

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    // Use the shared radix tree match
    var tree_result = RouteTree.MatchResult{};
    if (RouteTree.match(s.radix_root, path, &tree_result)) {
        if (tree_result.terminal_idx) |idx| {
            if (idx < s.routes.items.len) {
                const entry = s.routes.items[idx];
                if (entry.route.method == method) {
                    // Transfer params from shared result to gateway result
                    var result = MatchResult{ .route = entry.route };
                    result.matched_route_idx = idx;
                    result.param_count = tree_result.param_count;
                    result.params = tree_result.params;
                    return result;
                }
            }
        }
    }

    return null;
}

/// Check whether a request to `path` is allowed under the configured rate limiter.
pub fn checkRateLimit(path: []const u8) RateLimitResult {
    const s = gw_state orelse return .{};

    // Must hold exclusive lock for the entire lookup+consume to prevent
    // use-after-free if removeRoute destroys the limiter between lookup and use.
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const limiter = s.route_limiters.get(path) orelse return .{ .allowed = true };
    const result = limiter.tryConsume(nowNs());
    if (!result.allowed) {
        _ = s.stat_rate_limited.fetchAdd(1, .monotonic);
    }
    return result;
}

/// Record a success/failure for circuit breaker state tracking on `upstream`.
pub fn recordUpstreamResult(upstream: []const u8, success: bool) void {
    const s = gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    // Get or create circuit breaker
    const cb = s.circuit_breakers.get(upstream) orelse blk: {
        const new_cb = s.allocator.create(CircuitBreaker) catch return;
        new_cb.* = CircuitBreaker.init(s.config.circuit_breaker);
        s.circuit_breakers.put(s.allocator, upstream, new_cb) catch {
            s.allocator.destroy(new_cb);
            return;
        };
        break :blk new_cb;
    };

    if (success) {
        cb.recordSuccess();
    } else {
        cb.recordFailure(nowNs());
        _ = s.stat_upstream_errors.fetchAdd(1, .monotonic);
        if (cb.inner.getState() == .open) {
            _ = s.stat_cb_trips.fetchAdd(1, .monotonic);
        }
    }
}

/// Snapshot route count, request/error counters, and latency histogram.
pub fn stats() GatewayStats {
    const s = gw_state orelse return .{};
    return .{
        .total_requests = s.stat_total_requests.load(.monotonic),
        .active_routes = @intCast(s.routes.items.len),
        .rate_limited_count = s.stat_rate_limited.load(.monotonic),
        .circuit_breaker_trips = s.stat_cb_trips.load(.monotonic),
        .upstream_errors = s.stat_upstream_errors.load(.monotonic),
        .avg_latency_ms = s.latency.avgMs(),
    };
}

/// Query the circuit breaker state for an upstream service.
pub fn getCircuitState(upstream: []const u8) CircuitBreakerState {
    const s = gw_state orelse return .closed;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();
    const cb = s.circuit_breakers.get(upstream) orelse return .closed;
    return switch (cb.inner.getState()) {
        .closed => .closed,
        .open => .open,
        .half_open => .half_open,
    };
}

/// Force-close the circuit breaker for an upstream, clearing failure counters.
pub fn resetCircuit(upstream: []const u8) void {
    const s = gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();
    if (s.circuit_breakers.get(upstream)) |cb| {
        cb.reset();
    }
}

fn pathMatchesRoute(request_path: []const u8, route_path: []const u8) bool {
    return routing.pathMatchesRoute(request_path, route_path);
}

// ── Request Dispatch (wires rate limiter, circuit breaker, latency) ───

/// HTTP status codes used by the gateway dispatch path.
pub const HttpStatus = enum(u16) {
    ok = 200,
    too_many_requests = 429,
    service_unavailable = 503,
    bad_gateway = 502,
    not_found = 404,
};

/// Result of a fully dispatched gateway request, including middleware checks.
pub const RequestResult = struct {
    status: HttpStatus = .ok,
    match: ?MatchResult = null,
    rate_limit: ?RateLimitResult = null,
    latency_ns: u64 = 0,
};

/// Dispatch a request through the full gateway pipeline:
///   1. Route matching
///   2. Rate limit check (per-route, if configured)
///   3. Circuit breaker check (per-upstream)
///   4. Handler invocation (via caller-supplied function)
///   5. Record upstream result in circuit breaker
///   6. Record latency in histogram
///
/// The `handler` callback receives the matched route and should return `true`
/// on upstream success, `false` on upstream error. If `handler` is `null`, the
/// pipeline still runs steps 1-3 and records a synthetic success (useful for
/// dry-run / health-check probes).
pub fn dispatchRequest(
    path: []const u8,
    method: HttpMethod,
    handler: ?*const fn (route: Route) bool,
) GatewayError!RequestResult {
    const s = gw_state orelse return error.FeatureDisabled;
    const start_ns = nowNs();

    // Step 1: Route matching
    const match_result = try matchRoute(path, method);
    if (match_result == null) {
        return .{ .status = .not_found };
    }
    const matched = match_result.?;

    // Step 2: Rate limit check (uses the route path as key)
    const rl_result = checkRateLimit(path);
    if (!rl_result.allowed) {
        return .{
            .status = .too_many_requests,
            .match = matched,
            .rate_limit = rl_result,
        };
    }

    // Step 3: Circuit breaker check (uses the upstream as key)
    const upstream = matched.route.upstream;
    const cb_allowed = checkCircuitBreaker(upstream);
    if (!cb_allowed) {
        return .{
            .status = .service_unavailable,
            .match = matched,
            .rate_limit = rl_result,
        };
    }

    // Step 4: Invoke handler
    const handler_success = if (handler) |h| h(matched.route) else true;

    // Step 5: Record upstream result in circuit breaker
    recordUpstreamResult(upstream, handler_success);

    // Step 6: Record latency
    const end_ns = nowNs();
    const latency_ns: u64 = if (end_ns > start_ns)
        @intCast(end_ns - start_ns)
    else
        0;

    s.rw_lock.lock();
    s.latency.record(latency_ns);
    s.rw_lock.unlock();

    return .{
        .status = if (handler_success) .ok else .bad_gateway,
        .match = matched,
        .rate_limit = rl_result,
        .latency_ns = latency_ns,
    };
}

/// Check whether the circuit breaker for `upstream` allows a request through.
/// Returns `true` if the circuit is closed or half-open (probe allowed),
/// `false` if open.
pub fn checkCircuitBreaker(upstream: []const u8) bool {
    const s = gw_state orelse return true;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const cb = s.circuit_breakers.get(upstream) orelse return true;
    return cb.isAllowed(nowNs());
}

/// Record a request's latency in the gateway histogram.
pub fn recordLatency(latency_ns: u64) void {
    const s = gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();
    s.latency.record(latency_ns);
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
    const s = gw_state.?;
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
    const s = gw_state.?;
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
    const s = gw_state.?;
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

    const s = gw_state.?;
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
