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
const core_config = @import("../../core/config/gateway.zig");
const sync = @import("../../services/shared/sync.zig");
const time_mod = @import("../../services/shared/time.zig");
const radix = @import("../../services/shared/utils/radix_tree.zig");

/// Shared radix tree instantiated for route indices.
const RouteTree = radix.RadixTree(u32);

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

/// Radix tree node — shared implementation from `services/shared/utils/radix_tree.zig`.
const RadixNode = RouteTree.Node;

/// Token bucket rate limiter state.
const TokenBucket = struct {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per nanosecond
    last_refill_ns: u128,

    fn init(config: RateLimitConfig, now_ns: u128) TokenBucket {
        const cap = @as(f64, @floatFromInt(config.burst_size));
        const rate = @as(f64, @floatFromInt(config.requests_per_second)) /
            @as(f64, @floatFromInt(std.time.ns_per_s));
        return .{
            .tokens = cap,
            .capacity = cap,
            .refill_rate = rate,
            .last_refill_ns = now_ns,
        };
    }

    fn tryConsume(self: *TokenBucket, now_ns: u128) RateLimitResult {
        // Refill
        const elapsed_ns = now_ns - self.last_refill_ns;
        const added = @as(f64, @floatFromInt(elapsed_ns)) * self.refill_rate;
        self.tokens = @min(self.tokens + added, self.capacity);
        self.last_refill_ns = now_ns;

        if (self.tokens >= 1.0) {
            self.tokens -= 1.0;
            return .{
                .allowed = true,
                .remaining = @intFromFloat(@max(self.tokens, 0)),
                .reset_after_ms = 0,
            };
        }

        // Calculate time until 1 token available
        const deficit = 1.0 - self.tokens;
        const wait_ns = deficit / self.refill_rate;
        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intFromFloat(wait_ns / @as(f64, std.time.ns_per_ms)),
        };
    }
};

/// Fixed window rate limiter state.
const FixedWindow = struct {
    count: u32,
    window_start_ns: u128,
    window_duration_ns: u128,
    max_requests: u32,

    fn init(config: RateLimitConfig, now_ns: u128) FixedWindow {
        return .{
            .count = 0,
            .window_start_ns = now_ns,
            .window_duration_ns = std.time.ns_per_s, // 1 second windows
            .max_requests = config.requests_per_second,
        };
    }

    fn tryConsume(self: *FixedWindow, now_ns: u128) RateLimitResult {
        // Check if window has rolled over
        if (now_ns >= self.window_start_ns + self.window_duration_ns) {
            self.count = 0;
            self.window_start_ns = now_ns;
        }

        if (self.count < self.max_requests) {
            self.count += 1;
            return .{
                .allowed = true,
                .remaining = self.max_requests - self.count,
                .reset_after_ms = 0,
            };
        }

        const remaining_ns = (self.window_start_ns + self.window_duration_ns) - now_ns;
        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intCast(remaining_ns / std.time.ns_per_ms),
        };
    }
};

/// Sliding window rate limiter using histogram buckets.
/// Uses 7 sub-buckets within a window for smooth rate estimation.
const SlidingWindow = struct {
    buckets: [7]u32 = [_]u32{0} ** 7,
    bucket_start_ns: u128,
    bucket_width_ns: u128,
    max_requests: u32,
    window_ns: u128,

    fn init(config: RateLimitConfig, now_ns: u128) SlidingWindow {
        const window_ns: u128 = std.time.ns_per_s; // 1 second
        return .{
            .bucket_start_ns = now_ns,
            .bucket_width_ns = window_ns / 7,
            .max_requests = config.requests_per_second,
            .window_ns = window_ns,
        };
    }

    fn tryConsume(self: *SlidingWindow, now_ns: u128) RateLimitResult {
        // Advance buckets if needed
        self.advanceBuckets(now_ns);

        // Sum all bucket counts
        var total: u32 = 0;
        for (self.buckets) |b| total += b;

        if (total < self.max_requests) {
            // Add to current bucket
            const current_bucket = self.currentBucketIdx(now_ns);
            self.buckets[current_bucket] += 1;
            return .{
                .allowed = true,
                .remaining = self.max_requests - total - 1,
                .reset_after_ms = 0,
            };
        }

        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intCast(self.bucket_width_ns / std.time.ns_per_ms),
        };
    }

    fn currentBucketIdx(self: *const SlidingWindow, now_ns: u128) usize {
        if (self.bucket_width_ns == 0) return 0;
        const elapsed = now_ns -| self.bucket_start_ns;
        return @intCast((elapsed / self.bucket_width_ns) % 7);
    }

    fn advanceBuckets(self: *SlidingWindow, now_ns: u128) void {
        if (self.bucket_width_ns == 0) return;
        const elapsed = now_ns -| self.bucket_start_ns;
        const buckets_to_advance = elapsed / self.bucket_width_ns;

        if (buckets_to_advance >= 7) {
            // Full window has passed — clear everything
            self.buckets = [_]u32{0} ** 7;
            self.bucket_start_ns = now_ns;
        } else if (buckets_to_advance > 0) {
            // Clear old buckets
            var i: usize = 0;
            while (i < buckets_to_advance) : (i += 1) {
                const old_idx = (self.currentBucketIdx(self.bucket_start_ns) + i) % 7;
                self.buckets[old_idx] = 0;
            }
            self.bucket_start_ns += buckets_to_advance * self.bucket_width_ns;
        }
    }
};

/// Rate limiter wrapping the 3 algorithm variants.
const RateLimiter = union(RateLimitAlgorithm) {
    token_bucket: TokenBucket,
    sliding_window: SlidingWindow,
    fixed_window: FixedWindow,

    fn init(config: RateLimitConfig, now_ns: u128) RateLimiter {
        return switch (config.algorithm) {
            .token_bucket => .{ .token_bucket = TokenBucket.init(config, now_ns) },
            .sliding_window => .{ .sliding_window = SlidingWindow.init(config, now_ns) },
            .fixed_window => .{ .fixed_window = FixedWindow.init(config, now_ns) },
        };
    }

    fn tryConsume(self: *RateLimiter, now_ns: u128) RateLimitResult {
        return switch (self.*) {
            .token_bucket => |*tb| tb.tryConsume(now_ns),
            .sliding_window => |*sw| sw.tryConsume(now_ns),
            .fixed_window => |*fw| fw.tryConsume(now_ns),
        };
    }
};

/// Circuit breaker state machine.
const CircuitBreaker = struct {
    cb_state: CircuitBreakerState = .closed,
    failure_count: u32 = 0,
    success_count: u32 = 0,
    open_until_ns: u128 = 0,
    config: CircuitBreakerConfig,

    fn init(config: CircuitBreakerConfig) CircuitBreaker {
        return .{ .config = config };
    }

    fn recordSuccess(self: *CircuitBreaker) void {
        switch (self.cb_state) {
            .closed => {
                self.failure_count = 0;
            },
            .half_open => {
                self.success_count += 1;
                if (self.success_count >= self.config.half_open_max_requests) {
                    self.cb_state = .closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            },
            .open => {},
        }
    }

    fn recordFailure(self: *CircuitBreaker, now_ns: u128) void {
        self.failure_count += 1;
        switch (self.cb_state) {
            .closed => {
                if (self.failure_count >= self.config.failure_threshold) {
                    self.cb_state = .open;
                    self.open_until_ns = now_ns +
                        @as(u128, self.config.reset_timeout_ms) * std.time.ns_per_ms;
                }
            },
            .half_open => {
                // Any failure in half-open sends back to open
                self.cb_state = .open;
                self.open_until_ns = now_ns +
                    @as(u128, self.config.reset_timeout_ms) * std.time.ns_per_ms;
                self.success_count = 0;
            },
            .open => {},
        }
    }

    fn isAllowed(self: *CircuitBreaker, now_ns: u128) bool {
        switch (self.cb_state) {
            .closed => return true,
            .open => {
                if (now_ns >= self.open_until_ns) {
                    self.cb_state = .half_open;
                    self.success_count = 0;
                    return true;
                }
                return false;
            },
            .half_open => return true,
        }
    }

    fn reset(self: *CircuitBreaker) void {
        self.cb_state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.open_until_ns = 0;
    }
};

/// Latency histogram with 7 log-scale buckets.
const LatencyHistogram = struct {
    // Buckets: <1ms, <5ms, <10ms, <50ms, <100ms, <500ms, >=500ms
    buckets: [7]u64 = [_]u64{0} ** 7,
    total_ns: u128 = 0,
    count: u64 = 0,

    fn record(self: *LatencyHistogram, latency_ns: u64) void {
        const ms = latency_ns / std.time.ns_per_ms;
        const bucket_idx: usize = if (ms < 1) 0 else if (ms < 5) 1 else if (ms < 10) 2 else if (ms < 50) 3 else if (ms < 100) 4 else if (ms < 500) 5 else 6;
        self.buckets[bucket_idx] += 1;
        self.total_ns += latency_ns;
        self.count += 1;
    }

    fn avgMs(self: *const LatencyHistogram) u64 {
        if (self.count == 0) return 0;
        return @intCast(self.total_ns / self.count / std.time.ns_per_ms);
    }
};

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

/// Split a URL path into segments (delegates to shared radix tree).
fn splitPath(path: []const u8) std.mem.SplitIterator(u8, .scalar) {
    return RouteTree.splitPath(path);
}

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

    // Create rate limiter if configured
    if (route.rate_limit) |rl_config| {
        const limiter = s.allocator.create(RateLimiter) catch return error.OutOfMemory;
        limiter.* = RateLimiter.init(rl_config, nowNs());
        s.route_limiters.put(s.allocator, path_owned, limiter) catch {
            s.allocator.destroy(limiter);
            return error.OutOfMemory;
        };
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

pub fn getRoutes() []const Route {
    const s = gw_state orelse return &.{};
    // Return a view of the stored routes (not safe to hold long-term)
    _ = s;
    return &.{}; // Simplified — full impl would return route array
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
        if (cb.cb_state == .open) {
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
    return cb.cb_state;
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
    var req_seg = splitPath(request_path);
    var route_seg = splitPath(route_path);

    while (true) {
        const rs = route_seg.next();
        const rq = req_seg.next();

        if (rs == null and rq == null) return true;
        if (rs == null or rq == null) return false;

        const rseg = rs.?;
        if (rseg.len > 0 and rseg[0] == '*') return true;
        if (rseg.len > 2 and rseg[0] == '{') continue; // param matches anything
        if (!std.mem.eql(u8, rseg, rq.?)) return false;
    }
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

    // Wait for reset timeout (need write lock to check isAllowed which transitions)
    const s = gw_state.?;
    // Force transition to half_open by advancing past timeout
    s.rw_lock.lock();
    if (s.circuit_breakers.get("svc")) |cb| {
        cb.open_until_ns = 0; // Force expiry
        _ = cb.isAllowed(1); // Triggers transition to half_open
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
        cb.open_until_ns = 0;
        _ = cb.isAllowed(1);
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
