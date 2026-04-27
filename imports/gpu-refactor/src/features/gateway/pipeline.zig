const std = @import("std");
const circuit_breaker_mod = @import("circuit_breaker.zig");
const gateway_types = @import("types.zig");
const gateway_routes = @import("routes.zig");
const gateway_state = @import("state.zig");

const CircuitBreaker = circuit_breaker_mod.CircuitBreaker;

pub const HttpStatus = enum(u16) {
    ok = 200,
    too_many_requests = 429,
    service_unavailable = 503,
    bad_gateway = 502,
    not_found = 404,
};

pub const RequestResult = struct {
    status: HttpStatus = .ok,
    match: ?gateway_types.MatchResult = null,
    rate_limit: ?gateway_types.RateLimitResult = null,
    latency_ns: u64 = 0,
};

pub fn checkRateLimit(path: []const u8) gateway_types.RateLimitResult {
    const s = gateway_state.gw_state orelse return .{};

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const limiter = s.route_limiters.get(path) orelse return .{ .allowed = true };
    const result = limiter.tryConsume(gateway_state.nowNs());
    if (!result.allowed) {
        _ = s.stat_rate_limited.fetchAdd(1, .monotonic);
    }
    return result;
}

pub fn recordUpstreamResult(upstream: []const u8, success: bool) void {
    const s = gateway_state.gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

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
        cb.recordFailure(gateway_state.nowNs());
        _ = s.stat_upstream_errors.fetchAdd(1, .monotonic);
        if (cb.inner.getState() == .open) {
            _ = s.stat_cb_trips.fetchAdd(1, .monotonic);
        }
    }
}

pub fn stats() gateway_types.GatewayStats {
    const s = gateway_state.gw_state orelse return .{};
    return .{
        .total_requests = s.stat_total_requests.load(.monotonic),
        .active_routes = @intCast(s.routes.items.len),
        .rate_limited_count = s.stat_rate_limited.load(.monotonic),
        .circuit_breaker_trips = s.stat_cb_trips.load(.monotonic),
        .upstream_errors = s.stat_upstream_errors.load(.monotonic),
        .avg_latency_ms = s.latency.avgMs(),
    };
}

pub fn getCircuitState(upstream: []const u8) gateway_types.CircuitBreakerState {
    const s = gateway_state.gw_state orelse return .closed;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const cb = s.circuit_breakers.get(upstream) orelse return .closed;
    return switch (cb.inner.getState()) {
        .closed => .closed,
        .open => .open,
        .half_open => .half_open,
    };
}

pub fn resetCircuit(upstream: []const u8) void {
    const s = gateway_state.gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.circuit_breakers.get(upstream)) |cb| {
        cb.reset();
    }
}

pub fn dispatchRequest(
    path: []const u8,
    method: gateway_types.HttpMethod,
    handler: ?*const fn (route: gateway_types.Route) bool,
) gateway_types.GatewayError!RequestResult {
    const s = gateway_state.gw_state orelse return error.FeatureDisabled;
    const start_ns = gateway_state.nowNs();

    const match_result = try gateway_routes.matchRoute(path, method);
    if (match_result == null) {
        return .{ .status = .not_found };
    }
    const matched = match_result.?;

    const rl_result = checkRateLimit(path);
    if (!rl_result.allowed) {
        return .{
            .status = .too_many_requests,
            .match = matched,
            .rate_limit = rl_result,
        };
    }

    const upstream = matched.route.upstream;
    const cb_allowed = checkCircuitBreaker(upstream);
    if (!cb_allowed) {
        return .{
            .status = .service_unavailable,
            .match = matched,
            .rate_limit = rl_result,
        };
    }

    const handler_success = if (handler) |h| h(matched.route) else true;
    recordUpstreamResult(upstream, handler_success);

    const end_ns = gateway_state.nowNs();
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

pub fn checkCircuitBreaker(upstream: []const u8) bool {
    const s = gateway_state.gw_state orelse return true;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const cb = s.circuit_breakers.get(upstream) orelse return true;
    return cb.isAllowed(gateway_state.nowNs());
}

pub fn recordLatency(latency_ns: u64) void {
    const s = gateway_state.gw_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();
    s.latency.record(latency_ns);
}

test {
    std.testing.refAllDecls(@This());
}
