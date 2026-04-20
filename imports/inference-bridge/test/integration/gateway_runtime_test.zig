//! Deeper public-runtime integration coverage for the gateway facade.

const std = @import("std");
const abi = @import("abi");

const gateway = abi.gateway;

test "gateway runtime: init and deinit toggle initialization state" {
    gateway.deinit();
    try std.testing.expect(!gateway.isInitialized());

    try gateway.init(std.testing.allocator, gateway.GatewayConfig.defaults());
    try std.testing.expect(gateway.isInitialized());

    gateway.deinit();
    try std.testing.expect(!gateway.isInitialized());
}

test "gateway runtime: addRoute/removeRoute and radix reindex remain coherent" {
    gateway.deinit();
    try gateway.init(std.testing.allocator, gateway.GatewayConfig.defaults());
    defer gateway.deinit();

    try gateway.addRoute(.{ .path = "/one", .method = .GET, .upstream = "http://one:80" });
    try gateway.addRoute(.{ .path = "/two", .method = .GET, .upstream = "http://two:80" });
    try gateway.addRoute(.{ .path = "/three", .method = .GET, .upstream = "http://three:80" });
    try std.testing.expectEqual(@as(usize, 3), gateway.getRouteCount());

    const removed = try gateway.removeRoute("/two");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 2), gateway.getRouteCount());

    const match = try gateway.matchRoute("/three", .GET);
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(u32, 1), match.?.matched_route_idx.?);
}

test "gateway runtime: dispatchRequest succeeds through the public facade" {
    gateway.deinit();
    try gateway.init(std.testing.allocator, gateway.GatewayConfig.defaults());
    defer gateway.deinit();

    try gateway.addRoute(.{ .path = "/ok", .method = .GET, .upstream = "http://ok:80" });

    const result = try gateway.dispatchRequest("/ok", .GET, null);
    try std.testing.expectEqual(gateway.HttpStatus.ok, result.status);
    try std.testing.expect(result.match != null);
    try std.testing.expectEqualStrings("/ok", result.match.?.route.path);
}

test "gateway runtime: dispatchRequest reports rate-limit rejection" {
    gateway.deinit();
    try gateway.init(std.testing.allocator, gateway.GatewayConfig.defaults());
    defer gateway.deinit();

    try gateway.addRoute(.{
        .path = "/limited",
        .method = .GET,
        .upstream = "http://limited:80",
        .rate_limit = .{
            .requests_per_second = 1,
            .burst_size = 1,
            .algorithm = .token_bucket,
        },
    });

    const first = try gateway.dispatchRequest("/limited", .GET, null);
    try std.testing.expectEqual(gateway.HttpStatus.ok, first.status);

    const second = try gateway.dispatchRequest("/limited", .GET, null);
    try std.testing.expectEqual(gateway.HttpStatus.too_many_requests, second.status);
    try std.testing.expect(second.rate_limit != null);
    try std.testing.expect(!second.rate_limit.?.allowed);
}

test "gateway runtime: dispatchRequest reports circuit-breaker rejection" {
    gateway.deinit();
    try gateway.init(std.testing.allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 1,
            .reset_timeout_ms = 60_000,
            .half_open_max_requests = 1,
        },
    });
    defer gateway.deinit();

    try gateway.addRoute(.{ .path = "/flaky", .method = .GET, .upstream = "http://flaky:80" });
    gateway.recordUpstreamResult("http://flaky:80", false);

    try std.testing.expectEqual(
        gateway.CircuitBreakerState.open,
        gateway.getCircuitState("http://flaky:80"),
    );

    const result = try gateway.dispatchRequest("/flaky", .GET, null);
    try std.testing.expectEqual(gateway.HttpStatus.service_unavailable, result.status);
}

test "gateway runtime: upstream handler failures are recorded publicly" {
    gateway.deinit();
    try gateway.init(std.testing.allocator, gateway.GatewayConfig.defaults());
    defer gateway.deinit();

    try gateway.addRoute(.{ .path = "/fail", .method = .POST, .upstream = "http://fail:80" });

    const failing_handler: *const fn (gateway.Route) bool = &struct {
        fn handler(_: gateway.Route) bool {
            return false;
        }
    }.handler;

    const result = try gateway.dispatchRequest("/fail", .POST, failing_handler);
    try std.testing.expectEqual(gateway.HttpStatus.bad_gateway, result.status);
    try std.testing.expectEqual(@as(u64, 1), gateway.stats().upstream_errors);
}

test {
    std.testing.refAllDecls(@This());
}
