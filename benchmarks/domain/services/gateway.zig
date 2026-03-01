//! Gateway Benchmarks
//!
//! Performance measurement for the API gateway module:
//! - Route registration throughput (static routes)
//! - Route matching (static, parameterized, wildcard paths)
//! - Rate limiter throughput (token bucket, sliding window, fixed window)
//! - Mixed workload (70% match, 20% rate check, 10% route add)

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

pub const GatewayBenchConfig = struct {
    route_counts: []const usize = &.{ 50, 200, 1000 },
    match_iterations: []const usize = &.{ 1000, 5000, 10_000 },
    rate_limit_iterations: []const usize = &.{ 1000, 10_000 },
    mixed_iterations: []const usize = &.{ 1000, 5000 },
};

// ── Helpers ──────────────────────────────────────────────────────────

fn generateStaticPath(buf: *[64]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "/api/v1/route-{d:0>6}", .{i}) catch
        return "/api/v1/route-000000";
    return len;
}

fn generateParamPath(buf: *[64]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "/api/users/{d}/profile", .{i}) catch
        return "/api/users/0/profile";
    return len;
}

fn generateWildcardLookup(buf: *[64]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "/static/assets/file-{d}.css", .{i}) catch
        return "/static/assets/file-0.css";
    return len;
}

const methods = [_]abi.features.gateway.HttpMethod{
    .GET, .POST, .PUT, .DELETE, .PATCH, .HEAD, .OPTIONS,
};

fn methodForIndex(i: usize) abi.features.gateway.HttpMethod {
    return methods[i % methods.len];
}

// ── Route Registration Benchmark ─────────────────────────────────────

fn benchRouteRegistration(allocator: std.mem.Allocator, count: usize) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = @intCast(@min(count * 2, 100_000)),
        .default_timeout_ms = 30000,
    });
    defer gateway.deinit();

    var path_buf: [64]u8 = undefined;
    for (0..count) |i| {
        const path = generateStaticPath(&path_buf, i);
        try gateway.addRoute(.{
            .path = path,
            .method = methodForIndex(i),
            .upstream = "http://backend:8080",
            .timeout_ms = 5000,
        });
    }
}

// ── Static Route Matching Benchmark ──────────────────────────────────

fn benchMatchStatic(
    allocator: std.mem.Allocator,
    route_count: usize,
    match_count: usize,
) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = @intCast(@min(route_count * 2, 100_000)),
        .default_timeout_ms = 30000,
    });
    defer gateway.deinit();

    // Pre-populate routes
    var path_buf: [64]u8 = undefined;
    for (0..route_count) |i| {
        const path = generateStaticPath(&path_buf, i);
        try gateway.addRoute(.{
            .path = path,
            .method = methodForIndex(i),
            .upstream = "http://backend:8080",
            .timeout_ms = 5000,
        });
    }

    // Match against known routes
    for (0..match_count) |i| {
        const idx = i % route_count;
        const path = generateStaticPath(&path_buf, idx);
        const result = try gateway.matchRoute(path, methodForIndex(idx));
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Parameterized Route Matching Benchmark ───────────────────────────

fn benchMatchParam(allocator: std.mem.Allocator, match_count: usize) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = 256,
        .default_timeout_ms = 30000,
    });
    defer gateway.deinit();

    // Register parameterized routes
    try gateway.addRoute(.{
        .path = "/api/users/{id}/profile",
        .method = .GET,
        .upstream = "http://users-svc:8080",
        .timeout_ms = 5000,
    });
    try gateway.addRoute(.{
        .path = "/api/users/{id}/settings",
        .method = .GET,
        .upstream = "http://users-svc:8080",
        .timeout_ms = 5000,
    });
    try gateway.addRoute(.{
        .path = "/api/orders/{order_id}/items/{item_id}",
        .method = .GET,
        .upstream = "http://orders-svc:8080",
        .timeout_ms = 5000,
    });

    var path_buf: [64]u8 = undefined;
    for (0..match_count) |i| {
        const path = generateParamPath(&path_buf, i);
        const result = try gateway.matchRoute(path, .GET);
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Wildcard Route Matching Benchmark ────────────────────────────────

fn benchMatchWildcard(allocator: std.mem.Allocator, match_count: usize) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = 256,
        .default_timeout_ms = 30000,
    });
    defer gateway.deinit();

    // Register wildcard route
    try gateway.addRoute(.{
        .path = "/static/*",
        .method = .GET,
        .upstream = "http://cdn:8080",
        .timeout_ms = 10000,
    });

    var path_buf: [64]u8 = undefined;
    for (0..match_count) |i| {
        const path = generateWildcardLookup(&path_buf, i);
        const result = try gateway.matchRoute(path, .GET);
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Rate Limiter Throughput Benchmarks ───────────────────────────────

fn benchRateLimitTokenBucket(
    allocator: std.mem.Allocator,
    count: usize,
) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = 256,
        .default_timeout_ms = 30000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .token_bucket,
        },
    });
    defer gateway.deinit();

    try gateway.addRoute(.{
        .path = "/api/limited",
        .method = .GET,
        .upstream = "http://backend:8080",
        .timeout_ms = 5000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .token_bucket,
        },
    });

    for (0..count) |_| {
        const result = gateway.checkRateLimit("/api/limited");
        std.mem.doNotOptimizeAway(&result);
    }
}

fn benchRateLimitSlidingWindow(
    allocator: std.mem.Allocator,
    count: usize,
) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = 256,
        .default_timeout_ms = 30000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .sliding_window,
        },
    });
    defer gateway.deinit();

    try gateway.addRoute(.{
        .path = "/api/limited",
        .method = .GET,
        .upstream = "http://backend:8080",
        .timeout_ms = 5000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .sliding_window,
        },
    });

    for (0..count) |_| {
        const result = gateway.checkRateLimit("/api/limited");
        std.mem.doNotOptimizeAway(&result);
    }
}

fn benchRateLimitFixedWindow(
    allocator: std.mem.Allocator,
    count: usize,
) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = 256,
        .default_timeout_ms = 30000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .fixed_window,
        },
    });
    defer gateway.deinit();

    try gateway.addRoute(.{
        .path = "/api/limited",
        .method = .GET,
        .upstream = "http://backend:8080",
        .timeout_ms = 5000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .fixed_window,
        },
    });

    for (0..count) |_| {
        const result = gateway.checkRateLimit("/api/limited");
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Mixed Workload Benchmark ─────────────────────────────────────────

fn benchMixedWorkload(allocator: std.mem.Allocator, count: usize) !void {
    const gateway = abi.features.gateway;
    try gateway.init(allocator, .{
        .max_routes = @intCast(@min(count * 2, 100_000)),
        .default_timeout_ms = 30000,
        .rate_limit = .{
            .requests_per_second = 1000,
            .burst_size = 2000,
            .algorithm = .token_bucket,
        },
    });
    defer gateway.deinit();

    // Seed initial routes
    const seed_count = @max(count / 10, 10);
    var path_buf: [64]u8 = undefined;
    for (0..seed_count) |i| {
        const path = generateStaticPath(&path_buf, i);
        try gateway.addRoute(.{
            .path = path,
            .method = methodForIndex(i),
            .upstream = "http://backend:8080",
            .timeout_ms = 5000,
            .rate_limit = .{
                .requests_per_second = 1000,
                .burst_size = 2000,
                .algorithm = .token_bucket,
            },
        });
    }

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var route_idx: usize = seed_count;

    for (0..count) |_| {
        const roll = rand.float(f32);
        if (roll < 0.70) {
            // 70% route match
            const idx = rand.intRangeLessThan(usize, 0, seed_count);
            const path = generateStaticPath(&path_buf, idx);
            const result = gateway.matchRoute(
                path,
                methodForIndex(idx),
            ) catch null;
            std.mem.doNotOptimizeAway(&result);
        } else if (roll < 0.90) {
            // 20% rate limit check
            const idx = rand.intRangeLessThan(usize, 0, seed_count);
            const path = generateStaticPath(&path_buf, idx);
            const result = gateway.checkRateLimit(path);
            std.mem.doNotOptimizeAway(&result);
        } else {
            // 10% route add
            const path = generateStaticPath(&path_buf, route_idx);
            gateway.addRoute(.{
                .path = path,
                .method = methodForIndex(route_idx),
                .upstream = "http://backend:8080",
                .timeout_ms = 5000,
            }) catch {};
            route_idx += 1;
        }
    }
}

// ── Runner ───────────────────────────────────────────────────────────

pub fn runGatewayBenchmarks(
    allocator: std.mem.Allocator,
    config: GatewayBenchConfig,
) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                        GATEWAY BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Route registration throughput
    std.debug.print("[Route Registration Throughput]\n", .{});
    for (config.route_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "add_route_{d}",
            .{count},
        ) catch "add_route";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/registration",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchRouteRegistration(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Static route matching
    std.debug.print("\n[Static Route Matching]\n", .{});
    for (config.match_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "match_static_{d}",
            .{iters},
        ) catch "match_static";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/match",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchMatchStatic(a, 200, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Parameterized route matching
    std.debug.print("\n[Parameterized Route Matching]\n", .{});
    for (config.match_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "match_param_{d}",
            .{iters},
        ) catch "match_param";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/match_param",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchMatchParam(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Wildcard route matching
    std.debug.print("\n[Wildcard Route Matching]\n", .{});
    for (config.match_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "match_wildcard_{d}",
            .{iters},
        ) catch "match_wildcard";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/match_wildcard",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchMatchWildcard(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Rate limiter throughput — token bucket
    std.debug.print("\n[Rate Limiter: Token Bucket]\n", .{});
    for (config.rate_limit_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "rate_token_bucket_{d}",
            .{iters},
        ) catch "rate_token_bucket";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/rate_limit",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchRateLimitTokenBucket(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Rate limiter throughput — sliding window
    std.debug.print("\n[Rate Limiter: Sliding Window]\n", .{});
    for (config.rate_limit_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "rate_sliding_window_{d}",
            .{iters},
        ) catch "rate_sliding_window";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/rate_limit",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchRateLimitSlidingWindow(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Rate limiter throughput — fixed window
    std.debug.print("\n[Rate Limiter: Fixed Window]\n", .{});
    for (config.rate_limit_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "rate_fixed_window_{d}",
            .{iters},
        ) catch "rate_fixed_window";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/rate_limit",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchRateLimitFixedWindow(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Mixed workload (70% match, 20% rate check, 10% add)
    std.debug.print("\n[Mixed Workload (70/20/10 match/rate/add)]\n", .{});
    for (config.mixed_iterations) |iters| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "mixed_{d}",
            .{iters},
        ) catch "mixed";

        const result = try runner.run(
            .{
                .name = name,
                .category = "gateway/mixed",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchMixedWorkload(a, c);
                }
            }.bench,
            .{ allocator, iters },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runGatewayBenchmarks(allocator, .{});
}

test "gateway benchmarks compile" {
    const allocator = std.testing.allocator;
    try benchRouteRegistration(allocator, 10);
    try benchMatchStatic(allocator, 10, 20);
    try benchMatchParam(allocator, 10);
    try benchMatchWildcard(allocator, 10);
    try benchRateLimitTokenBucket(allocator, 10);
    try benchRateLimitSlidingWindow(allocator, 10);
    try benchRateLimitFixedWindow(allocator, 10);
    try benchMixedWorkload(allocator, 50);
}
