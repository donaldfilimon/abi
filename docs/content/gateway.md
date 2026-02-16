---
title: "Gateway"
description: "API gateway with routing, rate limiting, and circuit breaker"
section: "Infrastructure"
order: 2
---

# Gateway

The Gateway module provides an API gateway with radix-tree route matching,
three rate limiting algorithms, a circuit breaker state machine, middleware
chains, and latency tracking -- all in a single-process, zero-allocation hot
path.

- **Build flag:** `-Denable-gateway=true` (default: enabled)
- **Namespace:** `abi.gateway`
- **Source:** `src/features/gateway/`

## Overview

The gateway acts as the front door for incoming HTTP requests. It matches
request paths against registered routes using a shared radix tree (the same
implementation used by the [Pages](pages.html) module via
`services/shared/utils/radix_tree.zig`), enforces per-route rate limits, and
tracks upstream health through circuit breakers.

Key capabilities:

- **Radix-tree routing** -- O(path-segments) route matching with path parameters (`{id}`) and wildcards (`*`)
- **Three rate limiting algorithms** -- Token bucket, sliding window, and fixed window, configurable per route
- **Circuit breaker state machine** -- Automatic open/closed/half-open transitions based on upstream failure rates
- **Middleware chain** -- Auth, CORS, access logging, request ID, response transform
- **Latency histogram** -- 7-bucket histogram for p50/p99 estimation
- **Concurrent access** -- RwLock protects route lookups; atomic counters for stats

### Shared Infrastructure

The gateway uses two shared infrastructure modules:

- `services/shared/resilience/circuit_breaker.zig` -- Parameterized circuit breaker with `.atomic`, `.mutex`, or `.none` sync strategies
- `services/shared/security/rate_limit.zig` -- HTTP/API-level rate limiting with per-key tracking, bans, and whitelist

## Quick Start

```zig
const abi = @import("abi");

// Initialize via Framework
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withGatewayDefaults()
    .build();
defer framework.deinit();

// Register routes
try abi.gateway.addRoute(.{
    .path = "/api/users",
    .method = .GET,
    .upstream = "user-service",
});

try abi.gateway.addRoute(.{
    .path = "/api/users/{id}",
    .method = .GET,
    .upstream = "user-service",
});

try abi.gateway.addRoute(.{
    .path = "/api/search",
    .method = .POST,
    .upstream = "search-service",
    .rate_limit = .{
        .requests_per_second = 50,
        .burst_size = 100,
        .algorithm = .sliding_window,
    },
});

// Match an incoming request
if (try abi.gateway.matchRoute("/api/users/42", .GET)) |match| {
    std.debug.print("Upstream: {s}\n", .{match.route.upstream});
    // Extract path parameters
    for (match.params[0..match.param_count]) |p| {
        std.debug.print("  {s} = {s}\n", .{ p.name, p.value });
    }
}

// Check rate limit before forwarding
const rl = abi.gateway.checkRateLimit("/api/search");
if (!rl.allowed) {
    // Return 429 Too Many Requests
}

// Record upstream result for circuit breaker
abi.gateway.recordUpstreamResult("search-service", true);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context |
| `GatewayConfig` | Max routes, default timeout, rate limit and circuit breaker defaults |
| `Route` | A registered route: path, method, upstream, timeout, rate limit, middlewares |
| `HttpMethod` | GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS |
| `MatchResult` | Matched route with up to 8 extracted path parameters |
| `GatewayStats` | Snapshot of request counts, rate limit hits, CB trips, latency |
| `RateLimitResult` | Whether a request is allowed, remaining tokens, reset time |
| `GatewayError` | Error set: FeatureDisabled, RouteNotFound, RateLimitExceeded, CircuitOpen, etc. |

### Rate Limiting

| Type | Description |
|------|-------------|
| `RateLimitConfig` | Requests per second, burst size, algorithm selection |
| `RateLimitAlgorithm` | `token_bucket`, `sliding_window`, or `fixed_window` |

### Circuit Breaker

| Type | Description |
|------|-------------|
| `CircuitBreakerConfig` | Failure threshold, reset timeout, half-open max requests |
| `CircuitBreakerState` | `closed` (healthy), `open` (tripped), `half_open` (probing) |

### Middleware

| Type | Description |
|------|-------------|
| `MiddlewareType` | `auth`, `rate_limit`, `circuit_breaker`, `access_log`, `cors`, `response_transform`, `request_id` |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator, config) !void` | Initialize the gateway singleton |
| `deinit() void` | Tear down the gateway and free all state |
| `isEnabled() bool` | Returns `true` if gateway is compiled in |
| `isInitialized() bool` | Returns `true` if the singleton is active |
| `addRoute(route) !void` | Register a route with optional per-route rate limit |
| `removeRoute(path) !bool` | Remove all routes under a path; rebuilds radix tree |
| `getRoutes() []const Route` | Snapshot of all registered routes (max 256) |
| `getRouteCount() usize` | Number of active routes |
| `matchRoute(path, method) !?MatchResult` | Match a request against the radix tree |
| `checkRateLimit(path) RateLimitResult` | Check whether a request is allowed |
| `recordUpstreamResult(upstream, success) void` | Feed success/failure into the circuit breaker |
| `stats() GatewayStats` | Snapshot of counters and latency |
| `getCircuitState(upstream) CircuitBreakerState` | Query circuit breaker state for an upstream |
| `resetCircuit(upstream) void` | Force-close a circuit breaker, clearing failure counters |

## Configuration

Gateway is configured through the `GatewayConfig` struct:

```zig
const config = abi.gateway.GatewayConfig{
    .max_routes = 256,
    .default_timeout_ms = 30_000,
    .rate_limit = .{
        .requests_per_second = 100,
        .burst_size = 200,
        .algorithm = .token_bucket,
    },
    .circuit_breaker = .{
        .failure_threshold = 5,
        .reset_timeout_ms = 30_000,
        .half_open_max_requests = 3,
    },
    .enable_access_log = true,
    .enable_response_transform = false,
};
```

### Circuit Breaker State Machine

```
         success
  closed ---------> closed
    |
    | failure_threshold reached
    v
   open ------------> half_open (after reset_timeout_ms)
                          |
                 success  |  failure
                 closed <-+-> open
```

- **Closed** -- Normal operation; failures are counted
- **Open** -- All requests short-circuit with `error.CircuitOpen`
- **Half-open** -- A limited number of probe requests are allowed through

## CLI Commands

The gateway module does not have a dedicated CLI command. Use the gateway API
programmatically or through the Framework builder.

## Examples

See `examples/gateway.zig` for a complete working example that registers
routes, matches incoming paths, extracts parameters, and queries circuit
breaker state:

```bash
zig build run-gateway
```

## Disabling at Build Time

```bash
# Compile without gateway support
zig build -Denable-gateway=false
```

When disabled, all public functions return `error.FeatureDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures so downstream code compiles without conditional guards.

## Related

- [Pages](pages.html) -- Dashboard UI pages sharing the same radix tree router
- [Network](network.html) -- Distributed compute and node management
- [Web](web.html) -- HTTP client utilities and middleware
