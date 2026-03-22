---
title: gateway API
purpose: Generated API reference for gateway
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2962+08416b44f
---

# gateway

> Gateway Module

API gateway with radix-tree route matching, 3 rate limiting algorithms
(token bucket, sliding window, fixed window), circuit breaker state
machine, middleware chain, and latency tracking.

Architecture:
- Radix tree for O(path_segments) route matching with params and wildcards
- Per-route rate limiters (token bucket, sliding/fixed window)
- Per-upstream circuit breakers (closed → open → half_open → closed)
- Latency histogram with 7 buckets for p50/p99 estimation
- RwLock for concurrent route lookups

**Source:** [`src/features/gateway/mod.zig`](../../src/features/gateway/mod.zig)

**Build flag:** `-Dfeat_gateway=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-gatewayconfig-gatewayerror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: GatewayConfig) GatewayError!void`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L161)

Initialize the API gateway singleton with routing, rate limiting,
and circuit breaker configuration.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L167)

Tear down the gateway, freeing all routes and internal state.

### <a id="pub-fn-addroute-route-route-gatewayerror-void"></a>`pub fn addRoute(route: Route) GatewayError!void`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L184)

Register an API route (method + path pattern). Supports path parameters
(`{id}`) and wildcards (`*`).

### <a id="pub-fn-removeroute-path-const-u8-gatewayerror-bool"></a>`pub fn removeRoute(path: []const u8) GatewayError!bool`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L245)

Remove all routes registered under a given path. Returns `true` if any were removed.

### <a id="pub-fn-matchroute-path-const-u8-method-httpmethod-gatewayerror-matchresult"></a>`pub fn matchRoute(path: []const u8, method: HttpMethod) GatewayError!?MatchResult`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L299)

Match an incoming request path and method against the radix tree.
Returns the matching route and extracted path parameters, or `null`.

### <a id="pub-fn-checkratelimit-path-const-u8-ratelimitresult"></a>`pub fn checkRateLimit(path: []const u8) RateLimitResult`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L328)

Check whether a request to `path` is allowed under the configured rate limiter.

### <a id="pub-fn-recordupstreamresult-upstream-const-u8-success-bool-void"></a>`pub fn recordUpstreamResult(upstream: []const u8, success: bool) void`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L345)

Record a success/failure for circuit breaker state tracking on `upstream`.

### <a id="pub-fn-stats-gatewaystats"></a>`pub fn stats() GatewayStats`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L373)

Snapshot route count, request/error counters, and latency histogram.

### <a id="pub-fn-getcircuitstate-upstream-const-u8-circuitbreakerstate"></a>`pub fn getCircuitState(upstream: []const u8) CircuitBreakerState`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L386)

Query the circuit breaker state for an upstream service.

### <a id="pub-fn-resetcircuit-upstream-const-u8-void"></a>`pub fn resetCircuit(upstream: []const u8) void`

<sup>**fn**</sup> | [source](../../src/features/gateway/mod.zig#L399)

Force-close the circuit breaker for an upstream, clearing failure counters.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
