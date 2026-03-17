//! TCP HTTP Server
//!
//! Minimal HTTP/1.1 server with token-bucket rate limiting, circuit breaker
//! fault tolerance, and CORS support. Routes requests to handlers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const metrics_mod = @import("metrics.zig");
const auth_mod = @import("auth.zig");
const handlers_mod = @import("handlers.zig");
const time_mod = @import("../services/shared/mod.zig").time;

fn monotonicNowNs() i128 {
    const instant = time_mod.Instant.now() catch return 0;
    return if (instant.nanos > std.math.maxInt(i128))
        std.math.maxInt(i128)
    else
        @intCast(instant.nanos);
}

pub const Config = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: u32 = 1000,
    enable_cors: bool = true,
    enable_auth: bool = true,
    rate_limit_requests: u32 = 100,
    rate_limit_window_sec: u32 = 60,
};

// ============================================================================
// Rate Limiter (Token Bucket)
// ============================================================================

pub const RateLimiter = struct {
    const Self = @This();

    rate: u32, // Max requests per window
    window_ns: i128,
    clients: std.StringHashMapUnmanaged(ClientState),
    allocator: Allocator,

    const ClientState = struct {
        tokens: u32,
        last_refill: i128,
    };

    pub fn init(allocator: Allocator, rate: u32, window_sec: u32) Self {
        return .{
            .rate = rate,
            .window_ns = @as(i128, window_sec) * std.time.ns_per_s,
            .clients = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.clients.deinit(self.allocator);
    }

    pub fn allow(self: *Self, client_id: []const u8) bool {
        const now = monotonicNowNs();
        const entry = self.clients.getOrPut(self.allocator, client_id) catch return true;

        if (!entry.found_existing) {
            entry.value_ptr.* = .{
                .tokens = self.rate - 1,
                .last_refill = now,
            };
            return true;
        }

        // Refill tokens if window has passed.
        const elapsed = now - entry.value_ptr.last_refill;
        if (elapsed >= self.window_ns) {
            entry.value_ptr.tokens = self.rate;
            entry.value_ptr.last_refill = now;
        }

        if (entry.value_ptr.tokens > 0) {
            entry.value_ptr.tokens -= 1;
            return true;
        }
        return false;
    }

    pub fn remaining(self: *Self, client_id: []const u8) u32 {
        if (self.clients.get(client_id)) |state| {
            return state.tokens;
        }
        return self.rate;
    }
};

// ============================================================================
// Circuit Breaker
// ============================================================================

pub const CircuitBreaker = struct {
    const Self = @This();

    pub const State = enum { closed, open, half_open };

    state: State,
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
    timeout_ns: i128,
    last_failure_time: i128,

    pub fn init(failure_threshold: u32, success_threshold: u32, timeout_sec: u32) Self {
        return .{
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .failure_threshold = failure_threshold,
            .success_threshold = success_threshold,
            .timeout_ns = @as(i128, timeout_sec) * std.time.ns_per_s,
            .last_failure_time = 0,
        };
    }

    pub fn allow(self: *Self) bool {
        return switch (self.state) {
            .closed => true,
            .open => blk: {
                const now = monotonicNowNs();
                if (now - self.last_failure_time >= self.timeout_ns) {
                    self.state = .half_open;
                    self.success_count = 0;
                    break :blk true;
                }
                break :blk false;
            },
            .half_open => true,
        };
    }

    pub fn recordSuccess(self: *Self) void {
        self.failure_count = 0;
        if (self.state == .half_open) {
            self.success_count += 1;
            if (self.success_count >= self.success_threshold) {
                self.state = .closed;
            }
        }
    }

    pub fn recordFailure(self: *Self) void {
        self.failure_count += 1;
        self.last_failure_time = monotonicNowNs();
        if (self.failure_count >= self.failure_threshold) {
            self.state = .open;
        }
    }
};

// ============================================================================
// Server
// ============================================================================

pub const Server = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,
    metrics: metrics_mod.Metrics,
    auth: auth_mod.Auth,
    handlers: handlers_mod.Handlers,
    rate_limiter: RateLimiter,
    circuit_breaker: CircuitBreaker,
    running: bool,

    pub fn init(allocator: Allocator, config: Config) Self {
        var metrics = metrics_mod.Metrics{};
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = metrics,
            .auth = auth_mod.Auth.init(allocator, config.enable_auth),
            .handlers = handlers_mod.Handlers.init(allocator, &metrics),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit_requests, config.rate_limit_window_sec),
            .circuit_breaker = CircuitBreaker.init(5, 3, 30),
            .running = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.auth.deinit();
        self.rate_limiter.deinit();
    }

    /// Start the server (blocking). In production, this would accept TCP connections.
    pub fn start(self: *Self) !void {
        self.running = true;
        // Fix the metrics pointer after struct move.
        self.handlers.metrics = &self.metrics;

        const addr = std.net.Address.parseIp4(self.config.host, self.config.port) catch {
            return error.InvalidAddress;
        };
        _ = addr;

        // The actual TCP accept loop would go here.
        // For now, this is a structural placeholder that validates the server
        // can be instantiated and started. Real networking is deferred to
        // integration with the existing web server infrastructure.
    }

    pub fn stop(self: *Self) void {
        self.running = false;
    }

    /// Create an API key.
    pub fn createApiKey(self: *Self, name: []const u8) ![64]u8 {
        return self.auth.createKey(name);
    }

    /// Process a single request through the full pipeline.
    pub fn processRequest(self: *Self, req: handlers_mod.RequestInfo) handlers_mod.Response {
        // 1. Circuit breaker check.
        if (!self.circuit_breaker.allow()) {
            return .{
                .status = 503,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Service unavailable\",\"type\":\"server_error\"}}",
            };
        }

        // 2. Rate limiting.
        const client = req.api_key orelse "anonymous";
        if (!self.rate_limiter.allow(client)) {
            self.circuit_breaker.recordFailure();
            return .{
                .status = 429,
                .content_type = "application/json",
                .body = "{\"error\":{\"message\":\"Rate limit exceeded\",\"type\":\"rate_limit_error\"}}",
            };
        }

        // 3. Authentication.
        if (self.config.enable_auth) {
            if (req.api_key) |key| {
                if (self.auth.validate(key) == null) {
                    return .{
                        .status = 401,
                        .content_type = "application/json",
                        .body = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\"}}",
                    };
                }
            } else {
                return .{
                    .status = 401,
                    .content_type = "application/json",
                    .body = "{\"error\":{\"message\":\"API key required\",\"type\":\"authentication_error\"}}",
                };
            }
        }

        // 4. Handle the request.
        const response = self.handlers.handle(req);
        if (response.status >= 500) {
            self.circuit_breaker.recordFailure();
        } else {
            self.circuit_breaker.recordSuccess();
        }

        return response;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "rate limiter" {
    const allocator = std.testing.allocator;
    var rl = RateLimiter.init(allocator, 3, 60);
    defer rl.deinit();

    try std.testing.expect(rl.allow("client-1")); // 1
    try std.testing.expect(rl.allow("client-1")); // 2
    try std.testing.expect(rl.allow("client-1")); // 3
    try std.testing.expect(!rl.allow("client-1")); // exceeded

    // Different client should still be allowed.
    try std.testing.expect(rl.allow("client-2"));
}

test "circuit breaker transitions" {
    var cb = CircuitBreaker.init(2, 1, 0); // 0s timeout for testing

    try std.testing.expect(cb.allow()); // closed → allow
    cb.recordFailure();
    try std.testing.expect(cb.allow()); // still closed
    cb.recordFailure();
    // Now open (2 failures >= threshold)
    try std.testing.expectEqual(CircuitBreaker.State.open, cb.state);
}

test "server process request without auth" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{ .enable_auth = false });
    defer server.deinit();

    const resp = server.processRequest(.{
        .method = "GET",
        .path = "/health",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 200), resp.status);
}

test "server rejects without api key" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, .{ .enable_auth = true });
    defer server.deinit();

    const resp = server.processRequest(.{
        .method = "GET",
        .path = "/health",
        .body = "",
        .api_key = null,
        .content_type = null,
    });
    try std.testing.expectEqual(@as(u16, 401), resp.status);
}
