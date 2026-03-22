//! TCP HTTP Server
//!
//! Minimal HTTP/1.1 server with token-bucket rate limiting, circuit breaker
//! fault tolerance, and CORS support. Routes requests to handlers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const metrics_mod = @import("metrics.zig");
const auth_mod = @import("auth.zig");
const handlers_mod = @import("handlers.zig");
const time_mod = @import("../foundation/mod.zig").time;

const log = std.log.scoped(.api_server);

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

    pub const StartError = error{
        InvalidAddress,
        SocketCreateFailed,
        BindFailed,
        ListenFailed,
    };

    allocator: Allocator,
    config: Config,
    metrics: metrics_mod.Metrics,
    auth: auth_mod.Auth,
    handlers: handlers_mod.Handlers,
    rate_limiter: RateLimiter,
    circuit_breaker: CircuitBreaker,
    running: bool,
    listener: ?std.posix.socket_t,

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
            .listener = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.listener) |sock| {
            std.posix.close(sock);
            self.listener = null;
        }
        self.auth.deinit();
        self.rate_limiter.deinit();
    }

    /// Start the server (blocking). Binds to the configured host:port,
    /// then enters a single-threaded accept loop: accept → read → route → respond → close.
    /// Returns when `stop()` is called or a fatal socket error occurs.
    pub fn start(self: *Self) StartError!void {
        // Fix the metrics pointer after struct move.
        self.handlers.metrics = &self.metrics;

        // --- Create TCP listener socket ---
        const sock = std.posix.socket(
            std.posix.AF.INET,
            std.posix.SOCK.STREAM,
            0,
        ) catch {
            return error.SocketCreateFailed;
        };
        errdefer std.posix.close(sock);

        // Allow rapid restarts without TIME_WAIT blocking.
        const enable: i32 = 1;
        _ = std.posix.setsockopt(
            sock,
            std.posix.SOL.SOCKET,
            std.posix.SO.REUSEADDR,
            std.mem.asBytes(&enable),
        );

        // --- Bind ---
        const ip4 = Io.net.Ip4Address.parse(self.config.host, self.config.port) catch
            return error.InvalidAddress;

        const sin: std.c.sockaddr.in = .{
            .port = std.mem.nativeToBig(u16, ip4.port),
            .addr = @bitCast(ip4.bytes),
        };
        const sock_addr: *const std.posix.sockaddr = @ptrCast(&sin);
        std.posix.bind(sock, sock_addr, @sizeOf(std.c.sockaddr.in)) catch
            return error.BindFailed;

        // --- Listen ---
        std.posix.listen(sock, 128) catch
            return error.ListenFailed;

        self.listener = sock;
        self.running = true;
        log.info("API server listening on {s}:{d}", .{ self.config.host, self.config.port });

        // --- Accept loop (single-threaded) ---
        while (self.running) {
            var client_addr: std.posix.sockaddr = undefined;
            var addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);

            const client_sock = std.posix.accept(sock, &client_addr, &addr_len, 0) catch |err| {
                // If we're shutting down, the listener socket may have been closed.
                if (!self.running) break;
                log.warn("accept failed: {}", .{err});
                continue;
            };
            defer std.posix.close(client_sock);

            self.handleConnection(client_sock);
        }

        log.info("API server stopped", .{});
    }

    /// Process a single HTTP connection: read request, route, write response.
    fn handleConnection(self: *Self, client_sock: std.posix.socket_t) void {
        var buf: [8192]u8 = undefined;
        const n = std.posix.read(client_sock, &buf) catch |err| {
            log.warn("read error: {}", .{err});
            return;
        };
        if (n == 0) return; // Client closed immediately.

        const raw = buf[0..n];

        // Parse the HTTP request line: "METHOD /path HTTP/1.x\r\n..."
        const req_info = parseHttpRequest(raw) orelse {
            writeResponse(client_sock, 400, "application/json", "{\"error\":{\"message\":\"Bad request\",\"type\":\"invalid_request_error\"}}");
            return;
        };

        // Run the request through the full pipeline (rate limiter, auth, router).
        const response = self.processRequest(req_info);

        writeResponse(client_sock, response.status, response.content_type, response.body);
    }

    /// Write an HTTP/1.1 response to the client socket.
    fn writeResponse(sock: std.posix.socket_t, status: u16, content_type: []const u8, body: []const u8) void {
        const reason = httpReasonPhrase(status);
        var hdr_buf: [512]u8 = undefined;
        const header = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{
            status,
            reason,
            content_type,
            body.len,
        }) catch return;

        _ = std.posix.write(sock, header) catch return;
        _ = std.posix.write(sock, body) catch return;
    }

    /// Minimal HTTP request line parser. Extracts method, path, body, and headers.
    fn parseHttpRequest(raw: []const u8) ?handlers_mod.RequestInfo {
        // Find end of request line.
        const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return null;
        const request_line = raw[0..line_end];

        // Split "METHOD /path HTTP/1.x"
        var it = std.mem.splitScalar(u8, request_line, ' ');
        const method = it.next() orelse return null;
        const path = it.next() orelse return null;
        // HTTP version token ignored.

        // Find end of headers.
        const headers_end = std.mem.indexOf(u8, raw, "\r\n\r\n");
        const header_section = if (headers_end) |pos| raw[line_end + 2 .. pos] else raw[line_end + 2 ..];
        const body = if (headers_end) |pos| raw[pos + 4 ..] else "";

        // Extract API key from "Authorization: Bearer <key>" header.
        const api_key = extractHeader(header_section, "Authorization: Bearer ");
        const content_type = extractHeader(header_section, "Content-Type: ");

        return .{
            .method = method,
            .path = path,
            .body = body,
            .api_key = api_key,
            .content_type = content_type,
        };
    }

    /// Search headers for a specific prefix and return the value after it.
    fn extractHeader(headers: []const u8, prefix: []const u8) ?[]const u8 {
        var line_it = std.mem.splitSequence(u8, headers, "\r\n");
        while (line_it.next()) |line| {
            if (std.mem.startsWith(u8, line, prefix)) {
                return line[prefix.len..];
            }
        }
        return null;
    }

    /// Map status codes to reason phrases.
    fn httpReasonPhrase(status: u16) []const u8 {
        return switch (status) {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            405 => "Method Not Allowed",
            429 => "Too Many Requests",
            503 => "Service Unavailable",
            else => "Unknown",
        };
    }

    pub fn stop(self: *Self) void {
        self.running = false;
        // Close the listener socket to unblock accept().
        if (self.listener) |sock| {
            std.posix.close(sock);
            self.listener = null;
        }
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

test "parseHttpRequest parses GET" {
    const raw = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    const req = Server.parseHttpRequest(raw).?;
    try std.testing.expectEqualStrings("GET", req.method);
    try std.testing.expectEqualStrings("/health", req.path);
    try std.testing.expectEqualStrings("", req.body);
    try std.testing.expect(req.api_key == null);
}

test "parseHttpRequest parses POST with body and auth" {
    const raw = "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer sk-test-key\r\n\r\n{\"model\":\"abi-1\"}";
    const req = Server.parseHttpRequest(raw).?;
    try std.testing.expectEqualStrings("POST", req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path);
    try std.testing.expectEqualStrings("{\"model\":\"abi-1\"}", req.body);
    try std.testing.expectEqualStrings("sk-test-key", req.api_key.?);
    try std.testing.expectEqualStrings("application/json", req.content_type.?);
}

test "parseHttpRequest rejects garbage" {
    try std.testing.expect(Server.parseHttpRequest("not http") == null);
    try std.testing.expect(Server.parseHttpRequest("") == null);
}

test "extractHeader finds matching header" {
    const headers = "Host: localhost\r\nContent-Type: text/plain\r\nX-Custom: value";
    const ct = Server.extractHeader(headers, "Content-Type: ");
    try std.testing.expectEqualStrings("text/plain", ct.?);
}

test "extractHeader returns null for missing header" {
    const headers = "Host: localhost\r\n";
    try std.testing.expect(Server.extractHeader(headers, "Authorization: ") == null);
}

test "httpReasonPhrase returns correct phrases" {
    try std.testing.expectEqualStrings("OK", Server.httpReasonPhrase(200));
    try std.testing.expectEqualStrings("Not Found", Server.httpReasonPhrase(404));
    try std.testing.expectEqualStrings("Too Many Requests", Server.httpReasonPhrase(429));
    try std.testing.expectEqualStrings("Unknown", Server.httpReasonPhrase(999));
}
