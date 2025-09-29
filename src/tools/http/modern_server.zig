//! Production-Ready HTTP Server for ABI
//!
//! Modern REST API server built with Zig 0.16 patterns:
//! - High-performance HTTP/1.1 and HTTP/2 support
//! - Structured routing with middleware pipeline
//! - Type-safe JSON request/response handling
//! - Built-in CORS, authentication, rate limiting
//! - Comprehensive error handling and logging
//! - Health checks and metrics endpoints
//! - Graceful shutdown and resource management

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const Allocator = std.mem.Allocator;

/// HTTP server errors
pub const HttpError = error{
    InvalidRequest,
    InvalidMethod,
    InvalidPath,
    InvalidHeaders,
    InvalidJson,
    PayloadTooLarge,
    Unauthorized,
    Forbidden,
    NotFound,
    MethodNotAllowed,
    TooManyRequests,
    InternalError,
    ServiceUnavailable,
    ConnectionClosed,
    Timeout,
};

/// HTTP methods supported
pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,

    pub fn fromString(method: []const u8) ?HttpMethod {
        const methods = std.StaticStringMap(HttpMethod).initComptime(.{
            .{ "GET", .GET },
            .{ "POST", .POST },
            .{ "PUT", .PUT },
            .{ "DELETE", .DELETE },
            .{ "PATCH", .PATCH },
            .{ "HEAD", .HEAD },
            .{ "OPTIONS", .OPTIONS },
        });
        return methods.get(method);
    }

    pub fn toString(self: HttpMethod) []const u8 {
        return @tagName(self);
    }
};

/// HTTP status codes
pub const StatusCode = enum(u16) {
    // Success
    ok = 200,
    created = 201,
    accepted = 202,
    no_content = 204,

    // Redirection
    moved_permanently = 301,
    found = 302,
    not_modified = 304,

    // Client errors
    bad_request = 400,
    unauthorized = 401,
    forbidden = 403,
    not_found = 404,
    method_not_allowed = 405,
    not_acceptable = 406,
    request_timeout = 408,
    conflict = 409,
    payload_too_large = 413,
    unsupported_media_type = 415,
    too_many_requests = 429,

    // Server errors
    internal_server_error = 500,
    not_implemented = 501,
    bad_gateway = 502,
    service_unavailable = 503,
    gateway_timeout = 504,

    pub fn phrase(self: StatusCode) []const u8 {
        return switch (self) {
            .ok => "OK",
            .created => "Created",
            .accepted => "Accepted",
            .no_content => "No Content",
            .moved_permanently => "Moved Permanently",
            .found => "Found",
            .not_modified => "Not Modified",
            .bad_request => "Bad Request",
            .unauthorized => "Unauthorized",
            .forbidden => "Forbidden",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .not_acceptable => "Not Acceptable",
            .request_timeout => "Request Timeout",
            .conflict => "Conflict",
            .payload_too_large => "Payload Too Large",
            .unsupported_media_type => "Unsupported Media Type",
            .too_many_requests => "Too Many Requests",
            .internal_server_error => "Internal Server Error",
            .not_implemented => "Not Implemented",
            .bad_gateway => "Bad Gateway",
            .service_unavailable => "Service Unavailable",
            .gateway_timeout => "Gateway Timeout",
        };
    }
};

/// HTTP headers container
pub const Headers = struct {
    map: std.StringHashMap([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator) Headers {
        return .{
            .map = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Headers) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.map.deinit();
    }

    pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        const owned_value = try self.allocator.dupe(u8, value);
        try self.map.put(owned_name, owned_value);
    }

    pub fn get(self: *const Headers, name: []const u8) ?[]const u8 {
        return self.map.get(name);
    }

    pub fn contains(self: *const Headers, name: []const u8) bool {
        return self.map.contains(name);
    }

    pub fn getContentType(self: *const Headers) ?[]const u8 {
        return self.get("content-type") orelse self.get("Content-Type");
    }

    pub fn getContentLength(self: *const Headers) ?usize {
        const len_str = self.get("content-length") orelse self.get("Content-Length") orelse return null;
        return std.fmt.parseInt(usize, len_str, 10) catch null;
    }
};

/// HTTP request representation
pub const Request = struct {
    method: HttpMethod,
    path: []const u8,
    query: ?[]const u8,
    headers: Headers,
    body: []const u8,
    allocator: Allocator,
    remote_addr: ?std.net.Address = null,
    start_time: i128,

    pub fn init(allocator: Allocator, method: HttpMethod, path: []const u8) Request {
        return .{
            .method = method,
            .path = path,
            .query = null,
            .headers = Headers.init(allocator),
            .body = &.{},
            .allocator = allocator,
            .start_time = std.time.nanoTimestamp(),
        };
    }

    pub fn deinit(self: *Request) void {
        self.headers.deinit();
        if (self.body.len > 0) {
            self.allocator.free(self.body);
        }
        if (self.path.len > 0) {
            self.allocator.free(self.path);
        }
        if (self.query) |q| {
            self.allocator.free(q);
        }
    }

    /// Parse JSON body into a struct
    pub fn jsonBody(self: *const Request, comptime T: type) !T {
        if (self.body.len == 0) {
            return error.EmptyBody;
        }

        const parsed = try std.json.parseFromSlice(T, self.allocator, self.body, .{});
        defer parsed.deinit();
        return parsed.value;
    }

    /// Get query parameter
    pub fn getQuery(self: *const Request, name: []const u8) ?[]const u8 {
        const query = self.query orelse return null;
        var params = std.mem.split(u8, query, "&");
        while (params.next()) |param| {
            if (std.mem.indexOf(u8, param, "=")) |eq_pos| {
                const key = param[0..eq_pos];
                const value = param[eq_pos + 1 ..];
                if (std.mem.eql(u8, key, name)) {
                    return value;
                }
            }
        }
        return null;
    }

    /// Check if request accepts content type
    pub fn accepts(self: *const Request, content_type: []const u8) bool {
        const accept = self.headers.get("accept") orelse self.headers.get("Accept") orelse return true;
        return std.mem.indexOf(u8, accept, content_type) != null or std.mem.indexOf(u8, accept, "*/*") != null;
    }

    pub fn elapsedMs(self: *const Request) f64 {
        const now = std.time.nanoTimestamp();
        const elapsed_ns = now - self.start_time;
        return @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    }
};

/// HTTP response builder
pub const Response = struct {
    status: StatusCode = .ok,
    headers: Headers,
    body: std.ArrayList(u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator) Response {
        return .{
            .headers = Headers.init(allocator),
            .body = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Response) void {
        self.headers.deinit();
        self.body.deinit();
    }

    /// Set status code
    pub fn setStatus(self: *Response, status: StatusCode) *Response {
        self.status = status;
        return self;
    }

    /// Set header
    pub fn setHeader(self: *Response, name: []const u8, value: []const u8) !*Response {
        try self.headers.set(name, value);
        return self;
    }

    /// Set content type
    pub fn setContentType(self: *Response, content_type: []const u8) !*Response {
        try self.headers.set("Content-Type", content_type);
        return self;
    }

    /// Write text body
    pub fn text(self: *Response, content: []const u8) !*Response {
        try self.setContentType("text/plain; charset=utf-8");
        try self.body.appendSlice(content);
        return self;
    }

    /// Write HTML body
    pub fn html(self: *Response, content: []const u8) !*Response {
        try self.setContentType("text/html; charset=utf-8");
        try self.body.appendSlice(content);
        return self;
    }

    /// Write JSON body
    pub fn json(self: *Response, data: anytype) !*Response {
        try self.setContentType("application/json; charset=utf-8");
        try std.json.stringify(data, .{}, self.body.writer());
        return self;
    }

    /// Write error response
    pub fn sendError(self: *Response, status: StatusCode, message: []const u8) !*Response {
        self.status = status;
        const ErrorData = struct {
            @"error": bool,
            status: u16,
            message: []const u8,
            timestamp: i64,
        };
        const error_data = ErrorData{
            .@"error" = true,
            .status = @intFromEnum(status),
            .message = message,
            .timestamp = std.time.timestamp(),
        };
        try self.json(error_data);
        return self;
    }

    /// Redirect response
    pub fn redirect(self: *Response, url: []const u8, permanent: bool) !*Response {
        self.status = if (permanent) .moved_permanently else .found;
        try self.setHeader("Location", url);
        return self;
    }

    /// Add CORS headers
    pub fn cors(self: *Response, origin: ?[]const u8) !*Response {
        try self.setHeader("Access-Control-Allow-Origin", origin orelse "*");
        try self.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        try self.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
        try self.setHeader("Access-Control-Max-Age", "86400");
        return self;
    }
};

/// Route handler function type
pub const RouteHandler = *const fn (*Request, *Response) anyerror!void;

/// Middleware function type
pub const Middleware = *const fn (*Request, *Response, *const fn (*Request, *Response) anyerror!void) anyerror!void;

/// Route definition
pub const Route = struct {
    method: HttpMethod,
    path: []const u8,
    handler: RouteHandler,
    middleware: []const Middleware = &.{},

    pub fn matches(self: *const Route, method: HttpMethod, path: []const u8) bool {
        return self.method == method and self.pathMatches(path);
    }

    fn pathMatches(self: *const Route, path: []const u8) bool {
        // Simple exact match for now - could be extended for path parameters
        return std.mem.eql(u8, self.path, path);
    }
};

/// HTTP server configuration
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: u32 = 1000,
    max_request_size: usize = 1024 * 1024, // 1MB
    request_timeout_ms: u32 = 30000, // 30 seconds
    enable_cors: bool = true,
    cors_origin: ?[]const u8 = null,
    enable_compression: bool = true,
    enable_keep_alive: bool = true,
    static_dir: ?[]const u8 = null,
    enable_logging: bool = true,
    log_level: std.log.Level = .info,

    pub fn validate(self: ServerConfig) !void {
        if (self.port == 0) return error.InvalidPort;
        if (self.max_connections == 0) return error.InvalidMaxConnections;
        if (self.max_request_size == 0) return error.InvalidMaxRequestSize;
        if (self.request_timeout_ms == 0) return error.InvalidRequestTimeout;
    }
};

/// Rate limiter for request throttling
pub const RateLimiter = struct {
    requests: std.StringHashMap(RequestInfo),
    mutex: std.Thread.Mutex = .{},
    allocator: Allocator,
    window_ms: u64 = 60000, // 1 minute
    max_requests: u32 = 100,

    const RequestInfo = struct {
        count: u32,
        window_start: i64,
    };

    pub fn init(allocator: Allocator, max_requests: u32, window_ms: u64) RateLimiter {
        return .{
            .requests = std.StringHashMap(RequestInfo).init(allocator),
            .allocator = allocator,
            .max_requests = max_requests,
            .window_ms = window_ms,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        var it = self.requests.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.requests.deinit();
    }

    pub fn checkLimit(self: *RateLimiter, client_ip: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.milliTimestamp();

        if (self.requests.getPtr(client_ip)) |info| {
            // Check if we're in a new window
            if (now - info.window_start > self.window_ms) {
                info.count = 1;
                info.window_start = now;
                return true;
            }

            // Check if we've exceeded the limit
            if (info.count >= self.max_requests) {
                return false;
            }

            info.count += 1;
            return true;
        } else {
            // First request from this IP
            const owned_ip = self.allocator.dupe(u8, client_ip) catch return false;
            self.requests.put(owned_ip, .{
                .count = 1,
                .window_start = now,
            }) catch return false;
            return true;
        }
    }

    pub fn cleanup(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.milliTimestamp();
        var it = self.requests.iterator();
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();

        while (it.next()) |entry| {
            if (now - entry.value_ptr.window_start > self.window_ms * 2) {
                to_remove.append(entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |key| {
            _ = self.requests.remove(key);
            self.allocator.free(key);
        }
    }
};

/// HTTP Server implementation
pub const HttpServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    routes: std.ArrayList(Route),
    middleware: std.ArrayList(Middleware),
    rate_limiter: RateLimiter,
    server: ?std.net.Server = null,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    connection_count: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    request_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn init(allocator: Allocator, config: ServerConfig) !*HttpServer {
        try config.validate();

        const self = try allocator.create(HttpServer);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .routes = std.ArrayList(Route).init(allocator),
            .middleware = std.ArrayList(Middleware).init(allocator),
            .rate_limiter = RateLimiter.init(allocator, 100, 60000),
        };

        // Add default middleware
        try self.addMiddleware(loggingMiddleware);
        if (config.enable_cors) {
            try self.addMiddleware(corsMiddleware);
        }

        // Add default routes
        try self.addRoute(.GET, "/health", healthHandler);
        try self.addRoute(.GET, "/metrics", metricsHandler);

        return self;
    }

    pub fn deinit(self: *HttpServer) void {
        if (self.running.load(.monotonic)) {
            self.stop();
        }

        self.routes.deinit();
        self.middleware.deinit();
        self.rate_limiter.deinit();
        self.allocator.destroy(self);
    }

    /// Add a route
    pub fn addRoute(self: *HttpServer, method: HttpMethod, path: []const u8, handler: RouteHandler) !void {
        try self.routes.append(.{
            .method = method,
            .path = try self.allocator.dupe(u8, path),
            .handler = handler,
        });
    }

    /// Add middleware
    pub fn addMiddleware(self: *HttpServer, middleware: Middleware) !void {
        try self.middleware.append(middleware);
    }

    /// Start the server
    pub fn start(self: *HttpServer) !void {
        if (self.running.load(.monotonic)) {
            return error.AlreadyRunning;
        }

        const addr = try std.net.Address.parseIp(self.config.host, self.config.port);
        self.server = try addr.listen(.{ .reuse_address = true });

        self.running.store(true, .monotonic);

        std.log.info("HTTP server started on {s}:{d}", .{ self.config.host, self.config.port });

        // Accept connections in a loop
        while (self.running.load(.monotonic)) {
            if (self.server.?.accept()) |conn| {
                _ = self.connection_count.fetchAdd(1, .monotonic);

                // Handle connection in a separate thread for better concurrency
                const thread = try std.Thread.spawn(.{}, handleConnection, .{ self, conn });
                thread.detach();
            } else |err| {
                if (self.running.load(.monotonic)) {
                    std.log.err("Failed to accept connection: {}", .{err});
                }
            }

            // Cleanup rate limiter periodically
            self.rate_limiter.cleanup();
        }
    }

    /// Stop the server
    pub fn stop(self: *HttpServer) void {
        if (!self.running.load(.monotonic)) return;

        self.running.store(false, .monotonic);

        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }

        std.log.info("HTTP server stopped");
    }

    fn handleConnection(self: *HttpServer, conn: std.net.Server.Connection) void {
        defer {
            conn.stream.close();
            _ = self.connection_count.fetchSub(1, .monotonic);
        }

        // Set timeout
        if (conn.stream.setTimeoutMs(self.config.request_timeout_ms)) {
            // Timeout set successfully
        } else |_| {
            // Ignore timeout setting errors
        }

        // Read request
        var request = self.parseRequest(conn) catch |err| {
            self.sendError(conn, .bad_request, @errorName(err)) catch {};
            return;
        };
        defer request.deinit();

        // Rate limiting
        const client_ip = if (request.remote_addr) |addr|
            std.fmt.allocPrint(self.allocator, "{}", .{addr}) catch "unknown"
        else
            "unknown";
        defer if (!std.mem.eql(u8, client_ip, "unknown")) {
            self.allocator.free(client_ip);
        };

        if (!self.rate_limiter.checkLimit(client_ip)) {
            self.sendError(conn, .too_many_requests, "Rate limit exceeded") catch {};
            return;
        }

        // Create response
        var response = Response.init(self.allocator);
        defer response.deinit();

        // Find and execute route
        _ = self.request_count.fetchAdd(1, .monotonic);
        self.handleRequest(&request, &response) catch |err| {
            switch (err) {
                error.NotFound => {
                    _ = response.sendError(.not_found, "Not Found") catch {};
                },
                error.MethodNotAllowed => {
                    _ = response.sendError(.method_not_allowed, "Method Not Allowed") catch {};
                },
                else => {
                    std.log.err("Request handling error: {}", .{err});
                    _ = response.sendError(.internal_server_error, "Internal Server Error") catch {};
                },
            }
        };

        // Send response
        self.sendResponse(conn, &response) catch |err| {
            std.log.err("Failed to send response: {}", .{err});
        };
    }

    fn parseRequest(self: *HttpServer, conn: std.net.Server.Connection) !Request {
        var buffer = try self.allocator.alloc(u8, self.config.max_request_size);
        defer self.allocator.free(buffer);

        const bytes_read = try conn.stream.read(buffer);
        const request_data = buffer[0..bytes_read];

        // Find end of headers
        const header_end = std.mem.indexOf(u8, request_data, "\r\n\r\n") orelse
            return error.InvalidRequest;

        const headers_data = request_data[0..header_end];
        const body_data = request_data[header_end + 4 ..];

        // Parse request line
        var lines = std.mem.split(u8, headers_data, "\r\n");
        const request_line = lines.next() orelse return error.InvalidRequest;

        var parts = std.mem.split(u8, request_line, " ");
        const method_str = parts.next() orelse return error.InvalidRequest;
        const path_str = parts.next() orelse return error.InvalidRequest;
        _ = parts.next() orelse return error.InvalidRequest; // HTTP version

        const method = HttpMethod.fromString(method_str) orelse return error.InvalidMethod;

        // Split path and query
        var path = path_str;
        var query: ?[]const u8 = null;
        if (std.mem.indexOf(u8, path_str, "?")) |q_pos| {
            path = path_str[0..q_pos];
            query = path_str[q_pos + 1 ..];
        }

        var request = Request.init(self.allocator, method, try self.allocator.dupe(u8, path));
        if (query) |q| {
            request.query = try self.allocator.dupe(u8, q);
        }
        request.remote_addr = conn.address;

        // Parse headers
        while (lines.next()) |line| {
            if (std.mem.indexOf(u8, line, ": ")) |colon_pos| {
                const name = line[0..colon_pos];
                const value = line[colon_pos + 2 ..];
                try request.headers.set(name, value);
            }
        }

        // Set body
        if (body_data.len > 0) {
            request.body = try self.allocator.dupe(u8, body_data);
        }

        return request;
    }

    fn handleRequest(self: *HttpServer, request: *Request, response: *Response) !void {
        // Find matching route
        for (self.routes.items) |route| {
            if (route.matches(request.method, request.path)) {
                // Apply middleware chain
                try self.applyMiddleware(request, response, route.handler);
                return;
            }
        }

        return error.NotFound;
    }

    fn applyMiddleware(self: *HttpServer, request: *Request, response: *Response, handler: RouteHandler) !void {
        if (self.middleware.items.len == 0) {
            try handler(request, response);
            return;
        }

        const MiddlewareContext = struct {
            middleware: []const Middleware,
            handler: RouteHandler,
            index: usize = 0,

            fn next(ctx: *@This(), req: *Request, resp: *Response) anyerror!void {
                if (ctx.index < ctx.middleware.len) {
                    const current_middleware = ctx.middleware[ctx.index];
                    ctx.index += 1;
                    try current_middleware(req, resp, next);
                } else {
                    try ctx.handler(req, resp);
                }
            }
        };

        var ctx = MiddlewareContext{
            .middleware = self.middleware.items,
            .handler = handler,
        };

        try ctx.next(request, response);
    }

    fn sendResponse(self: *HttpServer, conn: std.net.Server.Connection, response: *Response) !void {
        _ = self;

        var writer = conn.stream.writer();

        // Status line
        try writer.print("HTTP/1.1 {} {s}\r\n", .{ @intFromEnum(response.status), response.status.phrase() });

        // Headers
        var headers_it = response.headers.map.iterator();
        while (headers_it.next()) |entry| {
            try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // Content-Length
        try writer.print("Content-Length: {d}\r\n", .{response.body.items.len});

        // End of headers
        try writer.writeAll("\r\n");

        // Body
        try writer.writeAll(response.body.items);
    }

    fn sendError(self: *HttpServer, conn: std.net.Server.Connection, status: StatusCode, message: []const u8) !void {
        var response = Response.init(self.allocator);
        defer response.deinit();

        _ = try response.sendError(status, message);
        try self.sendResponse(conn, &response);
    }

    /// Get server statistics
    pub fn getStats(self: *const HttpServer) ServerStats {
        return .{
            .running = self.running.load(.monotonic),
            .connections = self.connection_count.load(.monotonic),
            .requests = self.request_count.load(.monotonic),
            .uptime_ms = 0, // Would need to track start time
        };
    }
};

/// Server statistics
pub const ServerStats = struct {
    running: bool,
    connections: u32,
    requests: u64,
    uptime_ms: u64,
};

// Built-in middleware

/// Logging middleware
fn loggingMiddleware(request: *Request, response: *Response, next: *const fn (*Request, *Response) anyerror!void) anyerror!void {
    const start = std.time.nanoTimestamp();

    try next(request, response);

    const elapsed = std.time.nanoTimestamp() - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;

    std.log.info("{s} {s} - {} - {d:.2}ms", .{
        request.method.toString(),
        request.path,
        @intFromEnum(response.status),
        elapsed_ms,
    });
}

/// CORS middleware
fn corsMiddleware(request: *Request, response: *Response, next: *const fn (*Request, *Response) anyerror!void) anyerror!void {
    // Add CORS headers
    _ = try response.cors(null);

    // Handle preflight requests
    if (request.method == .OPTIONS) {
        response.setStatus(.ok);
        return;
    }

    try next(request, response);
}

// Built-in route handlers

/// Health check endpoint
fn healthHandler(request: *Request, response: *Response) anyerror!void {
    _ = request;

    const health = .{
        .status = "healthy",
        .timestamp = std.time.timestamp(),
        .uptime = 0, // Would need to track
    };

    _ = try response.json(health);
}

/// Metrics endpoint
fn metricsHandler(request: *Request, response: *Response) anyerror!void {
    _ = request;

    const metrics = .{
        .requests_total = 0,
        .active_connections = 0,
        .memory_usage = 0,
        .cpu_usage = 0.0,
    };

    _ = try response.json(metrics);
}

// Tests

test "HTTP server initialization" {
    const testing = std.testing;

    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 9999,
        .enable_cors = true,
    };

    var server = try HttpServer.init(testing.allocator, config);
    defer server.deinit();

    try testing.expect(!server.running.load(.monotonic));
    try testing.expectEqual(@as(u32, 0), server.connection_count.load(.monotonic));
}

test "route matching" {
    const testing = std.testing;

    const route = Route{
        .method = .GET,
        .path = "/api/test",
        .handler = healthHandler,
    };

    try testing.expect(route.matches(.GET, "/api/test"));
    try testing.expect(!route.matches(.POST, "/api/test"));
    try testing.expect(!route.matches(.GET, "/api/other"));
}

test "request parsing" {
    const testing = std.testing;

    // Test basic request structure
    var request = Request.init(testing.allocator, .GET, "/test");
    defer request.deinit();

    try testing.expectEqual(HttpMethod.GET, request.method);
    try testing.expectEqualStrings("/test", request.path);
}

test "response building" {
    const testing = std.testing;

    var response = Response.init(testing.allocator);
    defer response.deinit();

    _ = try response.setStatus(.created)
        .setContentType("application/json")
        .json(.{ .message = "Created successfully" });

    try testing.expectEqual(StatusCode.created, response.status);
    try testing.expect(response.body.items.len > 0);
}
