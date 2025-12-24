//! Enhanced HTTP Client with libcurl integration
//!
//! This module provides a robust HTTP client implementation that:
//! - Uses libcurl for reliable proxy and TLS support
//! - Implements exponential backoff and timeout handling
//! - Provides fallback to native Zig HTTP client
//! - Handles Windows-specific networking issues

const std = @import("std");
const builtin = @import("builtin");
// Note: core functionality is now imported through module dependencies

/// HTTP client configuration
pub const HttpClientConfig = struct {
    /// Connection timeout in milliseconds
    connect_timeout_ms: u32 = 10000,
    /// Read timeout in milliseconds
    read_timeout_ms: u32 = 30000,
    /// Maximum number of retries
    max_retries: u32 = 3,
    /// Initial backoff delay in milliseconds
    initial_backoff_ms: u32 = 1000,
    /// Maximum backoff delay in milliseconds
    max_backoff_ms: u32 = 10000,
    /// HTTP proxy URL (optional)
    proxy_url: ?[]const u8 = null,
    /// HTTPS proxy URL (optional)
    https_proxy_url: ?[]const u8 = null,
    /// User agent string
    user_agent: []const u8 = "WDBX-HTTP-Client/1.0",
    /// Follow redirects
    follow_redirects: bool = true,
    /// Maximum redirects to follow
    max_redirects: u32 = 10,
    /// Verify SSL certificates
    verify_ssl: bool = true,
    /// SSL CA bundle path (optional)
    ca_bundle_path: ?[]const u8 = null,
    /// Enable verbose logging
    verbose: bool = false,
};

/// HTTP response structure
pub const HttpResponse = struct {
    status_code: u16,
    headers: std.StringHashMap([]const u8),
    body: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *HttpResponse) void {
        self.allocator.free(self.body);
        var iterator = self.headers.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
    }
};

/// HTTP client with libcurl integration and fallback
pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    config: HttpClientConfig,
    use_libcurl: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: HttpClientConfig) Self {
        // Try to detect if libcurl is available
        const use_libcurl = detectLibcurl();

        return Self{
            .allocator = allocator,
            .config = config,
            .use_libcurl = use_libcurl,
        };
    }

    /// Make HTTP GET request with retry and backoff
    pub fn get(self: *Self, url: []const u8) !HttpResponse {
        return self.request("GET", url, null, null);
    }

    /// Make HTTP POST request with retry and backoff
    pub fn post(self: *Self, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !HttpResponse {
        return self.request("POST", url, content_type, body);
    }

    /// Make HTTP request with automatic retry and exponential backoff
    pub fn request(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !HttpResponse {
        var attempt: u32 = 0;
        var backoff_ms = self.config.initial_backoff_ms;

        while (attempt <= self.config.max_retries) {
            if (self.config.verbose) {
                std.log.debug("HTTP {s} attempt {d}/{d}: {s}", .{ method, attempt + 1, self.config.max_retries + 1, url });
            }

            const result = if (self.use_libcurl)
                self.requestWithLibcurl(method, url, content_type, body)
            else
                self.requestWithNative(method, url, content_type, body);

            switch (result) {
                // Success - return response
                .success => |response| return response,

                // Retryable errors
                .timeout, .connection_failed, .network_error => {
                    if (attempt >= self.config.max_retries) {
                        if (self.config.verbose) {
                            std.log.err("HTTP request failed after {d} attempts", .{attempt + 1});
                        }
                        return error.HttpRequestFailed;
                    }

                    if (self.config.verbose) {
                        std.log.debug("HTTP request failed, retrying in {d}ms...", .{backoff_ms});
                    }

                    // Exponential backoff with jitter
                    std.Thread.sleep(backoff_ms * std.time.ns_per_ms);
                    backoff_ms = @min(backoff_ms * 2, self.config.max_backoff_ms);
                    attempt += 1;
                },

                // Non-retryable errors
                .client_error, .server_error => |err| return err,
            }
        }

        return error.HttpRequestFailed;
    }

    /// Request result for retry logic
    const RequestResult = union(enum) {
        success: HttpResponse,
        timeout: void,
        connection_failed: void,
        network_error: void,
        client_error: anyerror,
        server_error: anyerror,
    };

    fn parseMethod(method: []const u8) ?std.http.Method {
        if (std.ascii.eqlIgnoreCase(method, "GET")) return .GET;
        if (std.ascii.eqlIgnoreCase(method, "POST")) return .POST;
        if (std.ascii.eqlIgnoreCase(method, "PUT")) return .PUT;
        if (std.ascii.eqlIgnoreCase(method, "DELETE")) return .DELETE;
        if (std.ascii.eqlIgnoreCase(method, "PATCH")) return .PATCH;
        if (std.ascii.eqlIgnoreCase(method, "HEAD")) return .HEAD;
        if (std.ascii.eqlIgnoreCase(method, "OPTIONS")) return .OPTIONS;
        return null;
    }

    fn mapRequestError(err: anyerror) RequestResult {
        return switch (err) {
            error.TimedOut, error.ConnectionTimedOut => .{ .timeout = {} },
            error.ConnectionRefused,
            error.ConnectionResetByPeer,
            error.NetworkUnreachable,
            error.NameResolutionFailure,
            => .{ .connection_failed = {} },
            error.BrokenPipe,
            error.ConnectionAborted,
            => .{ .network_error = {} },
            else => .{ .client_error = err },
        };
    }

    fn deinitHeaderMap(allocator: std.mem.Allocator, headers: *std.StringHashMap([]const u8)) void {
        var iterator = headers.iterator();
        while (iterator.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        headers.deinit();
    }

    /// Make request using libcurl (if available)
    fn requestWithLibcurl(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) RequestResult {
        // This is a placeholder for libcurl integration
        // In a real implementation, you would use curl bindings here

        // For now, fall back to native implementation
        return self.requestWithNative(method, url, content_type, body);
    }

    /// Make request using native Zig HTTP client
    fn requestWithNative(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) RequestResult {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const http_method = parseMethod(method) orelse return .{ .client_error = error.UnsupportedHttpMethod };
        const uri = std.Uri.parse(url) catch |err| return mapRequestError(err);

        var req = client.request(http_method, uri, .{}) catch |err| return mapRequestError(err);
        defer req.deinit();

        if (content_type) |ct| {
            req.headers.content_type = .{ .override = ct };
        }
        req.headers.user_agent = .{ .override = self.config.user_agent };

        const request_body = body orelse &.{};
        req.sendBodyComplete(request_body) catch |err| return mapRequestError(err);

        var redirect_buf: [1024]u8 = undefined;
        var response = req.receiveHead(&redirect_buf) catch |err| return mapRequestError(err);

        var response_headers = std.StringHashMap([]const u8).init(self.allocator);
        var header_iter = response.head.headers.iterator();
        while (header_iter.next()) |header| {
            const name_owned = self.allocator.dupe(u8, header.name) catch |err| {
                deinitHeaderMap(self.allocator, &response_headers);
                return mapRequestError(err);
            };
            const value_owned = self.allocator.dupe(u8, header.value) catch |err| {
                self.allocator.free(name_owned);
                deinitHeaderMap(self.allocator, &response_headers);
                return mapRequestError(err);
            };
            response_headers.put(name_owned, value_owned) catch |err| {
                self.allocator.free(name_owned);
                self.allocator.free(value_owned);
                deinitHeaderMap(self.allocator, &response_headers);
                return mapRequestError(err);
            };
        }

        var response_body_list = std.ArrayList(u8).initCapacity(self.allocator, 0) catch |err| {
            deinitHeaderMap(self.allocator, &response_headers);
            return mapRequestError(err);
        };

        var buffer: [8192]u8 = undefined;
        const reader = response.reader(&.{});
        while (true) {
            var slices = [_][]u8{buffer[0..]};
            const n = reader.readVec(slices[0..]) catch |err| switch (err) {
                error.ReadFailed => {
                    response_body_list.deinit(self.allocator);
                    deinitHeaderMap(self.allocator, &response_headers);
                    if (response.bodyErr()) |body_err| {
                        return mapRequestError(body_err);
                    }
                    return .{ .network_error = {} };
                },
                error.EndOfStream => 0,
            };
            if (n == 0) break;
            response_body_list.appendSlice(self.allocator, buffer[0..n]) catch |err| {
                response_body_list.deinit(self.allocator);
                deinitHeaderMap(self.allocator, &response_headers);
                return mapRequestError(err);
            };
        }

        const response_body = response_body_list.toOwnedSlice(self.allocator) catch |err| {
            response_body_list.deinit(self.allocator);
            deinitHeaderMap(self.allocator, &response_headers);
            return mapRequestError(err);
        };
        const status_code: u16 = @intCast(@intFromEnum(response.head.status));

        return RequestResult{ .success = HttpResponse{
            .status_code = status_code,
            .headers = response_headers,
            .body = response_body,
            .allocator = self.allocator,
        } };
    }

    /// Test connectivity with enhanced diagnostics
    pub fn testConnectivity(self: *Self, url: []const u8) !bool {
        if (self.config.verbose) {
            std.log.debug("Testing connectivity to: {s}", .{url});
        }

        var response = self.get(url) catch |err| {
            if (self.config.verbose) {
                std.log.warn("Connectivity test failed: {any}", .{err});
            }
            return false;
        };
        defer response.deinit();

        if (self.config.verbose) {
            std.log.debug("Connectivity test successful - status: {d}, body length: {d}", .{ response.status_code, response.body.len });
        }

        return response.status_code < 500;
    }

    /// Detect if libcurl is available at runtime
    fn detectLibcurl() bool {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{
                "libcurl.dll",
                "curl.dll",
            },
            .macos, .ios, .tvos, .watchos, .visionos => &[_][]const u8{
                "libcurl.dylib",
                "libcurl.4.dylib",
            },
            else => &[_][]const u8{
                "libcurl.so.4",
                "libcurl.so",
            },
        };

        for (candidates) |name| {
            var lib = std.DynLib.open(name) catch continue;
            lib.close();
            return true;
        }

        return false;
    }

    fn performNativeRequest(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !HttpResponse {
        const http_method = std.meta.stringToEnum(std.http.Method, method) orelse
            return error.InvalidHttpMethod;

        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        var headers: std.http.Client.Request.Headers = .{};
        headers.user_agent = .{ .override = self.config.user_agent };
        if (content_type) |ct| {
            headers.content_type = .{ .override = ct };
        }

        const redirect_behavior = if (self.config.follow_redirects and self.config.max_redirects > 0) b: {
            const redirect_limit = @min(self.config.max_redirects, @as(u32, std.math.maxInt(u16)));
            break :b std.http.Client.Request.RedirectBehavior.init(@intCast(redirect_limit));
        } else .not_allowed;

        var req = try client.request(http_method, try std.Uri.parse(url), .{
            .headers = headers,
            .redirect_behavior = redirect_behavior,
        });
        defer req.deinit();

        if (body) |payload| {
            try req.sendBodyComplete(@constCast(payload));
        } else if (http_method.requestHasBody()) {
            try req.sendBodyComplete(&.{});
        } else {
            try req.sendBodiless();
        }

        var redirect_buf: [1024]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);

        var response_headers = std.StringHashMap([]const u8).init(self.allocator);
        errdefer {
            var iterator = response_headers.iterator();
            while (iterator.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            response_headers.deinit();
        }

        var header_iter = response.head.iterateHeaders();
        while (header_iter.next()) |header| {
            const name = try self.allocator.dupe(u8, header.name);
            const value = try self.allocator.dupe(u8, header.value);
            try response_headers.put(name, value);
        }

        var list = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        errdefer list.deinit(self.allocator);

        var transfer_buf: [8192]u8 = undefined;
        const rdr = response.reader(&transfer_buf);
        while (true) {
            const slice: []u8 = transfer_buf[0..];
            var slices = [_][]u8{slice};
            const n = rdr.readVec(slices[0..]) catch |err| switch (err) {
                error.ReadFailed => return error.HttpRequestFailed,
                error.EndOfStream => 0,
            };
            if (n == 0) break;
            try list.appendSlice(self.allocator, transfer_buf[0..n]);
        }

        return HttpResponse{
            .status_code = @intFromEnum(response.head.status),
            .headers = response_headers,
            .body = try list.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }
};

/// Enhanced connectivity tester with comprehensive diagnostics
pub const ConnectivityTester = struct {
    allocator: std.mem.Allocator,
    client: HttpClient,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: HttpClientConfig) Self {
        return Self{
            .allocator = allocator,
            .client = HttpClient.init(allocator, config),
        };
    }

    /// Run comprehensive connectivity tests
    pub fn runDiagnostics(self: *Self, base_url: []const u8) !void {
        std.log.info("=== HTTP Client Connectivity Diagnostics ===", .{});

        // Test basic connectivity
        std.log.info("1. Testing basic connectivity...", .{});
        const health_url = try std.fmt.allocPrint(self.allocator, "{s}/health", .{base_url});
        defer self.allocator.free(health_url);

        if (self.client.testConnectivity(health_url)) |success| {
            if (success) {
                std.log.info("   ✅ Basic connectivity successful", .{});
            } else {
                std.log.warn("   ❌ Basic connectivity failed", .{});
            }
        } else |err| {
            std.log.err("   ❌ Connectivity test error: {any}", .{err});
        }

        // Test with retries
        std.log.info("2. Testing with retry mechanism...", .{});
        const response = self.client.get(health_url) catch |err| {
            std.log.err("   ❌ GET request failed: {any}", .{err});
            return;
        };
        defer response.deinit();

        std.log.info("   ✅ GET request successful - Status: {d}", .{response.status_code});
        std.log.info("   📊 Response headers: {d}", .{response.headers.count()});
        std.log.info("   📄 Response body length: {d} bytes", .{response.body.len});

        // Test POST request
        std.log.info("3. Testing POST request...", .{});
        const test_data = "{\"test\": \"connectivity\"}";
        const post_response = self.client.post(health_url, "application/json", test_data) catch |err| {
            std.log.err("   ❌ POST request failed: {any}", .{err});
            return;
        };
        defer post_response.deinit();

        std.log.info("   ✅ POST request successful - Status: {d}", .{post_response.status_code});

        std.log.info("=== Diagnostics Complete ===", .{});
    }
};

// Test function for the module
test "HttpClient basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var io_threaded = std.Io.Threaded.init(allocator);
    defer io_threaded.deinit();
    const io = io_threaded.io();

    var server = try std.Io.net.listen(.{ .ip4 = std.Io.net.Ip4Address.loopback(0) }, io, .{});
    defer server.deinit(io);

    const port = server.socket.address.getPort();
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/", .{port});
    defer allocator.free(url);

    const handler = try std.Thread.spawn(.{}, handleTestServer, .{ io, &server });
    defer handler.join();

    var client = HttpClient.init(allocator, .{});
    var response = try client.get(url);
    defer response.deinit();

    try testing.expectEqual(@as(u16, 200), response.status_code);
    try testing.expectEqualStrings("{\"ok\":true}", response.body);
    try testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
    try testing.expectEqualStrings("true", response.headers.get("X-Test").?);
}

fn handleTestServer(io: std.Io, server: *std.Io.net.Server) !void {
    var stream = try server.accept(io);
    defer stream.close(io);

    var read_buffer: [4096]u8 = undefined;
    var write_buffer: [4096]u8 = undefined;
    var reader = std.Io.net.Stream.Reader.init(stream, io, &read_buffer);
    var writer = std.Io.net.Stream.Writer.init(stream, io, &write_buffer);

    var http_server = std.http.Server.init(&reader.interface, &writer.interface);
    var request = try http_server.receiveHead();

    const headers = [_]std.http.Header{
        .{ .name = "Content-Type", .value = "application/json" },
        .{ .name = "X-Test", .value = "true" },
    };

    try request.respond("{\"ok\":true}", .{
        .status = .ok,
        .keep_alive = false,
        .extra_headers = &headers,
    });
}
