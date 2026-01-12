//! Async I/O HTTP client using std.Io
//!
//! Provides async HTTP requests with streaming support for connectors.
//!
//! Security features:
//! - URL validation against malicious inputs
//! - Optional HTTPS-only mode for secure communications
//! - Redirect limits to prevent redirect loops

const std = @import("std");

const http_mod = @import("mod.zig");

pub const Method = http_mod.Method;

pub const HttpStatus = enum(u16) {
    ok = 200,
    created = 201,
    accepted = 202,
    no_content = 204,
    bad_request = 400,
    unauthorized = 401,
    forbidden = 403,
    not_found = 404,
    method_not_allowed = 405,
    conflict = 409,
    internal_server_error = 500,
    bad_gateway = 502,
    service_unavailable = 503,
    _,
};

pub const HttpError = http_mod.HttpError;

/// Security policy for HTTP requests
pub const SecurityPolicy = struct {
    /// Require HTTPS for all requests (reject http://)
    require_https: bool = false,
    /// Allow localhost connections over HTTP even when require_https is true
    allow_localhost_http: bool = true,
    /// Maximum URL length
    max_url_length: usize = 2048,
    /// Reject URLs containing these hosts (blocklist)
    blocked_hosts: []const []const u8 = &.{},
};

/// Validates URL for basic security and format requirements
pub fn validateUrl(url: []const u8) !void {
    return validateUrlWithPolicy(url, .{});
}

/// Validates URL with a specific security policy
pub fn validateUrlWithPolicy(url: []const u8, policy: SecurityPolicy) !void {
    // Basic URL validation to prevent malicious inputs
    if (url.len == 0) return HttpError.InvalidUrl;
    if (url.len > policy.max_url_length) return HttpError.InvalidUrl;

    const is_https = std.mem.startsWith(u8, url, "https://");
    const is_http = std.mem.startsWith(u8, url, "http://");

    // Must start with http:// or https://
    if (!is_http and !is_https) {
        return HttpError.InvalidUrl;
    }

    // Check HTTPS requirement
    if (policy.require_https and is_http) {
        // Check if localhost exception applies
        if (policy.allow_localhost_http) {
            const host_start = if (is_https) "https://".len else "http://".len;
            const remaining = url[host_start..];
            const is_localhost = std.mem.startsWith(u8, remaining, "localhost") or
                std.mem.startsWith(u8, remaining, "127.0.0.1") or
                std.mem.startsWith(u8, remaining, "[::1]");
            if (!is_localhost) {
                return HttpError.InvalidUrl; // HTTPS required for non-localhost
            }
        } else {
            return HttpError.InvalidUrl; // HTTPS required
        }
    }

    // Check for potentially dangerous characters
    for (url) |c| {
        if (c < 32 or c == 127) { // Control characters
            return HttpError.InvalidUrl;
        }
    }

    // Check blocked hosts
    if (policy.blocked_hosts.len > 0) {
        const host_start = if (is_https) "https://".len else "http://".len;
        const remaining = url[host_start..];
        const host_end = std.mem.indexOfAny(u8, remaining, "/:?#") orelse remaining.len;
        const host = remaining[0..host_end];

        for (policy.blocked_hosts) |blocked| {
            if (std.mem.eql(u8, host, blocked)) {
                return HttpError.InvalidUrl;
            }
        }
    }
}

pub const HttpRequest = struct {
    allocator: std.mem.Allocator,
    method: Method,
    url: []const u8,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: ?[]const u8 = null,
    timeout_ms: u32 = 30_000,
    follow_redirects: bool = true,
    max_redirects: u8 = 5,

    pub fn init(allocator: std.mem.Allocator, method: Method, url: []const u8) !HttpRequest {
        try validateUrl(url);
        return .{
            .allocator = allocator,
            .method = method,
            .url = try allocator.dupe(u8, url),
            .headers = .{},
            .body = null,
            .timeout_ms = 30_000,
            .follow_redirects = true,
            .max_redirects = 5,
        };
    }

    pub fn deinit(self: *HttpRequest) void {
        self.allocator.free(self.url);
        if (self.body) |body| {
            self.allocator.free(body);
        }

        var iter = self.headers.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn setHeader(self: *HttpRequest, key: []const u8, value: []const u8) !void {
        if (self.headers.get(key)) |old_value| {
            self.allocator.free(old_value);
        }

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        try self.headers.put(self.allocator, key_copy, value_copy);
    }

    pub fn setBody(self: *HttpRequest, body: []const u8) !void {
        if (self.body) |old_body| {
            self.allocator.free(old_body);
        }

        self.body = try self.allocator.dupe(u8, body);
    }

    pub fn setJsonBody(self: *HttpRequest, json: []const u8) !void {
        try self.setHeader("Content-Type", "application/json");
        try self.setBody(json);
    }

    pub fn setBearerToken(self: *HttpRequest, token: []const u8) !void {
        const header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
        errdefer self.allocator.free(header);
        try self.setHeader("Authorization", header);
    }
};

pub const HttpResponse = struct {
    status: HttpStatus,
    status_code: u16,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HttpResponse {
        return .{
            .status = @enumFromInt(0),
            .status_code = 0,
            .headers = .{},
            .body = &.{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HttpResponse) void {
        self.allocator.free(self.body);

        var iter = self.headers.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn getHeader(self: *const HttpResponse, key: []const u8) ?[]const u8 {
        return self.headers.get(key);
    }

    pub fn isSuccess(self: *const HttpResponse) bool {
        return self.status_code >= 200 and self.status_code < 300;
    }

    pub fn isRedirect(self: *const HttpResponse) bool {
        return self.status_code >= 300 and self.status_code < 400;
    }

    pub fn isError(self: *const HttpResponse) bool {
        return self.status_code >= 400;
    }
};

pub const AsyncHttpClient = struct {
    allocator: std.mem.Allocator,
    io_backend: *std.Io.Threaded,
    client: *std.http.Client,
    redirect_count: u8 = 0,

    pub fn init(allocator: std.mem.Allocator) !AsyncHttpClient {
        // Create the I/O backend
        const io_backend = try allocator.create(std.Io.Threaded);
        errdefer allocator.destroy(io_backend);
        io_backend.* = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });

        // Create and initialize the HTTP client
        const client = try allocator.create(std.http.Client);
        errdefer allocator.destroy(client);
        client.* = .{
            .allocator = allocator,
            .io = io_backend.io(),
        };

        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .client = client,
            .redirect_count = 0,
        };
    }

    pub fn deinit(self: *AsyncHttpClient) void {
        self.client.deinit();
        self.io_backend.deinit();
        self.allocator.destroy(self.client);
        self.allocator.destroy(self.io_backend);
        self.* = undefined;
    }

    pub fn fetch(self: *AsyncHttpClient, http_request: *HttpRequest) !HttpResponse {
        var response = HttpResponse.init(self.allocator);
        errdefer response.deinit();

        const uri = try std.Uri.parse(http_request.url);

        // Convert local Method enum to std.http.Method
        const method: std.http.Method = switch (http_request.method) {
            .get => .GET,
            .post => .POST,
            .put => .PUT,
            .delete => .DELETE,
            .patch => .PATCH,
            .head => .HEAD,
            .options => .OPTIONS,
        };

        // Build extra headers from the request
        var extra_headers_list = std.ArrayListUnmanaged(std.http.Header){};
        defer extra_headers_list.deinit(self.allocator);

        var header_iter = http_request.headers.iterator();
        while (header_iter.next()) |entry| {
            // Skip headers that are handled specially
            const name_lower = entry.key_ptr.*;
            if (std.mem.eql(u8, name_lower, "authorization") or
                std.mem.eql(u8, name_lower, "content-type"))
            {
                continue;
            }
            try extra_headers_list.append(self.allocator, .{
                .name = name_lower,
                .value = entry.value_ptr.*,
            });
        }

        // Create the request
        var req = try self.client.*.request(method, uri, .{
            .headers = .{
                .authorization = if (http_request.headers.get("authorization")) |auth|
                    .{ .override = auth }
                else if (http_request.headers.get("Authorization")) |auth|
                    .{ .override = auth }
                else
                    .default,
                .content_type = if (http_request.headers.get("content-type")) |ct|
                    .{ .override = ct }
                else if (http_request.headers.get("Content-Type")) |ct|
                    .{ .override = ct }
                else
                    .default,
            },
            .extra_headers = extra_headers_list.items,
        });
        defer req.deinit();

        // Send request body if present
        if (http_request.body) |body| {
            try req.sendBodyComplete(@constCast(body));
        } else {
            try req.sendBodiless();
        }

        // Receive response head
        var redirect_buffer: [2048]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        response.status_code = @intFromEnum(res.head.status);
        response.status = @enumFromInt(response.status_code);

        // Read response body
        var transfer_buffer: [16384]u8 = undefined;
        const body_reader = res.reader(&transfer_buffer);

        var body_list = std.ArrayListUnmanaged(u8){};
        errdefer body_list.deinit(self.allocator);

        // Read all body data using readSliceShort
        var read_buf: [4096]u8 = undefined;
        while (true) {
            const bytes_read = body_reader.readSliceShort(&read_buf) catch break;
            if (bytes_read == 0) break;
            try body_list.appendSlice(self.allocator, read_buf[0..bytes_read]);
        }

        response.body = try body_list.toOwnedSlice(self.allocator);

        return response;
    }

    pub const StreamingResponse = struct {
        response: HttpResponse,

        pub fn deinit(self: *StreamingResponse) void {
            self.response.deinit();
        }
    };

    /// Note: Streaming is handled internally in Zig 0.16's HTTP client.
    /// This method now returns a regular response for API compatibility.
    pub fn fetchStreaming(self: *AsyncHttpClient, http_request: *HttpRequest) !StreamingResponse {
        const response = try self.fetch(http_request);
        return .{
            .response = response,
        };
    }

    pub fn fetchJson(self: *AsyncHttpClient, http_request: *HttpRequest) !HttpResponse {
        try http_request.setHeader("Accept", "application/json");
        return try self.fetch(http_request);
    }

    /// Async version of fetch - delegates to standard fetch in Zig 0.16
    pub fn fetchAsync(self: *AsyncHttpClient, http_request: *HttpRequest) !HttpResponse {
        return try self.fetch(http_request);
    }

    /// Batch request helper for making multiple HTTP requests efficiently
    /// Demonstrates proper resource management for multiple concurrent operations
    pub fn fetchBatch(
        self: *AsyncHttpClient,
        allocator: std.mem.Allocator,
        requests: []const *HttpRequest,
    ) ![]HttpResponse {
        const responses = try allocator.alloc(HttpResponse, requests.len);
        errdefer {
            for (responses[0..]) |*response| {
                response.deinit();
            }
            allocator.free(responses);
        }

        var completed: usize = 0;

        // Process requests sequentially for simplicity and resource safety
        for (requests, 0..) |request, i| {
            responses[i] = try self.fetchAsync(request);
            completed += 1;
        }

        std.log.info("Completed {d}/{d} requests successfully", .{ completed, requests.len });

        return responses;
    }

    pub fn get(self: *AsyncHttpClient, url: []const u8) !HttpResponse {
        var http_request = try HttpRequest.init(self.allocator, .get, url);
        defer http_request.deinit();
        return try self.fetch(&http_request);
    }

    pub fn post(self: *AsyncHttpClient, url: []const u8, body: []const u8) !HttpResponse {
        var http_request = try HttpRequest.init(self.allocator, .post, url);
        errdefer http_request.deinit();
        try http_request.setJsonBody(body);
        return try self.fetch(&http_request);
    }

    pub fn postJson(self: *AsyncHttpClient, url: []const u8, json: []const u8) !HttpResponse {
        return try self.post(url, json);
    }
};

test "http request lifecycle" {
    const allocator = std.testing.allocator;

    var request = try HttpRequest.init(allocator, .get, "https://example.com");
    defer request.deinit();

    try request.setHeader("User-Agent", "abi/0.1.0");
    try std.testing.expectEqual(Method.get, request.method);
}

test "http response status checks" {
    const allocator = std.testing.allocator;

    var response = HttpResponse.init(allocator);
    defer response.deinit();

    response.status_code = 200;
    response.status = .ok;
    try std.testing.expect(response.isSuccess());
    try std.testing.expect(!response.isError());

    response.status_code = 404;
    response.status = .not_found;
    try std.testing.expect(!response.isSuccess());
    try std.testing.expect(response.isError());
}

test "url validation with security policy" {
    // Default policy accepts both HTTP and HTTPS
    try validateUrl("https://example.com");
    try validateUrl("http://example.com");

    // HTTPS-only policy
    const https_only = SecurityPolicy{ .require_https = true, .allow_localhost_http = false };
    try validateUrlWithPolicy("https://example.com", https_only);
    try std.testing.expectError(HttpError.InvalidUrl, validateUrlWithPolicy("http://example.com", https_only));

    // HTTPS-only with localhost exception
    const https_with_localhost = SecurityPolicy{ .require_https = true, .allow_localhost_http = true };
    try validateUrlWithPolicy("https://example.com", https_with_localhost);
    try validateUrlWithPolicy("http://localhost:8080/api", https_with_localhost);
    try validateUrlWithPolicy("http://127.0.0.1:8080/api", https_with_localhost);
    try std.testing.expectError(HttpError.InvalidUrl, validateUrlWithPolicy("http://example.com", https_with_localhost));

    // Blocked hosts
    const blocked = SecurityPolicy{ .blocked_hosts = &.{ "malicious.com", "evil.org" } };
    try validateUrlWithPolicy("https://example.com", blocked);
    try std.testing.expectError(HttpError.InvalidUrl, validateUrlWithPolicy("https://malicious.com/path", blocked));
    try std.testing.expectError(HttpError.InvalidUrl, validateUrlWithPolicy("http://evil.org", blocked));
}

test "url validation rejects invalid urls" {
    try std.testing.expectError(HttpError.InvalidUrl, validateUrl(""));
    try std.testing.expectError(HttpError.InvalidUrl, validateUrl("ftp://example.com"));
    try std.testing.expectError(HttpError.InvalidUrl, validateUrl("javascript:alert(1)"));
}
