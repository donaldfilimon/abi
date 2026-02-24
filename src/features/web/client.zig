//! HTTP Client Implementation
//!
//! This module provides the core HTTP client functionality for the web module.
//! It wraps Zig's standard library HTTP client with convenient utilities.
//!
//! ## Thread Safety
//!
//! Each `HttpClient` instance manages its own I/O backend and should be used
//! from a single thread. For multi-threaded access, use the global functions
//! in `mod.zig` which provide mutex protection.
//!
//! ## Memory Management
//!
//! Response bodies are allocated using the client's allocator. Use
//! `freeResponse()` to properly deallocate response memory.

const std = @import("std");

const http_utils = @import("../../services/shared/utils.zig").http;

/// HTTP-specific errors that can occur during requests.
/// These are re-exported from the shared http utilities module.
pub const HttpError = http_utils.HttpError;

/// HTTP response structure containing status code and body.
///
/// The `body` field is allocated memory that must be freed using
/// `HttpClient.freeResponse()` or `allocator.free()`.
pub const Response = struct {
    /// HTTP status code (e.g., 200, 404, 500).
    status: u16,
    /// Response body as raw bytes. Caller owns this memory.
    body: []const u8,
};

/// Configuration options for HTTP requests.
///
/// Provides control over response size limits, user agent, redirect behavior,
/// and custom headers. Security limits are enforced to prevent resource exhaustion.
pub const RequestOptions = struct {
    /// Maximum bytes to read from response body. Default: 1MB.
    /// Hard limit: 100MB to prevent memory exhaustion attacks.
    /// Values exceeding the hard limit are automatically capped.
    max_response_bytes: usize = 1024 * 1024,

    /// User-Agent header value sent with requests.
    user_agent: []const u8 = "abi-http",

    /// Whether to automatically follow HTTP redirects (3xx responses).
    follow_redirects: bool = true,

    /// Maximum number of redirects to follow. Prevents infinite redirect loops.
    redirect_limit: u16 = 3,

    /// Content-Type header for requests with a body.
    /// For JSON requests, this is automatically set to "application/json".
    content_type: ?[]const u8 = null,

    /// Additional headers to include in the request.
    /// Useful for authentication tokens, custom headers, etc.
    extra_headers: []const std.http.Header = &.{},

    /// Hard upper limit for response size (100MB).
    /// This cannot be exceeded even if max_response_bytes is set higher.
    /// Protects against memory exhaustion from malicious or misconfigured servers.
    pub const MAX_ALLOWED_RESPONSE_BYTES: usize = 100 * 1024 * 1024;

    /// Returns the effective max response bytes, capped at the hard limit.
    /// This ensures security limits are always enforced regardless of user configuration.
    pub fn effectiveMaxResponseBytes(self: RequestOptions) usize {
        return @min(self.max_response_bytes, MAX_ALLOWED_RESPONSE_BYTES);
    }
};

/// Synchronous HTTP client for making web requests.
///
/// This client manages its own I/O backend and provides methods for
/// GET and POST requests with configurable options. It properly handles
/// Zig 0.16's I/O backend requirements.
///
/// ## Memory Management
///
/// - Call `deinit()` when finished to release resources
/// - Response bodies must be freed with `freeResponse()` or `allocator.free()`
///
/// ## Example
///
/// ```zig
/// var client = try HttpClient.init(allocator);
/// defer client.deinit();
///
/// const response = try client.get("https://api.example.com/data");
/// defer client.freeResponse(response);
///
/// std.debug.print("Status: {d}, Body: {s}\n", .{ response.status, response.body });
/// ```
pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    client: std.http.Client,

    /// Initialize a new HTTP client.
    ///
    /// Creates an I/O backend for network operations. The client is ready
    /// to make requests immediately after initialization.
    ///
    /// Returns an error if I/O backend initialization fails.
    pub fn init(allocator: std.mem.Allocator) !HttpClient {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        const client = std.http.Client{
            .allocator = allocator,
            .io = io_backend.io(),
        };
        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .client = client,
        };
    }

    /// Deinitialize the client and release all resources.
    ///
    /// After calling this, the client should not be used. Any outstanding
    /// response bodies remain valid and must still be freed separately.
    pub fn deinit(self: *HttpClient) void {
        self.client.deinit();
        self.io_backend.deinit();
        self.* = undefined;
    }

    /// Perform an HTTP GET request with default options.
    ///
    /// Shorthand for `getWithOptions(url, .{})`.
    pub fn get(self: *HttpClient, url: []const u8) !Response {
        return self.getWithOptions(url, .{});
    }

    /// Perform an HTTP GET request with custom options.
    ///
    /// Allows configuration of response size limits, redirects, headers, etc.
    pub fn getWithOptions(self: *HttpClient, url: []const u8, options: RequestOptions) !Response {
        return self.requestWithOptions(.GET, url, null, options);
    }

    /// Perform an HTTP POST request with a JSON body.
    ///
    /// Automatically sets Content-Type to "application/json".
    /// The body should be valid JSON encoded as UTF-8.
    pub fn postJson(self: *HttpClient, url: []const u8, body: []const u8) !Response {
        return self.requestWithOptions(.POST, url, body, .{
            .content_type = "application/json",
        });
    }

    /// Perform an HTTP request with full control over method, body, and options.
    ///
    /// This is the core request method used by `get()` and `postJson()`.
    /// Use this for PUT, DELETE, PATCH, or other HTTP methods.
    ///
    /// ## Parameters
    /// - `method`: HTTP method (GET, POST, PUT, DELETE, etc.)
    /// - `url`: Full URL including scheme (https://...)
    /// - `body`: Optional request body (null for bodiless requests)
    /// - `options`: Request configuration options
    ///
    /// ## Errors
    /// - `InvalidUrl`: URL parsing failed
    /// - `InvalidRequest`: Body provided for method that doesn't support it
    /// - `ResponseTooLarge`: Response exceeds configured size limit
    /// - `ReadFailed`: Error reading response body
    pub fn requestWithOptions(
        self: *HttpClient,
        method: std.http.Method,
        url: []const u8,
        body: ?[]const u8,
        options: RequestOptions,
    ) !Response {
        const uri = std.Uri.parse(url) catch return error.InvalidUrl;
        var request_options: std.http.Client.RequestOptions = .{};
        request_options.headers.user_agent = .{ .override = options.user_agent };
        request_options.redirect_behavior = if (options.follow_redirects)
            std.http.Client.Request.RedirectBehavior.init(options.redirect_limit)
        else
            .not_allowed;
        request_options.extra_headers = options.extra_headers;

        if (options.content_type) |content_type| {
            request_options.headers.content_type = .{ .override = content_type };
        }

        var req = try self.client.request(method, uri, request_options);
        defer req.deinit();

        if (body) |payload| {
            if (!method.requestHasBody()) return error.InvalidRequest;
            var send_buffer: [4096]u8 = undefined;
            var body_writer = try req.sendBody(&send_buffer);
            try body_writer.writer.writeAll(payload);
            try body_writer.end();
        } else {
            try req.sendBodiless();
        }

        var redirect_buffer: [4096]u8 = undefined;
        var response = try req.receiveHead(&redirect_buffer);

        var transfer_buffer: [4096]u8 = undefined;
        const reader = response.reader(&transfer_buffer);
        const response_body = try readAllAlloc(
            reader,
            self.allocator,
            options.effectiveMaxResponseBytes(),
        );
        return .{
            .status = @intFromEnum(response.head.status),
            .body = response_body,
        };
    }

    /// Free a response body allocated by this client.
    ///
    /// This is equivalent to `self.allocator.free(response.body)` but
    /// provides a cleaner API that matches the request/response pattern.
    pub fn freeResponse(self: *HttpClient, response: Response) void {
        self.allocator.free(response.body);
    }
};

fn readAllAlloc(
    reader: *std.Io.Reader,
    allocator: std.mem.Allocator,
    max_bytes: usize,
) HttpError![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var buffer: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(buffer[0..]) catch
            return error.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > max_bytes) return error.ResponseTooLarge;
        list.appendSlice(allocator, buffer[0..n]) catch return error.ResponseTooLarge;
        if (n < buffer.len) break;
    }
    return list.toOwnedSlice(allocator) catch return error.ResponseTooLarge;
}

// ============================================================================
// Tests
// ============================================================================

test "request options default values" {
    const options = RequestOptions{};
    try std.testing.expectEqual(@as(usize, 1024 * 1024), options.max_response_bytes);
    try std.testing.expectEqualStrings("abi-http", options.user_agent);
    try std.testing.expect(options.follow_redirects);
    try std.testing.expectEqual(@as(u16, 3), options.redirect_limit);
    try std.testing.expectEqual(@as(?[]const u8, null), options.content_type);
}

test "request options custom values" {
    const options = RequestOptions{
        .max_response_bytes = 2048,
        .user_agent = "custom-agent",
        .follow_redirects = false,
        .redirect_limit = 5,
        .content_type = "text/plain",
    };
    try std.testing.expectEqual(@as(usize, 2048), options.max_response_bytes);
    try std.testing.expectEqualStrings("custom-agent", options.user_agent);
    try std.testing.expect(!options.follow_redirects);
    try std.testing.expectEqual(@as(u16, 5), options.redirect_limit);
    try std.testing.expectEqualStrings("text/plain", options.content_type.?);
}

test "response struct" {
    const response = Response{
        .status = 200,
        .body = "OK",
    };
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expectEqualStrings("OK", response.body);
}

test "request options effective max response bytes" {
    // Default should be 1MB
    const default_options = RequestOptions{};
    try std.testing.expectEqual(@as(usize, 1024 * 1024), default_options.effectiveMaxResponseBytes());

    // Custom value under limit should be used
    const small_options = RequestOptions{ .max_response_bytes = 2048 };
    try std.testing.expectEqual(@as(usize, 2048), small_options.effectiveMaxResponseBytes());

    // Value exceeding hard limit should be capped
    const large_options = RequestOptions{ .max_response_bytes = 200 * 1024 * 1024 };
    try std.testing.expectEqual(RequestOptions.MAX_ALLOWED_RESPONSE_BYTES, large_options.effectiveMaxResponseBytes());
}

test {
    std.testing.refAllDecls(@This());
}
