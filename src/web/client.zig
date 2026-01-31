const std = @import("std");

const http_utils = @import("../shared/utils.zig").http;
pub const HttpError = http_utils.HttpError;

pub const Response = struct {
    status: u16,
    body: []const u8,
};

pub const RequestOptions = struct {
    /// Maximum bytes to read from response body. Default: 1MB.
    /// Hard limit: 100MB to prevent memory exhaustion attacks.
    max_response_bytes: usize = 1024 * 1024,
    user_agent: []const u8 = "abi-http",
    follow_redirects: bool = true,
    redirect_limit: u16 = 3,
    content_type: ?[]const u8 = null,
    extra_headers: []const std.http.Header = &.{},

    /// Hard upper limit for response size (100MB).
    /// This cannot be exceeded even if max_response_bytes is set higher.
    pub const MAX_ALLOWED_RESPONSE_BYTES: usize = 100 * 1024 * 1024;

    /// Returns the effective max response bytes, capped at the hard limit.
    pub fn effectiveMaxResponseBytes(self: RequestOptions) usize {
        return @min(self.max_response_bytes, MAX_ALLOWED_RESPONSE_BYTES);
    }
};

pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    client: std.http.Client,

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

    pub fn deinit(self: *HttpClient) void {
        self.client.deinit();
        self.io_backend.deinit();
        self.* = undefined;
    }

    pub fn get(self: *HttpClient, url: []const u8) !Response {
        return self.getWithOptions(url, .{});
    }

    pub fn getWithOptions(self: *HttpClient, url: []const u8, options: RequestOptions) !Response {
        return self.requestWithOptions(.GET, url, null, options);
    }

    pub fn postJson(self: *HttpClient, url: []const u8, body: []const u8) !Response {
        return self.requestWithOptions(.POST, url, body, .{
            .content_type = "application/json",
        });
    }

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
