//! Async I/O HTTP client using std.Io
//!
//! Provides async HTTP requests with streaming support for connectors.

const std = @import("std");

pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
};

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

pub const HttpError = error{
    InvalidUrl,
    ConnectionFailed,
    RequestFailed,
    Timeout,
    InvalidResponse,
    RedirectExceeded,
};

pub const HttpRequest = struct {
    method: HttpMethod,
    url: []const u8,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8 = null,
    timeout_ms: u32 = 30_000,
    follow_redirects: bool = true,
    max_redirects: u8 = 5,

    pub fn init(allocator: std.mem.Allocator, method: HttpMethod, url: []const u8) !HttpRequest {
        return .{
            .method = method,
            .url = try allocator.dupe(u8, url),
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = null,
            .timeout_ms = 30_000,
            .follow_redirects = true,
            .max_redirects = 5,
        };
    }

    pub fn deinit(self: *HttpRequest) void {
        const allocator = self.headers.allocator;
        allocator.free(self.url);
        if (self.body) |body| {
            allocator.free(body);
        }

        var iter = self.headers.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        self.* = undefined;
    }

    pub fn setHeader(self: *HttpRequest, key: []const u8, value: []const u8) !void {
        const allocator = self.headers.allocator;

        if (self.headers.get(key)) |old_value| {
            allocator.free(old_value);
        }

        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);

        const value_copy = try allocator.dupe(u8, value);
        errdefer allocator.free(value_copy);

        try self.headers.put(key_copy, value_copy);
    }

    pub fn setBody(self: *HttpRequest, body: []const u8) !void {
        const allocator = self.headers.allocator;

        if (self.body) |old_body| {
            allocator.free(old_body);
        }

        self.body = try allocator.dupe(u8, body);
    }

    pub fn setJsonBody(self: *HttpRequest, json: []const u8) !void {
        try self.setHeader("Content-Type", "application/json");
        try self.setBody(json);
    }

    pub fn setBearerToken(self: *HttpRequest, token: []const u8) !void {
        const header = try std.fmt.allocPrint(self.headers.allocator, "Bearer {s}", .{token});
        errdefer self.headers.allocator.free(header);
        try self.setHeader("Authorization", header);
    }
};

pub const HttpResponse = struct {
    status: HttpStatus,
    status_code: u16,
    headers: std.StringHashMap([]const u8),
    body: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HttpResponse {
        return .{
            .status = ._,
            .status_code = 0,
            .headers = std.StringHashMap([]const u8).init(allocator),
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
        self.headers.deinit();
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
    client: *std.http.Client,
    redirect_count: u8 = 0,

    pub fn init(allocator: std.mem.Allocator) !AsyncHttpClient {
        const client = try allocator.create(std.http.Client);
        client.* = .{ .allocator = allocator };

        return .{
            .allocator = allocator,
            .client = client,
            .redirect_count = 0,
        };
    }

    pub fn deinit(self: *AsyncHttpClient) void {
        self.client.deinit();
        self.allocator.destroy(self.client);
        self.* = undefined;
    }

    pub fn fetch(self: *AsyncHttpClient, request: *HttpRequest) !HttpResponse {
        var response = HttpResponse.init(self.allocator);
        errdefer response.deinit();

        const uri = try std.Uri.parse(request.url);

        var server_header_buffer: [8192]u8 = undefined;
        var http_res = try self.client.open(.{
            .method = std.http.Method.fromEnum(@intFromEnum(request.method)),
            .location = uri,
            .server_header_buffer = &server_header_buffer,
            .headers = .{
                .authorization = if (request.headers.get("Authorization")) |auth| auth else "",
                .accept = if (request.headers.get("Accept")) |accept| accept else "*/*",
                .content_type = if (request.headers.get("Content-Type")) |ct| ct else "",
            },
        });

        defer http_res.deinit();

        if (request.body) |body| {
            try http_res.writeAll(body);
        }

        try http_res.finish();

        response.status_code = @intCast(http_res.status);
        response.status = @enumFromInt(response.status_code);

        var body_reader = http_res.reader();
        var body_list = std.ArrayList(u8).init(self.allocator);
        errdefer body_list.deinit();

        try body_reader.readAllArrayList(&body_list, 10 * 1024 * 1024);
        response.body = try body_list.toOwnedSlice();

        return response;
    }

    pub fn fetchJson(self: *AsyncHttpClient, request: *HttpRequest) !HttpResponse {
        try request.setHeader("Accept", "application/json");
        return try self.fetch(request);
    }

    pub fn get(self: *AsyncHttpClient, url: []const u8) !HttpResponse {
        var request = try HttpRequest.init(self.allocator, .GET, url);
        defer request.deinit();
        return try self.fetch(&request);
    }

    pub fn post(self: *AsyncHttpClient, url: []const u8, body: []const u8) !HttpResponse {
        var request = try HttpRequest.init(self.allocator, .POST, url);
        errdefer request.deinit();
        try request.setJsonBody(body);
        return try self.fetch(&request);
    }

    pub fn postJson(self: *AsyncHttpClient, url: []const u8, json: []const u8) !HttpResponse {
        return try self.post(url, json);
    }
};

test "http request lifecycle" {
    const allocator = std.testing.allocator;

    var request = try HttpRequest.init(allocator, .GET, "https://example.com");
    defer request.deinit();

    try request.setHeader("User-Agent", "abi/0.1.0");
    try std.testing.expectEqual(HttpMethod.GET, request.method);
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
