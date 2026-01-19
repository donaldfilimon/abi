const std = @import("std");

const http_utils = @import("../shared/utils/http/mod.zig");
pub const HttpError = http_utils.HttpError;

pub const Response = struct {
    status: u16,
    body: []const u8,
};

pub const RequestOptions = struct {
    max_response_bytes: usize = 1024 * 1024,
    user_agent: []const u8 = "abi-http",
    follow_redirects: bool = true,
    redirect_limit: u16 = 3,
    content_type: ?[]const u8 = null,
    extra_headers: []const std.http.Header = &.{},
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
            options.max_response_bytes,
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
