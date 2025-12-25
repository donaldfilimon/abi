const std = @import("std");

pub const HttpError = error{
    InvalidUrl,
    ResponseTooLarge,
    ReadFailed,
};

pub const Response = struct {
    status: u16,
    body: []const u8,
};

pub const RequestOptions = struct {
    max_response_bytes: usize = 1024 * 1024,
    user_agent: []const u8 = "abi-http",
    follow_redirects: bool = true,
    redirect_limit: u16 = 3,
};

pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator) !HttpClient {
        const io_backend = std.Io.Threaded.init(allocator);
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
        const uri = std.Uri.parse(url) catch return HttpError.InvalidUrl;
        var request_options: std.http.Client.RequestOptions = .{};
        request_options.headers.user_agent = .{ .override = options.user_agent };
        request_options.redirect_behavior = if (options.follow_redirects)
            std.http.Client.Request.RedirectBehavior.init(options.redirect_limit)
        else
            .not_allowed;

        var req = try self.client.request(.GET, uri, request_options);
        defer req.deinit();
        try req.sendBodiless();

        var redirect_buffer: [4096]u8 = undefined;
        var response = try req.receiveHead(&redirect_buffer);

        var transfer_buffer: [4096]u8 = undefined;
        const reader = response.reader(&transfer_buffer);
        const body = try readAllAlloc(reader, self.allocator, options.max_response_bytes);
        return .{
            .status = @intFromEnum(response.head.status),
            .body = body,
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
    var list = std.ArrayList(u8).empty;
    errdefer list.deinit(allocator);

    var buffer: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(buffer[0..]) catch
            return HttpError.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > max_bytes) return HttpError.ResponseTooLarge;
        try list.appendSlice(allocator, buffer[0..n]);
        if (n < buffer.len) break;
    }
    return list.toOwnedSlice(allocator);
}
