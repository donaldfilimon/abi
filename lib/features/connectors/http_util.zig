const std = @import("std");

pub const HttpResponse = struct {
    status: u16,
    body: []u8,
};

pub fn buildJson(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    return std.json.stringifyAlloc(allocator, value, .{});
}

pub fn joinUrl(allocator: std.mem.Allocator, base: []const u8, path: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, path, "http://") or std.mem.startsWith(u8, path, "https://")) {
        return allocator.dupe(u8, path);
    }

    const base_has_slash = std.mem.endsWith(u8, base, "/");
    const path_has_slash = std.mem.startsWith(u8, path, "/");

    if (base_has_slash and path_has_slash) {
        return std.fmt.allocPrint(allocator, "{s}{s}", .{ base, path[1..] });
    }
    if (!base_has_slash and !path_has_slash) {
        return std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, path });
    }
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ base, path });
}

pub fn postJson(
    allocator: std.mem.Allocator,
    url: []const u8,
    extra_headers: []const std.http.Header,
    body: []const u8,
) !HttpResponse {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var headers: std.http.Client.Request.Headers = .{};
    headers.content_type = .{ .override = "application/json" };

    var req = try client.request(.POST, try std.Uri.parse(url), .{
        .headers = headers,
        .extra_headers = extra_headers,
    });
    defer req.deinit();

    try req.sendBodyComplete(@constCast(body));

    var redirect_buf: [1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    var list = try std.ArrayList(u8).initCapacity(allocator, 0);
    defer list.deinit(allocator);

    var buf: [8192]u8 = undefined;
    const rdr = response.reader(&buf);
    while (true) {
        const slice: []u8 = buf[0..];
        var slices = [_][]u8{slice};
        const n = rdr.readVec(slices[0..]) catch |err| switch (err) {
            error.ReadFailed => return error.NetworkError,
            error.EndOfStream => 0,
        };
        if (n == 0) break;
        try list.appendSlice(allocator, buf[0..n]);
    }

    return .{ .status = @intFromEnum(response.head.status), .body = try list.toOwnedSlice(allocator) };
}
