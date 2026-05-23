const std = @import("std");
const connector = @import("connector.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;

pub fn httpPostForm(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    extra_headers: []const std.http.Header,
) ConnectorError!Response {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/x-www-form-urlencoded" } },
        .extra_headers = extra_headers,
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = false,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);
    return .{
        .status = @intCast(@intFromEnum(result.status)),
        .body = try response_writer.toOwnedSlice(),
        .owned = true,
    };
}

pub fn httpPostJson(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    extra_headers: []const std.http.Header,
) ConnectorError!Response {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .extra_headers = extra_headers,
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = false,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);
    return .{
        .status = @intCast(@intFromEnum(result.status)),
        .body = try response_writer.toOwnedSlice(),
        .owned = true,
    };
}

fn mapHttpError(err: anyerror) ConnectorError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Timeout => error.Timeout,
        error.AuthenticationError => error.AuthenticationError,
        else => error.ConnectionFailed,
    };
}

fn mapHttpStatus(status: std.http.Status) ConnectorError!void {
    const code: u16 = @intCast(@intFromEnum(status));
    return switch (code) {
        200...299 => {},
        401, 403 => ConnectorError.AuthenticationError,
        408, 504 => ConnectorError.Timeout,
        429 => ConnectorError.RateLimited,
        else => ConnectorError.InvalidResponse,
    };
}

pub fn joinUrl(allocator: std.mem.Allocator, base_url: []const u8, path: []const u8) ConnectorError![]u8 {
    if (base_url.len == 0 or path.len == 0) return ConnectorError.ConnectionFailed;
    const base_has_slash = base_url[base_url.len - 1] == '/';
    const path_has_slash = path[0] == '/';
    if (base_has_slash and path_has_slash) return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path[1..] });
    if (!base_has_slash and !path_has_slash) return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_url, path });
    return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path });
}

pub fn bearerHeader(allocator: std.mem.Allocator, api_key: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
}

pub fn botHeader(allocator: std.mem.Allocator, token: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
}

pub fn basicAuthHeader(allocator: std.mem.Allocator, username: []const u8, password: []const u8) ConnectorError![]u8 {
    const combined = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ username, password });
    defer allocator.free(combined);

    var encoder = std.base64.standard.Encoder.init(.{});
    const encoded_len = encoder.calcLength(combined.len);
    const encoded = try allocator.alloc(u8, encoded_len);
    defer allocator.free(encoded);

    _ = encoder.encode(encoded, combined);
    return try std.fmt.allocPrint(allocator, "Basic {s}", .{encoded});
}
