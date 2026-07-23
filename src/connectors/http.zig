const std = @import("std");
const connector = @import("connector.zig");
const sse = @import("sse_stream.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;

pub const StreamCallback = sse.StreamCallback;
pub const StreamChunk = sse.StreamChunk;

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

pub fn httpGetJson(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
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
        .method = .GET,
        .headers = .{ .content_type = .{ .override = "application/json" } },
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
        error.ReadFailed, error.ConnectionRefused, error.ConnectionResetByPeer, error.HostUnreachable, error.NetworkUnreachable, error.UnknownHostName, error.WouldBlock, error.Canceled, error.Unexpected, error.SystemResources, error.AddressInUse, error.AddressUnavailable, error.AddressFamilyUnsupported, error.ProtocolUnsupportedBySystem, error.ProtocolUnsupportedByAddressFamily, error.SocketModeUnsupported, error.OptionUnsupported, error.ConnectionPending, error.AccessDenied, error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded, error.NetworkDown, error.NameServerFailure, error.InvalidDnsARecord, error.InvalidDnsAAAARecord, error.InvalidDnsCnameRecord, error.InvalidHostName, error.ResolvConfParseFailed, error.NoAddressReturned, error.DetectingNetworkConfigurationFailed, error.TlsInitializationFailed, error.UnsupportedUriScheme, error.UriMissingHost, error.CertificateBundleLoadFailure => error.ConnectionFailed,
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

/// True if `s` starts with `prefix` (case-insensitively) and the prefix ends
/// at a real host-name boundary — end of string, or a `:`/`/` separator.
/// A bare prefix match would let `http://127.0.0.1.evil.com` or
/// `http://127.0.0.1@evil.com` slip through as if they were loopback.
fn hasHostPrefix(s: []const u8, prefix: []const u8) bool {
    if (s.len < prefix.len or !std.ascii.eqlIgnoreCase(s[0..prefix.len], prefix)) return false;
    if (s.len == prefix.len) return true;
    return switch (s[prefix.len]) {
        ':', '/' => true,
        else => false,
    };
}

/// Live connector base URLs must use HTTPS so API keys are never sent over
/// cleartext HTTP (TM-007). Loopback URLs (http://127.0.0.1 / http://localhost)
/// are exempted so local integration tests and loopback services work without TLS.
pub fn requireHttpsBaseUrl(base_url: []const u8) ConnectorError!void {
    const https_prefix = "https://";
    if (base_url.len >= https_prefix.len and std.ascii.eqlIgnoreCase(base_url[0..https_prefix.len], https_prefix)) return;
    if (hasHostPrefix(base_url, "http://127.0.0.1")) return;
    if (hasHostPrefix(base_url, "http://localhost")) return;
    return ConnectorError.InsecureBaseUrl;
}

pub fn joinUrl(allocator: std.mem.Allocator, base_url: []const u8, path: []const u8) ConnectorError![]u8 {
    if (base_url.len == 0 or path.len == 0) return ConnectorError.ConnectionFailed;
    try requireHttpsBaseUrl(base_url);
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

    const encoder = std.base64.standard.Encoder;
    const encoded_len = encoder.calcSize(combined.len);
    const encoded = try allocator.alloc(u8, encoded_len);
    defer allocator.free(encoded);

    const encoded_slice = encoder.encode(encoded, combined);
    return try std.fmt.allocPrint(allocator, "Basic {s}", .{encoded_slice});
}

/// Streaming SSE response handler. Parses Server-Sent Events from an OpenAI- or
/// Anthropic-compatible streaming response and invokes the callback for each
/// token delta. The callback receives `done=true` with empty `delta` when the
/// stream ends. `extra_headers` may carry provider auth (e.g. `x-api-key`).
pub fn httpPostJsonStreaming(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
    extra_headers: []const std.http.Header,
) ConnectorError![]const u8 {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var headers_buf: [16]std.http.Header = undefined;
    var header_count: usize = 0;
    headers_buf[header_count] = .{ .name = "Accept", .value = "text/event-stream" };
    header_count += 1;
    headers_buf[header_count] = .{ .name = "Cache-Control", .value = "no-cache" };
    header_count += 1;
    for (extra_headers) |h| {
        if (header_count >= headers_buf.len) return ConnectorError.InvalidResponse;
        headers_buf[header_count] = h;
        header_count += 1;
    }

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .extra_headers = headers_buf[0..header_count],
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = true,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);

    return try sse.parseSseStream(allocator, response_writer.written(), on_chunk, callback_ctx);
}

/// Streaming SSE response handler with **true incremental HTTP read**.
/// Uses the lower-level `std.http.Client.request` API to parse SSE events
/// as they arrive over the network, without buffering the entire response.
/// The callback receives `done=true` with empty `delta` when the stream ends.
/// `extra_headers` may carry provider auth (e.g. Anthropic `x-api-key`).
pub fn httpPostJsonStreamingIncremental(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
    extra_headers: []const std.http.Header,
) ConnectorError![]const u8 {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    const uri = std.Uri.parse(url) catch return ConnectorError.ConnectionFailed;

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var headers_buf: [16]std.http.Header = undefined;
    var header_count: usize = 0;
    headers_buf[header_count] = .{ .name = "Accept", .value = "text/event-stream" };
    header_count += 1;
    headers_buf[header_count] = .{ .name = "Cache-Control", .value = "no-cache" };
    header_count += 1;
    for (extra_headers) |h| {
        if (header_count >= headers_buf.len) return ConnectorError.InvalidResponse;
        headers_buf[header_count] = h;
        header_count += 1;
    }

    // Create request using lower-level API for streaming access
    var req = (blk: {
        const r = std.http.Client.request(&client, .POST, uri, .{
            .headers = .{
                .content_type = .{ .override = "application/json" },
            },
            .extra_headers = headers_buf[0..header_count],
            .redirect_behavior = .unhandled,
            .keep_alive = true,
        }) catch |err| {
            break :blk mapHttpError(err);
        };
        break :blk r;
    }) catch |err| return err;
    defer req.deinit();

    // Send request body
    const mutable_body = @constCast(body);
    req.sendBodyComplete(mutable_body) catch |err| {
        return mapHttpError(err);
    };

    // Receive response headers
    var redirect_buffer: [8192]u8 = undefined;
    var response = req.receiveHead(&redirect_buffer) catch |err| {
        return mapHttpError(err);
    };
    try mapHttpStatus(response.head.status);

    // Get decompressing reader for incremental body reading
    var transfer_buffer: [64]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    var decompress_buffer: []u8 = &.{};
    if (response.head.content_encoding != .identity) {
        decompress_buffer = try allocator.alloc(u8, switch (response.head.content_encoding) {
            .zstd => std.compress.zstd.default_window_len,
            .deflate, .gzip => std.compress.flate.max_window_len,
            else => return ConnectorError.InvalidResponse,
        });
        defer allocator.free(decompress_buffer);
    }
    const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);

    // Parse SSE incrementally as data arrives
    return try sse.parseSseStreamIncremental(allocator, reader, on_chunk, callback_ctx);
}

test "requireHttpsBaseUrl accepts https and rejects cleartext schemes" {
    try requireHttpsBaseUrl("https://api.openai.com");
    try requireHttpsBaseUrl("HTTPS://api.anthropic.com/v1");
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://api.openai.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("ftp://example.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl(""));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("https:/bad"));
}

test "requireHttpsBaseUrl loopback exemption checks a real host boundary" {
    // Legitimate loopback forms must still pass.
    try requireHttpsBaseUrl("http://127.0.0.1");
    try requireHttpsBaseUrl("http://127.0.0.1:8080");
    try requireHttpsBaseUrl("http://127.0.0.1/v1");
    try requireHttpsBaseUrl("http://localhost");
    try requireHttpsBaseUrl("http://localhost:11434");
    try requireHttpsBaseUrl("HTTP://LOCALHOST/v1");

    // A bare prefix match must not exempt a spoofed non-loopback host — these
    // all start with the loopback prefix but resolve to an attacker host and
    // must be rejected as insecure, not silently treated as loopback.
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://127.0.0.1.evil.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://127.0.0.1@evil.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://127.0.0.10"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://localhost.evil.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://localhost@evil.com"));
}

test "joinUrl requires https and joins path segments" {
    const allocator = std.testing.allocator;
    const a = try joinUrl(allocator, "https://api.example.com", "/v1/chat");
    defer allocator.free(a);
    try std.testing.expectEqualStrings("https://api.example.com/v1/chat", a);

    const b = try joinUrl(allocator, "https://api.example.com/", "v1/chat");
    defer allocator.free(b);
    try std.testing.expectEqualStrings("https://api.example.com/v1/chat", b);

    try std.testing.expectError(error.InsecureBaseUrl, joinUrl(allocator, "http://api.example.com", "/v1"));
}

test {
    std.testing.refAllDecls(@This());
}
