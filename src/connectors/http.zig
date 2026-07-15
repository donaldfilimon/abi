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

/// Live connector base URLs must use HTTPS so API keys are never sent over
/// cleartext HTTP (TM-007). Loopback URLs (http://127.0.0.1 / http://localhost)
/// are exempted so local integration tests and loopback services work without TLS.
pub fn requireHttpsBaseUrl(base_url: []const u8) ConnectorError!void {
    const https_prefix = "https://";
    if (base_url.len >= https_prefix.len and std.ascii.eqlIgnoreCase(base_url[0..https_prefix.len], https_prefix)) return;
    const loopback_http = "http://127.0.0.1";
    if (base_url.len >= loopback_http.len and std.ascii.eqlIgnoreCase(base_url[0..loopback_http.len], loopback_http)) return;
    const localhost_http = "http://localhost";
    if (base_url.len >= localhost_http.len and std.ascii.eqlIgnoreCase(base_url[0..localhost_http.len], localhost_http)) return;
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

/// Callback for SSE streaming chunks. `delta` contains the token text (empty when done).
/// Returns error to abort streaming.
pub const StreamCallback = *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void;

/// A single token delta from an SSE streaming response.
pub const StreamChunk = struct {
    delta: []const u8,
    done: bool,
};

/// Streaming SSE response handler. Parses Server-Sent Events from an OpenAI-compatible
/// streaming response and invokes the callback for each token delta.
/// The callback receives `done=true` with empty `delta` when the stream ends.
pub fn httpPostJsonStreaming(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    // We need to read the response body incrementally for SSE parsing
    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .extra_headers = &[_]std.http.Header{
            .{ .name = "Accept", .value = "text/event-stream" },
            .{ .name = "Cache-Control", .value = "no-cache" },
        },
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = true,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);

    // Parse SSE stream from response body; returned slice is the full
    // concatenated token text (owned by caller).
    return try parseSseStream(allocator, response_writer.written(), on_chunk, callback_ctx);
}

/// Streaming SSE response handler with **true incremental HTTP read**.
/// Uses the lower-level `std.http.Client.request` API to parse SSE events
/// as they arrive over the network, without buffering the entire response.
/// The callback receives `done=true` with empty `delta` when the stream ends.
pub fn httpPostJsonStreamingIncremental(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    const uri = std.Uri.parse(url) catch return ConnectorError.ConnectionFailed;

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    // Create request using lower-level API for streaming access
    var req = (blk: {
        const r = std.http.Client.request(&client, .POST, uri, .{
            .headers = .{
                .content_type = .{ .override = "application/json" },
            },
            .extra_headers = &[_]std.http.Header{
                .{ .name = "Accept", .value = "text/event-stream" },
                .{ .name = "Cache-Control", .value = "no-cache" },
            },
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
    return try parseSseStreamIncremental(allocator, reader, on_chunk, callback_ctx);
}

fn parseSseStreamIncremental(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    var accumulated = std.ArrayListUnmanaged(u8).empty;
    errdefer accumulated.deinit(allocator);

    // Read in chunks and parse SSE events incrementally
    var read_buf: [4096]u8 = undefined;
    var line_buf = std.ArrayListUnmanaged(u8).empty;
    defer line_buf.deinit(allocator);
    var data_buffer = std.ArrayListUnmanaged(u8).empty;
    defer data_buffer.deinit(allocator);

    while (true) {
        var n: usize = 0;
        blk: {
            n = try std.Io.Reader.readSliceShort(reader, read_buf[0..]);
            if (n == 0) break :blk;
        }
        if (n == 0) break;

        var i: usize = 0;
        while (i < n) {
            const byte = read_buf[i];
            i += 1;

            if (byte == '\n') {
                // End of line - process it
                const line = std.mem.trim(u8, line_buf.items, "\r");
                if (line.len == 0) {
                    // Empty line = end of SSE event
                    if (data_buffer.items.len > 0) {
                        const event_data = try data_buffer.toOwnedSlice(allocator);
                        defer allocator.free(event_data);
                        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
                        data_buffer = std.ArrayListUnmanaged(u8).empty;
                    }
                } else if (std.mem.startsWith(u8, line, "data:")) {
                    const data = std.mem.trim(u8, line[5..], " \t");
                    // OpenAI SSE sentinel: data: [DONE]
                    if (std.mem.eql(u8, data, "[DONE]")) {
                        try on_chunk(callback_ctx, .{ .delta = "", .done = true });
                    } else {
                        try data_buffer.appendSlice(allocator, data);
                        try data_buffer.append(allocator, '\n');
                    }
                }
                // Ignore other SSE fields (event:, id:, retry:)
                line_buf = std.ArrayListUnmanaged(u8).empty;
            } else {
                // Accumulate line
                try line_buf.append(allocator, byte);
            }
        }
    }

    // Handle final event if no trailing newline
    if (data_buffer.items.len > 0) {
        const event_data = try data_buffer.toOwnedSlice(allocator);
        defer allocator.free(event_data);
        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
    }

    // Ensure a terminal done event even if the server omitted finish_reason/[DONE]
    try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    return try accumulated.toOwnedSlice(allocator);
}

fn parseSseStream(
    allocator: std.mem.Allocator,
    body: []const u8,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
) ConnectorError![]const u8 {
    var lines = std.mem.splitScalar(u8, body, '\n');
    var data_buffer = std.ArrayListUnmanaged(u8).empty;
    defer data_buffer.deinit(allocator);
    var accumulated = std.ArrayListUnmanaged(u8).empty;
    errdefer accumulated.deinit(allocator);

    while (lines.next()) |line| {
        // Trim trailing \r if present
        const trimmed = if (line.len > 0 and line[line.len - 1] == '\r') line[0 .. line.len - 1] else line;

        if (trimmed.len == 0) {
            // Empty line = end of event
            if (data_buffer.items.len > 0) {
                const event_data = try data_buffer.toOwnedSlice(allocator);
                defer allocator.free(event_data);
                try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
                data_buffer = std.ArrayListUnmanaged(u8).empty;
            }
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "data:")) {
            const data = std.mem.trim(u8, trimmed[5..], " \t");
            // OpenAI SSE sentinel: data: [DONE]
            if (std.mem.eql(u8, data, "[DONE]")) {
                try on_chunk(callback_ctx, .{ .delta = "", .done = true });
                continue;
            }
            try data_buffer.appendSlice(allocator, data);
            try data_buffer.append(allocator, '\n');
        }
        // Ignore other SSE fields (event:, id:, retry:)
    }

    // Handle final event if no trailing newline
    if (data_buffer.items.len > 0) {
        const event_data = try data_buffer.toOwnedSlice(allocator);
        defer allocator.free(event_data);
        try processSseEvent(event_data, allocator, on_chunk, callback_ctx, &accumulated);
    }

    // Ensure a terminal done event even if the server omitted finish_reason/[DONE]
    try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    return try accumulated.toOwnedSlice(allocator);
}

fn processSseEvent(
    event_data: []const u8,
    allocator: std.mem.Allocator,
    on_chunk: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void,
    callback_ctx: *anyopaque,
    accumulated: *std.ArrayListUnmanaged(u8),
) ConnectorError!void {
    // Skip blank event payloads
    const trimmed_event = std.mem.trim(u8, event_data, " \t\n\r");
    if (trimmed_event.len == 0) return;

    // Parse JSON from event data
    // Expected format: {"choices":[{"delta":{"content":"token"},"finish_reason":null}]}
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed_event, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();

    const root = parsed.value.object;
    const choices = root.get("choices") orelse return;
    if (choices.array.items.len == 0) return;

    const choice = choices.array.items[0].object;
    const delta = choice.get("delta") orelse return;
    const content = delta.object.get("content") orelse return;
    if (content != .string) return;
    const finish_reason = choice.get("finish_reason");

    const token = content.string;
    // finish_reason is null (JSON null) while streaming, becomes "stop"/"length" when done
    const done = if (finish_reason) |fr| fr != .null else false;

    if (token.len > 0) {
        try accumulated.appendSlice(allocator, token);
        try on_chunk(callback_ctx, .{ .delta = token, .done = false });
    }
    if (done) {
        try on_chunk(callback_ctx, .{ .delta = "", .done = true });
    }
}

test "parseSseEvent parses token delta" {
    const allocator = std.testing.allocator;
    var context = struct {
        called: bool = false,
        last_len: usize = 0,
        last_done: bool = false,
    }{};
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const self: *struct { called: bool, last_len: usize, last_done: bool } = @ptrCast(@alignCast(ctx));
            self.called = true;
            // delta is borrowed from the parse buffer — only length is durable here.
            self.last_len = chunk.delta.len;
            self.last_done = chunk.done;
        }
    }.call;

    try processSseEvent(
        \\{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
    , allocator, callback, &context, &accumulated);

    try std.testing.expect(context.called);
    try std.testing.expectEqual(@as(usize, 5), context.last_len);
    try std.testing.expect(!context.last_done);
    // Accumulator owns a durable copy of the token text.
    try std.testing.expectEqualStrings("Hello", accumulated.items);
}

test "parseSseEvent handles finish_reason" {
    const allocator = std.testing.allocator;
    var context = struct {
        called: bool = false,
        last_done: bool = false,
    }{};
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const self: *struct { called: bool, last_done: bool } = @ptrCast(@alignCast(ctx));
            self.called = true;
            self.last_done = chunk.done;
        }
    }.call;

    try processSseEvent(
        \\{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}
    , allocator, callback, &context, &accumulated);

    try std.testing.expect(context.called);
    try std.testing.expect(context.last_done);
}

test "parseSseStream accumulates multi-token SSE and forwards callback_ctx" {
    const allocator = std.testing.allocator;
    var tokens = std.ArrayListUnmanaged(u8).empty;
    defer tokens.deinit(allocator);

    const callback: *const fn (ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void = struct {
        fn call(ctx: *anyopaque, chunk: StreamChunk) ConnectorError!void {
            const acc: *std.ArrayListUnmanaged(u8) = @ptrCast(@alignCast(ctx));
            if (chunk.delta.len > 0) {
                acc.appendSlice(std.testing.allocator, chunk.delta) catch return ConnectorError.InvalidResponse;
            }
        }
    }.call;

    const body =
        \\data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}
        \\
        \\data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}
        \\
        \\data: [DONE]
        \\
    ;
    const full = try parseSseStream(allocator, body, callback, &tokens);
    defer allocator.free(full);
    try std.testing.expectEqualStrings("Hello", full);
    try std.testing.expectEqualStrings("Hello", tokens.items);
}

test {
    std.testing.refAllDecls(@This());
}

test "requireHttpsBaseUrl accepts https and rejects cleartext schemes" {
    try requireHttpsBaseUrl("https://api.openai.com");
    try requireHttpsBaseUrl("HTTPS://api.anthropic.com/v1");
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("http://api.openai.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("ftp://example.com"));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl(""));
    try std.testing.expectError(error.InsecureBaseUrl, requireHttpsBaseUrl("https:/bad"));
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
