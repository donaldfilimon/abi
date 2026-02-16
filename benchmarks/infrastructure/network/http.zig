//! HTTP request parsing, header parsing, and query string parsing benchmarks.

const std = @import("std");

// ============================================================================
// HTTP Request Parsing
// ============================================================================

pub const HttpRequest = struct {
    allocator: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    version: []const u8,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: []const u8,
};

pub fn parseHttpRequest(allocator: std.mem.Allocator, raw: []const u8) !HttpRequest {
    var request = HttpRequest{
        .allocator = allocator,
        .method = "",
        .path = "",
        .version = "",
        .headers = .{},
        .body = "",
    };
    errdefer request.headers.deinit(allocator);

    var lines = std.mem.splitSequence(u8, raw, "\r\n");

    if (lines.next()) |request_line| {
        var parts = std.mem.splitScalar(u8, request_line, ' ');
        request.method = parts.next() orelse "";
        request.path = parts.next() orelse "";
        request.version = parts.next() orelse "";
    }

    while (lines.next()) |line| {
        if (line.len == 0) break;
        if (std.mem.indexOf(u8, line, ": ")) |sep| {
            try request.headers.put(allocator, line[0..sep], line[sep + 2 ..]);
        }
    }

    request.body = lines.rest();
    return request;
}

pub fn generateHttpRequest(allocator: std.mem.Allocator, path_segments: usize, header_count: usize, body_size: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, "GET /");
    for (0..path_segments) |i| {
        if (i > 0) try buffer.appendSlice(allocator, "/");
        var segment_buf: [32]u8 = undefined;
        const segment = std.fmt.bufPrint(&segment_buf, "segment{d}", .{i}) catch "segment";
        try buffer.appendSlice(allocator, segment);
    }
    try buffer.appendSlice(allocator, " HTTP/1.1\r\n");

    try buffer.appendSlice(allocator, "Host: example.com\r\n");
    try buffer.appendSlice(allocator, "User-Agent: benchmark/1.0\r\n");
    try buffer.appendSlice(allocator, "Accept: */*\r\n");

    for (0..header_count) |i| {
        var header_buf: [64]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf, "X-Custom-Header-{d}: value-{d}\r\n", .{ i, i }) catch "";
        try buffer.appendSlice(allocator, header);
    }

    var len_buf: [32]u8 = undefined;
    const content_len = std.fmt.bufPrint(&len_buf, "Content-Length: {d}\r\n", .{body_size}) catch "";
    try buffer.appendSlice(allocator, content_len);
    try buffer.appendSlice(allocator, "\r\n");
    try buffer.appendNTimes(allocator, 'X', body_size);

    return buffer.toOwnedSlice(allocator);
}

pub fn benchHttpParsing(allocator: std.mem.Allocator, raw: []const u8) !void {
    var request = try parseHttpRequest(allocator, raw);
    defer request.headers.deinit(allocator);
    std.mem.doNotOptimizeAway(&request);
}

// ============================================================================
// Header Parsing
// ============================================================================

pub const ParsedHeaders = struct {
    allocator: std.mem.Allocator,
    headers: std.StringHashMapUnmanaged([]const u8),

    pub fn deinit(self: *ParsedHeaders) void {
        self.headers.deinit(self.allocator);
    }
};

pub fn parseHeaders(allocator: std.mem.Allocator, raw: []const u8) !ParsedHeaders {
    var result = ParsedHeaders{ .allocator = allocator, .headers = .{} };
    errdefer result.headers.deinit(allocator);

    var lines = std.mem.splitSequence(u8, raw, "\r\n");
    while (lines.next()) |line| {
        if (line.len == 0) break;
        if (std.mem.indexOf(u8, line, ": ")) |sep| {
            try result.headers.put(allocator, line[0..sep], line[sep + 2 ..]);
        }
    }
    return result;
}

pub fn generateHeaders(allocator: std.mem.Allocator, count: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    try buffer.appendSlice(allocator, "Host: api.example.com\r\n");
    try buffer.appendSlice(allocator, "User-Agent: BenchmarkClient/1.0\r\n");
    try buffer.appendSlice(allocator, "Accept: application/json\r\n");
    try buffer.appendSlice(allocator, "Accept-Language: en-US,en;q=0.9\r\n");
    try buffer.appendSlice(allocator, "Accept-Encoding: gzip, deflate, br\r\n");
    try buffer.appendSlice(allocator, "Connection: keep-alive\r\n");
    try buffer.appendSlice(allocator, "Cache-Control: no-cache\r\n");

    for (0..count) |i| {
        var header_buf: [128]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf, "X-Custom-Header-{d}: custom-value-{d}-with-some-extra-content\r\n", .{ i, i }) catch "";
        try buffer.appendSlice(allocator, header);
    }

    try buffer.appendSlice(allocator, "\r\n");
    return buffer.toOwnedSlice(allocator);
}

pub fn benchHeaderParsing(allocator: std.mem.Allocator, headers_raw: []const u8) !void {
    var parsed = try parseHeaders(allocator, headers_raw);
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

// ============================================================================
// Query String Parsing
// ============================================================================

pub const QueryParams = struct {
    allocator: std.mem.Allocator,
    params: std.StringHashMapUnmanaged([]const u8),

    pub fn deinit(self: *QueryParams) void {
        self.params.deinit(self.allocator);
    }
};

pub fn parseQueryString(allocator: std.mem.Allocator, query: []const u8) !QueryParams {
    var result = QueryParams{ .allocator = allocator, .params = .{} };
    errdefer result.params.deinit(allocator);

    var pairs = std.mem.splitScalar(u8, query, '&');
    while (pairs.next()) |pair| {
        if (std.mem.indexOf(u8, pair, "=")) |sep| {
            try result.params.put(allocator, pair[0..sep], pair[sep + 1 ..]);
        } else {
            try result.params.put(allocator, pair, "");
        }
    }
    return result;
}

pub fn benchQueryStringParsing(allocator: std.mem.Allocator, query: []const u8) !void {
    var parsed = try parseQueryString(allocator, query);
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

// ============================================================================
// JSON Benchmarks
// ============================================================================

pub fn generateJsonObject(allocator: std.mem.Allocator, depth: usize, width: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);
    try writeJsonObject(allocator, &buffer, depth, width, 0);
    return buffer.toOwnedSlice(allocator);
}

fn writeJsonObject(allocator: std.mem.Allocator, buffer: *std.ArrayListUnmanaged(u8), depth: usize, width: usize, current: usize) !void {
    try buffer.appendSlice(allocator, "{");
    for (0..width) |i| {
        if (i > 0) try buffer.appendSlice(allocator, ",");
        var field_buf: [32]u8 = undefined;
        const field = std.fmt.bufPrint(&field_buf, "\"field{d}\":", .{i}) catch "";
        try buffer.appendSlice(allocator, field);

        if (current < depth) {
            try writeJsonObject(allocator, buffer, depth, width, current + 1);
        } else {
            var val_buf: [32]u8 = undefined;
            switch (i % 4) {
                0 => {
                    const val = std.fmt.bufPrint(&val_buf, "{d}", .{i * 100}) catch "0";
                    try buffer.appendSlice(allocator, val);
                },
                1 => {
                    const val = std.fmt.bufPrint(&val_buf, "\"{d}\"", .{i}) catch "\"\"";
                    try buffer.appendSlice(allocator, val);
                },
                2 => try buffer.appendSlice(allocator, "true"),
                3 => try buffer.appendSlice(allocator, "null"),
                else => unreachable,
            }
        }
    }
    try buffer.appendSlice(allocator, "}");
}

pub fn benchJsonParsing(allocator: std.mem.Allocator, json: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

pub fn benchJsonStringify(allocator: std.mem.Allocator, value: anytype) !void {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    const writer = &aw.writer;
    try std.json.stringify(value, .{}, writer);
    const payload = try aw.toOwnedSlice();
    defer allocator.free(payload);
    std.mem.doNotOptimizeAway(payload.ptr);
}
