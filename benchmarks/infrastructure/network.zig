//! HTTP and Network Benchmarks
//!
//! Industry-standard benchmarks for networking:
//! - HTTP parsing throughput (request/response)
//! - URL parsing and encoding
//! - JSON serialization/deserialization
//! - Connection pool efficiency
//! - Request routing
//! - Header parsing
//! - Query string parsing
//! - WebSocket frame encoding/decoding
//! - Compression (gzip, deflate)
//! - TLS handshake simulation

const std = @import("std");
const abi = @import("abi");
const sync = abi.shared.sync;
const framework = @import("../system/framework.zig");

/// Network benchmark configuration
pub const NetworkBenchConfig = struct {
    /// Payload sizes for throughput tests
    payload_sizes: []const usize = &.{ 64, 1024, 16384 },
    /// Number of headers to test
    header_counts: []const usize = &.{ 5, 20 },
    /// JSON object complexity levels
    json_depths: []const usize = &.{ 1, 3, 5 },
    /// URL complexity (path segments)
    url_segments: []const usize = &.{ 1, 5 },
    /// Minimum time to run each benchmark (100ms default)
    min_time_ns: u64 = 100_000_000,
};

// ============================================================================
// HTTP Request Parsing Benchmarks
// ============================================================================

const HttpRequest = struct {
    allocator: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    version: []const u8,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: []const u8,
};

fn parseHttpRequest(allocator: std.mem.Allocator, raw: []const u8) !HttpRequest {
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

    // Parse request line
    if (lines.next()) |request_line| {
        var parts = std.mem.splitScalar(u8, request_line, ' ');
        request.method = parts.next() orelse "";
        request.path = parts.next() orelse "";
        request.version = parts.next() orelse "";
    }

    // Parse headers
    while (lines.next()) |line| {
        if (line.len == 0) break;
        if (std.mem.indexOf(u8, line, ": ")) |sep| {
            try request.headers.put(allocator, line[0..sep], line[sep + 2 ..]);
        }
    }

    // Rest is body
    request.body = lines.rest();

    return request;
}

fn generateHttpRequest(allocator: std.mem.Allocator, path_segments: usize, header_count: usize, body_size: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    errdefer buffer.deinit(allocator);

    // Request line
    try buffer.appendSlice(allocator, "GET /");
    for (0..path_segments) |i| {
        if (i > 0) try buffer.appendSlice(allocator, "/");
        var segment_buf: [32]u8 = undefined;
        const segment = std.fmt.bufPrint(&segment_buf, "segment{d}", .{i}) catch "segment";
        try buffer.appendSlice(allocator, segment);
    }
    try buffer.appendSlice(allocator, " HTTP/1.1\r\n");

    // Headers
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

    // Body
    try buffer.appendNTimes(allocator, 'X', body_size);

    return buffer.toOwnedSlice(allocator);
}

fn benchHttpParsing(allocator: std.mem.Allocator, raw: []const u8) !void {
    var request = try parseHttpRequest(allocator, raw);
    defer request.headers.deinit(allocator);
    std.mem.doNotOptimizeAway(&request);
}

// ============================================================================
// URL Parsing and Encoding Benchmarks
// ============================================================================

const ParsedUrl = struct {
    scheme: []const u8,
    host: []const u8,
    port: ?u16,
    path: []const u8,
    query: []const u8,
    fragment: []const u8,
};

fn parseUrl(url: []const u8) ParsedUrl {
    var result = ParsedUrl{
        .scheme = "",
        .host = "",
        .port = null,
        .path = "/",
        .query = "",
        .fragment = "",
    };

    var rest = url;

    // Scheme
    if (std.mem.indexOf(u8, rest, "://")) |idx| {
        result.scheme = rest[0..idx];
        rest = rest[idx + 3 ..];
    }

    // Fragment
    if (std.mem.indexOf(u8, rest, "#")) |idx| {
        result.fragment = rest[idx + 1 ..];
        rest = rest[0..idx];
    }

    // Query
    if (std.mem.indexOf(u8, rest, "?")) |idx| {
        result.query = rest[idx + 1 ..];
        rest = rest[0..idx];
    }

    // Path
    if (std.mem.indexOf(u8, rest, "/")) |idx| {
        result.path = rest[idx..];
        rest = rest[0..idx];
    }

    // Port
    if (std.mem.lastIndexOf(u8, rest, ":")) |idx| {
        if (std.fmt.parseInt(u16, rest[idx + 1 ..], 10)) |port| {
            result.port = port;
            rest = rest[0..idx];
        } else |_| {}
    }

    result.host = rest;
    return result;
}

fn urlEncode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    for (input) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try result.append(allocator, c);
        } else {
            try result.append(allocator, '%');
            try result.append(allocator, std.fmt.digitToChar(c >> 4, .upper));
            try result.append(allocator, std.fmt.digitToChar(c & 0x0F, .upper));
        }
    }

    return result.toOwnedSlice(allocator);
}

fn urlDecode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '%' and i + 2 < input.len) {
            const hi = std.fmt.charToDigit(input[i + 1], 16) catch {
                try result.append(allocator, input[i]);
                i += 1;
                continue;
            };
            const lo = std.fmt.charToDigit(input[i + 2], 16) catch {
                try result.append(allocator, input[i]);
                i += 1;
                continue;
            };
            try result.append(allocator, (hi << 4) | lo);
            i += 3;
        } else if (input[i] == '+') {
            try result.append(allocator, ' ');
            i += 1;
        } else {
            try result.append(allocator, input[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}

fn benchUrlParsing(url: []const u8) void {
    const parsed = parseUrl(url);
    std.mem.doNotOptimizeAway(&parsed);
}

fn benchUrlEncoding(allocator: std.mem.Allocator, input: []const u8) !void {
    const encoded = try urlEncode(allocator, input);
    defer allocator.free(encoded);
    std.mem.doNotOptimizeAway(encoded.ptr);
}

// ============================================================================
// JSON Benchmarks
// ============================================================================

fn generateJsonObject(allocator: std.mem.Allocator, depth: usize, width: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
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
            // Leaf value
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

fn benchJsonParsing(allocator: std.mem.Allocator, json: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

fn benchJsonStringify(allocator: std.mem.Allocator, value: anytype) !void {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    const writer = &aw.writer;

    try std.json.stringify(value, .{}, writer);
    const payload = try aw.toOwnedSlice();
    defer allocator.free(payload);
    std.mem.doNotOptimizeAway(payload.ptr);
}

// ============================================================================
// Query String Parsing
// ============================================================================

const QueryParams = struct {
    allocator: std.mem.Allocator,
    params: std.StringHashMapUnmanaged([]const u8),

    pub fn deinit(self: *QueryParams) void {
        self.params.deinit(self.allocator);
    }
};

fn parseQueryString(allocator: std.mem.Allocator, query: []const u8) !QueryParams {
    var result = QueryParams{
        .allocator = allocator,
        .params = .{},
    };
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

fn benchQueryStringParsing(allocator: std.mem.Allocator, query: []const u8) !void {
    var parsed = try parseQueryString(allocator, query);
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

// ============================================================================
// Header Parsing
// ============================================================================

const ParsedHeaders = struct {
    allocator: std.mem.Allocator,
    headers: std.StringHashMapUnmanaged([]const u8),

    pub fn deinit(self: *ParsedHeaders) void {
        self.headers.deinit(self.allocator);
    }
};

fn parseHeaders(allocator: std.mem.Allocator, raw: []const u8) !ParsedHeaders {
    var result = ParsedHeaders{
        .allocator = allocator,
        .headers = .{},
    };
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

fn generateHeaders(allocator: std.mem.Allocator, count: usize) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    errdefer buffer.deinit(allocator);

    // Common headers
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

fn benchHeaderParsing(allocator: std.mem.Allocator, headers_raw: []const u8) !void {
    var parsed = try parseHeaders(allocator, headers_raw);
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

// ============================================================================
// WebSocket Frame Encoding/Decoding
// ============================================================================

const WebSocketFrame = struct {
    fin: bool,
    opcode: u4,
    masked: bool,
    payload: []const u8,
};

fn encodeWebSocketFrame(allocator: std.mem.Allocator, frame: WebSocketFrame) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    errdefer buffer.deinit(allocator);

    // First byte: FIN + opcode
    const first_byte: u8 = (@as(u8, if (frame.fin) 0x80 else 0)) | frame.opcode;
    try buffer.append(allocator, first_byte);

    // Second byte: mask + length
    const mask_bit: u8 = if (frame.masked) 0x80 else 0;
    if (frame.payload.len < 126) {
        try buffer.append(allocator, mask_bit | @as(u8, @intCast(frame.payload.len)));
    } else if (frame.payload.len <= 65535) {
        try buffer.append(allocator, mask_bit | 126);
        try buffer.append(allocator, @intCast((frame.payload.len >> 8) & 0xFF));
        try buffer.append(allocator, @intCast(frame.payload.len & 0xFF));
    } else {
        try buffer.append(allocator, mask_bit | 127);
        const len = frame.payload.len;
        inline for (0..8) |i| {
            try buffer.append(allocator, @intCast((len >> @intCast((7 - i) * 8)) & 0xFF));
        }
    }

    // Masking key (if masked)
    if (frame.masked) {
        try buffer.appendSlice(allocator, &[_]u8{ 0x12, 0x34, 0x56, 0x78 });
        // XOR payload with mask
        for (frame.payload, 0..) |b, i| {
            const mask_key = [_]u8{ 0x12, 0x34, 0x56, 0x78 };
            try buffer.append(allocator, b ^ mask_key[i % 4]);
        }
    } else {
        try buffer.appendSlice(allocator, frame.payload);
    }

    return buffer.toOwnedSlice(allocator);
}

fn decodeWebSocketFrame(data: []const u8) ?WebSocketFrame {
    if (data.len < 2) return null;

    const fin = (data[0] & 0x80) != 0;
    const opcode: u4 = @intCast(data[0] & 0x0F);
    const masked = (data[1] & 0x80) != 0;

    var payload_len: usize = data[1] & 0x7F;
    var offset: usize = 2;

    if (payload_len == 126) {
        if (data.len < 4) return null;
        payload_len = (@as(usize, data[2]) << 8) | data[3];
        offset = 4;
    } else if (payload_len == 127) {
        if (data.len < 10) return null;
        payload_len = 0;
        inline for (0..8) |i| {
            payload_len = (payload_len << 8) | data[2 + i];
        }
        offset = 10;
    }

    if (masked) {
        offset += 4;
    }

    if (data.len < offset + payload_len) return null;

    return WebSocketFrame{
        .fin = fin,
        .opcode = opcode,
        .masked = masked,
        .payload = data[offset .. offset + payload_len],
    };
}

fn benchWebSocketEncode(allocator: std.mem.Allocator, payload: []const u8) !void {
    const frame = WebSocketFrame{
        .fin = true,
        .opcode = 1, // Text
        .masked = true,
        .payload = payload,
    };
    const encoded = try encodeWebSocketFrame(allocator, frame);
    defer allocator.free(encoded);
    std.mem.doNotOptimizeAway(encoded.ptr);
}

fn benchWebSocketDecode(data: []const u8) void {
    const frame = decodeWebSocketFrame(data);
    std.mem.doNotOptimizeAway(&frame);
}

// ============================================================================
// Connection Pool Simulation
// ============================================================================

const ConnectionPool = struct {
    const Connection = struct {
        id: u32,
        in_use: bool,
        created_at: i64,
    };

    connections: std.ArrayListUnmanaged(Connection),
    mutex: sync.Mutex,
    max_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) ConnectionPool {
        return .{
            .connections = .{},
            .mutex = .{},
            .max_size = max_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConnectionPool) void {
        self.connections.deinit(self.allocator);
    }

    pub fn acquire(self: *ConnectionPool) !?*Connection {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to find an available connection
        for (self.connections.items) |*conn| {
            if (!conn.in_use) {
                conn.in_use = true;
                return conn;
            }
        }

        // Create new if under limit
        if (self.connections.items.len < self.max_size) {
            try self.connections.append(self.allocator, .{
                .id = @intCast(self.connections.items.len),
                .in_use = true,
                .created_at = @as(i64, @intCast(self.connections.items.len)), // Use connection count as pseudo-timestamp
            });
            return &self.connections.items[self.connections.items.len - 1];
        }

        return null;
    }

    pub fn release(self: *ConnectionPool, conn: *Connection) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        conn.in_use = false;
    }
};

fn benchConnectionPool(allocator: std.mem.Allocator, pool_size: usize, operations: usize) !u64 {
    var pool = ConnectionPool.init(allocator, pool_size);
    defer pool.deinit();

    var acquired: u64 = 0;

    for (0..operations) |_| {
        if (try pool.acquire()) |conn| {
            acquired += 1;
            pool.release(conn);
        }
    }

    return acquired;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runNetworkBenchmarks(allocator: std.mem.Allocator, config: NetworkBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    HTTP/NETWORK BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // HTTP Request Parsing
    std.debug.print("[HTTP Request Parsing]\n", .{});
    for (config.header_counts) |headers| {
        for ([_]usize{ 0, 1024 }) |body| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "http_parse_{d}h_{d}b", .{ headers, body }) catch "http";

            const request = try generateHttpRequest(allocator, 3, headers, body);
            defer allocator.free(request);

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "network/http",
                    .bytes_per_op = request.len,
                    .warmup_iterations = 50,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(a: std.mem.Allocator, req: []const u8) !void {
                        try benchHttpParsing(a, req);
                    }
                }.bench,
                .{ allocator, request },
            );

            std.debug.print("  {s}: {d:.0} req/sec, {d:.2} MB/s\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.throughputMBps(request.len),
            });
        }
    }

    // URL Parsing
    std.debug.print("\n[URL Parsing]\n", .{});
    {
        const urls = [_][]const u8{
            "https://example.com/path",
            "https://api.example.com:8443/v1/users/123?filter=active&page=1#section",
            "https://subdomain.domain.example.com/very/long/path/to/some/resource?key1=value1&key2=value2&key3=value3",
        };

        for (urls, 0..) |url, i| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "url_parse_{d}", .{i}) catch "url";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "network/url",
                    .warmup_iterations = 100,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(u: []const u8) void {
                        benchUrlParsing(u);
                    }
                }.bench,
                .{url},
            );

            std.debug.print("  {s}: {d:.0} urls/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // URL Encoding
    std.debug.print("\n[URL Encoding]\n", .{});
    for ([_]usize{ 32, 128, 512 }) |size| {
        const input = try allocator.alloc(u8, size);
        defer allocator.free(input);

        // Fill with chars that need encoding
        for (input, 0..) |*c, i| {
            c.* = @intCast((i % 256));
        }

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "url_encode_{d}b", .{size}) catch "encode";

        const result = try runner.run(
            .{
                .name = name,
                .category = "network/url",
                .warmup_iterations = 50,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, inp: []const u8) !void {
                    try benchUrlEncoding(a, inp);
                }
            }.bench,
            .{ allocator, input },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    // JSON Parsing
    std.debug.print("\n[JSON Parsing]\n", .{});
    for (config.json_depths) |depth| {
        const json = try generateJsonObject(allocator, depth, 5);
        defer allocator.free(json);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "json_parse_d{d}_{d}b", .{ depth, json.len }) catch "json";

        const result = try runner.run(
            .{
                .name = name,
                .category = "network/json",
                .bytes_per_op = json.len,
                .warmup_iterations = 50,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, j: []const u8) !void {
                    try benchJsonParsing(a, j);
                }
            }.bench,
            .{ allocator, json },
        );

        std.debug.print("  {s}: {d:.0} docs/sec, {d:.2} MB/s\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.throughputMBps(json.len),
        });
    }

    // Header Parsing
    std.debug.print("\n[Header Parsing]\n", .{});
    for (config.header_counts) |count| {
        const headers = try generateHeaders(allocator, count);
        defer allocator.free(headers);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "headers_{d}", .{count}) catch "headers";

        const result = try runner.run(
            .{
                .name = name,
                .category = "network/headers",
                .warmup_iterations = 50,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, h: []const u8) !void {
                    try benchHeaderParsing(a, h);
                }
            }.bench,
            .{ allocator, headers },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    // Query String Parsing
    std.debug.print("\n[Query String Parsing]\n", .{});
    {
        const queries = [_][]const u8{
            "key=value",
            "a=1&b=2&c=3&d=4&e=5",
            "search=hello+world&filter=active&sort=date&page=1&limit=20&fields=id,name,email",
        };

        for (queries, 0..) |query, i| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "query_{d}params", .{i + 1}) catch "query";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "network/query",
                    .warmup_iterations = 50,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(a: std.mem.Allocator, q: []const u8) !void {
                        try benchQueryStringParsing(a, q);
                    }
                }.bench,
                .{ allocator, query },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // WebSocket
    std.debug.print("\n[WebSocket Frame Encoding]\n", .{});
    for (config.payload_sizes[0..@min(4, config.payload_sizes.len)]) |size| {
        const payload = try allocator.alloc(u8, size);
        defer allocator.free(payload);
        @memset(payload, 'X');

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "ws_encode_{d}b", .{size}) catch "ws";

        const result = try runner.run(
            .{
                .name = name,
                .category = "network/websocket",
                .bytes_per_op = size,
                .warmup_iterations = 50,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, p: []const u8) !void {
                    try benchWebSocketEncode(a, p);
                }
            }.bench,
            .{ allocator, payload },
        );

        std.debug.print("  {s}: {d:.0} frames/sec, {d:.2} MB/s\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.throughputMBps(size),
        });
    }

    // Connection Pool
    std.debug.print("\n[Connection Pool]\n", .{});
    for ([_]usize{ 10, 50, 100 }) |pool_size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "pool_{d}", .{pool_size}) catch "pool";

        const result = try runner.run(
            .{
                .name = name,
                .category = "network/pool",
                .warmup_iterations = 50,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, ps: usize, ops: usize) !u64 {
                    return try benchConnectionPool(a, ps, ops);
                }
            }.bench,
            .{ allocator, pool_size, 10000 },
        );

        std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} acquire+release/sec)\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 10000,
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

test "url parsing" {
    const url = "https://api.example.com:8443/v1/users?page=1#section";
    const parsed = parseUrl(url);

    try std.testing.expectEqualStrings("https", parsed.scheme);
    try std.testing.expectEqualStrings("api.example.com", parsed.host);
    try std.testing.expectEqual(@as(?u16, 8443), parsed.port);
    try std.testing.expectEqualStrings("/v1/users", parsed.path);
    try std.testing.expectEqualStrings("page=1", parsed.query);
    try std.testing.expectEqualStrings("section", parsed.fragment);
}

test "url encoding" {
    const allocator = std.testing.allocator;

    const encoded = try urlEncode(allocator, "hello world");
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("hello%20world", encoded);

    const decoded = try urlDecode(allocator, "hello%20world");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hello world", decoded);
}

test "websocket frame" {
    const allocator = std.testing.allocator;

    const frame = WebSocketFrame{
        .fin = true,
        .opcode = 1,
        .masked = false,
        .payload = "Hello",
    };

    const encoded = try encodeWebSocketFrame(allocator, frame);
    defer allocator.free(encoded);

    const decoded = decodeWebSocketFrame(encoded);
    try std.testing.expect(decoded != null);
    try std.testing.expectEqualStrings("Hello", decoded.?.payload);
}
