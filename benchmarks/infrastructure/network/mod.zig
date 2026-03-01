//! HTTP and Network Benchmarks
//!
//! Industry-standard benchmarks for networking:
//! - HTTP parsing throughput (request/response)
//! - URL parsing and encoding
//! - JSON serialization/deserialization
//! - Connection pool efficiency
//! - Header parsing, query string parsing
//! - WebSocket frame encoding/decoding
//!
//! Split into sub-modules for navigability:
//! - `http`: HTTP request/header/query/JSON parsing
//! - `url`: URL parsing, encoding, decoding
//! - `websocket`: WebSocket frame encode/decode
//! - `pool`: Connection pool simulation

const std = @import("std");
const framework = @import("../../system/framework.zig");

pub const http = @import("http.zig");
pub const url = @import("url.zig");
pub const websocket = @import("websocket.zig");
pub const pool = @import("pool.zig");

/// Network benchmark configuration
pub const NetworkBenchConfig = struct {
    payload_sizes: []const usize = &.{ 64, 1024, 16384 },
    header_counts: []const usize = &.{ 5, 20 },
    json_depths: []const usize = &.{ 1, 3, 5 },
    url_segments: []const usize = &.{ 1, 5 },
    min_time_ns: u64 = 100_000_000,
};

/// Run all network benchmarks (called as `run` by infrastructure/mod.zig)
pub fn run(allocator: std.mem.Allocator) !void {
    try runNetworkBenchmarks(allocator, .{});
}

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
            const request = try http.generateHttpRequest(allocator, 3, headers, body);
            defer allocator.free(request);

            const result = try runner.run(
                .{ .name = name, .category = "network/http", .bytes_per_op = request.len, .warmup_iterations = 50, .min_time_ns = 100_000_000 },
                struct {
                    fn bench(a: std.mem.Allocator, req: []const u8) !void {
                        try http.benchHttpParsing(a, req);
                    }
                }.bench,
                .{ allocator, request },
            );
            std.debug.print("  {s}: {d:.0} req/sec, {d:.2} MB/s\n", .{ name, result.stats.opsPerSecond(), result.stats.throughputMBps(request.len) });
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
        for (urls, 0..) |u, i| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "url_parse_{d}", .{i}) catch "url";
            const result = try runner.run(
                .{ .name = name, .category = "network/url", .warmup_iterations = 100, .min_time_ns = 100_000_000 },
                struct {
                    fn bench(uu: []const u8) void {
                        url.benchUrlParsing(uu);
                    }
                }.bench,
                .{u},
            );
            std.debug.print("  {s}: {d:.0} urls/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // URL Encoding
    std.debug.print("\n[URL Encoding]\n", .{});
    for ([_]usize{ 32, 128, 512 }) |size| {
        const input = try allocator.alloc(u8, size);
        defer allocator.free(input);
        for (input, 0..) |*c, i| c.* = @intCast((i % 256));

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "url_encode_{d}b", .{size}) catch "encode";
        const result = try runner.run(
            .{ .name = name, .category = "network/url", .warmup_iterations = 50, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, inp: []const u8) !void {
                    try url.benchUrlEncoding(a, inp);
                }
            }.bench,
            .{ allocator, input },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    // JSON Parsing
    std.debug.print("\n[JSON Parsing]\n", .{});
    for (config.json_depths) |depth| {
        const json = try http.generateJsonObject(allocator, depth, 5);
        defer allocator.free(json);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "json_parse_d{d}_{d}b", .{ depth, json.len }) catch "json";
        const result = try runner.run(
            .{ .name = name, .category = "network/json", .bytes_per_op = json.len, .warmup_iterations = 50, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, j: []const u8) !void {
                    try http.benchJsonParsing(a, j);
                }
            }.bench,
            .{ allocator, json },
        );
        std.debug.print("  {s}: {d:.0} docs/sec, {d:.2} MB/s\n", .{ name, result.stats.opsPerSecond(), result.stats.throughputMBps(json.len) });
    }

    // Header Parsing
    std.debug.print("\n[Header Parsing]\n", .{});
    for (config.header_counts) |count| {
        const headers = try http.generateHeaders(allocator, count);
        defer allocator.free(headers);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "headers_{d}", .{count}) catch "headers";
        const result = try runner.run(
            .{ .name = name, .category = "network/headers", .warmup_iterations = 50, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, h: []const u8) !void {
                    try http.benchHeaderParsing(a, h);
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
                .{ .name = name, .category = "network/query", .warmup_iterations = 50, .min_time_ns = 100_000_000 },
                struct {
                    fn bench(a: std.mem.Allocator, q: []const u8) !void {
                        try http.benchQueryStringParsing(a, q);
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
            .{ .name = name, .category = "network/websocket", .bytes_per_op = size, .warmup_iterations = 50, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, p: []const u8) !void {
                    try websocket.benchWebSocketEncode(a, p);
                }
            }.bench,
            .{ allocator, payload },
        );
        std.debug.print("  {s}: {d:.0} frames/sec, {d:.2} MB/s\n", .{ name, result.stats.opsPerSecond(), result.stats.throughputMBps(size) });
    }

    // Connection Pool
    std.debug.print("\n[Connection Pool]\n", .{});
    for ([_]usize{ 10, 50, 100 }) |pool_size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "pool_{d}", .{pool_size}) catch "pool";
        const result = try runner.run(
            .{ .name = name, .category = "network/pool", .warmup_iterations = 50, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, ps: usize, ops: usize) !u64 {
                    return try pool.benchConnectionPool(a, ps, ops);
                }
            }.bench,
            .{ allocator, pool_size, 10000 },
        );
        std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} acquire+release/sec)\n", .{ name, result.stats.opsPerSecond(), result.stats.opsPerSecond() * 10000 });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

test {
    _ = http;
    _ = url;
    _ = websocket;
    _ = pool;
}
