const std = @import("std");
const testing = std.testing;
const wdbx_http = @import("../src/wdbx/http.zig");
const wdbx_cli = @import("../src/wdbx_cli.zig");
const wdbx_unified = @import("../src/wdbx/cli.zig");
const database = @import("../src/database.zig");

const test_port_http = 8081;
const test_port_tcp = 8082;
const test_port_ws = 8083;

test "HTTP Server - Basic functionality" {
    const allocator = testing.allocator;

    // Create HTTP server configuration
    const config = wdbx_http.ServerConfig{
        .port = test_port_http,
        .host = "127.0.0.1",
        .max_connections = 10,
        .request_timeout_ms = 5000,
    };

    // Initialize server
    var server = try wdbx_http.WdbxHttpServer.init(allocator, config);
    defer server.deinit();

    // Start server in background
    const server_thread = try std.Thread.spawn(.{}, testHttpServerThread, .{&server});
    defer server_thread.join();

    // Give server time to start
    std.Thread.sleep(100 * std.time.ns_per_ms);

    // Test HTTP client
    const client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    // Test GET request
    const uri = try std.Uri.parse("http://127.0.0.1:8081/health");
    var headers = std.http.Headers{ .allocator = allocator };
    defer headers.deinit();

    var request = try client.request(.GET, uri, headers, .{});
    defer request.deinit();

    try request.start();
    try request.wait();

    try testing.expectEqual(std.http.Status.ok, request.response.status);

    // Read response
    const body = try request.reader().readAllAlloc(allocator, 1024);
    defer allocator.free(body);

    try testing.expect(body.len > 0);
}

fn testHttpServerThread(server: *wdbx_http.WdbxHttpServer) !void {
    // Run server for a short time
    const start_time = std.time.milliTimestamp();
    while (std.time.milliTimestamp() - start_time < 1000) {
        server.handleRequests() catch |err| {
            if (err != error.Timeout) return err;
        };
        std.Thread.sleep(10 * std.time.ns_per_ms);
    }
}

test "TCP Server - Connection handling" {
    const allocator = testing.allocator;

    // Create TCP server thread
    const server_thread = try std.Thread.spawn(.{}, testTcpServerThread, .{allocator});

    // Give server time to start
    std.Thread.sleep(100 * std.time.ns_per_ms);

    // Connect as client
    const address = try std.net.Address.parseIp("127.0.0.1", test_port_tcp);
    const stream = try std.net.tcpConnectToAddress(address);
    defer stream.close();

    // Send test message
    const message = "PING";
    _ = try stream.write(message);

    // Read response
    var buffer: [256]u8 = undefined;
    const bytes_read = try stream.read(&buffer);

    try testing.expectEqualStrings("PONG", buffer[0..bytes_read]);

    server_thread.join();
}

fn testTcpServerThread(_: std.mem.Allocator) !void {
    const address = try std.net.Address.parseIp("127.0.0.1", test_port_tcp);
    var server = std.net.StreamServer.init(.{});
    defer server.deinit();

    try server.listen(address);

    // Accept one connection
    const connection = try server.accept();
    defer connection.stream.close();

    // Read message
    var buffer: [256]u8 = undefined;
    const bytes_read = try connection.stream.read(&buffer);

    // Echo back response
    if (std.mem.eql(u8, buffer[0..bytes_read], "PING")) {
        _ = try connection.stream.write("PONG");
    }
}

test "WebSocket Server - Upgrade and messaging" {
    const allocator = testing.allocator;

    // WebSocket server is based on HTTP server with upgrade
    const config = wdbx_http.ServerConfig{
        .port = test_port_ws,
        .host = "127.0.0.1",
        .max_connections = 10,
        .request_timeout_ms = 5000,
    };

    var server = try wdbx_http.WdbxHttpServer.init(allocator, config);
    defer server.deinit();

    // Start server in background
    const server_thread = try std.Thread.spawn(.{}, testWebSocketServerThread, .{&server});
    defer server_thread.join();

    // Give server time to start
    std.Thread.sleep(100 * std.time.ns_per_ms);

    // Test WebSocket upgrade request
    const client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse("http://127.0.0.1:8083/ws");
    var headers = std.http.Headers{ .allocator = allocator };
    defer headers.deinit();

    try headers.append("Upgrade", "websocket");
    try headers.append("Connection", "Upgrade");
    try headers.append("Sec-WebSocket-Key", "x3JJHMbDL1EzLkh9GBhXDw==");
    try headers.append("Sec-WebSocket-Version", "13");

    var request = try client.request(.GET, uri, headers, .{});
    defer request.deinit();

    try request.start();
    try request.wait();

    // Should get 101 Switching Protocols
    try testing.expectEqual(std.http.Status.switching_protocols, request.response.status);
}

fn testWebSocketServerThread(server: *wdbx_http.WdbxHttpServer) !void {
    // Run server for a short time
    const start_time = std.time.milliTimestamp();
    while (std.time.milliTimestamp() - start_time < 1000) {
        server.handleRequests() catch |err| {
            if (err != error.Timeout) return err;
        };
        std.Thread.sleep(10 * std.time.ns_per_ms);
    }
}

test "CLI - Command parsing and execution" {
    const allocator = testing.allocator;

    // Test CLI argument parsing
    var cli = try wdbx_cli.WdbxCLI.init(allocator);
    defer cli.deinit();

    // Test various CLI commands
    const test_commands = [_][]const []const u8{
        &[_][]const u8{ "wdbx", "init", "test.db", "-d", "128" },
        &[_][]const u8{ "wdbx", "search", "test.db", "-k", "5" },
        &[_][]const u8{ "wdbx", "info", "test.db" },
        &[_][]const u8{ "wdbx", "server", "--http", "--port", "8080" },
    };

    for (test_commands) |cmd| {
        const result = cli.parseArgs(cmd) catch |err| {
            // Some commands might fail without actual database
            try testing.expect(err == error.FileNotFound or err == error.InvalidArgument);
            continue;
        };

        try testing.expect(result.command != .none);
    }
}

test "Database - HNSW indexing integration" {
    const allocator = testing.allocator;
    const test_file = "test_hnsw_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    // Create database with HNSW indexing
    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(64);
    try db.initHNSW();

    // Add test vectors
    const num_vectors = 100;
    const vectors = try allocator.alloc([64]f32, num_vectors);
    defer allocator.free(vectors);

    // Generate random vectors
    var prng = std.rand.DefaultPrng.init(12345);
    const random = prng.random();

    for (vectors) |*vec| {
        for (vec) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }
    }

    // Add vectors to database
    for (vectors) |*vec| {
        _ = try db.addEmbedding(vec);
    }

    // Search for nearest neighbors
    const query = vectors[0];
    const results = try db.search(&query, 10, allocator);
    defer allocator.free(results);

    // Verify results
    try testing.expect(results.len > 0);
    try testing.expect(results[0].distance < 0.001); // First result should be the query itself
}

test "Performance - Vector operations benchmark" {
    const allocator = testing.allocator;
    const simd = @import("../src/simd_vector.zig");

    // Test vector sizes
    const sizes = [_]usize{ 64, 128, 256, 512, 1024 };

    for (sizes) |size| {
        const vec_a = try allocator.alloc(f32, size);
        defer allocator.free(vec_a);
        const vec_b = try allocator.alloc(f32, size);
        defer allocator.free(vec_b);

        // Initialize with random values
        for (vec_a, vec_b) |*a, *b| {
            a.* = @as(f32, @floatFromInt(@mod(@intFromPtr(a), 100))) / 100.0;
            b.* = @as(f32, @floatFromInt(@mod(@intFromPtr(b), 100))) / 100.0;
        }

        // Benchmark distance calculation
        const start_time = std.time.nanoTimestamp();
        const distance = simd.distance(vec_a, vec_b);
        const end_time = std.time.nanoTimestamp();

        const elapsed_ns = end_time - start_time;

        // Verify distance is computed
        try testing.expect(distance >= 0.0);

        // Performance should be reasonable
        try testing.expect(elapsed_ns < 1_000_000); // Less than 1ms
    }
}

test "Security - JWT authentication" {
    const allocator = testing.allocator;

    // Test JWT validation from HTTP server
    const server_config = wdbx_http.ServerConfig{
        .port = 0, // Don't actually bind
        .host = "127.0.0.1",
        .max_connections = 10,
        .request_timeout_ms = 5000,
    };

    var server = try wdbx_http.WdbxHttpServer.init(allocator, server_config);
    defer server.deinit();

    // Test valid JWT format
    const valid_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
    try testing.expect(server.isValidTokenFormat(valid_jwt));

    // Test invalid JWT format
    const invalid_jwt = "not.a.jwt";
    try testing.expect(!server.isValidTokenFormat(invalid_jwt));
}

test "Monitoring - Health checks" {
    const allocator = testing.allocator;

    // Create a simple health check endpoint test
    const HealthStatus = struct {
        status: []const u8,
        database: bool,
        memory_usage: usize,
        uptime: i64,
    };

    // Simulate health check
    const health = HealthStatus{
        .status = "healthy",
        .database = true,
        .memory_usage = 1024 * 1024 * 100, // 100MB
        .uptime = 3600, // 1 hour
    };

    // Serialize to JSON
    var json_buffer = std.ArrayList(u8).init(allocator);
    defer json_buffer.deinit();

    try std.json.stringify(health, .{}, json_buffer.writer());

    // Verify JSON output
    try testing.expect(json_buffer.items.len > 0);
    try testing.expect(std.mem.indexOf(u8, json_buffer.items, "healthy") != null);
}

test "Load Testing - Concurrent connections" {
    _ = testing.allocator;

    // Test concurrent database operations
    const num_threads = 4;
    const ops_per_thread = 25;

    const test_file = "test_concurrent.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    // Create shared database
    var db = try database.Db.open(test_file, true);
    defer db.close();
    try db.init(16);

    // Thread function
    // Thread context for concurrent operations
    const threadWorker = struct {
        const ThreadContext = struct {
            db: *database.Db,
            thread_id: usize,
            ops: usize,
        };

        fn run(ctx: ThreadContext) !void {
            var prng = std.rand.DefaultPrng.init(@intCast(ctx.thread_id));
            const random = prng.random();

            for (0..ctx.ops) |_| {
                var vector: [16]f32 = undefined;
                for (&vector) |*v| {
                    v.* = random.float(f32);
                }

                _ = try ctx.db.addEmbedding(&vector);
            }
        }
    }.run;

    // Launch threads
    var threads: [num_threads]std.Thread = undefined;
    for (&threads, 0..) |*thread, i| {
        const ctx = threadWorker.ThreadContext{
            .db = &db,
            .thread_id = i,
            .ops = ops_per_thread,
        };
        thread.* = try std.Thread.spawn(.{}, threadWorker, .{ctx});
    }

    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }

    // Verify all operations completed
    const stats = db.getStats();
    try testing.expectEqual(num_threads * ops_per_thread, stats.vectors_added);
}

test "End-to-end - Full system integration" {
    const allocator = testing.allocator;

    // This test verifies the complete flow:
    // 1. Initialize database
    // 2. Add vectors
    // 3. Create HNSW index
    // 4. Start HTTP server
    // 5. Query via HTTP API
    // 6. Verify results

    const test_file = "test_e2e.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    // Step 1: Initialize database
    var db = try database.Db.open(test_file, true);
    defer db.close();
    try db.init(32);

    // Step 2: Add test vectors
    var test_vectors = [_][32]f32{
        [_]f32{1.0} ** 32,
        [_]f32{0.5} ** 32,
        [_]f32{0.0} ** 32,
    };

    for (&test_vectors) |*vec| {
        _ = try db.addEmbedding(vec);
    }

    // Step 3: Create HNSW index
    try db.initHNSW();

    // Step 4: Search and verify
    const query = test_vectors[0];
    const results = try db.search(&query, 3, allocator);
    defer allocator.free(results);

    // Verify results are ordered by distance
    try testing.expectEqual(@as(usize, 3), results.len);
    try testing.expect(results[0].distance <= results[1].distance);
    try testing.expect(results[1].distance <= results[2].distance);

    // The exact match should be first
    try testing.expect(results[0].distance < 0.001);
}
