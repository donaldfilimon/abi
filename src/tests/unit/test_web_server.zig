const std = @import("std");
const testing = std.testing;
const web_server = @import("web_server");

// Test web server functionality
test "Web server configuration" {
    // Test default configuration
    const config = web_server.WebConfig{};

    try testing.expectEqual(@as(u16, 3000), config.port);
    try testing.expectEqualStrings("127.0.0.1", config.host);
    try testing.expectEqual(@as(u32, 1000), config.max_connections);
    try testing.expect(config.enable_cors);
    try testing.expect(config.log_requests);
    try testing.expectEqual(@as(usize, 1024 * 1024), config.max_body_size);
    try testing.expectEqual(@as(u32, 30), config.timeout_seconds);
    try testing.expect(config.static_dir == null);
}

test "Web server configuration customization" {
    // Test custom configuration
    const config = web_server.WebConfig{
        .port = 8080,
        .host = "0.0.0.0",
        .max_connections = 500,
        .enable_cors = false,
        .log_requests = false,
        .max_body_size = 2 * 1024 * 1024,
        .timeout_seconds = 60,
        .static_dir = "./public",
    };

    try testing.expectEqual(@as(u16, 8080), config.port);
    try testing.expectEqualStrings("0.0.0.0", config.host);
    try testing.expectEqual(@as(u32, 500), config.max_connections);
    try testing.expect(!config.enable_cors);
    try testing.expect(!config.log_requests);
    try testing.expectEqual(@as(usize, 2 * 1024 * 1024), config.max_body_size);
    try testing.expectEqual(@as(u32, 60), config.timeout_seconds);
    try testing.expectEqualStrings("./public", config.static_dir.?);
}

test "Web server initialization" {
    const allocator = testing.allocator;
    const config = web_server.WebConfig{};

    // Test server creation
    const server = try web_server.WebServer.init(allocator, config);
    // Non-null pointer check
    try testing.expect(@intFromPtr(server) != 0);
    try testing.expectEqual(config.port, server.config.port);
    try testing.expectEqual(config.host, server.config.host);

    // Test server cleanup
    server.deinit();
}

test "Web server multiple configurations" {
    const allocator = testing.allocator;

    // Test development configuration
    const dev_config = web_server.WebConfig{
        .port = 3000,
        .host = "127.0.0.1",
        .max_connections = 100,
        .enable_cors = true,
        .log_requests = true,
    };

    const dev_server = try web_server.WebServer.init(allocator, dev_config);
    defer dev_server.deinit();

    // Test production configuration
    const prod_config = web_server.WebConfig{
        .port = 80,
        .host = "0.0.0.0",
        .max_connections = 10000,
        .enable_cors = false,
        .log_requests = false,
        .max_body_size = 10 * 1024 * 1024, // 10MB
        .timeout_seconds = 120,
    };

    const prod_server = try web_server.WebServer.init(allocator, prod_config);
    defer prod_server.deinit();

    // Verify configurations are independent
    try testing.expect(dev_server.config.port != prod_server.config.port);
    try testing.expect(!std.mem.eql(u8, dev_server.config.host, prod_server.config.host));
    try testing.expect(dev_server.config.max_connections != prod_server.config.max_connections);
}

test "Web server memory management" {
    const allocator = testing.allocator;
    const config = web_server.WebConfig{};

    // Test multiple server instances
    const server1 = try web_server.WebServer.init(allocator, config);
    const server2 = try web_server.WebServer.init(allocator, config);

    try testing.expect(server1 != server2);

    // Clean up
    server1.deinit();
    server2.deinit();
}

test "Web server configuration validation" {
    const allocator = testing.allocator;

    // Test valid configurations
    const valid_configs = [_]web_server.WebConfig{
        .{ .port = 80 }, // HTTP
        .{ .port = 443 }, // HTTPS
        .{ .port = 8080 }, // Common development port
        .{ .port = 3000 }, // Node.js default
        .{ .port = 8000 }, // Python default
    };

    for (valid_configs) |config| {
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(server.config.port == config.port);
        server.deinit();
    }
}

test "Web server host validation" {
    const allocator = testing.allocator;

    // Test different host configurations
    const host_configs = [_][]const u8{
        "127.0.0.1", // localhost
        "0.0.0.0", // all interfaces
        "192.168.1.1", // private IP
        "10.0.0.1", // private IP
        "172.16.0.1", // private IP
    };

    for (host_configs) |host| {
        const config = web_server.WebConfig{ .host = host };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expectEqualStrings(host, server.config.host);
        server.deinit();
    }
}

test "Web server timeout configuration" {
    const allocator = testing.allocator;

    // Test various timeout values
    const timeout_values = [_]u32{ 1, 5, 10, 30, 60, 120, 300, 600 };

    for (timeout_values) |timeout| {
        const config = web_server.WebConfig{ .timeout_seconds = timeout };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expectEqual(timeout, server.config.timeout_seconds);
        server.deinit();
    }
}

test "Web server body size limits" {
    const allocator = testing.allocator;

    // Test various body size limits
    const size_limits = [_]usize{
        1024, // 1KB
        1024 * 1024, // 1MB
        2 * 1024 * 1024, // 2MB
        10 * 1024 * 1024, // 10MB
        100 * 1024 * 1024, // 100MB
    };

    for (size_limits) |limit| {
        const config = web_server.WebConfig{ .max_body_size = limit };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expectEqual(limit, server.config.max_body_size);
        server.deinit();
    }
}

test "Web server feature flags" {
    const allocator = testing.allocator;

    // Test CORS settings
    {
        const config = web_server.WebConfig{ .enable_cors = true };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(server.config.enable_cors);
        server.deinit();
    }

    {
        const config = web_server.WebConfig{ .enable_cors = false };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(!server.config.enable_cors);
        server.deinit();
    }

    // Test logging settings
    {
        const config = web_server.WebConfig{ .log_requests = true };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(server.config.log_requests);
        server.deinit();
    }

    {
        const config = web_server.WebConfig{ .log_requests = false };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(!server.config.log_requests);
        server.deinit();
    }
}

test "Web server static directory configuration" {
    const allocator = testing.allocator;

    // Test with static directory
    const static_dirs = [_][]const u8{
        "./public",
        "./static",
        "./www",
        "/var/www/html",
        "C:\\inetpub\\wwwroot",
    };

    for (static_dirs) |dir| {
        const config = web_server.WebConfig{ .static_dir = dir };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expectEqualStrings(dir, server.config.static_dir.?);
        server.deinit();
    }

    // Test without static directory
    {
        const config = web_server.WebConfig{ .static_dir = null };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expect(server.config.static_dir == null);
        server.deinit();
    }
}

test "Web server connection limits" {
    const allocator = testing.allocator;

    // Test various connection limits
    const connection_limits = [_]u32{ 1, 10, 100, 1000, 10000, 100000 };

    for (connection_limits) |limit| {
        const config = web_server.WebConfig{ .max_connections = limit };
        const server = try web_server.WebServer.init(allocator, config);
        try testing.expectEqual(limit, server.config.max_connections);
        server.deinit();
    }
}

test "Web server configuration copying" {
    const allocator = testing.allocator;
    const original_config = web_server.WebConfig{
        .port = 8080,
        .host = "example.com",
        .max_connections = 500,
        .enable_cors = false,
        .log_requests = false,
        .max_body_size = 5 * 1024 * 1024,
        .timeout_seconds = 45,
        .static_dir = "./test_static",
    };

    // Create server with config
    const server = try web_server.WebServer.init(allocator, original_config);
    defer server.deinit();

    // Verify config is properly copied
    try testing.expectEqual(original_config.port, server.config.port);
    try testing.expectEqualStrings(original_config.host, server.config.host);
    try testing.expectEqual(original_config.max_connections, server.config.max_connections);
    try testing.expectEqual(original_config.enable_cors, server.config.enable_cors);
    try testing.expectEqual(original_config.log_requests, server.config.log_requests);
    try testing.expectEqual(original_config.max_body_size, server.config.max_body_size);
    try testing.expectEqual(original_config.timeout_seconds, server.config.timeout_seconds);
    try testing.expectEqualStrings(original_config.static_dir.?, server.config.static_dir.?);
}

test "Web server performance configuration" {
    const allocator = testing.allocator;

    // Test high-performance configuration
    const high_perf_config = web_server.WebConfig{
        .max_connections = 50000,
        .max_body_size = 50 * 1024 * 1024, // 50MB
        .timeout_seconds = 5, // Fast timeout
        .enable_cors = false, // Skip CORS for performance
        .log_requests = false, // Skip logging for performance
    };

    const server = try web_server.WebServer.init(allocator, high_perf_config);
    defer server.deinit();

    try testing.expectEqual(@as(u32, 50000), server.config.max_connections);
    try testing.expectEqual(@as(usize, 50 * 1024 * 1024), server.config.max_body_size);
    try testing.expectEqual(@as(u32, 5), server.config.timeout_seconds);
    try testing.expect(!server.config.enable_cors);
    try testing.expect(!server.config.log_requests);
}

test "web server test helper routes" {
    const alloc = testing.allocator;
    const server = try web_server.WebServer.init(alloc, .{});
    defer server.deinit();

    const body = try server.handlePathForTest("/health", alloc);
    defer alloc.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "healthy") != null);

    const body2 = try server.handlePathForTest("/api/status", alloc);
    defer alloc.free(body2);
    try testing.expect(std.mem.indexOf(u8, body2, "running") != null);
}

test "web server socket responds to /health" {
    const alloc = testing.allocator;
    var srv = try web_server.WebServer.init(alloc, .{ .port = 30807, .host = "127.0.0.1" });
    defer srv.deinit();

    var th = try std.Thread.spawn(.{}, web_server.WebServer.startOnce, .{srv});
    defer th.join();

    // Give the server a moment to start listening
    std.Thread.sleep(10 * std.time.ns_per_ms);

    const addr = try std.net.Address.parseIp("127.0.0.1", 30807);
    var conn = try std.net.tcpConnectToAddress(addr);
    defer conn.close();

    const req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";

    const builtin = @import("builtin");
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;

        // Send request using socket-specific send
        const sent: c_int = windows.ws2_32.send(conn.handle, @ptrCast(req.ptr), @intCast(req.len), 0);
        try testing.expect(sent != windows.ws2_32.SOCKET_ERROR);

        // Receive response using socket-specific recv
        var buf: [1024]u8 = undefined;
        const max_len: c_int = @intCast(@min(buf.len, @as(usize, @intCast(std.math.maxInt(c_int)))));
        const n_recv: c_int = windows.ws2_32.recv(conn.handle, @ptrCast(&buf[0]), max_len, 0);
        try testing.expect(n_recv != windows.ws2_32.SOCKET_ERROR);

        const body = buf[0..@intCast(n_recv)];
        try testing.expect(std.mem.indexOf(u8, body, "200") != null);
        try testing.expect(std.mem.indexOf(u8, body, "healthy") != null);
    } else {
        _ = try conn.writeAll(req);

        var buf: [1024]u8 = undefined;
        const n = try conn.read(&buf);
        const body = buf[0..n];
        try testing.expect(std.mem.indexOf(u8, body, "200") != null);
        try testing.expect(std.mem.indexOf(u8, body, "healthy") != null);
    }
}
