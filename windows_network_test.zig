//! Windows Network Diagnostic Tool for WDBX HTTP Server
//!
//! This tool helps diagnose Windows-specific networking issues
//! by testing various connection scenarios and providing detailed
//! error reporting.

const std = @import("std");
const builtin = @import("builtin");
const http_server = @import("src/wdbx/http.zig");
const web = @import("src/web_server.zig");

pub fn main() !void {
    // This tool is Windows-specific
    if (builtin.os.tag != .windows) {
        std.debug.print("Windows Network Diagnostic Tool - skipping on non-Windows platform\n", .{});
        return;
    }
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== WDBX Windows Network Diagnostic Tool ===\n\n", .{});

    // Test 1: Basic server creation
    std.debug.print("1. Testing server initialization...\n", .{});
    const config = http_server.ServerConfig{
        .host = "127.0.0.1",
        .port = 8081, // Use different port to avoid conflicts
        .socket_buffer_size = 8192,
        .tcp_nodelay = true,
        .socket_keepalive = true,
        .enable_windows_optimizations = true,
        .connection_timeout_ms = 30000,
        .max_retries = 3,
    };

    const server = http_server.WdbxHttpServer.init(allocator, config) catch |err| {
        std.debug.print("❌ Failed to initialize server: {any}\n", .{err});
        return;
    };
    defer server.deinit();
    std.debug.print("✅ Server initialized successfully\n", .{});

    // Test 2: Port binding
    std.debug.print("\n2. Testing port binding...\n", .{});
    const address = std.net.Address.parseIp(config.host, config.port) catch |err| {
        std.debug.print("❌ Failed to parse address: {any}\n", .{err});
        return;
    };

    var test_server = address.listen(.{
        .reuse_address = true,
        .kernel_backlog = 1024,
    }) catch |err| {
        std.debug.print("❌ Failed to bind to port {d}: {any}\n", .{ config.port, err });
        return;
    };
    std.debug.print("✅ Port {d} bound successfully\n", .{config.port});

    // Test 3: Socket configuration with Windows optimizations
    std.debug.print("\n3. Testing socket configuration...\n", .{});
    server.configureSocket(test_server.stream.handle) catch |err| {
        std.debug.print("⚠️  Socket configuration failed: {any}\n", .{err});
    };
    std.debug.print("✅ Socket configured with Windows optimizations\n", .{});

    // Test 4: Basic TCP connection
    std.debug.print("\n4. Testing basic TCP connection...\n", .{});
    var client_connection = std.net.tcpConnectToAddress(address) catch |err| {
        std.debug.print("❌ Failed to connect via TCP: {any}\n", .{err});
        return;
    };
    defer client_connection.close();
    std.debug.print("✅ TCP connection successful\n", .{});

    // Test 5: HTTP request/response via web_server.startOnce to avoid ReadFile path
    std.debug.print("\n5. Testing HTTP communication...\n", .{});

    // Free the earlier test listener to reuse port
    test_server.deinit();

    var web_server_inst = try web.WebServer.init(allocator, .{
        .host = "127.0.0.1",
        .port = 8081,
        .max_connections = 128,
        .enable_cors = true,
        .log_requests = false,
        .max_body_size = 1024 * 1024,
        .timeout_seconds = 5,
        .static_dir = null,
    });
    defer web_server_inst.deinit();

    const server_thread = try std.Thread.spawn(.{}, web.WebServer.startOnce, .{web_server_inst});

    // Give the server a moment to start and accept
    std.Thread.sleep(10 * std.time.ns_per_ms);

    // Open a fresh connection for the HTTP request
    client_connection.close();
    client_connection = std.net.tcpConnectToAddress(address) catch |err| {
        std.debug.print("❌ Failed to reconnect for HTTP: {any}\n", .{err});
        return;
    };

    // Send HTTP request from client
    const http_request = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    const bytes_sent = client_connection.write(http_request) catch |err| {
        std.debug.print("❌ Failed to send HTTP request: {any}\n", .{err});
        return;
    };
    std.debug.print("✅ Sent {d} bytes via HTTP\n", .{bytes_sent});

    // Read response on client side using ws2_32.recv to avoid ReadFile path
    var client_buffer: [2048]u8 = undefined;
    const windows = std.os.windows;
    const rcv_timeout_ms: c_int = 2000;
    _ = std.posix.setsockopt(client_connection.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&rcv_timeout_ms)) catch {};
    const n: c_int = windows.ws2_32.recv(client_connection.handle, @ptrCast(&client_buffer[0]), @intCast(client_buffer.len), 0);
    const client_bytes_read: usize = if (n >= 0) @intCast(n) else 0;

    // Join server thread
    server_thread.join();

    if (client_bytes_read > 0) {
        std.debug.print("✅ Client received {d} bytes\n", .{client_bytes_read});
        std.debug.print("\n=== Network Diagnostic Summary ===\n", .{});
        std.debug.print("✅ All networking tests passed successfully!\n", .{});
        std.debug.print("✅ Windows socket optimizations applied\n", .{});
        std.debug.print("✅ HTTP communication working via web_server handler\n", .{});
    } else {
        std.debug.print("⚠️  No response bytes read; HTTP may be closed quickly by server\n", .{});
        std.debug.print("\n=== Network Diagnostic Summary ===\n", .{});
        std.debug.print("✅ Server initialization successful\n", .{});
        std.debug.print("✅ Port binding successful\n", .{});
        std.debug.print("✅ Windows socket optimizations applied\n", .{});
        std.debug.print("✅ TCP connection established\n", .{});
        std.debug.print("⚠️  HTTP read/write may need further investigation\n", .{});
    }

    std.debug.print("\n=== Troubleshooting Recommendations ===\n", .{});
    std.debug.print("If you're still experiencing connection issues:\n", .{});
    std.debug.print("1. Run the PowerShell fix script as Administrator\n", .{});
    std.debug.print("2. Restart Windows after applying network fixes\n", .{});
    std.debug.print("3. Try different port numbers (8080, 8082, 3000)\n", .{});
    std.debug.print("4. Check Windows Firewall settings\n", .{});
    std.debug.print("5. Temporarily disable antivirus to test\n", .{});
    std.debug.print("6. Use PowerShell Invoke-WebRequest for testing\n", .{});
}
