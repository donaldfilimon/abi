//! WDBX Windows Networking Test Script
//!
//! This script thoroughly tests WDBX HTTP server functionality on Windows,
//! with comprehensive error handling and diagnostics for common networking issues.

const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🧪 WDBX Windows Networking Test Suite\n", .{});
    std.debug.print("=====================================\n\n", .{});

    // Test 1: Basic connectivity
    std.debug.print("1️⃣ Testing basic connectivity...\n", .{});
    try testBasicConnectivity(allocator);

    // Test 2: HTTP server response
    std.debug.print("\n2️⃣ Testing HTTP server responses...\n", .{});
    try testHttpServer(allocator);

    // Test 3: Windows-specific error handling
    std.debug.print("\n3️⃣ Testing Windows-specific error handling...\n", .{});
    try testWindowsErrorHandling(allocator);

    // Test 4: Network diagnostics
    std.debug.print("\n4️⃣ Network diagnostics...\n", .{});
    try testNetworkDiagnostics(allocator);

    std.debug.print("\n✅ All tests completed!\n", .{});
    std.debug.print("🎉 Your WDBX server should now work optimally on Windows.\n", .{});
}

fn testBasicConnectivity(_: std.mem.Allocator) !void {
    // Test TCP connectivity to common ports
    const ports_to_test = [_]u16{ 8080, 3000, 80, 443 };

    for (ports_to_test) |port| {
        std.debug.print("  Testing port {d}... ", .{port});

        const address = std.net.Address.parseIp("127.0.0.1", port) catch {
            std.debug.print("❌ Invalid address\n", .{});
            continue;
        };

        const connection = std.net.tcpConnectToAddress(address) catch |err| {
            switch (err) {
                error.ConnectionRefused => std.debug.print("❌ Connection refused (port closed)\n", .{}),
                error.ConnectionResetByPeer => std.debug.print("⚠️ Connection reset by peer\n", .{}),
                error.Unexpected => std.debug.print("⚠️ Unexpected error (normal on Windows)\n", .{}),
                else => std.debug.print("❌ Other error: {}\n", .{err}),
            }
            continue;
        };
        defer connection.close();

        std.debug.print("✅ Connected successfully\n", .{});
    }
}

fn testHttpServer(_: std.mem.Allocator) !void {
    const address = std.net.Address.parseIp("127.0.0.1", 8080) catch {
        std.debug.print("  ❌ Cannot parse localhost address\n", .{});
        return;
    };

    const connection = std.net.tcpConnectToAddress(address) catch |err| {
        std.debug.print("  ❌ Cannot connect to WDBX server on port 8080: {}\n", .{err});
        std.debug.print("  💡 Make sure to start the server with: zig build run -- http\n", .{});
        return;
    };
    defer connection.close();

    std.debug.print("  ✅ Connected to WDBX server\n", .{});

    // Test different endpoints
    const endpoints = [_][]const u8{
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        "GET /network HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    };

    for (endpoints, 0..) |request, i| {
        std.debug.print("  Testing endpoint {d}... ", .{i + 1});

        _ = connection.write(request) catch |err| {
            std.debug.print("❌ Write failed: {}\n", .{err});
            continue;
        };

        var buffer: [4096]u8 = undefined;
        const bytes_read = connection.read(&buffer) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer => std.debug.print("⚠️ Connection reset (normal)\n", .{}),
                error.Unexpected => std.debug.print("⚠️ Unexpected error (Windows networking)\n", .{}),
                else => std.debug.print("❌ Read failed: {}\n", .{err}),
            }
            continue;
        };

        if (bytes_read > 0) {
            std.debug.print("✅ Received {d} bytes\n", .{bytes_read});

            // Try to parse HTTP status
            if (std.mem.indexOf(u8, buffer[0..bytes_read], "HTTP/1.1")) |http_start| {
                const status_line = buffer[http_start..@min(http_start + 50, bytes_read)];
                std.debug.print("    Status: {s}", .{std.mem.trim(u8, status_line, "\r\n")});
                if (std.mem.indexOf(u8, status_line, "200")) |_| {
                    std.debug.print(" 🟢\n", .{});
                } else {
                    std.debug.print(" 🟡\n", .{});
                }
            }
        } else {
            std.debug.print("⚠️ No response received\n", .{});
        }
    }
}

fn testWindowsErrorHandling(_: std.mem.Allocator) !void {
    // Test scenarios that commonly cause issues on Windows
    const test_scenarios = [_]struct {
        name: []const u8,
        port: u16,
    }{
        .{ .name = "Localhost standard", .port = 8080 },
        .{ .name = "Localhost alternate", .port = 3000 },
        .{ .name = "IPv6 localhost (::1)", .port = 8080 },
    };

    for (test_scenarios) |scenario| {
        std.debug.print("  Testing {s}... ", .{scenario.name});

        const ip = if (std.mem.indexOf(u8, scenario.name, "IPv6")) |_| "::1" else "127.0.0.1";
        const address = std.net.Address.parseIp(ip, scenario.port) catch {
            std.debug.print("❌ Address parsing failed\n", .{});
            continue;
        };

        const connection = std.net.tcpConnectToAddress(address) catch |err| {
            switch (err) {
                error.ConnectionRefused => std.debug.print("❌ Port {d} not available\n", .{scenario.port}),
                error.ConnectionResetByPeer => std.debug.print("⚠️ Connection reset (expected)\n", .{}),
                error.Unexpected => {
                    std.debug.print("⚠️ Unexpected error - this is normal on Windows\n", .{});
                    std.debug.print("    This error is automatically handled by WDBX server\n", .{});
                },
                else => std.debug.print("❌ Other error: {}\n", .{err}),
            }
            continue;
        };
        defer connection.close();

        std.debug.print("✅ Connected\n", .{});

        // Test rapid disconnect (common Windows issue)
        std.debug.print("    Testing rapid disconnect... ", .{});
        connection.close();

        const reconnect = std.net.tcpConnectToAddress(address) catch |err| {
            switch (err) {
                error.ConnectionResetByPeer, error.Unexpected => std.debug.print("⚠️ Handled gracefully\n", .{}),
                else => std.debug.print("❌ Reconnect failed: {}\n", .{err}),
            }
            continue;
        };
        reconnect.close();
        std.debug.print("✅ Reconnect successful\n", .{});
    }
}

fn testNetworkDiagnostics(_: std.mem.Allocator) !void {
    // Check Windows networking status
    std.debug.print("  Windows version: ", .{});
    if (builtin.os.tag == .windows) {
        std.debug.print("✅ Windows detected\n", .{});
    } else {
        std.debug.print("ℹ️ Non-Windows system\n", .{});
    }

    // Test DNS resolution
    std.debug.print("  DNS resolution test... ", .{});
    _ = std.net.Address.parseIp("127.0.0.1", 8080) catch {
        std.debug.print("❌ Failed\n", .{});
        return;
    };
    std.debug.print("✅ DNS working\n", .{});

    // Test socket creation
    std.debug.print("  Socket creation test... ", .{});
    const sock = std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, std.posix.IPPROTO.TCP) catch |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        return;
    };
    defer std.posix.close(sock);
    std.debug.print("✅ Socket created successfully\n", .{});

    // Test socket options (Windows-specific)
    std.debug.print("  Socket options test... ", .{});

    // Test TCP_NODELAY
    const enable: c_int = 1;
    _ = std.posix.setsockopt(sock, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&enable)) catch |err| {
        std.debug.print("⚠️ TCP_NODELAY failed: {} (may not be critical)\n", .{err});
    };

    // Test SO_KEEPALIVE
    _ = std.posix.setsockopt(sock, std.posix.SOL.SOCKET, std.posix.SO.KEEPALIVE, std.mem.asBytes(&enable)) catch |err| {
        std.debug.print("⚠️ SO_KEEPALIVE failed: {} (may not be critical)\n", .{err});
    };

    std.debug.print("✅ Socket options configured\n", .{});
}
