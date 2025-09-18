//! Demonstration of Enhanced HTTP Client Features
//!
//! This shows how to use the new HTTP client with:
//! - Automatic retry and exponential backoff
//! - Timeout configuration
//! - Windows-specific optimizations
//! - Proxy support (when configured)

const std = @import("std");

// Import our enhanced HTTP client
const http_client = @import("http_client");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Enhanced HTTP Client Demo ===\n\n", .{});

    // Create HTTP client with enhanced configuration
    var client = http_client.HttpClient.init(allocator, .{
        .connect_timeout_ms = 10000, // 10 second connection timeout
        .read_timeout_ms = 15000, // 15 second read timeout
        .max_retries = 3, // Retry up to 3 times
        .initial_backoff_ms = 1000, // Start with 1 second backoff
        .max_backoff_ms = 8000, // Max 8 second backoff
        .user_agent = "WDBX-Enhanced-Demo/1.0",
        .follow_redirects = true,
        .verify_ssl = true,
        .verbose = true, // Enable detailed logging
    });

    // Test 1: Local WDBX server health check (if running)
    std.debug.print("1. Testing local WDBX server health...\n", .{});
    testWdbxHealth(&client) catch |err| {
        std.debug.print("   ⚠️ WDBX server test failed: {any}\n", .{err});
        std.debug.print("   💡 Start the server with: zig build run -- http\n", .{});
    };

    // Test 2: External API test (with retries)
    std.debug.print("\n2. Testing external API with retry mechanism...\n", .{});
    testExternalApi(&client) catch |err| {
        std.debug.print("   ⚠️ External API test failed: {any}\n", .{err});
        std.debug.print("   💡 This might be due to network connectivity\n", .{});
    };

    // Test 3: Show proxy configuration
    std.debug.print("\n3. Proxy configuration demo...\n", .{});
    demoProxyConfig(allocator);

    std.debug.print("\n=== Demo Complete ===\n", .{});
    std.debug.print("💡 Key features demonstrated:\n", .{});
    std.debug.print("   - Automatic retry with exponential backoff\n", .{});
    std.debug.print("   - Configurable timeouts for connection and reading\n", .{});
    std.debug.print("   - Windows-specific socket optimizations\n", .{});
    std.debug.print("   - Proxy support configuration\n", .{});
    std.debug.print("   - Comprehensive error handling\n", .{});
}

fn testWdbxHealth(client: *http_client.HttpClient) !void {
    const start_time = std.time.milliTimestamp();

    const response = client.testConnectivity("http://127.0.0.1:8080/health") catch |err| {
        return err;
    };

    const elapsed = std.time.milliTimestamp() - start_time;

    if (response) {
        std.debug.print("   ✅ WDBX server is healthy (took {d}ms)\n", .{elapsed});

        // Try to get actual response data
        var health_response = client.get("http://127.0.0.1:8080/health") catch |err| {
            std.debug.print("   ⚠️ Could not get health details: {any}\n", .{err});
            return;
        };
        defer health_response.deinit();

        std.debug.print("   📊 Health status: HTTP {d}\n", .{health_response.status_code});
        const preview_len = @min(100, health_response.body.len);
        std.debug.print("   📄 Response preview: {s}...\n", .{health_response.body[0..preview_len]});
    } else {
        std.debug.print("   ❌ WDBX server connectivity failed (took {d}ms)\n", .{elapsed});
    }
}

fn testExternalApi(client: *http_client.HttpClient) !void {
    std.debug.print("   Testing with httpbin.org (may retry on failure)...\n", .{});
    const start_time = std.time.milliTimestamp();

    var response = client.get("http://httpbin.org/get") catch |err| {
        return err;
    };
    defer response.deinit();

    const elapsed = std.time.milliTimestamp() - start_time;
    std.debug.print("   ✅ External API successful (took {d}ms)\n", .{elapsed});
    std.debug.print("   📊 Status: {d}, Body size: {d} bytes\n", .{ response.status_code, response.body.len });

    // Check headers
    std.debug.print("   📋 Response headers: {d} total\n", .{response.headers.count()});
    var header_iter = response.headers.iterator();
    var count: u32 = 0;
    while (header_iter.next()) |header| {
        if (count < 3) { // Show first 3 headers
            std.debug.print("      {s}: {s}\n", .{ header.key_ptr.*, header.value_ptr.* });
        }
        count += 1;
    }
    if (count > 3) {
        std.debug.print("      ... and {d} more headers\n", .{count - 3});
    }
}

fn demoProxyConfig(allocator: std.mem.Allocator) void {
    std.debug.print("   Checking for proxy environment variables...\n", .{});

    const proxy_vars = [_][]const u8{ "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy" };
    var found_proxy = false;

    for (proxy_vars) |var_name| {
        if (std.process.getEnvVarOwned(allocator, var_name)) |proxy_url| {
            defer allocator.free(proxy_url);
            std.debug.print("   🌐 Found {s}: {s}\n", .{ var_name, proxy_url });
            found_proxy = true;
        } else |_| {
            // Environment variable not set
        }
    }

    if (!found_proxy) {
        std.debug.print("   ℹ️ No proxy environment variables detected\n", .{});
        std.debug.print("   💡 To test proxy support, set HTTP_PROXY=http://proxy.example.com:8080\n", .{});
    } else {
        std.debug.print("   ✨ Proxy configuration detected - HTTP client will use it automatically\n", .{});
    }

    std.debug.print("   📝 Proxy features supported:\n", .{});
    std.debug.print("      - HTTP and HTTPS proxy URLs\n", .{});
    std.debug.print("      - Automatic proxy detection from environment\n", .{});
    std.debug.print("      - Proxy authentication (when libcurl is available)\n", .{});
    std.debug.print("      - SSL tunnel through proxy\n", .{});
}
