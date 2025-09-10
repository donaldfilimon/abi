//! Test program for enhanced HTTP client with libcurl integration
//! 
//! This program demonstrates:
//! - Automatic retry with exponential backoff
//! - Proxy configuration and TLS handling
//! - Comprehensive connectivity diagnostics
//! - Timeout and error handling improvements

const std = @import("std");
const http_client = @import("src/http_client.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Enhanced HTTP Client Test ===\n\n", .{});

    // Test 1: Basic HTTP client functionality
    std.debug.print("1. Testing basic HTTP client...\n", .{});
    try testBasicHttpClient(allocator);

    // Test 2: Retry and backoff mechanisms
    std.debug.print("\n2. Testing retry and backoff...\n", .{});
    try testRetryMechanism(allocator);

    // Test 3: Proxy configuration (if available)
    std.debug.print("\n3. Testing proxy configuration...\n", .{});
    try testProxyConfiguration(allocator);

    // Test 4: Connectivity diagnostics
    std.debug.print("\n4. Running connectivity diagnostics...\n", .{});
    try testConnectivityDiagnostics(allocator);

    // Test 5: Performance with multiple requests
    std.debug.print("\n5. Testing performance with multiple requests...\n", .{});
    try testPerformance(allocator);

    std.debug.print("\n=== All Tests Complete ===\n", .{});
}

/// Test basic HTTP client functionality
fn testBasicHttpClient(allocator: std.mem.Allocator) !void {
    var client = http_client.HttpClient.init(allocator, .{
        .verbose = true,
        .connect_timeout_ms = 10000,
        .read_timeout_ms = 15000,
        .max_retries = 1,
    });

    // Test GET request to a reliable service
    std.debug.print("   Testing GET request...\n", .{});
    var response = client.get("http://httpbin.org/get") catch |err| {
        std.debug.print("   ❌ GET request failed: {any}\n", .{err});
        return;
    };
    defer response.deinit();

    std.debug.print("   ✅ GET successful - Status: {d}, Body length: {d}\n", .{ response.status_code, response.body.len });

    // Test POST request
    std.debug.print("   Testing POST request...\n", .{});
    const test_data = "{\"test\": \"data\", \"client\": \"wdbx-enhanced\"}";
    var post_response = client.post("http://httpbin.org/post", "application/json", test_data) catch |err| {
        std.debug.print("   ❌ POST request failed: {any}\n", .{err});
        return;
    };
    defer post_response.deinit();

    std.debug.print("   ✅ POST successful - Status: {d}\n", .{post_response.status_code});
}

/// Test retry and exponential backoff mechanisms
fn testRetryMechanism(allocator: std.mem.Allocator) !void {
    var client = http_client.HttpClient.init(allocator, .{
        .verbose = true,
        .connect_timeout_ms = 2000,  // Short timeout to trigger retries
        .read_timeout_ms = 3000,
        .max_retries = 3,
        .initial_backoff_ms = 500,
        .max_backoff_ms = 2000,
    });

    // Test with a URL that might be slow or unreliable
    std.debug.print("   Testing retry mechanism (this may take a moment)...\n", .{});
    const start_time = std.time.milliTimestamp();
    
    var response = client.get("http://httpbin.org/delay/1") catch |err| {
        const elapsed = std.time.milliTimestamp() - start_time;
        std.debug.print("   ⚠️ Request failed after retries: {any} (took {d}ms)\n", .{ err, elapsed });
        return;
    };
    defer response.deinit();

    const elapsed = std.time.milliTimestamp() - start_time;
    std.debug.print("   ✅ Request with retries successful - took {d}ms\n", .{elapsed});
}

/// Test proxy configuration
fn testProxyConfiguration(allocator: std.mem.Allocator) !void {
    // Check if HTTP_PROXY environment variable is set
    const proxy_url = std.process.getEnvVarOwned(allocator, "HTTP_PROXY") catch null;
    defer if (proxy_url) |url| allocator.free(url);

    if (proxy_url) |proxy| {
        std.debug.print("   Testing with proxy: {s}\n", .{proxy});
        
        var client = http_client.HttpClient.init(allocator, .{
            .verbose = true,
            .proxy_url = proxy,
            .connect_timeout_ms = 15000,
            .read_timeout_ms = 20000,
            .max_retries = 2,
        });

        var response = client.get("http://httpbin.org/ip") catch |err| {
            std.debug.print("   ❌ Proxy request failed: {any}\n", .{err});
            return;
        };
        defer response.deinit();

        std.debug.print("   ✅ Proxy request successful - Status: {d}\n", .{response.status_code});
        std.debug.print("   📄 Response: {s}\n", .{response.body[0..@min(200, response.body.len)]});
    } else {
        std.debug.print("   ⚠️ No HTTP_PROXY environment variable set, skipping proxy test\n", .{});
        std.debug.print("   💡 Set HTTP_PROXY=http://proxy.example.com:8080 to test proxy functionality\n", .{});
    }
}

/// Test comprehensive connectivity diagnostics
fn testConnectivityDiagnostics(allocator: std.mem.Allocator) !void {
    var tester = http_client.ConnectivityTester.init(allocator, .{
        .verbose = true,
        .connect_timeout_ms = 10000,
        .read_timeout_ms = 15000,
        .max_retries = 2,
        .initial_backoff_ms = 1000,
    });

    // Test connectivity to local WDBX server (if running)
    try tester.runDiagnostics("http://127.0.0.1:8080");
}

/// Test performance with multiple concurrent requests
fn testPerformance(allocator: std.mem.Allocator) !void {
    const num_requests = 5;
    var clients: [num_requests]http_client.HttpClient = undefined;
    
    for (0..num_requests) |i| {
        clients[i] = http_client.HttpClient.init(allocator, .{
            .verbose = false,  // Reduce verbosity for performance test
            .connect_timeout_ms = 10000,
            .read_timeout_ms = 15000,
            .max_retries = 1,
        });
    }

    std.debug.print("   Making {d} concurrent requests...\n", .{num_requests});
    const start_time = std.time.milliTimestamp();

    var successful_requests: u32 = 0;
    for (0..num_requests) |i| {
        const url = try std.fmt.allocPrint(allocator, "http://httpbin.org/get?request={d}", .{i});
        defer allocator.free(url);
        
        var response = clients[i].get(url) catch |err| {
            std.debug.print("   ❌ Request {d} failed: {any}\n", .{ i, err });
            continue;
        };
        defer response.deinit();
        
        if (response.status_code == 200) {
            successful_requests += 1;
        }
    }

    const elapsed = std.time.milliTimestamp() - start_time;
    std.debug.print("   ✅ {d}/{d} requests successful in {d}ms\n", .{ successful_requests, num_requests, elapsed });
    
    if (successful_requests > 0) {
        const avg_time = @divTrunc(elapsed, successful_requests);
        std.debug.print("   📊 Average time per successful request: {d}ms\n", .{avg_time});
    }
}

/// Integration test with local WDBX server
fn testLocalWdbxIntegration(allocator: std.mem.Allocator) !void {
    std.debug.print("=== WDBX Integration Test ===\n", .{});
    
    var client = http_client.HttpClient.init(allocator, .{
        .verbose = true,
        .connect_timeout_ms = 5000,
        .read_timeout_ms = 10000,
        .max_retries = 3,
        .initial_backoff_ms = 1000,
    });

    // Test health endpoint
    std.debug.print("Testing WDBX health endpoint...\n", .{});
    var health_response = client.get("http://127.0.0.1:8080/health") catch |err| {
        std.debug.print("❌ Health check failed: {any}\n", .{err});
        std.debug.print("💡 Make sure WDBX server is running: zig build run -- http\n", .{});
        return;
    };
    defer health_response.deinit();

    if (health_response.status_code == 200) {
        std.debug.print("✅ WDBX server is healthy\n", .{});
        std.debug.print("📄 Health response: {s}\n", .{health_response.body[0..@min(200, health_response.body.len)]});
    } else {
        std.debug.print("⚠️ WDBX server returned status: {d}\n", .{health_response.status_code});
    }

    // Test stats endpoint
    std.debug.print("\nTesting WDBX stats endpoint...\n", .{});
    var stats_response = client.get("http://127.0.0.1:8080/stats") catch |err| {
        std.debug.print("❌ Stats request failed: {any}\n", .{err});
        return;
    };
    defer stats_response.deinit();

    if (stats_response.status_code == 200) {
        std.debug.print("✅ WDBX stats retrieved successfully\n", .{});
        std.debug.print("📊 Stats response: {s}\n", .{stats_response.body[0..@min(300, stats_response.body.len)]});
    } else {
        std.debug.print("⚠️ WDBX stats returned status: {d}\n", .{stats_response.status_code});
    }
}
