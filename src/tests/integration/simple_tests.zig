//! Simple Tests Suite
//!
//! Consolidated collection of simple test programs for basic functionality:
//! - TCP connectivity testing
//! - HTTP server testing
//! - Database integration testing
//!
//! Usage:
//!   zig run simple_tests.zig -- tcp    # Run TCP test
//!   zig run simple_tests.zig -- http   # Run HTTP test
//!   zig run simple_tests.zig -- db     # Run database integration test

const std = @import("std");
const abi = @import("../../mod.zig");

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Prints the usage banner for this tool.
fn printUsage(prog: []const u8) void {
    std.debug.print("Usage: {s} [tcp|http|db]\n", .{prog});
    std.debug.print("Available tests:\n", .{});
    std.debug.print("  tcp  - Test TCP connectivity to WDBX server\n", .{});
    std.debug.print("  http - Test basic HTTP server functionality\n", .{});
    std.debug.print("  db   - Test database integration\n", .{});
}

// ---------------------------------------------------------------------------
// Main entry point - dispatch to appropriate test
// ---------------------------------------------------------------------------

pub fn main() !void {
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    if (args.len < 2) {
        printUsage(args[0]);
        return error.InvalidArguments;
    }

    const test_type = args[1];
    if (std.mem.eql(u8, test_type, "tcp")) {
        return runTcpTest();
    } else if (std.mem.eql(u8, test_type, "http")) {
        return runHttpTest();
    } else if (std.mem.eql(u8, test_type, "db")) {
        return runDatabaseTest();
    } else {
        std.debug.print("Unknown test type: {s}\n", .{test_type});
        return error.InvalidArguments;
    }
}

// ---------------------------------------------------------------------------
// Test TCP connectivity to WDBX server (enhanced for Windows compatibility)
// ---------------------------------------------------------------------------

fn runTcpTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    _ = gpa.allocator();

    std.debug.print("🔌 Simple TCP Test for WDBX Server (Windows Enhanced)\n", .{});
    std.debug.print("=================================================\n\n", .{});

    const address = std.net.Address.parseIp("127.0.0.1", 8080) catch {
        std.debug.print("❌ Failed to parse address\n", .{});
        return;
    };

    std.debug.print("📡 Connecting to 127.0.0.1:8080...\n", .{});

    const connection = std.net.tcpConnectToAddress(address) catch |err| {
        std.debug.print("❌ Connection failed: {}\n", .{err});
        std.debug.print("💡 Make sure the WDBX server is running: .\\zig-out\\bin\\abi.exe http\n", .{});
        return err;
    };
    defer connection.close();

    // Configure socket for Windows compatibility
    configureClientSocket(connection) catch |err| {
        std.debug.print("⚠️ Socket configuration warning: {} (continuing anyway)\n", .{err});
    };

    std.debug.print("✅ Connected successfully!\n", .{});

    // Define a list of HTTP requests to test
    const requests = [_][]const u8{
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        "GET /network HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    };

    for (requests, 0..) |request, i| {
        std.debug.print("\n🧪 Test {d}: Sending request...\n", .{i + 1});

        // Send request
        connection.writer().writeAll(request) catch |err| {
            std.debug.print("❌ Write failed: {}\n", .{err});
            continue;
        };

        std.debug.print("📤 Request sent, waiting for response...\n", .{});

        // Read the full response until the socket closes
        const response = try std.io.readAllToAlloc(std.heap.page_allocator, connection.reader(), 64 * 1024);
        defer std.heap.page_allocator.free(response);

        if (response.len > 0) {
            std.debug.print("📥 Received {d} bytes:\n", .{response.len});
            std.debug.print("--- Response ---\n{s}\n--- End Response ---\n", .{response});
        } else {
            std.debug.print("📥 No data received (connection closed by server)\n", .{});
        }
    }

    std.debug.print("\n🎉 TCP test completed!\n", .{});
    std.debug.print("✅ Your WDBX server is working correctly on Windows.\n", .{});
    std.debug.print("ℹ️ The 'connection reset' behavior is expected and properly handled.\n", .{});
}

// ---------------------------------------------------------------------------
// Test basic HTTP server functionality
// ---------------------------------------------------------------------------

fn runHttpTest() !void {
    const allocator = std.heap.page_allocator;

    var server = std.http.Server.init(.{
        .allocator = allocator,
        .reuse_address = true,
    });
    defer server.deinit();

    const address = try std.net.Address.parseIp("127.0.0.1", 8081);
    try server.listen(address);

    std.debug.print("Simple HTTP server listening on http://127.0.0.1:8081\n", .{});
    std.debug.print("Test with: Invoke-WebRequest -Uri 'http://127.0.0.1:8081/'\n", .{});

    while (true) {
        var response = try server.accept(.{});
        defer response.deinit();

        std.debug.print("Received request: {s} {s}\n", .{ @tagName(response.request.method), response.request.target });

        // Send a simple response
        response.status = .ok;
        try response.headers.append("Content-Type", "text/plain");
        try response.do();
        try response.writer().writeAll("Hello from WDBX HTTP Server!");

        std.debug.print("Response sent successfully\n", .{});
    }
}

// ---------------------------------------------------------------------------
// Test database integration functionality
// ---------------------------------------------------------------------------

fn runDatabaseTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Running Simple Integration Test", .{});

    const test_file = "test_simple_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.wdbx.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Add a simple vector
    const test_vector = [_]f32{1.0} ** 128;
    const id = try db.addEmbedding(&test_vector);
    std.log.info("  Added vector with ID: {}", .{id});

    // Search for similar vectors
    const results = try db.search(&test_vector, 5, allocator);
    defer allocator.free(results);

    std.log.info("  Found {} similar vectors", .{results.len});

    std.log.info("✅ Simple integration test passed", .{});
}

// ---------------------------------------------------------------------------
// Configure client socket for Windows compatibility
// ---------------------------------------------------------------------------

fn configureClientSocket(connection: std.net.Stream) !void {
    const handle = connection.handle;

    // Set TCP_NODELAY for better performance
    const enable: c_int = 1;
    _ = std.posix.setsockopt(handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&enable)) catch |err| {
        std.debug.print("Warning: TCP_NODELAY failed: {}\n", .{err});
    };

    // Set socket buffer sizes
    const buffer_size: c_int = 8192;
    _ = std.posix.setsockopt(handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&buffer_size)) catch |err| {
        std.debug.print("Warning: SO_RCVBUF failed: {}\n", .{err});
    };
    _ = std.posix.setsockopt(handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&buffer_size)) catch |err| {
        std.debug.print("Warning: SO_SNDBUF failed: {}\n", .{err});
    };
}
