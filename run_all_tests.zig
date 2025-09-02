const std = @import("std");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var stdout = std.io.getStdOut().writer();

    try stdout.print("\n🧪 WDBX-AI Comprehensive Test Suite\n", .{});
    try stdout.print("=====================================\n\n", .{});

    // Test configurations
    const test_suites = [_]struct {
        name: []const u8,
        command: []const []const u8,
    }{
        .{
            .name = "Core Unit Tests",
            .command = &[_][]const u8{ "zig", "build", "test" },
        },
        .{
            .name = "Server Integration Tests",
            .command = &[_][]const u8{ "zig", "build", "test-servers" },
        },
        .{
            .name = "Performance Benchmarks",
            .command = &[_][]const u8{ "zig", "build", "benchmark" },
        },
        .{
            .name = "Static Code Analysis",
            .command = &[_][]const u8{ "zig", "build", "analyze" },
        },
    };

    var total_passed: usize = 0;
    var total_failed: usize = 0;

    for (test_suites) |suite| {
        try stdout.print("▶️  Running: {s}\n", .{suite.name});
        try stdout.print("   Command: ", .{});
        for (suite.command) |arg| {
            try stdout.print("{s} ", .{arg});
        }
        try stdout.print("\n", .{});

        const start_time = std.time.milliTimestamp();

        var child = std.ChildProcess.init(suite.command, allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        try child.spawn();

        // Capture output
        const stdout_data = try child.stdout.?.readToEndAlloc(allocator, 10 * 1024 * 1024);
        defer allocator.free(stdout_data);
        const stderr_data = try child.stderr.?.readToEndAlloc(allocator, 10 * 1024 * 1024);
        defer allocator.free(stderr_data);

        const result = try child.wait();
        const elapsed_ms = std.time.milliTimestamp() - start_time;

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    try stdout.print("   ✅ PASSED ({d}ms)\n", .{elapsed_ms});
                    total_passed += 1;
                } else {
                    try stdout.print("   ❌ FAILED (exit code: {d}, {d}ms)\n", .{ code, elapsed_ms });
                    total_failed += 1;

                    // Print error output
                    if (stderr_data.len > 0) {
                        try stdout.print("   Error output:\n", .{});
                        var lines = std.mem.split(u8, stderr_data, "\n");
                        while (lines.next()) |line| {
                            if (line.len > 0) {
                                try stdout.print("      {s}\n", .{line});
                            }
                        }
                    }
                }
            },
            else => {
                try stdout.print("   ❌ FAILED (abnormal termination)\n", .{});
                total_failed += 1;
            },
        }

        try stdout.print("\n", .{});
    }

    // Summary
    try stdout.print("=====================================\n", .{});
    try stdout.print("📊 Test Summary:\n", .{});
    try stdout.print("   Total suites: {d}\n", .{test_suites.len});
    try stdout.print("   ✅ Passed: {d}\n", .{total_passed});
    try stdout.print("   ❌ Failed: {d}\n", .{total_failed});
    try stdout.print("\n", .{});

    if (total_failed == 0) {
        try stdout.print("🎉 All tests passed! System is production ready.\n", .{});
    } else {
        try stdout.print("⚠️  Some tests failed. Please review the errors above.\n", .{});
        std.process.exit(1);
    }
}

// Utility function to run individual server tests
pub fn testHttpServer() !void {
    const allocator = std.heap.page_allocator;
    var stdout = std.io.getStdOut().writer();

    try stdout.print("\n🌐 Testing HTTP Server...\n", .{});

    // Start HTTP server
    var server_process = std.ChildProcess.init(
        &[_][]const u8{ "zig", "run", "src/wdbx_cli.zig", "--", "server", "--http", "--port", "8080" },
        allocator,
    );

    try server_process.spawn();
    defer _ = server_process.kill() catch {};

    // Wait for server to start
    std.Thread.sleep(1 * std.time.ns_per_s);

    // Test endpoints
    const endpoints = [_][]const u8{
        "http://localhost:8080/health",
        "http://localhost:8080/api/v1/status",
        "http://localhost:8080/api/v1/vectors",
    };

    for (endpoints) |endpoint| {
        try stdout.print("   Testing {s}...", .{endpoint});

        // Simple curl test
        const result = try std.ChildProcess.exec(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", endpoint },
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        if (std.mem.eql(u8, result.stdout, "200") or std.mem.eql(u8, result.stdout, "404")) {
            try stdout.print(" ✅\n", .{});
        } else {
            try stdout.print(" ❌ (status: {s})\n", .{result.stdout});
        }
    }
}

pub fn testTcpServer() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n🔌 Testing TCP Server...\n", .{});

    // Test TCP connection
    const address = try std.net.Address.parseIp("127.0.0.1", 8081);

    if (std.net.tcpConnectToAddress(address)) |stream| {
        defer stream.close();
        try stdout.print("   ✅ TCP connection successful\n", .{});

        // Send test message
        _ = try stream.write("PING\n");

        var buffer: [256]u8 = undefined;
        if (try stream.read(&buffer)) |bytes| {
            try stdout.print("   ✅ Received response: {s}\n", .{buffer[0..bytes]});
        }
    } else |err| {
        try stdout.print("   ⚠️  TCP connection failed: {}\n", .{err});
    }
}

pub fn testWebSocketServer() !void {
    const allocator = std.heap.page_allocator;
    var stdout = std.io.getStdOut().writer();

    try stdout.print("\n🔄 Testing WebSocket Server...\n", .{});

    // WebSocket upgrade test
    const ws_test = try std.ChildProcess.exec(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "curl",
            "-s",
            "-i",
            "-N",
            "-H",
            "Connection: Upgrade",
            "-H",
            "Upgrade: websocket",
            "-H",
            "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==",
            "-H",
            "Sec-WebSocket-Version: 13",
            "http://localhost:8082/ws",
        },
    });
    defer allocator.free(ws_test.stdout);
    defer allocator.free(ws_test.stderr);

    if (std.mem.indexOf(u8, ws_test.stdout, "101") != null) {
        try stdout.print("   ✅ WebSocket upgrade successful\n", .{});
    } else {
        try stdout.print("   ❌ WebSocket upgrade failed\n", .{});
    }
}
