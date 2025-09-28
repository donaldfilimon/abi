const std = @import("std");
const metrics = @import("../src/metrics.zig");

test "metrics endpoint returns expected metric" {
    const allocator = std.heap.page_allocator;

    // Start the metrics server in a background thread.
    _ = try std.Thread.spawn(.{}, metrics.startMetricsServer, .{allocator});

    // Give the server a moment to start listening.
    std.time.sleep(std.time.ns_per_ms * 100);

    // Connect to the server.
    var client = try std.net.StreamSocket.connect(.{
        .address = .{
            .ip = .{ .ipv4 = .{ .bytes = .{ 127, 0, 0, 1 } } },
            .port = 9100,
        },
        .allocator = allocator,
    });
    defer client.close();

    // Send a minimal HTTP GET request.
    const writer = client.writer();
    try writer.writeAll(
        \\GET /metrics HTTP/1.1\r\n
        \\Host: localhost\r\n
        \\Connection: close\r\n
        \\\
    );
    try writer.flush();

    // Read the full HTTP response.
    const reader = client.reader();
    var buf: [2048]u8 = undefined;
    const n = try reader.readAll(&buf);
    const resp = buf[0..n];

    // Verify the response contains our dummy metric line.
    std.testing.expect(std.mem.indexOf(u8, resp, "abi_exporter_time_seconds 12345.6789") != null);
}
