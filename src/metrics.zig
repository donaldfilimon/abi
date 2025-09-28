const std = @import("std");

/// Starts a simple HTTP metrics exporter on port 9100.
/// The server listens for incoming connections and replies with a
/// plain‑text payload that follows the Prometheus `/metrics` format.
pub fn startMetricsServer(allocator: std.mem.Allocator) !void {
    var server = std.net.Server.init(.{ .reuse_address = true });
    defer server.deinit();

    try server.listen(.{ .port = 9100 });
    std.log.info("Metrics server listening on :9100", .{});

    while (true) {
        const socket = try server.accept(allocator);
        defer socket.close();

        const reader = socket.reader();
        var line_buf: [256]u8 = undefined;
        _ = try reader.readUntilDelimiterOrEof(line_buf[0..], '\n');

        const payload =
            "# HELP abi_exporter_time_seconds A dummy counter metric\n" ++
            "# TYPE abi_exporter_time_seconds counter\n" ++
            "abi_exporter_time_seconds 12345.6789\n";

        const writer = socket.writer();

        // Write HTTP header using writer.writeAll
        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: text/plain; charset=utf-8\r\n");
        const lenStr = try std.fmt.allocPrint(allocator, "{}\r\n", .{payload.len});
        defer allocator.free(lenStr);
        try writer.writeAll("Content-Length: ");
        try writer.writeAll(lenStr);
        try writer.writeAll("Connection: close\r\n");
        try writer.writeAll("\r\n");

        // Write payload
        try writer.writeAll(payload);
        try writer.flush();
    }
}

pub fn main() !void {
    try startMetricsServer(std.heap.page_allocator);
}
