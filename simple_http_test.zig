const std = @import("std");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a simple HTTP server
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
