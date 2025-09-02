const std = @import("std");
const wdbx_http = @import("src/wdbx/http.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create HTTP server configuration
    const config = wdbx_http.ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .enable_cors = true,
        .enable_auth = false, // Disable auth for testing
    };

    // Initialize server
    var server = try wdbx_http.WdbxHttpServer.init(allocator, config);
    defer server.deinit();

    // Open a test database
    try server.openDatabase("test_http.db");

    std.debug.print("Starting HTTP server on http://{s}:{}\n", .{ config.host, config.port });
    std.debug.print("Test endpoints:\n", .{});
    std.debug.print("  GET  http://{s}:{}/health\n", .{ config.host, config.port });
    std.debug.print("  GET  http://{s}:{}/stats\n", .{ config.host, config.port });
    std.debug.print("  GET  http://{s}:{}/query?vec=1.0,2.0,3.0\n", .{ config.host, config.port });
    std.debug.print("  POST http://{s}:{}/add\n", .{ config.host, config.port });
    std.debug.print("\nPress Ctrl+C to stop the server\n", .{});

    // Start the server
    try server.start();
}
