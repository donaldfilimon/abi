//! Enhanced Web Server Example
//!
//! Demonstrates the production-ready enhanced web server with
//! middleware, routing, and AI integration capabilities.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create enhanced web server
    const config = abi.web.enhanced_web_server.WebServerConfig{
        .port = 8080,
        .enable_cors = true,
        .enable_websocket = true,
        .enable_rate_limiting = true,
    };

    var server = try abi.web.enhanced_web_server.EnhancedWebServer.init(allocator, config);
    defer server.deinit();

    // Start the server
    std.log.info("Starting enhanced web server on port {}", .{config.port});
    try server.start();

    // Keep the server running
    while (true) {
        std.time.sleep(1_000_000_000); // Sleep for 1 second
    }
}
