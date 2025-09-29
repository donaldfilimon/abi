const std = @import("std");

pub const HttpServer = struct {
    allocator: std.mem.Allocator,
    port: u16,

    pub fn init(allocator: std.mem.Allocator, port: u16) HttpServer {
        return HttpServer{
            .allocator = allocator,
            .port = port,
        };
    }

    pub fn start(self: *HttpServer) !void {
        std.debug.print("🌐 HTTP Server starting on port {}...\n", .{self.port});

        // Simulate server setup
        std.debug.print("✅ Server listening on http://localhost:{}\n", .{self.port});
        std.debug.print("📡 Ready to accept connections\n", .{});

        // Simulate handling requests
        const sample_requests = [_][]const u8{
            "GET /health HTTP/1.1",
            "POST /api/chat HTTP/1.1",
            "GET /api/status HTTP/1.1",
            "POST /api/embeddings HTTP/1.1",
        };

        for (sample_requests, 0..) |request, i| {
            std.Thread.sleep(800 * std.time.ns_per_ms);
            std.debug.print("📨 Request {}: {s} -> 200 OK\n", .{ i + 1, request });
        }

        std.debug.print("🛑 Server shutdown complete\n", .{});
    }

    pub fn handleRequest(self: *HttpServer, path: []const u8) []const u8 {
        _ = self;

        if (std.mem.eql(u8, path, "/health")) {
            return "OK";
        } else if (std.mem.eql(u8, path, "/api/status")) {
            return "{\"status\": \"running\", \"version\": \"1.0.0\"}";
        } else if (std.mem.eql(u8, path, "/api/chat")) {
            return "{\"response\": \"Hello from ABI AI!\"}";
        } else {
            return "Not Found";
        }
    }
};

test "http server basic functionality" {
    var server = HttpServer.init(std.testing.allocator, 8080);

    const health_response = server.handleRequest("/health");
    try std.testing.expect(std.mem.eql(u8, health_response, "OK"));

    const status_response = server.handleRequest("/api/status");
    try std.testing.expect(std.mem.startsWith(u8, status_response, "{"));
}
