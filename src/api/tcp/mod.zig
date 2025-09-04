//! TCP API module for WDBX-AI
//!
//! Provides TCP server functionality for the WDBX vector database.

const std = @import("std");
const core = @import("../../core/mod.zig");

pub const TcpServer = struct {
    allocator: std.mem.Allocator,
    port: u16,
    running: bool,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, port: u16) Self {
        return Self{
            .allocator = allocator,
            .port = port,
            .running = false,
        };
    }
    
    pub fn start(self: *Self) !void {
        self.running = true;
        core.log.info("TCP server starting on port {d}", .{self.port});
        // TCP server implementation would go here
    }
    
    pub fn stop(self: *Self) void {
        self.running = false;
        core.log.info("TCP server stopped", .{});
    }
    
    pub fn isRunning(self: Self) bool {
        return self.running;
    }
};

test "tcp server" {
    const testing = std.testing;
    
    var server = TcpServer.init(testing.allocator, 8081);
    try testing.expect(!server.isRunning());
    
    try server.start();
    try testing.expect(server.isRunning());
    
    server.stop();
    try testing.expect(!server.isRunning());
}