//! TCP API Module
const std = @import("std");

pub const TcpServer = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TcpServer {
        return TcpServer{ .allocator = allocator };
    }
    
    pub fn deinit(self: *TcpServer) void {
        _ = self;
    }
    
    pub fn start(self: *TcpServer) !void {
        _ = self;
        std.debug.print("TCP server starting...\n", .{});
    }
};