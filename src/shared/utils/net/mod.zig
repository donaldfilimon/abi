//! Network Utilities Module
//!
//! Network-related utilities and functions

const std = @import("std");

/// Resolve hostname to IP address
pub fn resolveHost(allocator: std.mem.Allocator, hostname: []const u8) !std.net.Address {
    return std.net.tcpConnectToHost(allocator, hostname, 80);
}

/// Check if port is open on host
pub fn isPortOpen(allocator: std.mem.Allocator, host: []const u8, port: u16) bool {
    std.net.tcpConnectToHost(allocator, host, port) catch return false;
    return true;
}

/// Parse IP address string
pub fn parseAddress(address: []const u8) !std.net.Address {
    return std.net.Address.parseIp(address, 0);
}

test {
    std.testing.refAllDecls(@This());
}

test "parseAddress" {
    const addr = try parseAddress("127.0.0.1");
    try std.testing.expect(addr.getPort() == 0);
}
