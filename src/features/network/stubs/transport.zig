const std = @import("std");

pub const TcpTransport = struct {
    pub const TransportStats = struct {
        messages_sent: u64 = 0,
        messages_received: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
    };

    pub fn init(_: std.mem.Allocator, _: TransportConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const TransportConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    port: u16 = 9000,
    max_connections: usize = 100,
};

pub const TransportError = error{
    NetworkDisabled,
    ConnectionRefused,
    BindFailed,
    SendFailed,
};

pub const MessageType = enum { request, response, heartbeat, data };

pub const MessageHeader = struct {
    msg_type: MessageType = .request,
    payload_size: u32 = 0,
    sequence: u64 = 0,
};

pub const PeerConnection = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    connected: bool = false,
};

pub const RpcSerializer = struct {
    pub fn serialize(_: std.mem.Allocator, _: anytype) ![]const u8 {
        return error.NetworkDisabled;
    }
    pub fn deserialize(_: type, _: std.mem.Allocator, _: []const u8) !void {
        return error.NetworkDisabled;
    }
};

pub fn parseAddress(_: []const u8) !struct { host: []const u8, port: u16 } {
    return error.NetworkDisabled;
}
