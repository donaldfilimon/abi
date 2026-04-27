const std = @import("std");

pub const RaftTransport = struct {
    pub const RaftTransportStats = struct {
        votes_sent: u64 = 0,
        votes_received: u64 = 0,
        append_entries_sent: u64 = 0,
        append_entries_received: u64 = 0,
    };

    pub fn init(_: std.mem.Allocator, _: RaftTransportConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const RaftTransportConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    port: u16 = 9001,
};

pub const PeerAddress = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    port: u16 = 0,
};

test {
    std.testing.refAllDecls(@This());
}
