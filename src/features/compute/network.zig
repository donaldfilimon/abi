const std = @import("std");

pub const Transport = enum {
    rdma,
    tcp,
};

pub const NetworkContext = struct {
    allocator: std.mem.Allocator,
    transport: Transport,

    pub fn deinit(self: *NetworkContext) void {
        _ = self;
    }
};

pub fn isAvailable(_: Transport) bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator, transport: Transport) !NetworkContext {
    _ = allocator;
    _ = transport;
    return error.TransportUnavailable;
}

pub fn send(_: *NetworkContext, _: []const u8) !void {
    return error.TransportUnavailable;
}

pub fn receive(_: *NetworkContext, _: []u8) !usize {
    return error.TransportUnavailable;
}
