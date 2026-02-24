const std = @import("std");

pub const ConnectionPool = struct {
    pub fn init(_: std.mem.Allocator, _: ConnectionPoolConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ConnectionPoolConfig = struct {
    max_connections: usize = 10,
    idle_timeout_ms: u64 = 60_000,
    connection_timeout_ms: u64 = 5000,
};

pub const PooledConnection = struct {
    id: u64 = 0,
    state: ConnectionState = .idle,
};

pub const ConnectionState = enum { idle, active, closed };

pub const ConnectionStats = struct {
    active: usize = 0,
    idle: usize = 0,
    total_created: u64 = 0,
};

pub const HostKey = struct {
    host: []const u8 = "",
    port: u16 = 0,
};

pub const PoolStats = struct {
    total_pools: usize = 0,
    total_connections: usize = 0,
    active_connections: usize = 0,
};

pub const PoolBuilder = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn build(_: *@This()) !ConnectionPool {
        return error.NetworkDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
