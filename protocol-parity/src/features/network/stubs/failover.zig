const std = @import("std");

pub const FailoverManager = struct {
    pub fn init(_: std.mem.Allocator, _: FailoverConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const FailoverConfig = struct {
    max_failover_attempts: u32 = 3,
    failover_timeout_ms: u64 = 30_000,
};

pub const FailoverState = enum { normal, failing_over, failed, recovered };

pub const FailoverEvent = struct {
    timestamp_ms: i64 = 0,
    from_node: []const u8 = "",
    to_node: []const u8 = "",
    state: FailoverState = .normal,
};

test {
    std.testing.refAllDecls(@This());
}
