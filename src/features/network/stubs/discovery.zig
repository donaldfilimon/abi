const std = @import("std");

pub const ServiceDiscovery = struct {
    pub fn init(_: std.mem.Allocator, _: DiscoveryConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .static,
    refresh_interval_ms: u64 = 10_000,
};

pub const DiscoveryBackend = enum { static, dns, multicast, consul };

pub const ServiceInstance = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    address: []const u8 = "",
    port: u16 = 0,
    status: ServiceStatus = .unknown,
};

pub const ServiceStatus = enum { unknown, healthy, unhealthy, draining };

pub const DiscoveryError = error{
    NetworkDisabled,
    ServiceNotFound,
    DiscoveryFailed,
};

pub fn generateServiceId(_: std.mem.Allocator, _: []const u8) ![]const u8 {
    return error.NetworkDisabled;
}

pub fn base64Encode(_: std.mem.Allocator, _: []const u8) ![]const u8 {
    return error.NetworkDisabled;
}

pub fn base64Decode(_: std.mem.Allocator, _: []const u8) ![]const u8 {
    return error.NetworkDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
