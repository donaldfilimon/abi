//! Stub for Network feature when disabled
const std = @import("std");

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};

pub const NodeStatus = enum {
    healthy,
    degraded,
    offline,
};

pub const NodeInfo = struct {
    id: []const u8,
    address: []const u8,
    status: NodeStatus = .healthy,
    last_seen_ms: i64,
};

pub const NodeRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) NodeRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *NodeRegistry) void {
        _ = self;
    }

    pub fn register(self: *NodeRegistry, id: []const u8, address: []const u8) !void {
        _ = self;
        _ = id;
        _ = address;
        return NetworkError.NetworkDisabled;
    }

    pub fn unregister(self: *NodeRegistry, id: []const u8) bool {
        _ = self;
        _ = id;
        return false;
    }

    pub fn touch(self: *NodeRegistry, id: []const u8) bool {
        _ = self;
        _ = id;
        return false;
    }

    pub fn setStatus(self: *NodeRegistry, id: []const u8, status: NodeStatus) bool {
        _ = self;
        _ = id;
        _ = status;
        return false;
    }

    pub fn list(self: *NodeRegistry) []const NodeInfo {
        _ = self;
        return &.{};
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return NetworkError.NetworkDisabled;
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) !void {
    _ = allocator;
    _ = config;
    return NetworkError.NetworkDisabled;
}

pub fn deinit() void {}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    return NetworkError.NotInitialized;
}

pub fn defaultConfig() ?NetworkConfig {
    return null;
}
