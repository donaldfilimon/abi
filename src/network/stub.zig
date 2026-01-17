//! Network Stub Module

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    NetworkDisabled,
    ConnectionFailed,
    NodeNotFound,
    ConsensusFailed,
    Timeout,
};

pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};
pub const NetworkState = enum { disconnected, connected };
pub const Node = struct {};
pub const NodeStatus = enum { healthy, degraded, offline };
pub const NodeInfo = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    status: NodeStatus = .healthy,
    last_seen_ms: i64 = 0,
};
pub const NodeRegistry = struct {
    pub fn register(_: *NodeRegistry, _: []const u8, _: []const u8) Error!void {
        return error.NetworkDisabled;
    }
    pub fn unregister(_: *NodeRegistry, _: []const u8) bool {
        return false;
    }
    pub fn touch(_: *NodeRegistry, _: []const u8) bool {
        return false;
    }
    pub fn setStatus(_: *NodeRegistry, _: []const u8, _: NodeStatus) bool {
        return false;
    }
    pub fn list(_: *NodeRegistry) []const NodeInfo {
        return &.{};
    }
};

pub const Context = struct {
    pub const State = enum {
        disconnected,
        connecting,
        connected,
        error_state,
    };

    pub fn init(_: std.mem.Allocator, _: config_module.NetworkConfig) Error!*Context {
        return error.NetworkDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn connect(_: *Context) Error!void {
        return error.NetworkDisabled;
    }

    pub fn disconnect(_: *Context) void {}

    pub fn getState(_: *Context) State {
        return .disconnected;
    }

    pub fn discoverPeers(_: *Context) Error![]NodeInfo {
        return error.NetworkDisabled;
    }

    pub fn sendTask(_: *Context, _: []const u8, _: anytype) Error!void {
        return error.NetworkDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn defaultRegistry() Error!*NodeRegistry {
    return error.NetworkDisabled;
}

pub fn defaultConfig() ?NetworkConfig {
    return null;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.NetworkDisabled;
}

pub fn deinit() void {}
