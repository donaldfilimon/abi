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

pub const NetworkConfig = struct {};
pub const NetworkState = enum { disconnected, connected };
pub const Node = struct {};
pub const NodeInfo = struct {};
pub const NodeRegistry = struct {};

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

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.NetworkDisabled;
}

pub fn deinit() void {}
