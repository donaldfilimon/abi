//! Network Stub Module

const std = @import("std");
const config_module = @import("../config/mod.zig");

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const Error = types.Error;
pub const NetworkConfig = types.NetworkConfig;
pub const NetworkState = types.NetworkState;
pub const Node = types.Node;
pub const NodeStatus = types.NodeStatus;
pub const NodeInfo = types.NodeInfo;
pub const NodeRegistry = types.NodeRegistry;

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
