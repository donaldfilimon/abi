//! Network Stub Module
//!
//! This module provides API-compatible no-op implementations for all public
//! network functions when the network feature is disabled at compile time.
//! All functions return `error.NetworkDisabled` or empty/default values as
//! appropriate.
//!
//! The network module encompasses:
//! - Distributed compute coordination
//! - Raft consensus protocol
//! - Node discovery and peer management
//! - Task distribution across cluster nodes
//! - Network state management
//!
//! To enable the real implementation, build with `-Denable-network=true`.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

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

pub fn initWithConfig(_: std.mem.Allocator, _: NetworkConfig) Error!void {
    return error.NetworkDisabled;
}

pub fn deinit() void {}
