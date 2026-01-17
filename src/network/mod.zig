//! Network Module
//!
//! Distributed compute network with node discovery and Raft consensus.
//!
//! ## Features
//! - Node discovery and registration
//! - Raft consensus for leader election
//! - Task distribution and load balancing
//! - Automatic failover

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/network
const features_network = @import("../features/network/mod.zig");

pub const NetworkConfig = features_network.NetworkConfig;
pub const NetworkState = features_network.NetworkState;
pub const Node = features_network.Node;
pub const NodeInfo = features_network.NodeInfo;
pub const NodeRegistry = features_network.NodeRegistry;
pub const NodeStatus = features_network.NodeStatus;

pub const Error = error{
    NetworkDisabled,
    ConnectionFailed,
    NodeNotFound,
    ConsensusFailed,
    Timeout,
};

/// Network context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.NetworkConfig,
    state: State = .disconnected,

    pub const State = enum {
        disconnected,
        connecting,
        connected,
        error_state,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.NetworkConfig) !*Context {
        if (!isEnabled()) return error.NetworkDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.disconnect();
        self.allocator.destroy(self);
    }

    /// Connect to the network.
    pub fn connect(self: *Context) !void {
        if (self.state == .connected) return;
        self.state = .connecting;
        // Network connection logic
        self.state = .connected;
    }

    /// Disconnect from the network.
    pub fn disconnect(self: *Context) void {
        self.state = .disconnected;
    }

    /// Get current state.
    pub fn getState(self: *Context) State {
        return self.state;
    }

    /// Discover peers.
    pub fn discoverPeers(self: *Context) ![]NodeInfo {
        if (self.state != .connected) {
            try self.connect();
        }
        return &.{};
    }

    /// Send a task to a remote node.
    pub fn sendTask(self: *Context, node_id: []const u8, task: anytype) !void {
        _ = self;
        _ = node_id;
        _ = task;
    }
};

pub fn isEnabled() bool {
    return build_options.enable_network;
}

pub fn isInitialized() bool {
    return features_network.isInitialized();
}

pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.NetworkDisabled;
    features_network.init(allocator) catch return error.NetworkDisabled;
}

pub fn deinit() void {
    features_network.deinit();
}

/// Get the default node registry.
pub fn defaultRegistry() Error!*NodeRegistry {
    if (!isEnabled()) return error.NetworkDisabled;
    return features_network.defaultRegistry() catch return error.NetworkDisabled;
}

/// Get the default network configuration.
pub fn defaultConfig() ?NetworkConfig {
    if (!isEnabled()) return null;
    return features_network.defaultConfig();
}
