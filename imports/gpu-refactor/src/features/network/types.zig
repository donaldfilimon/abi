//! Shared types for the network feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

/// Primary error set for network operations.
pub const Error = error{
    NetworkDisabled,
    ConnectionFailed,
    NodeNotFound,
    ConsensusFailed,
    Timeout,
};

/// Error set for module-level network operations.
pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

/// Configuration for a network cluster.
pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};
