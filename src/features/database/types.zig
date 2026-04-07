//! Shared types for the database feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.
//!
//! Note: the canonical public WDBX types now live under the `Store`, `Context`,
//! `memory`, `storage`, `distributed`, and `retrieval` namespaces.
//! This file covers only the feature-level error set.

pub const FrameworkError = error{
    DatabaseDisabled,
    ConnectionFailed,
    QueryFailed,
    IndexError,
    StorageError,
};

<<<<<<< Updated upstream
pub const DatabaseFeatureError = FrameworkError;
=======
/// Represents a block in the "Liquid Glass" memory model.
/// Supports vector embeddings, MVCC via timestamps, and cryptographic integrity.
pub const LiquidGlassMemory = struct {
    shard_id: u32,
    vector_dimension: u16,
    commit_timestamp: i64,
    integrity_hash: [32]u8,
    skip_pointer: ?usize,
};

/// Defines a vector/semantic storage endpoint for the distributed memory model.
pub const SemanticEndpoint = struct {
    endpoint_url: []const u8,
    is_active: bool,
    latency_ms: u32,
};
>>>>>>> Stashed changes
