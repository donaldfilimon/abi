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

pub const DatabaseFeatureError = FrameworkError;
