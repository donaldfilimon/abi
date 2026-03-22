//! Shared types for the database feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.
//!
//! Note: most database types (SearchResult, DatabaseHandle, Stats, etc.)
//! live in `src/core/database/` and are re-exported by both mod and stub.
//! This file covers only the feature-level error set.

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
    FeatureDisabled,
};
