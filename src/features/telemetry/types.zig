//! Shared types for the telemetry feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by telemetry operations.
pub const TelemetryError = error{
    FeatureDisabled,
    OutOfMemory,
};

pub const Error = TelemetryError;
