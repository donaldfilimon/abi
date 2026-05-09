//! Shared types for the desktop feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by desktop integration operations.
pub const DesktopError = error{
    DesktopDisabled,
    PlatformUnsupported,
    IntegrationFailed,
    OutOfMemory,
};

pub const Error = DesktopError;

/// Desktop context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,
};
