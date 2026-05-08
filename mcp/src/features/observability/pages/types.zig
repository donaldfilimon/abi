//! Shared types for the dashboard pages feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

/// Errors returned by dashboard page operations.
pub const PagesError = error{
    PagesDisabled,
    PageNotFound,
    RenderFailed,
    InvalidData,
};

pub const Error = PagesError;

/// Page metadata for the dashboard.
pub const PageInfo = struct {
    name: []const u8,
    path: []const u8,
    is_active: bool = false,
};
