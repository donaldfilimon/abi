//! Shared types for the documents feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by document parsing operations.
pub const DocumentsError = error{
    DocumentsDisabled,
    ParseFailed,
    UnsupportedFormat,
    InvalidInput,
    OutOfMemory,
};

pub const Error = DocumentsError;

/// Documents context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,
};
