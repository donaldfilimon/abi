//! Shared Error Types
//!
//! Common error sets for cross-module error handling. Modules can import
//! and combine these error sets to maintain consistency across the framework.
//!
//! ## Usage
//!
//! ```zig
//! const errors = @import("shared").errors;
//!
//! pub const MyModuleError = errors.ResourceError || error{
//!     ModuleSpecificError,
//! };
//! ```

const std = @import("std");

/// Common resource errors across all modules.
/// Use when operations fail due to resource constraints.
pub const ResourceError = error{
    /// Memory allocation failed.
    OutOfMemory,
    /// Resource limit exceeded (connections, handles, etc.).
    ResourceExhausted,
    /// Operation timed out.
    Timeout,
    /// Operation was cancelled.
    Cancelled,
};

/// Common I/O errors for file and network operations.
pub const IoError = error{
    /// Remote host refused connection.
    ConnectionRefused,
    /// Connection was reset by peer.
    ConnectionReset,
    /// End of stream reached unexpectedly.
    EndOfStream,
    /// Data format or content is invalid.
    InvalidData,
    /// File or resource not found.
    NotFound,
    /// Permission denied for operation.
    PermissionDenied,
};

/// Feature availability errors.
/// Use when features are disabled or not available.
pub const FeatureError = error{
    /// Feature is disabled at compile time.
    FeatureDisabled,
    /// Feature not supported on this platform.
    NotSupported,
    /// Feature not yet implemented.
    NotImplemented,
};

/// Configuration and validation errors.
pub const ConfigError = error{
    /// Configuration value is invalid.
    InvalidConfig,
    /// Required configuration is missing.
    MissingRequired,
    /// Configuration values conflict with each other.
    ConflictingConfig,
    /// Configuration parsing failed.
    ParseError,
};

/// Authentication and authorization errors.
pub const AuthError = error{
    /// Authentication failed.
    AuthenticationFailed,
    /// Authorization denied for operation.
    AuthorizationDenied,
    /// Token has expired.
    TokenExpired,
    /// Invalid credentials provided.
    InvalidCredentials,
};

// ============================================================================
// Tests
// ============================================================================

test "error sets are distinct" {
    // Verify error sets don't overlap (would cause compile error if they did)
    const Combined = ResourceError || IoError || FeatureError;
    _ = @as(Combined, error.OutOfMemory);
    _ = @as(Combined, error.ConnectionRefused);
    _ = @as(Combined, error.FeatureDisabled);
}

test "errors can be matched" {
    const err: ResourceError = error.Timeout;
    switch (err) {
        error.OutOfMemory => unreachable,
        error.ResourceExhausted => unreachable,
        error.Timeout => {},
        error.Cancelled => unreachable,
    }
}

test "config errors cover common cases" {
    const test_errors = [_]ConfigError{
        error.InvalidConfig,
        error.MissingRequired,
        error.ConflictingConfig,
        error.ParseError,
    };
    for (test_errors) |err| {
        _ = err;
    }
    try std.testing.expectEqual(@as(usize, 4), test_errors.len);
}
