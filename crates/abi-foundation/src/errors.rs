//! The framework-wide error set.
//!
//! Ported from `src/foundation/errors.zig`. The `Display` strings are part of
//! the CLI's observable output, so they are reproduced verbatim.

use std::fmt;

/// Canonical error kinds surfaced across the ABI framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiError {
    /// Configuration failed to load or validate.
    InvalidConfig,
    /// A path was malformed or outside an allowed root.
    InvalidPath,
    /// A required file was absent.
    FileNotFound,
    /// The process lacked permission for the operation.
    PermissionDenied,
    /// An allocation failed.
    OutOfMemory,
    /// An unexpected internal invariant was violated.
    InternalError,
    /// A subsystem was used before initialization.
    NotInitialized,
    /// A subsystem was initialized twice.
    AlreadyInitialized,
    /// The operation is not valid in the current state.
    InvalidState,
    /// The operation exceeded its deadline.
    Timeout,
    /// A network connection could not be established.
    ConnectionFailed,
    /// Input could not be parsed.
    ParseError,
    /// Input failed validation.
    ValidationError,
    /// The requested operation is not supported on this build or platform.
    UnsupportedOperation,
    /// The requested entity does not exist.
    NotFound,
}

impl AbiError {
    /// The human-readable description of this error.
    ///
    /// These strings are load-bearing: they appear in CLI output and in MCP
    /// error payloads, so they match the Zig `formatError` exactly.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::InvalidConfig => "invalid configuration",
            Self::InvalidPath => "invalid path",
            Self::FileNotFound => "file not found",
            Self::PermissionDenied => "permission denied",
            Self::OutOfMemory => "out of memory",
            Self::InternalError => "internal error",
            Self::NotInitialized => "not initialized",
            Self::AlreadyInitialized => "already initialized",
            Self::InvalidState => "invalid state",
            Self::Timeout => "operation timed out",
            Self::ConnectionFailed => "connection failed",
            Self::ParseError => "parse error",
            Self::ValidationError => "validation error",
            Self::UnsupportedOperation => "unsupported operation",
            Self::NotFound => "not found",
        }
    }

    /// The Zig-side error name (`error.InvalidConfig` → `"InvalidConfig"`).
    ///
    /// Retained because `help-json` and some MCP payloads report the symbolic
    /// name rather than the prose description.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidConfig => "InvalidConfig",
            Self::InvalidPath => "InvalidPath",
            Self::FileNotFound => "FileNotFound",
            Self::PermissionDenied => "PermissionDenied",
            Self::OutOfMemory => "OutOfMemory",
            Self::InternalError => "InternalError",
            Self::NotInitialized => "NotInitialized",
            Self::AlreadyInitialized => "AlreadyInitialized",
            Self::InvalidState => "InvalidState",
            Self::Timeout => "Timeout",
            Self::ConnectionFailed => "ConnectionFailed",
            Self::ParseError => "ParseError",
            Self::ValidationError => "ValidationError",
            Self::UnsupportedOperation => "UnsupportedOperation",
            Self::NotFound => "NotFound",
        }
    }

    /// Resolve a symbolic error name back to its variant.
    ///
    /// Replaces Zig's `describeError(anyerror)`, which matched `@errorName`
    /// against the error set at comptime.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        const ALL: [AbiError; 15] = [
            AbiError::InvalidConfig,
            AbiError::InvalidPath,
            AbiError::FileNotFound,
            AbiError::PermissionDenied,
            AbiError::OutOfMemory,
            AbiError::InternalError,
            AbiError::NotInitialized,
            AbiError::AlreadyInitialized,
            AbiError::InvalidState,
            AbiError::Timeout,
            AbiError::ConnectionFailed,
            AbiError::ParseError,
            AbiError::ValidationError,
            AbiError::UnsupportedOperation,
            AbiError::NotFound,
        ];
        ALL.into_iter().find(|candidate| candidate.name() == name)
    }

    /// Describe an arbitrary error name, falling back to `"unknown error"`.
    #[must_use]
    pub fn describe_name(name: &str) -> &str {
        match Self::from_name(name) {
            Some(err) => err.describe(),
            None => "unknown error",
        }
    }
}

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.describe())
    }
}

impl std::error::Error for AbiError {}

impl From<std::io::Error> for AbiError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => Self::FileNotFound,
            std::io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            std::io::ErrorKind::TimedOut => Self::Timeout,
            std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::NotConnected => Self::ConnectionFailed,
            std::io::ErrorKind::OutOfMemory => Self::OutOfMemory,
            std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => Self::ParseError,
            _ => Self::InternalError,
        }
    }
}

impl From<serde_json::Error> for AbiError {
    fn from(_: serde_json::Error) -> Self {
        Self::ParseError
    }
}

/// A contextual wrapper carrying the source location of a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    /// The underlying error kind.
    pub error_type: AbiError,
    /// Operator-facing detail about what failed.
    pub message: String,
    /// The source file that raised the error.
    pub source: String,
    /// The line within `source`.
    pub line: u32,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} ({}:{})",
            self.error_type, self.message, self.source, self.line
        )
    }
}

impl std::error::Error for ErrorContext {}

/// Build an [`ErrorContext`] at the current source location.
#[macro_export]
macro_rules! error_context {
    ($kind:expr, $($arg:tt)*) => {
        $crate::errors::ErrorContext {
            error_type: $kind,
            message: format!($($arg)*),
            source: file!().to_string(),
            line: line!(),
        }
    };
}

/// Convenience alias for fallible framework operations.
pub type Result<T> = std::result::Result<T, AbiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_matches_zig_strings() {
        assert_eq!(AbiError::InvalidConfig.describe(), "invalid configuration");
        assert_eq!(AbiError::FileNotFound.describe(), "file not found");
        assert_eq!(AbiError::OutOfMemory.describe(), "out of memory");
        assert_eq!(AbiError::NotFound.describe(), "not found");
        assert_eq!(AbiError::Timeout.describe(), "operation timed out");
    }

    #[test]
    fn display_uses_description() {
        assert_eq!(AbiError::InvalidPath.to_string(), "invalid path");
    }

    #[test]
    fn name_roundtrips_through_from_name() {
        for name in [
            "InvalidConfig",
            "InvalidPath",
            "FileNotFound",
            "PermissionDenied",
            "OutOfMemory",
            "InternalError",
            "NotInitialized",
            "AlreadyInitialized",
            "InvalidState",
            "Timeout",
            "ConnectionFailed",
            "ParseError",
            "ValidationError",
            "UnsupportedOperation",
            "NotFound",
        ] {
            let err = AbiError::from_name(name).expect("known error name");
            assert_eq!(err.name(), name);
        }
    }

    #[test]
    fn describe_name_falls_back_for_unknown() {
        assert_eq!(AbiError::describe_name("NopeNotReal"), "unknown error");
        assert_eq!(AbiError::describe_name("NotFound"), "not found");
    }

    #[test]
    fn io_errors_map_onto_the_error_set() {
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        assert_eq!(AbiError::from(not_found), AbiError::FileNotFound);

        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert_eq!(AbiError::from(denied), AbiError::PermissionDenied);

        let refused = std::io::Error::from(std::io::ErrorKind::ConnectionRefused);
        assert_eq!(AbiError::from(refused), AbiError::ConnectionFailed);
    }

    #[test]
    fn error_context_renders_location() {
        let ctx = error_context!(AbiError::InvalidState, "scheduler already {}", "running");
        assert_eq!(ctx.error_type, AbiError::InvalidState);
        assert_eq!(ctx.message, "scheduler already running");
        assert!(
            ctx.to_string()
                .starts_with("invalid state: scheduler already running (")
        );
        assert!(ctx.line > 0);
    }
}
