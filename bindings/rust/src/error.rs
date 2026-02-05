//! Error types for ABI Rust bindings.

use crate::ffi::AbiError;
use thiserror::Error;

/// Result type for ABI operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using the ABI framework.
#[derive(Debug, Error)]
pub enum Error {
    /// Initialization failed.
    #[error("Initialization failed")]
    InitFailed,

    /// Invalid argument passed to a function.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Out of memory.
    #[error("Out of memory")]
    OutOfMemory,

    /// Framework or component not initialized.
    #[error("Not initialized")]
    NotInitialized,

    /// Framework or component already initialized.
    #[error("Already initialized")]
    AlreadyInitialized,

    /// Requested feature is disabled.
    #[error("Feature disabled")]
    FeatureDisabled,

    /// Operation timed out.
    #[error("Operation timed out")]
    Timeout,

    /// I/O error occurred.
    #[error("I/O error")]
    IoError,

    /// GPU not available.
    #[error("GPU unavailable")]
    GpuUnavailable,

    /// Database error occurred.
    #[error("Database error")]
    DatabaseError,

    /// Network error occurred.
    #[error("Network error")]
    NetworkError,

    /// AI operation error.
    #[error("AI error")]
    AiError,

    /// Unknown error from the C library.
    #[error("Unknown error: code {0}")]
    Unknown(i32),

    /// Null pointer returned from C library.
    #[error("Null pointer returned")]
    NullPointer,

    /// Invalid UTF-8 string from C library.
    #[error("Invalid UTF-8 string")]
    InvalidUtf8,
}

impl From<AbiError> for Error {
    fn from(err: AbiError) -> Self {
        match err {
            AbiError::Ok => panic!("Cannot convert Ok to Error"),
            AbiError::InitFailed => Error::InitFailed,
            AbiError::AlreadyInitialized => Error::AlreadyInitialized,
            AbiError::NotInitialized => Error::NotInitialized,
            AbiError::OutOfMemory => Error::OutOfMemory,
            AbiError::InvalidArgument => Error::InvalidArgument(String::new()),
            AbiError::FeatureDisabled => Error::FeatureDisabled,
            AbiError::Timeout => Error::Timeout,
            AbiError::IoError => Error::IoError,
            AbiError::GpuUnavailable => Error::GpuUnavailable,
            AbiError::DatabaseError => Error::DatabaseError,
            AbiError::NetworkError => Error::NetworkError,
            AbiError::AiError => Error::AiError,
            AbiError::Unknown => Error::Unknown(-99),
        }
    }
}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        AbiError::from(code).into()
    }
}

/// Check a C error code and convert to Result.
pub(crate) fn check_error(code: i32) -> Result<()> {
    let err = AbiError::from(code);
    if err == AbiError::Ok {
        Ok(())
    } else {
        Err(err.into())
    }
}
