//! Error types for ABI Rust bindings.

use crate::ffi::AbiError;
use thiserror::Error;

/// Result type for ABI operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using the ABI framework.
#[derive(Debug, Error)]
pub enum Error {
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
    #[error("Feature disabled: {0}")]
    FeatureDisabled(String),

    /// I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(String),

    /// Network error occurred.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// GPU error occurred.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Database error occurred.
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Agent error occurred.
    #[error("Agent error: {0}")]
    AgentError(String),

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
            AbiError::InvalidArgument => Error::InvalidArgument(String::new()),
            AbiError::OutOfMemory => Error::OutOfMemory,
            AbiError::NotInitialized => Error::NotInitialized,
            AbiError::AlreadyInitialized => Error::AlreadyInitialized,
            AbiError::FeatureDisabled => Error::FeatureDisabled(String::new()),
            AbiError::IoError => Error::IoError(String::new()),
            AbiError::NetworkError => Error::NetworkError(String::new()),
            AbiError::GpuError => Error::GpuError(String::new()),
            AbiError::DatabaseError => Error::DatabaseError(String::new()),
            AbiError::AgentError => Error::AgentError(String::new()),
            AbiError::Unknown => Error::Unknown(255),
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
