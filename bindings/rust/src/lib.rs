//! # ABI Framework Rust Bindings
//!
//! Safe Rust bindings for the ABI high-performance AI and vector database framework.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use abi::{Framework, Config};
//!
//! fn main() -> abi::Result<()> {
//!     // Initialize framework
//!     let framework = Framework::new(Config::default())?;
//!
//!     // Vector operations
//!     let a = vec![1.0, 2.0, 3.0, 4.0];
//!     let b = vec![4.0, 3.0, 2.0, 1.0];
//!     let similarity = abi::simd::cosine_similarity(&a, &b);
//!     println!("Cosine similarity: {}", similarity);
//!
//!     // Vector database
//!     let mut db = abi::database::VectorDatabase::new("embeddings", 4)?;
//!     db.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
//!     db.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
//!
//!     let results = db.search(&[0.9, 0.1, 0.0, 0.0], 2)?;
//!     for r in results {
//!         println!("ID: {}, Score: {:.4}", r.id, r.score);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - `simd` - SIMD-accelerated vector operations (default)
//! - `database` - Vector database with HNSW indexing (default)
//! - `gpu` - GPU acceleration (CUDA, Vulkan, Metal)
//! - `ai` - AI agent system
//! - `full` - All features enabled

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod ffi;
pub mod simd;
#[cfg(feature = "database")]
pub mod database;
#[cfg(feature = "gpu")]
pub mod gpu;

use std::ffi::CStr;
use std::ptr;
use thiserror::Error;

/// ABI error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Framework initialization failed
    #[error("initialization failed")]
    InitFailed,
    /// Framework already initialized
    #[error("already initialized")]
    AlreadyInitialized,
    /// Framework not initialized
    #[error("not initialized")]
    NotInitialized,
    /// Out of memory
    #[error("out of memory")]
    OutOfMemory,
    /// Invalid argument provided
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Feature is disabled
    #[error("feature disabled: {0}")]
    FeatureDisabled(String),
    /// Operation timed out
    #[error("operation timed out")]
    Timeout,
    /// I/O error
    #[error("I/O error")]
    Io,
    /// Unknown error
    #[error("unknown error: {0}")]
    Unknown(i32),
}

/// Result type for ABI operations
pub type Result<T> = std::result::Result<T, Error>;

impl From<ffi::abi_error_t> for Error {
    fn from(code: ffi::abi_error_t) -> Self {
        match code {
            ffi::ABI_ERROR_INIT_FAILED => Error::InitFailed,
            ffi::ABI_ERROR_ALREADY_INITIALIZED => Error::AlreadyInitialized,
            ffi::ABI_ERROR_NOT_INITIALIZED => Error::NotInitialized,
            ffi::ABI_ERROR_OUT_OF_MEMORY => Error::OutOfMemory,
            ffi::ABI_ERROR_INVALID_ARGUMENT => Error::InvalidArgument("unknown".into()),
            ffi::ABI_ERROR_FEATURE_DISABLED => Error::FeatureDisabled("unknown".into()),
            ffi::ABI_ERROR_TIMEOUT => Error::Timeout,
            ffi::ABI_ERROR_IO => Error::Io,
            _ => Error::Unknown(code),
        }
    }
}

fn check_error(code: ffi::abi_error_t) -> Result<()> {
    if code == ffi::ABI_OK {
        Ok(())
    } else {
        Err(Error::from(code))
    }
}

/// Framework configuration options
#[derive(Debug, Clone)]
pub struct Config {
    /// Enable AI features
    pub enable_ai: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Enable vector database
    pub enable_database: bool,
    /// Enable network features
    pub enable_network: bool,
    /// Enable web features
    pub enable_web: bool,
    /// Enable profiling
    pub enable_profiling: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enable_ai: true,
            enable_gpu: true,
            enable_database: true,
            enable_network: true,
            enable_web: true,
            enable_profiling: true,
        }
    }
}

impl Config {
    /// Create a minimal configuration with only essential features
    pub fn minimal() -> Self {
        Self {
            enable_ai: false,
            enable_gpu: false,
            enable_database: true,
            enable_network: false,
            enable_web: false,
            enable_profiling: false,
        }
    }

    /// Create configuration for AI workloads
    pub fn for_ai() -> Self {
        Self {
            enable_ai: true,
            enable_gpu: true,
            enable_database: true,
            enable_network: false,
            enable_web: false,
            enable_profiling: true,
        }
    }

    /// Convert to FFI options struct
    fn to_ffi(&self) -> ffi::abi_options_t {
        ffi::abi_options_t {
            enable_ai: self.enable_ai,
            enable_gpu: self.enable_gpu,
            enable_database: self.enable_database,
            enable_network: self.enable_network,
            enable_web: self.enable_web,
            enable_profiling: self.enable_profiling,
        }
    }
}

/// The main ABI framework handle
///
/// This is the entry point for using the ABI framework. It manages
/// the lifecycle of the underlying native library.
pub struct Framework {
    handle: ffi::abi_framework_t,
}

impl Framework {
    /// Initialize a new framework instance with the given configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use abi::{Framework, Config};
    ///
    /// let framework = Framework::new(Config::default())?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn new(config: Config) -> Result<Self> {
        let mut handle: ffi::abi_framework_t = ptr::null_mut();
        let options = config.to_ffi();

        let code = unsafe { ffi::abi_init_with_options(&options, &mut handle) };
        check_error(code)?;

        Ok(Self { handle })
    }

    /// Initialize with default configuration
    pub fn init() -> Result<Self> {
        Self::new(Config::default())
    }

    /// Check if a feature is enabled
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use abi::Framework;
    ///
    /// let framework = Framework::init()?;
    /// if framework.is_feature_enabled("gpu") {
    ///     println!("GPU acceleration available");
    /// }
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        let c_feature = std::ffi::CString::new(feature).unwrap();
        unsafe { ffi::abi_is_feature_enabled(self.handle, c_feature.as_ptr()) }
    }

    /// Get the framework version
    pub fn version() -> &'static str {
        unsafe {
            let ptr = ffi::abi_version();
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }
}

impl Drop for Framework {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::abi_shutdown(self.handle);
            }
        }
    }
}

// Framework is safe to send between threads
unsafe impl Send for Framework {}

/// Get the ABI framework version string
pub fn version() -> &'static str {
    Framework::version()
}

/// Check if SIMD operations are available
pub fn has_simd() -> bool {
    simd::is_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.enable_ai);
        assert!(config.enable_gpu);
        assert!(config.enable_database);
    }

    #[test]
    fn test_config_minimal() {
        let config = Config::minimal();
        assert!(!config.enable_ai);
        assert!(!config.enable_gpu);
        assert!(config.enable_database);
    }

    #[test]
    fn test_error_conversion() {
        let err = Error::from(ffi::ABI_ERROR_OUT_OF_MEMORY);
        assert!(matches!(err, Error::OutOfMemory));
    }
}
