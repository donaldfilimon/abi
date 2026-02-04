//! ABI Framework wrapper.

use crate::error::{check_error, Error, Result};
use crate::ffi::{self, AbiFramework, AbiOptions, AbiVersion};
use std::ffi::{CStr, CString};
use std::ptr;

/// ABI Framework instance.
///
/// This is the main entry point for using the ABI framework.
/// It must be initialized before using other ABI features.
///
/// # Example
///
/// ```no_run
/// use abi::Framework;
///
/// let framework = Framework::new().expect("Failed to initialize");
/// println!("ABI version: {}", framework.version());
/// ```
pub struct Framework {
    handle: AbiFramework,
}

impl Framework {
    /// Initialize the ABI framework with default options.
    pub fn new() -> Result<Self> {
        let mut handle: AbiFramework = ptr::null_mut();
        let err = unsafe { ffi::abi_init(&mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle })
    }

    /// Initialize the ABI framework with custom options.
    pub fn with_options(options: &Options) -> Result<Self> {
        let mut handle: AbiFramework = ptr::null_mut();
        let abi_options = options.to_ffi();
        let err = unsafe { ffi::abi_init_with_options(&abi_options, &mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle })
    }

    /// Get the framework handle for use with other ABI types.
    pub(crate) fn handle(&self) -> AbiFramework {
        self.handle
    }

    /// Get the ABI version string.
    pub fn version(&self) -> &'static str {
        unsafe {
            let ptr = ffi::abi_version();
            if ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
            }
        }
    }

    /// Get detailed version information.
    pub fn version_info(&self) -> VersionInfo {
        let mut ver = AbiVersion {
            major: 0,
            minor: 0,
            patch: 0,
            full: ptr::null(),
        };
        unsafe {
            ffi::abi_version_info(&mut ver);
        }

        VersionInfo {
            major: ver.major,
            minor: ver.minor,
            patch: ver.patch,
            full: unsafe {
                if ver.full.is_null() {
                    "unknown".to_string()
                } else {
                    CStr::from_ptr(ver.full)
                        .to_str()
                        .unwrap_or("unknown")
                        .to_string()
                }
            },
        }
    }

    /// Check if a feature is enabled.
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        let c_feature = CString::new(feature).unwrap_or_default();
        unsafe { ffi::abi_is_feature_enabled(self.handle, c_feature.as_ptr()) }
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

// Framework is not thread-safe by default
// Wrap in Arc<Mutex<Framework>> for multi-threaded use
unsafe impl Send for Framework {}

/// Framework initialization options.
#[derive(Debug, Clone)]
pub struct Options {
    /// Enable AI features (agents, LLM, embeddings).
    pub enable_ai: bool,
    /// Enable GPU acceleration.
    pub enable_gpu: bool,
    /// Enable vector database.
    pub enable_database: bool,
    /// Enable distributed networking.
    pub enable_network: bool,
    /// Enable web/HTTP utilities.
    pub enable_web: bool,
    /// Enable performance profiling.
    pub enable_profiling: bool,
}

impl Default for Options {
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

impl Options {
    /// Create a minimal options with only core features.
    pub fn minimal() -> Self {
        Self {
            enable_ai: false,
            enable_gpu: false,
            enable_database: false,
            enable_network: false,
            enable_web: false,
            enable_profiling: false,
        }
    }

    fn to_ffi(&self) -> AbiOptions {
        AbiOptions {
            enable_ai: self.enable_ai,
            enable_gpu: self.enable_gpu,
            enable_database: self.enable_database,
            enable_network: self.enable_network,
            enable_web: self.enable_web,
            enable_profiling: self.enable_profiling,
        }
    }
}

/// Detailed version information.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Major version number.
    pub major: i32,
    /// Minor version number.
    pub minor: i32,
    /// Patch version number.
    pub patch: i32,
    /// Full version string.
    pub full: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_init() {
        // Skip if library not available
        let result = Framework::new();
        if result.is_err() {
            eprintln!("Skipping test: library not available");
            return;
        }

        let framework = result.unwrap();
        let version = framework.version();
        assert!(!version.is_empty());
    }
}
