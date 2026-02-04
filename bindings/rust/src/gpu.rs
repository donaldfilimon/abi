//! GPU acceleration operations.

use crate::error::{check_error, Error, Result};
use crate::ffi::{self, AbiGpu, AbiGpuConfig};
use std::ffi::CStr;
use std::ptr;

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Auto-detect the best available backend.
    Auto = 0,
    /// NVIDIA CUDA.
    Cuda = 1,
    /// Vulkan (cross-platform).
    Vulkan = 2,
    /// Apple Metal (macOS/iOS).
    Metal = 3,
    /// WebGPU (browser/WASM).
    WebGpu = 4,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Auto
    }
}

/// GPU context for acceleration.
///
/// # Example
///
/// ```no_run
/// use abi::Gpu;
///
/// if Gpu::is_available() {
///     let gpu = Gpu::new().expect("Failed to initialize GPU");
///     println!("Using backend: {}", gpu.backend_name());
/// }
/// ```
pub struct Gpu {
    handle: AbiGpu,
}

impl Gpu {
    /// Initialize GPU context with auto-detected backend.
    pub fn new() -> Result<Self> {
        Self::with_config(GpuConfig::default())
    }

    /// Initialize GPU context with specific configuration.
    pub fn with_config(config: GpuConfig) -> Result<Self> {
        let ffi_config = AbiGpuConfig {
            backend: config.backend as i32,
            device_index: config.device_index as i32,
            enable_profiling: config.enable_profiling,
        };

        let mut handle: AbiGpu = ptr::null_mut();
        let err = unsafe { ffi::abi_gpu_init(&ffi_config, &mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle })
    }

    /// Check if any GPU backend is available.
    pub fn is_available() -> bool {
        unsafe { ffi::abi_gpu_is_available() }
    }

    /// Get the active backend name.
    pub fn backend_name(&self) -> &str {
        unsafe {
            let ptr = ffi::abi_gpu_backend_name(self.handle);
            if ptr.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::abi_gpu_shutdown(self.handle);
            }
        }
    }
}

unsafe impl Send for Gpu {}

/// GPU configuration.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Backend to use.
    pub backend: Backend,
    /// Device index (0 = first GPU).
    pub device_index: u32,
    /// Enable profiling.
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            device_index: 0,
            enable_profiling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        // This should work even without the library
        let _ = Gpu::is_available();
    }
}
