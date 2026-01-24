//! GPU acceleration module
//!
//! This module provides GPU-accelerated operations for matrix and vector
//! computations. Multiple backends are supported including CUDA, Vulkan, and Metal.
//!
//! # Example
//!
//! ```rust,no_run
//! use abi::gpu::{GpuContext, Backend};
//!
//! fn main() -> abi::Result<()> {
//!     // Check if GPU is available
//!     if !abi::gpu::is_available() {
//!         println!("No GPU available, using CPU fallback");
//!         return Ok(());
//!     }
//!
//!     // List available devices
//!     for device in abi::gpu::list_devices()? {
//!         println!("Found GPU: {} ({:?})", device.name, device.backend);
//!     }
//!
//!     // Create GPU context with auto-detected backend
//!     let gpu = GpuContext::new(Backend::Auto)?;
//!
//!     // Perform GPU-accelerated matrix multiplication
//!     let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
//!     let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
//!     let result = gpu.matrix_multiply(&a, &b, 2, 2, 2)?;
//!
//!     Ok(())
//! }
//! ```

use std::ffi::CStr;
use std::ptr;

use crate::ffi;
use crate::{check_error, Error, Result};

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Auto-detect best available backend
    Auto,
    /// Vulkan compute
    Vulkan,
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
    /// WebGPU
    WebGpu,
    /// OpenGL compute
    OpenGl,
    /// CPU fallback
    Cpu,
}

impl From<i32> for Backend {
    fn from(value: i32) -> Self {
        match value {
            1 => Backend::Vulkan,
            2 => Backend::Cuda,
            3 => Backend::Metal,
            4 => Backend::WebGpu,
            5 => Backend::OpenGl,
            6 => Backend::Cpu,
            _ => Backend::Auto,
        }
    }
}

impl From<Backend> for i32 {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Auto => 0,
            Backend::Vulkan => 1,
            Backend::Cuda => 2,
            Backend::Metal => 3,
            Backend::WebGpu => 4,
            Backend::OpenGl => 5,
            Backend::Cpu => 6,
        }
    }
}

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Backend used
    pub backend: Backend,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Number of compute units
    pub compute_units: i32,
}

/// Configuration for GPU context
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Backend to use
    pub backend: Backend,
    /// Device index (0 for first/default)
    pub device_index: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            device_index: 0,
        }
    }
}

impl Config {
    /// Create config for a specific backend
    pub fn with_backend(backend: Backend) -> Self {
        Self {
            backend,
            device_index: 0,
        }
    }

    /// Set device index
    pub fn device(mut self, index: i32) -> Self {
        self.device_index = index;
        self
    }
}

/// Check if any GPU is available
pub fn is_available() -> bool {
    unsafe { ffi::abi_gpu_is_available() }
}

/// List all available GPU devices
pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let mut devices: [ffi::abi_gpu_device_info_t; 16] = unsafe { std::mem::zeroed() };
    let mut count: usize = 0;

    let code = unsafe {
        ffi::abi_gpu_list_devices(devices.as_mut_ptr(), devices.len(), &mut count)
    };
    check_error(code)?;

    Ok(devices[..count]
        .iter()
        .map(|d| {
            let name = unsafe {
                CStr::from_ptr(d.name.as_ptr())
                    .to_str()
                    .unwrap_or("unknown")
                    .to_string()
            };
            DeviceInfo {
                name,
                backend: Backend::from(d.backend),
                total_memory: d.total_memory,
                free_memory: d.free_memory,
                compute_units: d.compute_units,
            }
        })
        .collect())
}

/// GPU context for performing accelerated computations
pub struct GpuContext {
    handle: ffi::abi_gpu_t,
}

impl GpuContext {
    /// Create a new GPU context with the specified backend
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let gpu = abi::gpu::GpuContext::new(abi::gpu::Backend::Auto)?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn new(backend: Backend) -> Result<Self> {
        Self::with_config(Config::with_backend(backend))
    }

    /// Create a GPU context with custom configuration
    pub fn with_config(config: Config) -> Result<Self> {
        let ffi_config = ffi::abi_gpu_config_t {
            backend: i32::from(config.backend),
            device_index: config.device_index,
        };

        let mut handle: ffi::abi_gpu_t = ptr::null_mut();
        let code = unsafe { ffi::abi_gpu_init(&ffi_config, &mut handle) };
        check_error(code)?;

        Ok(Self { handle })
    }

    /// Perform GPU-accelerated matrix multiplication
    ///
    /// Computes C = A * B where A is m x k, B is k x n, C is m x n.
    /// Matrices are in row-major order.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let gpu = abi::gpu::GpuContext::new(abi::gpu::Backend::Auto)?;
    /// let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    /// let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
    /// let result = gpu.matrix_multiply(&a, &b, 2, 2, 2)?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        if a.len() != m * k {
            return Err(Error::InvalidArgument("matrix A has wrong size".into()));
        }
        if b.len() != k * n {
            return Err(Error::InvalidArgument("matrix B has wrong size".into()));
        }

        let mut result = vec![0.0f32; m * n];

        let code = unsafe {
            ffi::abi_gpu_matrix_multiply(
                self.handle,
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                m,
                n,
                k,
            )
        };
        check_error(code)?;

        Ok(result)
    }

    /// Perform GPU-accelerated vector addition
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(Error::InvalidArgument("vectors have different lengths".into()));
        }

        let mut result = vec![0.0f32; a.len()];

        let code = unsafe {
            ffi::abi_gpu_vector_add(
                self.handle,
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                a.len(),
            )
        };
        check_error(code)?;

        Ok(result)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::abi_gpu_shutdown(self.handle);
            }
        }
    }
}

// GpuContext is safe to send between threads
unsafe impl Send for GpuContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_conversion() {
        assert_eq!(i32::from(Backend::Auto), 0);
        assert_eq!(i32::from(Backend::Cuda), 2);
        assert_eq!(Backend::from(1), Backend::Vulkan);
        assert_eq!(Backend::from(3), Backend::Metal);
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.backend, Backend::Auto);
        assert_eq!(config.device_index, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = Config::with_backend(Backend::Cuda).device(1);
        assert_eq!(config.backend, Backend::Cuda);
        assert_eq!(config.device_index, 1);
    }
}
