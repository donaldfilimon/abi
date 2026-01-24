//! Raw FFI bindings to the ABI C library
//!
//! These are low-level bindings. Prefer using the safe wrappers in the parent module.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use libc::{c_char, c_int, c_void, size_t};

// Error codes
pub type abi_error_t = c_int;

pub const ABI_OK: abi_error_t = 0;
pub const ABI_ERROR_INIT_FAILED: abi_error_t = -1;
pub const ABI_ERROR_ALREADY_INITIALIZED: abi_error_t = -2;
pub const ABI_ERROR_NOT_INITIALIZED: abi_error_t = -3;
pub const ABI_ERROR_OUT_OF_MEMORY: abi_error_t = -4;
pub const ABI_ERROR_INVALID_ARGUMENT: abi_error_t = -5;
pub const ABI_ERROR_FEATURE_DISABLED: abi_error_t = -6;
pub const ABI_ERROR_TIMEOUT: abi_error_t = -7;
pub const ABI_ERROR_IO: abi_error_t = -8;
pub const ABI_ERROR_UNKNOWN: abi_error_t = -99;

// Opaque handle types
pub type abi_framework_t = *mut c_void;
pub type abi_engine_t = *mut c_void;
pub type abi_database_t = *mut c_void;
pub type abi_gpu_t = *mut c_void;
pub type abi_task_id_t = u64;

/// Framework initialization options
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct abi_options_t {
    pub enable_ai: bool,
    pub enable_gpu: bool,
    pub enable_database: bool,
    pub enable_network: bool,
    pub enable_web: bool,
    pub enable_profiling: bool,
}

impl Default for abi_options_t {
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

/// Database configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct abi_database_config_t {
    pub name: *const c_char,
    pub dimension: size_t,
    pub initial_capacity: size_t,
}

/// Search result
#[repr(C)]
#[derive(Debug, Clone)]
pub struct abi_search_result_t {
    pub id: u64,
    pub score: f32,
    pub vector: *const f32,
    pub vector_len: size_t,
}

/// SIMD capabilities
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct abi_simd_caps_t {
    pub vector_size: size_t,
    pub has_simd: bool,
    pub arch: c_int,
}

// GPU backend enum
pub const ABI_GPU_BACKEND_AUTO: c_int = 0;
pub const ABI_GPU_BACKEND_VULKAN: c_int = 1;
pub const ABI_GPU_BACKEND_CUDA: c_int = 2;
pub const ABI_GPU_BACKEND_METAL: c_int = 3;
pub const ABI_GPU_BACKEND_WEBGPU: c_int = 4;
pub const ABI_GPU_BACKEND_OPENGL: c_int = 5;
pub const ABI_GPU_BACKEND_CPU: c_int = 6;

/// GPU configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct abi_gpu_config_t {
    pub backend: c_int,
    pub device_index: c_int,
}

/// GPU device information
#[repr(C)]
#[derive(Debug, Clone)]
pub struct abi_gpu_device_info_t {
    pub name: [c_char; 256],
    pub backend: c_int,
    pub total_memory: size_t,
    pub free_memory: size_t,
    pub compute_units: c_int,
}

#[cfg_attr(target_os = "windows", link(name = "abi"))]
#[cfg_attr(target_os = "linux", link(name = "abi"))]
#[cfg_attr(target_os = "macos", link(name = "abi"))]
extern "C" {
    // Framework lifecycle
    pub fn abi_init(out_framework: *mut abi_framework_t) -> abi_error_t;
    pub fn abi_init_with_options(
        options: *const abi_options_t,
        out_framework: *mut abi_framework_t,
    ) -> abi_error_t;
    pub fn abi_shutdown(framework: abi_framework_t);
    pub fn abi_version() -> *const c_char;
    pub fn abi_is_feature_enabled(framework: abi_framework_t, feature: *const c_char) -> bool;

    // SIMD operations
    pub fn abi_simd_get_caps(out_caps: *mut abi_simd_caps_t);
    pub fn abi_simd_available() -> bool;
    pub fn abi_simd_vector_add(a: *const f32, b: *const f32, result: *mut f32, len: size_t);
    pub fn abi_simd_vector_dot(a: *const f32, b: *const f32, len: size_t) -> f32;
    pub fn abi_simd_vector_l2_norm(v: *const f32, len: size_t) -> f32;
    pub fn abi_simd_cosine_similarity(a: *const f32, b: *const f32, len: size_t) -> f32;
    pub fn abi_simd_matrix_multiply(
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        m: size_t,
        n: size_t,
        k: size_t,
    );

    // Engine
    pub fn abi_engine_create(out_engine: *mut abi_engine_t) -> abi_error_t;
    pub fn abi_engine_destroy(engine: abi_engine_t);

    // Database
    pub fn abi_database_create(
        config: *const abi_database_config_t,
        out_db: *mut abi_database_t,
    ) -> abi_error_t;
    pub fn abi_database_close(db: abi_database_t);
    pub fn abi_database_insert(
        db: abi_database_t,
        id: u64,
        vector: *const f32,
        vector_len: size_t,
    ) -> abi_error_t;
    pub fn abi_database_search(
        db: abi_database_t,
        query: *const f32,
        query_len: size_t,
        k: size_t,
        out_results: *mut abi_search_result_t,
        out_count: *mut size_t,
    ) -> abi_error_t;
    pub fn abi_database_delete(db: abi_database_t, id: u64) -> abi_error_t;
    pub fn abi_database_count(db: abi_database_t) -> size_t;

    // GPU
    pub fn abi_gpu_init(config: *const abi_gpu_config_t, out_gpu: *mut abi_gpu_t) -> abi_error_t;
    pub fn abi_gpu_shutdown(gpu: abi_gpu_t);
    pub fn abi_gpu_is_available() -> bool;
    pub fn abi_gpu_list_devices(
        out_devices: *mut abi_gpu_device_info_t,
        max_devices: size_t,
        out_count: *mut size_t,
    ) -> abi_error_t;
    pub fn abi_gpu_matrix_multiply(
        gpu: abi_gpu_t,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        m: size_t,
        n: size_t,
        k: size_t,
    ) -> abi_error_t;
    pub fn abi_gpu_vector_add(
        gpu: abi_gpu_t,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        len: size_t,
    ) -> abi_error_t;
}

// Safe stubs for when native library isn't available (for testing/docs)
#[cfg(not(feature = "native"))]
mod stubs {
    use super::*;

    pub unsafe fn abi_version_stub() -> *const c_char {
        b"0.1.0-stub\0".as_ptr() as *const c_char
    }

    pub unsafe fn abi_simd_available_stub() -> bool {
        false
    }

    pub fn abi_simd_vector_dot_stub(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn abi_simd_cosine_similarity_stub(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    pub fn abi_simd_vector_l2_norm_stub(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_default() {
        let opts = abi_options_t::default();
        assert!(opts.enable_ai);
        assert!(opts.enable_gpu);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(ABI_OK, 0);
        assert!(ABI_ERROR_INIT_FAILED < 0);
    }
}
