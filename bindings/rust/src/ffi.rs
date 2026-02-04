//! Raw FFI bindings to the ABI C library.
//!
//! These are low-level bindings generated from the C headers.
//! Users should prefer the safe wrappers in the parent module.

use libc::{c_char, c_float, c_int, size_t};

/// Opaque handle to the ABI framework instance.
pub type AbiFramework = *mut std::ffi::c_void;

/// Opaque handle to a GPU context.
pub type AbiGpu = *mut std::ffi::c_void;

/// Opaque handle to a vector database.
pub type AbiDatabase = *mut std::ffi::c_void;

/// Opaque handle to an AI agent.
pub type AbiAgent = *mut std::ffi::c_void;

/// Error codes returned by ABI functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiError {
    Ok = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    AlreadyInitialized = 4,
    FeatureDisabled = 5,
    IoError = 6,
    NetworkError = 7,
    GpuError = 8,
    DatabaseError = 9,
    AgentError = 10,
    Unknown = 255,
}

impl From<i32> for AbiError {
    fn from(code: i32) -> Self {
        match code {
            0 => AbiError::Ok,
            1 => AbiError::InvalidArgument,
            2 => AbiError::OutOfMemory,
            3 => AbiError::NotInitialized,
            4 => AbiError::AlreadyInitialized,
            5 => AbiError::FeatureDisabled,
            6 => AbiError::IoError,
            7 => AbiError::NetworkError,
            8 => AbiError::GpuError,
            9 => AbiError::DatabaseError,
            10 => AbiError::AgentError,
            _ => AbiError::Unknown,
        }
    }
}

/// Framework initialization options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiOptions {
    pub enable_ai: bool,
    pub enable_gpu: bool,
    pub enable_database: bool,
    pub enable_network: bool,
    pub enable_web: bool,
    pub enable_profiling: bool,
}

impl Default for AbiOptions {
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

/// Vector database configuration.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AbiDatabaseConfig {
    pub name: *const c_char,
    pub dimension: size_t,
    pub initial_capacity: size_t,
}

/// GPU context configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiGpuConfig {
    pub backend: c_int,
    pub device_index: c_int,
    pub enable_profiling: bool,
}

impl Default for AbiGpuConfig {
    fn default() -> Self {
        Self {
            backend: 0, // auto
            device_index: 0,
            enable_profiling: false,
        }
    }
}

/// AI agent configuration.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AbiAgentConfig {
    pub name: *const c_char,
    pub persona: *const c_char,
    pub temperature: c_float,
    pub enable_history: bool,
}

/// Vector search result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiSearchResult {
    pub id: u64,
    pub score: c_float,
    pub vector: *const c_float,
    pub vector_len: size_t,
}

/// SIMD capability flags.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AbiSimdCaps {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

/// Version information.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiVersion {
    pub major: c_int,
    pub minor: c_int,
    pub patch: c_int,
    pub full: *const c_char,
}

// External C functions
#[link(name = "abi")]
extern "C" {
    // Framework lifecycle
    pub fn abi_init(out_framework: *mut AbiFramework) -> c_int;
    pub fn abi_init_with_options(
        options: *const AbiOptions,
        out_framework: *mut AbiFramework,
    ) -> c_int;
    pub fn abi_shutdown(framework: AbiFramework);
    pub fn abi_version() -> *const c_char;
    pub fn abi_version_info(out_version: *mut AbiVersion);
    pub fn abi_is_feature_enabled(framework: AbiFramework, feature: *const c_char) -> bool;

    // SIMD operations
    pub fn abi_simd_get_caps(out_caps: *mut AbiSimdCaps);
    pub fn abi_simd_available() -> bool;
    pub fn abi_simd_vector_add(
        a: *const c_float,
        b: *const c_float,
        result: *mut c_float,
        len: size_t,
    );
    pub fn abi_simd_vector_dot(a: *const c_float, b: *const c_float, len: size_t) -> c_float;
    pub fn abi_simd_vector_l2_norm(v: *const c_float, len: size_t) -> c_float;
    pub fn abi_simd_cosine_similarity(
        a: *const c_float,
        b: *const c_float,
        len: size_t,
    ) -> c_float;

    // Database operations
    pub fn abi_database_create(
        config: *const AbiDatabaseConfig,
        out_db: *mut AbiDatabase,
    ) -> c_int;
    pub fn abi_database_close(db: AbiDatabase);
    pub fn abi_database_insert(
        db: AbiDatabase,
        id: u64,
        vector: *const c_float,
        vector_len: size_t,
    ) -> c_int;
    pub fn abi_database_search(
        db: AbiDatabase,
        query: *const c_float,
        query_len: size_t,
        k: size_t,
        out_results: *mut AbiSearchResult,
        out_count: *mut size_t,
    ) -> c_int;
    pub fn abi_database_delete(db: AbiDatabase, id: u64) -> c_int;
    pub fn abi_database_count(db: AbiDatabase, out_count: *mut size_t) -> c_int;

    // GPU operations
    pub fn abi_gpu_init(config: *const AbiGpuConfig, out_gpu: *mut AbiGpu) -> c_int;
    pub fn abi_gpu_shutdown(gpu: AbiGpu);
    pub fn abi_gpu_is_available() -> bool;
    pub fn abi_gpu_backend_name(gpu: AbiGpu) -> *const c_char;

    // Agent operations
    pub fn abi_agent_create(
        framework: AbiFramework,
        config: *const AbiAgentConfig,
        out_agent: *mut AbiAgent,
    ) -> c_int;
    pub fn abi_agent_destroy(agent: AbiAgent);
    pub fn abi_agent_chat(
        agent: AbiAgent,
        message: *const c_char,
        out_response: *mut *mut c_char,
    ) -> c_int;
    pub fn abi_agent_clear_history(agent: AbiAgent) -> c_int;

    // Memory management
    pub fn abi_free_string(str: *mut c_char);
    pub fn abi_free_results(results: *mut AbiSearchResult, count: size_t);
}
