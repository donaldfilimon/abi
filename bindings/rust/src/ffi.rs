//! Raw FFI bindings to the ABI C library.
//!
//! These are low-level bindings matching the C headers (abi.h).
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

/// Error codes returned by ABI functions (negative integers).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiError {
    Ok = 0,
    InitFailed = -1,
    AlreadyInitialized = -2,
    NotInitialized = -3,
    OutOfMemory = -4,
    InvalidArgument = -5,
    FeatureDisabled = -6,
    Timeout = -7,
    IoError = -8,
    GpuUnavailable = -9,
    DatabaseError = -10,
    NetworkError = -11,
    AiError = -12,
    Unknown = -99,
}

impl From<i32> for AbiError {
    fn from(code: i32) -> Self {
        match code {
            0 => AbiError::Ok,
            -1 => AbiError::InitFailed,
            -2 => AbiError::AlreadyInitialized,
            -3 => AbiError::NotInitialized,
            -4 => AbiError::OutOfMemory,
            -5 => AbiError::InvalidArgument,
            -6 => AbiError::FeatureDisabled,
            -7 => AbiError::Timeout,
            -8 => AbiError::IoError,
            -9 => AbiError::GpuUnavailable,
            -10 => AbiError::DatabaseError,
            -11 => AbiError::NetworkError,
            -12 => AbiError::AiError,
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
    /// Agent name (null-terminated).
    pub name: *const c_char,
    /// Backend type (0=echo, 1=openai, 2=ollama, 3=huggingface, 4=local).
    pub backend: c_int,
    /// Model name (e.g., "gpt-4").
    pub model: *const c_char,
    /// System prompt (optional, NULL for none).
    pub system_prompt: *const c_char,
    /// Temperature (0.0 - 2.0).
    pub temperature: c_float,
    /// Top-p (0.0 - 1.0).
    pub top_p: c_float,
    /// Maximum generation tokens.
    pub max_tokens: u32,
    /// Enable conversation history.
    pub enable_history: bool,
}

/// Agent response from a send operation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiAgentResponse {
    /// Response text (null-terminated, valid until next send or destroy).
    pub text: *const c_char,
    /// Length of response text.
    pub length: size_t,
    /// Number of tokens used.
    pub tokens_used: u64,
}

/// Agent conversation statistics.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiAgentStats {
    pub history_length: size_t,
    pub user_messages: size_t,
    pub assistant_messages: size_t,
    pub total_characters: size_t,
    pub total_tokens_used: u64,
}

/// Vector search result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiSearchResult {
    pub id: u64,
    pub score: c_float,
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
    // Error handling
    pub fn abi_error_string(err: c_int) -> *const c_char;

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
        metadata: *const c_char,
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
        config: *const AbiAgentConfig,
        out_agent: *mut AbiAgent,
    ) -> c_int;
    pub fn abi_agent_destroy(agent: AbiAgent);
    pub fn abi_agent_send(
        agent: AbiAgent,
        message: *const c_char,
        out_response: *mut AbiAgentResponse,
    ) -> c_int;
    pub fn abi_agent_get_status(agent: AbiAgent) -> c_int;
    pub fn abi_agent_get_stats(agent: AbiAgent, out_stats: *mut AbiAgentStats) -> c_int;
    pub fn abi_agent_clear_history(agent: AbiAgent) -> c_int;
    pub fn abi_agent_set_temperature(agent: AbiAgent, temperature: c_float) -> c_int;
    pub fn abi_agent_set_max_tokens(agent: AbiAgent, max_tokens: u32) -> c_int;
    pub fn abi_agent_get_name(agent: AbiAgent) -> *const c_char;

    // Memory management
    pub fn abi_free_string(str: *mut c_char);
    pub fn abi_free_results(results: *mut AbiSearchResult, count: size_t);
}
