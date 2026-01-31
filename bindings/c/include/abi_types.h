// abi_types.h - Type definitions for ABI C bindings
// SPDX-License-Identifier: MIT

#ifndef ABI_TYPES_H
#define ABI_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "abi_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque Handle Types
// ============================================================================

/// Opaque handle to the ABI framework instance.
typedef struct abi_framework* abi_framework_t;

/// Opaque handle to a GPU context.
typedef struct abi_gpu* abi_gpu_t;

/// Opaque handle to a vector database.
typedef struct abi_database* abi_database_t;

/// Opaque handle to an AI agent.
typedef struct abi_agent* abi_agent_t;

// ============================================================================
// Configuration Structs
// ============================================================================

/// Framework initialization options.
typedef struct {
    bool enable_ai;         ///< Enable AI features (agents, LLM, embeddings)
    bool enable_gpu;        ///< Enable GPU acceleration
    bool enable_database;   ///< Enable vector database
    bool enable_network;    ///< Enable distributed networking
    bool enable_web;        ///< Enable web/HTTP utilities
    bool enable_profiling;  ///< Enable performance profiling
} abi_options_t;

/// Vector database configuration.
typedef struct {
    const char* name;           ///< Database name (null-terminated)
    size_t dimension;           ///< Vector dimension (e.g., 384, 768, 1536)
    size_t initial_capacity;    ///< Initial capacity hint
} abi_database_config_t;

/// GPU context configuration.
typedef struct {
    int backend;        ///< Backend: 0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu
    int device_index;   ///< Device index (0 = first GPU)
    bool enable_profiling;  ///< Enable GPU profiling
} abi_gpu_config_t;

/// AI agent configuration.
typedef struct {
    const char* name;       ///< Agent name (null-terminated)
    const char* persona;    ///< Persona name or NULL for default
    float temperature;      ///< Sampling temperature (0.0-2.0)
    bool enable_history;    ///< Enable conversation history
} abi_agent_config_t;

// ============================================================================
// Result Structs
// ============================================================================

/// Vector search result.
typedef struct {
    uint64_t id;            ///< Vector ID
    float score;            ///< Similarity score (higher = more similar)
    const float* vector;    ///< Pointer to vector data (valid until next search)
    size_t vector_len;      ///< Vector dimension
} abi_search_result_t;

/// SIMD capability flags.
typedef struct {
    bool sse;       ///< SSE support
    bool sse2;      ///< SSE2 support
    bool sse3;      ///< SSE3 support
    bool ssse3;     ///< SSSE3 support
    bool sse4_1;    ///< SSE4.1 support
    bool sse4_2;    ///< SSE4.2 support
    bool avx;       ///< AVX support
    bool avx2;      ///< AVX2 support
    bool avx512f;   ///< AVX-512F support
    bool neon;      ///< ARM NEON support
} abi_simd_caps_t;

/// Version information.
typedef struct {
    int major;              ///< Major version number
    int minor;              ///< Minor version number
    int patch;              ///< Patch version number
    const char* full;       ///< Full version string (do not free)
} abi_version_t;

// ============================================================================
// Default Initializers
// ============================================================================

/// Default options with all features enabled.
#define ABI_OPTIONS_DEFAULT { true, true, true, true, true, true }

/// Default database config (384-dim, 1000 capacity).
#define ABI_DATABASE_CONFIG_DEFAULT { "default", 384, 1000 }

/// Default GPU config (auto backend, first device).
#define ABI_GPU_CONFIG_DEFAULT { 0, 0, false }

/// Default agent config (assistant, temperature 0.7).
#define ABI_AGENT_CONFIG_DEFAULT { "assistant", NULL, 0.7f, true }

#ifdef __cplusplus
}
#endif

#endif // ABI_TYPES_H
