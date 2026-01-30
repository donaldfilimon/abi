// abi.h - Main header for ABI C bindings
// SPDX-License-Identifier: MIT
//
// ABI Framework C Bindings
// ========================
// C-compatible interface to the ABI framework for use from C, C++,
// and other languages via FFI.
//
// Example:
//   abi_framework_t fw = NULL;
//   if (abi_init(&fw) == ABI_OK) {
//       printf("ABI v%s\n", abi_version());
//       abi_shutdown(fw);
//   }

#ifndef ABI_H
#define ABI_H

#include "abi_types.h"
#include "abi_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Framework Lifecycle
// ============================================================================

/// Initialize the ABI framework with default options.
/// @param out_framework Pointer to receive the framework handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_init(abi_framework_t* out_framework);

/// Initialize the ABI framework with custom options.
/// @param options Configuration options
/// @param out_framework Pointer to receive the framework handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_init_with_options(const abi_options_t* options,
                                   abi_framework_t* out_framework);

/// Shutdown the framework and release all resources.
/// @param framework Framework handle (may be NULL)
void abi_shutdown(abi_framework_t framework);

/// Get the version string.
/// @return Static version string (do not free)
const char* abi_version(void);

/// Get detailed version information.
/// @param out_version Pointer to receive version info
void abi_version_info(abi_version_t* out_version);

/// Check if a feature is enabled.
/// @param framework Framework handle
/// @param feature Feature name ("ai", "gpu", "database", "network", "web")
/// @return true if enabled, false otherwise
bool abi_is_feature_enabled(abi_framework_t framework, const char* feature);

// ============================================================================
// SIMD Operations
// ============================================================================

/// Query CPU SIMD capabilities.
/// @param out_caps Pointer to receive capability flags
void abi_simd_get_caps(abi_simd_caps_t* out_caps);

/// Check if any SIMD instruction set is available.
/// @return true if SSE or NEON is available
bool abi_simd_available(void);

/// Vector element-wise addition: result[i] = a[i] + b[i]
/// @param a First input vector
/// @param b Second input vector
/// @param result Output vector (must be pre-allocated)
/// @param len Vector length
void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);

/// Vector dot product: sum(a[i] * b[i])
/// @param a First input vector
/// @param b Second input vector
/// @param len Vector length
/// @return Dot product result
float abi_simd_vector_dot(const float* a, const float* b, size_t len);

/// Vector L2 norm: sqrt(sum(v[i]^2))
/// @param v Input vector
/// @param len Vector length
/// @return L2 norm
float abi_simd_vector_l2_norm(const float* v, size_t len);

/// Cosine similarity between two vectors.
/// @param a First input vector
/// @param b Second input vector
/// @param len Vector length
/// @return Cosine similarity (-1.0 to 1.0)
float abi_simd_cosine_similarity(const float* a, const float* b, size_t len);

// ============================================================================
// Database Operations
// ============================================================================

/// Create a new vector database.
/// @param config Database configuration
/// @param out_db Pointer to receive database handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_database_create(const abi_database_config_t* config,
                                 abi_database_t* out_db);

/// Close a database and release resources.
/// @param db Database handle (may be NULL)
void abi_database_close(abi_database_t db);

/// Insert a vector into the database.
/// @param db Database handle
/// @param id Unique vector ID
/// @param vector Vector data
/// @param vector_len Vector dimension (must match database dimension)
/// @return ABI_OK on success, error code on failure
abi_error_t abi_database_insert(abi_database_t db, uint64_t id,
                                 const float* vector, size_t vector_len);

/// Search for similar vectors.
/// @param db Database handle
/// @param query Query vector
/// @param query_len Query dimension (must match database dimension)
/// @param k Maximum number of results
/// @param out_results Pre-allocated array of at least k results
/// @param out_count Receives actual number of results
/// @return ABI_OK on success, error code on failure
abi_error_t abi_database_search(abi_database_t db, const float* query,
                                 size_t query_len, size_t k,
                                 abi_search_result_t* out_results,
                                 size_t* out_count);

/// Delete a vector from the database.
/// @param db Database handle
/// @param id Vector ID to delete
/// @return ABI_OK on success, error code on failure
abi_error_t abi_database_delete(abi_database_t db, uint64_t id);

/// Get the number of vectors in the database.
/// @param db Database handle
/// @param out_count Receives vector count
/// @return ABI_OK on success, error code on failure
abi_error_t abi_database_count(abi_database_t db, size_t* out_count);

// ============================================================================
// GPU Operations
// ============================================================================

/// Initialize a GPU context.
/// @param config GPU configuration
/// @param out_gpu Pointer to receive GPU handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_gpu_init(const abi_gpu_config_t* config, abi_gpu_t* out_gpu);

/// Shutdown GPU context and release resources.
/// @param gpu GPU handle (may be NULL)
void abi_gpu_shutdown(abi_gpu_t gpu);

/// Check if any GPU backend is available.
/// @return true if at least one GPU backend can be initialized
bool abi_gpu_is_available(void);

/// Get the active GPU backend name.
/// @param gpu GPU handle
/// @return Static backend name string (do not free)
const char* abi_gpu_backend_name(abi_gpu_t gpu);

// ============================================================================
// Agent Operations
// ============================================================================

/// Create an AI agent.
/// @param framework Framework handle (must have AI enabled)
/// @param config Agent configuration
/// @param out_agent Pointer to receive agent handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_agent_create(abi_framework_t framework,
                              const abi_agent_config_t* config,
                              abi_agent_t* out_agent);

/// Destroy an agent and release resources.
/// @param agent Agent handle (may be NULL)
void abi_agent_destroy(abi_agent_t agent);

/// Send a message to the agent and get a response.
/// @param agent Agent handle
/// @param message Input message (null-terminated)
/// @param out_response Pointer to receive response string
/// @return ABI_OK on success, error code on failure
/// @note Caller must free response with abi_free_string()
abi_error_t abi_agent_chat(abi_agent_t agent, const char* message,
                            char** out_response);

/// Clear the agent's conversation history.
/// @param agent Agent handle
/// @return ABI_OK on success, error code on failure
abi_error_t abi_agent_clear_history(abi_agent_t agent);

// ============================================================================
// Memory Management
// ============================================================================

/// Free a string allocated by ABI functions.
/// @param str String to free (may be NULL)
void abi_free_string(char* str);

/// Free a search results array.
/// @param results Results array (may be NULL)
/// @param count Number of results
void abi_free_results(abi_search_result_t* results, size_t count);

#ifdef __cplusplus
}
#endif

#endif // ABI_H
