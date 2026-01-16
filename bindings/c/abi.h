/**
 * ABI Framework - C API Header
 *
 * This header provides C-compatible bindings to the ABI Framework.
 * Link against the compiled ABI library to use these functions.
 *
 * Version: 0.1.0
 * Zig Version: 0.16.x
 */

#ifndef ABI_H
#define ABI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version Information
 * ============================================================================ */

/** ABI Framework version */
#define ABI_VERSION_MAJOR 0
#define ABI_VERSION_MINOR 1
#define ABI_VERSION_PATCH 0
#define ABI_VERSION_STRING "0.1.0"

/* ============================================================================
 * Error Codes
 * ============================================================================ */

typedef enum {
    ABI_OK = 0,
    ABI_ERROR_INIT_FAILED = -1,
    ABI_ERROR_ALREADY_INITIALIZED = -2,
    ABI_ERROR_NOT_INITIALIZED = -3,
    ABI_ERROR_OUT_OF_MEMORY = -4,
    ABI_ERROR_INVALID_ARGUMENT = -5,
    ABI_ERROR_FEATURE_DISABLED = -6,
    ABI_ERROR_TIMEOUT = -7,
    ABI_ERROR_IO = -8,
    ABI_ERROR_UNKNOWN = -99
} abi_error_t;

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

/** Opaque handle to the ABI framework instance */
typedef struct abi_framework* abi_framework_t;

/** Opaque handle to a compute engine */
typedef struct abi_engine* abi_engine_t;

/** Opaque handle to a database instance */
typedef struct abi_database* abi_database_t;

/** Opaque handle to a GPU context */
typedef struct abi_gpu* abi_gpu_t;

/** Task identifier */
typedef uint64_t abi_task_id_t;

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */

/** Framework initialization options */
typedef struct {
    bool enable_ai;
    bool enable_gpu;
    bool enable_database;
    bool enable_network;
    bool enable_web;
    bool enable_profiling;
} abi_options_t;

/** Default options initializer */
#define ABI_OPTIONS_DEFAULT { \
    .enable_ai = true, \
    .enable_gpu = true, \
    .enable_database = true, \
    .enable_network = true, \
    .enable_web = true, \
    .enable_profiling = true \
}

/** Database configuration */
typedef struct {
    const char* name;
    size_t dimension;
    size_t initial_capacity;
} abi_database_config_t;

/** Search result */
typedef struct {
    uint64_t id;
    float score;
    const float* vector;
    size_t vector_len;
} abi_search_result_t;

/** SIMD capabilities */
typedef struct {
    size_t vector_size;
    bool has_simd;
    int arch;  /* 0=generic, 1=x86_64, 2=aarch64, 3=wasm */
} abi_simd_caps_t;

/* ============================================================================
 * Framework Lifecycle
 * ============================================================================ */

/**
 * Initialize the ABI framework with default options.
 * @param out_framework Pointer to receive framework handle
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_init(abi_framework_t* out_framework);

/**
 * Initialize the ABI framework with custom options.
 * @param options Configuration options
 * @param out_framework Pointer to receive framework handle
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_init_with_options(const abi_options_t* options, abi_framework_t* out_framework);

/**
 * Shutdown the framework and release all resources.
 * @param framework Framework handle to shutdown
 */
void abi_shutdown(abi_framework_t framework);

/**
 * Get the framework version string.
 * @return Null-terminated version string (do not free)
 */
const char* abi_version(void);

/**
 * Check if a feature is enabled.
 * @param framework Framework handle
 * @param feature Feature name ("ai", "gpu", "database", "network", "web", "monitoring")
 * @return true if enabled, false otherwise
 */
bool abi_is_feature_enabled(abi_framework_t framework, const char* feature);

/* ============================================================================
 * SIMD Operations
 * ============================================================================ */

/**
 * Get SIMD capabilities for the current platform.
 * @param out_caps Pointer to receive capabilities
 */
void abi_simd_get_caps(abi_simd_caps_t* out_caps);

/**
 * Check if SIMD is available.
 * @return true if SIMD is available
 */
bool abi_simd_available(void);

/**
 * Vector addition: result[i] = a[i] + b[i]
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector (must be pre-allocated)
 * @param len Length of all vectors
 */
void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);

/**
 * Vector dot product: sum(a[i] * b[i])
 * @param a First input vector
 * @param b Second input vector
 * @param len Length of vectors
 * @return Dot product result
 */
float abi_simd_vector_dot(const float* a, const float* b, size_t len);

/**
 * Vector L2 norm: sqrt(sum(v[i]^2))
 * @param v Input vector
 * @param len Length of vector
 * @return L2 norm
 */
float abi_simd_vector_l2_norm(const float* v, size_t len);

/**
 * Cosine similarity between two vectors.
 * @param a First vector
 * @param b Second vector
 * @param len Length of vectors
 * @return Cosine similarity in range [-1, 1]
 */
float abi_simd_cosine_similarity(const float* a, const float* b, size_t len);

/**
 * Matrix multiplication: result = a * b
 * @param a Matrix A (m x k, row-major)
 * @param b Matrix B (k x n, row-major)
 * @param result Output matrix (m x n, row-major, must be pre-allocated)
 * @param m Rows in A
 * @param n Columns in B
 * @param k Columns in A / Rows in B
 */
void abi_simd_matrix_multiply(const float* a, const float* b, float* result,
                               size_t m, size_t n, size_t k);

/* ============================================================================
 * Compute Engine
 * ============================================================================ */

/**
 * Create a compute engine with default configuration.
 * @param out_engine Pointer to receive engine handle
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_engine_create(abi_engine_t* out_engine);

/**
 * Destroy a compute engine and release resources.
 * @param engine Engine handle to destroy
 */
void abi_engine_destroy(abi_engine_t engine);

/* ============================================================================
 * Database Operations (when enabled)
 * ============================================================================ */

/**
 * Create a new vector database.
 * @param config Database configuration
 * @param out_db Pointer to receive database handle
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_database_create(const abi_database_config_t* config, abi_database_t* out_db);

/**
 * Close a database and release resources.
 * @param db Database handle
 */
void abi_database_close(abi_database_t db);

/**
 * Insert a vector into the database.
 * @param db Database handle
 * @param id Vector identifier
 * @param vector Vector data
 * @param vector_len Vector dimension
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_database_insert(abi_database_t db, uint64_t id,
                                 const float* vector, size_t vector_len);

/**
 * Search for similar vectors.
 * @param db Database handle
 * @param query Query vector
 * @param query_len Query vector dimension
 * @param k Number of results to return
 * @param out_results Array to receive results (must have space for k results)
 * @param out_count Pointer to receive actual number of results
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_database_search(abi_database_t db,
                                 const float* query, size_t query_len,
                                 size_t k,
                                 abi_search_result_t* out_results,
                                 size_t* out_count);

/**
 * Delete a vector from the database.
 * @param db Database handle
 * @param id Vector identifier to delete
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_database_delete(abi_database_t db, uint64_t id);

/**
 * Get number of vectors in database.
 * @param db Database handle
 * @return Number of vectors
 */
size_t abi_database_count(abi_database_t db);

/* ============================================================================
 * GPU Operations (when enabled)
 * ============================================================================ */

/**
 * Initialize GPU subsystem.
 * @param out_gpu Pointer to receive GPU handle
 * @return ABI_OK on success, error code on failure
 */
abi_error_t abi_gpu_init(abi_gpu_t* out_gpu);

/**
 * Shutdown GPU subsystem.
 * @param gpu GPU handle
 */
void abi_gpu_shutdown(abi_gpu_t gpu);

/**
 * Get number of available GPU backends.
 * @param gpu GPU handle
 * @return Number of backends
 */
size_t abi_gpu_backend_count(abi_gpu_t gpu);

/**
 * Get GPU backend name by index.
 * @param gpu GPU handle
 * @param index Backend index
 * @return Backend name (do not free)
 */
const char* abi_gpu_backend_name(abi_gpu_t gpu, size_t index);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Allocate memory through ABI's allocator.
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 */
void* abi_alloc(size_t size);

/**
 * Free memory allocated by abi_alloc.
 * @param ptr Pointer to free
 * @param size Size of allocation
 */
void abi_free(void* ptr, size_t size);

/**
 * Duplicate a string using ABI's allocator.
 * @param str String to duplicate
 * @return Duplicated string (caller must free with abi_free)
 */
char* abi_strdup(const char* str);

#ifdef __cplusplus
}
#endif

#endif /* ABI_H */
