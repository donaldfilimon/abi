/**
 * @file abi.h
 * @brief ABI Framework C Bindings
 *
 * This header provides C-compatible FFI bindings for the ABI framework.
 * All functions use C calling conventions and return C-compatible types.
 *
 * The bindings follow a handle-based pattern where opaque pointers are
 * returned to C code for managing framework resources.
 *
 * @example Basic Usage
 * @code
 * #include <abi.h>
 *
 * int main(void) {
 *     abi_framework_t *fw = NULL;
 *     int err = abi_init(&fw);
 *     if (err != ABI_OK) {
 *         fprintf(stderr, "Init failed: %s\n", abi_error_string(err));
 *         return 1;
 *     }
 *
 *     printf("ABI version: %s\n", abi_version());
 *
 *     abi_shutdown(fw);
 *     return 0;
 * }
 * @endcode
 */

#ifndef ABI_H
#define ABI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/** Operation completed successfully. */
#define ABI_OK                        0

/** Framework initialization failed. */
#define ABI_ERROR_INIT_FAILED        -1

/** Framework is already initialized. */
#define ABI_ERROR_ALREADY_INITIALIZED -2

/** Framework is not initialized. */
#define ABI_ERROR_NOT_INITIALIZED    -3

/** Memory allocation failed. */
#define ABI_ERROR_OUT_OF_MEMORY      -4

/** Invalid argument provided. */
#define ABI_ERROR_INVALID_ARGUMENT   -5

/** Feature is disabled at compile time. */
#define ABI_ERROR_FEATURE_DISABLED   -6

/** Operation timed out. */
#define ABI_ERROR_TIMEOUT            -7

/** I/O error occurred. */
#define ABI_ERROR_IO                 -8

/** GPU is not available. */
#define ABI_ERROR_GPU_UNAVAILABLE    -9

/** Database operation error. */
#define ABI_ERROR_DATABASE_ERROR     -10

/** Network operation error. */
#define ABI_ERROR_NETWORK_ERROR      -11

/** AI operation error. */
#define ABI_ERROR_AI_ERROR           -12

/** Unknown or unspecified error. */
#define ABI_ERROR_UNKNOWN            -99

/* ============================================================================
 * GPU Backend Constants
 * ============================================================================ */

/** Auto-detect GPU backend. */
#define ABI_GPU_BACKEND_AUTO    0

/** NVIDIA CUDA backend. */
#define ABI_GPU_BACKEND_CUDA    1

/** Vulkan backend. */
#define ABI_GPU_BACKEND_VULKAN  2

/** Apple Metal backend. */
#define ABI_GPU_BACKEND_METAL   3

/** WebGPU backend. */
#define ABI_GPU_BACKEND_WEBGPU  4

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

/** Opaque framework handle. */
typedef struct abi_framework abi_framework_t;

/** Opaque database handle. */
typedef struct abi_database abi_database_t;

/** Opaque GPU context handle. */
typedef struct abi_gpu abi_gpu_t;

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */

/**
 * Framework initialization options.
 *
 * Controls which features are enabled during framework initialization.
 * All features default to enabled (true).
 */
typedef struct abi_options {
    /** Enable AI agent and LLM features. */
    bool enable_ai;

    /** Enable GPU acceleration. */
    bool enable_gpu;

    /** Enable vector database features. */
    bool enable_database;

    /** Enable network/distributed features. */
    bool enable_network;

    /** Enable web/HTTP utilities. */
    bool enable_web;

    /** Enable performance profiling. */
    bool enable_profiling;
} abi_options_t;

/**
 * Detailed version information.
 */
typedef struct abi_version_info {
    /** Major version number. */
    int major;

    /** Minor version number. */
    int minor;

    /** Patch version number. */
    int patch;

    /** Full version string (e.g., "1.2.3"). */
    const char *full;
} abi_version_info_t;

/**
 * SIMD capability flags.
 *
 * Indicates which SIMD instruction sets are available on the current CPU.
 */
typedef struct abi_simd_caps {
    bool sse;       /**< SSE supported. */
    bool sse2;      /**< SSE2 supported. */
    bool sse3;      /**< SSE3 supported. */
    bool ssse3;     /**< SSSE3 supported. */
    bool sse4_1;    /**< SSE4.1 supported. */
    bool sse4_2;    /**< SSE4.2 supported. */
    bool avx;       /**< AVX supported. */
    bool avx2;      /**< AVX2 supported. */
    bool avx512f;   /**< AVX-512F supported. */
    bool neon;      /**< ARM NEON supported. */
} abi_simd_caps_t;

/**
 * Database configuration options.
 */
typedef struct abi_database_config {
    /** Database name (default: "default"). */
    const char *name;

    /** Vector dimension (default: 384). */
    size_t dimension;

    /** Initial capacity hint (default: 1000). */
    size_t initial_capacity;
} abi_database_config_t;

/**
 * GPU context configuration options.
 */
typedef struct abi_gpu_config {
    /** GPU backend (use ABI_GPU_BACKEND_* constants, default: AUTO). */
    int backend;

    /** GPU device index (default: 0). */
    int device_index;

    /** Enable GPU profiling (default: false). */
    bool enable_profiling;
} abi_gpu_config_t;

/**
 * Vector search result.
 */
typedef struct abi_search_result {
    /** Vector ID. */
    uint64_t id;

    /** Similarity score (higher is more similar). */
    float score;
} abi_search_result_t;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * Get a human-readable error message for an error code.
 *
 * @param err Error code returned by an ABI function.
 * @return Null-terminated error message string. The string is statically
 *         allocated and must not be freed.
 */
const char *abi_error_string(int err);

/* ============================================================================
 * Framework Lifecycle
 * ============================================================================ */

/**
 * Initialize the ABI framework with default options.
 *
 * This is equivalent to calling abi_init_with_options(NULL, out_framework).
 *
 * @param[out] out_framework Pointer to receive the framework handle.
 * @return ABI_OK on success, or an error code on failure.
 *
 * @see abi_init_with_options
 * @see abi_shutdown
 */
int abi_init(abi_framework_t **out_framework);

/**
 * Initialize the ABI framework with custom options.
 *
 * @param options Configuration options, or NULL for defaults.
 * @param[out] out_framework Pointer to receive the framework handle.
 * @return ABI_OK on success, or an error code on failure.
 *
 * @see abi_init
 * @see abi_shutdown
 */
int abi_init_with_options(const abi_options_t *options,
                          abi_framework_t **out_framework);

/**
 * Shutdown the framework and release all resources.
 *
 * After calling this function, the framework handle becomes invalid.
 *
 * @param framework Framework handle to shutdown. May be NULL (no-op).
 */
void abi_shutdown(abi_framework_t *framework);

/**
 * Get the framework version string.
 *
 * @return Null-terminated version string (e.g., "1.2.3"). The string is
 *         statically allocated and must not be freed.
 */
const char *abi_version(void);

/**
 * Get detailed version information.
 *
 * @param[out] out_version Pointer to receive version information.
 */
void abi_version_info(abi_version_info_t *out_version);

/**
 * Check if a feature is enabled.
 *
 * Valid feature names: "ai", "gpu", "database", "network", "web", "profiling".
 *
 * @param framework Framework handle (may be NULL for compile-time check).
 * @param feature Null-terminated feature name.
 * @return true if the feature is enabled, false otherwise.
 */
bool abi_is_feature_enabled(abi_framework_t *framework, const char *feature);

/* ============================================================================
 * SIMD Operations
 * ============================================================================ */

/**
 * Query CPU SIMD capabilities.
 *
 * @param[out] out_caps Pointer to receive capability flags.
 */
void abi_simd_get_caps(abi_simd_caps_t *out_caps);

/**
 * Check if any SIMD instruction set is available.
 *
 * @return true if SIMD is available, false otherwise.
 */
bool abi_simd_available(void);

/**
 * Vector element-wise addition: result[i] = a[i] + b[i].
 *
 * Uses SIMD acceleration when available.
 *
 * @param a First input vector.
 * @param b Second input vector.
 * @param[out] result Output vector.
 * @param len Number of elements in each vector.
 */
void abi_simd_vector_add(const float *a, const float *b, float *result,
                         size_t len);

/**
 * Compute the dot product of two vectors: sum(a[i] * b[i]).
 *
 * Uses SIMD acceleration when available.
 *
 * @param a First input vector.
 * @param b Second input vector.
 * @param len Number of elements in each vector.
 * @return The dot product.
 */
float abi_simd_vector_dot(const float *a, const float *b, size_t len);

/**
 * Compute the L2 norm (Euclidean length) of a vector: sqrt(sum(v[i]^2)).
 *
 * Uses SIMD acceleration when available.
 *
 * @param v Input vector.
 * @param len Number of elements.
 * @return The L2 norm.
 */
float abi_simd_vector_l2_norm(const float *v, size_t len);

/**
 * Compute the cosine similarity between two vectors.
 *
 * Returns a value in the range [-1, 1], where 1 means identical direction,
 * 0 means orthogonal, and -1 means opposite direction.
 *
 * @param a First input vector.
 * @param b Second input vector.
 * @param len Number of elements in each vector.
 * @return The cosine similarity.
 */
float abi_simd_cosine_similarity(const float *a, const float *b, size_t len);

/* ============================================================================
 * Database Operations
 * ============================================================================ */

/**
 * Create a new vector database.
 *
 * Requires the database feature to be enabled at compile time.
 *
 * @param config Database configuration, or NULL for defaults.
 * @param[out] out_db Pointer to receive the database handle.
 * @return ABI_OK on success, or an error code on failure.
 *
 * @see abi_database_close
 */
int abi_database_create(const abi_database_config_t *config,
                        abi_database_t **out_db);

/**
 * Close a database and release resources.
 *
 * After calling this function, the database handle becomes invalid.
 *
 * @param db Database handle to close. May be NULL (no-op).
 */
void abi_database_close(abi_database_t *db);

/**
 * Insert a vector into the database.
 *
 * @param db Database handle.
 * @param id Unique vector ID.
 * @param vector Vector data.
 * @param vector_len Number of elements in the vector.
 * @param metadata Optional null-terminated metadata string, or NULL.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_database_insert(abi_database_t *db, uint64_t id, const float *vector,
                        size_t vector_len, const char *metadata);

/**
 * Search for similar vectors.
 *
 * @param db Database handle.
 * @param query Query vector.
 * @param query_len Number of elements in the query vector.
 * @param k Maximum number of results to return.
 * @param[out] out_results Array to receive results (must have space for k).
 * @param[out] out_count Receives the actual number of results returned.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_database_search(abi_database_t *db, const float *query,
                        size_t query_len, size_t k,
                        abi_search_result_t *out_results, size_t *out_count);

/**
 * Delete a vector from the database.
 *
 * @param db Database handle.
 * @param id Vector ID to delete.
 * @return ABI_OK on success, ABI_ERROR_INVALID_ARGUMENT if not found.
 */
int abi_database_delete(abi_database_t *db, uint64_t id);

/**
 * Get the number of vectors in the database.
 *
 * @param db Database handle.
 * @param[out] out_count Receives the vector count.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_database_count(abi_database_t *db, size_t *out_count);

/* ============================================================================
 * GPU Operations
 * ============================================================================ */

/**
 * Initialize a GPU context.
 *
 * Requires the GPU feature to be enabled at compile time.
 *
 * @param config GPU configuration, or NULL for auto-detection.
 * @param[out] out_gpu Pointer to receive the GPU handle.
 * @return ABI_OK on success, or an error code on failure.
 *
 * @see abi_gpu_shutdown
 */
int abi_gpu_init(const abi_gpu_config_t *config, abi_gpu_t **out_gpu);

/**
 * Shutdown GPU context and release resources.
 *
 * After calling this function, the GPU handle becomes invalid.
 *
 * @param gpu GPU handle to shutdown. May be NULL (no-op).
 */
void abi_gpu_shutdown(abi_gpu_t *gpu);

/**
 * Check if any GPU backend is available.
 *
 * @return true if a GPU backend is available, false otherwise.
 */
bool abi_gpu_is_available(void);

/**
 * Get the active GPU backend name.
 *
 * @param gpu GPU handle, or NULL.
 * @return Backend name string (e.g., "cuda", "vulkan", "metal", "none",
 *         "disabled"). The string is statically allocated and must not
 *         be freed.
 */
const char *abi_gpu_backend_name(abi_gpu_t *gpu);

/* ============================================================================
 * Agent Operations
 * ============================================================================ */

/** Opaque agent handle. */
typedef struct abi_agent abi_agent_t;

/** Agent backend constants. */
#define ABI_AGENT_BACKEND_ECHO        0
#define ABI_AGENT_BACKEND_OPENAI      1
#define ABI_AGENT_BACKEND_OLLAMA      2
#define ABI_AGENT_BACKEND_HUGGINGFACE 3
#define ABI_AGENT_BACKEND_LOCAL       4

/** Agent status constants. */
#define ABI_AGENT_STATUS_READY  0
#define ABI_AGENT_STATUS_BUSY   1
#define ABI_AGENT_STATUS_ERROR  2

/**
 * Agent configuration options.
 */
typedef struct abi_agent_config {
    /** Agent name (required, null-terminated). */
    const char *name;

    /** Backend type (use ABI_AGENT_BACKEND_* constants, default: ECHO). */
    int backend;

    /** Model name (e.g., "gpt-4", "llama3.2"). */
    const char *model;

    /** System prompt (optional, NULL for no system prompt). */
    const char *system_prompt;

    /** Temperature for generation (0.0 - 2.0, default: 0.7). */
    float temperature;

    /** Top-p for generation (0.0 - 1.0, default: 0.9). */
    float top_p;

    /** Maximum tokens for generation (default: 1024). */
    uint32_t max_tokens;

    /** Enable conversation history (default: true). */
    bool enable_history;
} abi_agent_config_t;

/**
 * Agent response from a send operation.
 */
typedef struct abi_agent_response {
    /** Response text (null-terminated, valid until next send or destroy). */
    const char *text;

    /** Length of response text (excluding null terminator). */
    size_t length;

    /** Number of tokens used (if available from backend). */
    uint64_t tokens_used;
} abi_agent_response_t;

/**
 * Agent conversation statistics.
 */
typedef struct abi_agent_stats {
    /** Total messages in history. */
    size_t history_length;

    /** Number of user messages. */
    size_t user_messages;

    /** Number of assistant messages. */
    size_t assistant_messages;

    /** Total characters in history. */
    size_t total_characters;

    /** Total tokens used in session. */
    uint64_t total_tokens_used;
} abi_agent_stats_t;

/**
 * Create a new AI agent.
 *
 * Requires the AI feature to be enabled at compile time.
 *
 * @param config Agent configuration, or NULL for defaults.
 * @param[out] out_agent Pointer to receive the agent handle.
 * @return ABI_OK on success, or an error code on failure.
 *
 * @see abi_agent_destroy
 */
int abi_agent_create(const abi_agent_config_t *config,
                     abi_agent_t **out_agent);

/**
 * Destroy an agent and release all resources.
 *
 * After calling this function, the agent handle becomes invalid.
 *
 * @param agent Agent handle to destroy. May be NULL (no-op).
 */
void abi_agent_destroy(abi_agent_t *agent);

/**
 * Send a message to the agent and get a response.
 *
 * The response text is owned by the agent and valid until the next
 * send or destroy call.
 *
 * @param agent Agent handle.
 * @param message Null-terminated message string.
 * @param[out] out_response Pointer to receive the response.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_agent_send(abi_agent_t *agent, const char *message,
                   abi_agent_response_t *out_response);

/**
 * Get the current status of the agent.
 *
 * @param agent Agent handle.
 * @return ABI_AGENT_STATUS_* constant.
 */
int abi_agent_get_status(abi_agent_t *agent);

/**
 * Get agent conversation statistics.
 *
 * @param agent Agent handle.
 * @param[out] out_stats Pointer to receive statistics.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_agent_get_stats(abi_agent_t *agent, abi_agent_stats_t *out_stats);

/**
 * Clear the agent's conversation history.
 *
 * @param agent Agent handle.
 * @return ABI_OK on success, or an error code on failure.
 */
int abi_agent_clear_history(abi_agent_t *agent);

/**
 * Set the agent's temperature parameter.
 *
 * @param agent Agent handle.
 * @param temperature New temperature (0.0 - 2.0).
 * @return ABI_OK on success, ABI_ERROR_INVALID_ARGUMENT if out of range.
 */
int abi_agent_set_temperature(abi_agent_t *agent, float temperature);

/**
 * Set the agent's max tokens parameter.
 *
 * @param agent Agent handle.
 * @param max_tokens New max tokens value.
 * @return ABI_OK on success, ABI_ERROR_INVALID_ARGUMENT if invalid.
 */
int abi_agent_set_max_tokens(abi_agent_t *agent, uint32_t max_tokens);

/**
 * Get the agent's name.
 *
 * @param agent Agent handle.
 * @return Agent name string. The string is owned by the agent and
 *         must not be freed.
 */
const char *abi_agent_get_name(abi_agent_t *agent);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Free a string allocated by ABI functions.
 *
 * @param str String to free. May be NULL (no-op).
 */
void abi_free_string(char *str);

/**
 * Free a search results array.
 *
 * @param results Results array to free. May be NULL (no-op).
 * @param count Number of results in the array.
 */
void abi_free_results(abi_search_result_t *results, size_t count);

/* ============================================================================
 * Initialization Helpers
 * ============================================================================ */

/**
 * Initialize an options structure with default values.
 *
 * All features are enabled by default.
 *
 * @param[out] options Pointer to options structure to initialize.
 */
static inline void abi_options_init(abi_options_t *options) {
    options->enable_ai = true;
    options->enable_gpu = true;
    options->enable_database = true;
    options->enable_network = true;
    options->enable_web = true;
    options->enable_profiling = true;
}

/**
 * Initialize a database config structure with default values.
 *
 * @param[out] config Pointer to config structure to initialize.
 */
static inline void abi_database_config_init(abi_database_config_t *config) {
    config->name = "default";
    config->dimension = 384;
    config->initial_capacity = 1000;
}

/**
 * Initialize a GPU config structure with default values.
 *
 * @param[out] config Pointer to config structure to initialize.
 */
static inline void abi_gpu_config_init(abi_gpu_config_t *config) {
    config->backend = ABI_GPU_BACKEND_AUTO;
    config->device_index = 0;
    config->enable_profiling = false;
}

/**
 * Initialize an agent config structure with default values.
 *
 * @param[out] config Pointer to config structure to initialize.
 */
static inline void abi_agent_config_init(abi_agent_config_t *config) {
    config->name = "agent";
    config->backend = ABI_AGENT_BACKEND_ECHO;
    config->model = "gpt-4";
    config->system_prompt = NULL;
    config->temperature = 0.7f;
    config->top_p = 0.9f;
    config->max_tokens = 1024;
    config->enable_history = true;
}

#ifdef __cplusplus
}
#endif

#endif /* ABI_H */
