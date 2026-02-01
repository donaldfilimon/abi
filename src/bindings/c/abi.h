#ifndef ABI_H
#define ABI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types
// ============================================================================

typedef enum AbiStatus {
    ABI_SUCCESS = 0,
    ABI_ERROR_UNKNOWN = 1,
    ABI_ERROR_INVALID_ARGUMENT = 2,
    ABI_ERROR_OUT_OF_MEMORY = 3,
    ABI_ERROR_INITIALIZATION_FAILED = 4,
    ABI_ERROR_NOT_INITIALIZED = 5,
} AbiStatus;

typedef void* AbiHandle;

// ============================================================================
// Core
// ============================================================================

/**
 * Initialize the ABI framework.
 *
 * @return Handle to the framework instance, or NULL on failure.
 */
AbiHandle abi_init(void);

/**
 * Shutdown the ABI framework and release resources.
 *
 * @param handle Framework handle to shutdown.
 */
void abi_shutdown(AbiHandle handle);

/**
 * Get the framework version string.
 *
 * @return Pointer to a null-terminated version string.
 */
const char* abi_version(void);

// ============================================================================
// Database
// ============================================================================

/**
 * Create a new vector database.
 *
 * @param handle Framework handle.
 * @param dimension Vector dimension.
 * @param db_out Pointer to store the database handle.
 * @return Status code.
 */
AbiStatus abi_db_create(AbiHandle handle, uint32_t dimension, AbiHandle* db_out);

/**
 * Insert a vector into the database.
 *
 * @param db Database handle.
 * @param id Vector ID.
 * @param vector Pointer to float array of vector data.
 * @param vector_len Length of the vector (must match dimension).
 * @return Status code.
 */
AbiStatus abi_db_insert(AbiHandle db, uint64_t id, const float* vector, size_t vector_len);

/**
 * Search for similar vectors.
 *
 * @param db Database handle.
 * @param vector Query vector.
 * @param vector_len Length of query vector.
 * @param k Number of results to return.
 * @param ids_out Pointer to array to store result IDs (allocated by caller).
 * @param scores_out Pointer to array to store result scores (allocated by caller).
 * @return Status code.
 */
AbiStatus abi_db_search(AbiHandle db, const float* vector, size_t vector_len, uint32_t k, uint64_t* ids_out, float* scores_out);

/**
 * Destroy a vector database handle.
 *
 * @param db Database handle.
 */
void abi_db_destroy(AbiHandle db);

#ifdef __cplusplus
}
#endif

#endif // ABI_H
