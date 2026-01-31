// abi_errors.h - Error codes for ABI C bindings
// SPDX-License-Identifier: MIT

#ifndef ABI_ERRORS_H
#define ABI_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int abi_error_t;

#define ABI_OK                          0
#define ABI_ERROR_INIT_FAILED          -1
#define ABI_ERROR_ALREADY_INITIALIZED  -2
#define ABI_ERROR_NOT_INITIALIZED      -3
#define ABI_ERROR_OUT_OF_MEMORY        -4
#define ABI_ERROR_INVALID_ARGUMENT     -5
#define ABI_ERROR_FEATURE_DISABLED     -6
#define ABI_ERROR_TIMEOUT              -7
#define ABI_ERROR_IO                   -8
#define ABI_ERROR_GPU_UNAVAILABLE      -9
#define ABI_ERROR_DATABASE_ERROR      -10
#define ABI_ERROR_NETWORK_ERROR       -11
#define ABI_ERROR_AI_ERROR            -12
#define ABI_ERROR_UNKNOWN             -99

/// Get human-readable error message for an error code.
/// @param error The error code
/// @return Static string describing the error (do not free)
const char* abi_error_string(abi_error_t error);

#ifdef __cplusplus
}
#endif

#endif // ABI_ERRORS_H
