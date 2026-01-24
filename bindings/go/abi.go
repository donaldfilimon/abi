// Package abi provides Go bindings for the ABI Framework.
//
// The ABI Framework provides high-performance SIMD operations, vector database
// with HNSW indexing, GPU acceleration, and AI agent capabilities.
//
// Quick Start:
//
//	import "github.com/donaldfilimon/abi-go"
//
//	// SIMD operations (no initialization needed)
//	a := []float32{1.0, 2.0, 3.0, 4.0}
//	b := []float32{4.0, 3.0, 2.0, 1.0}
//	similarity := abi.CosineSimilarity(a, b)
//
//	// Framework initialization
//	fw, err := abi.NewFramework(abi.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer fw.Close()
package abi

/*
#cgo CFLAGS: -I../../bindings/c
#cgo LDFLAGS: -L../../zig-out/lib -labi

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

// Error codes
typedef int abi_error_t;
#define ABI_OK 0
#define ABI_ERROR_INIT_FAILED -1
#define ABI_ERROR_NOT_INITIALIZED -3
#define ABI_ERROR_OUT_OF_MEMORY -4
#define ABI_ERROR_INVALID_ARGUMENT -5
#define ABI_ERROR_FEATURE_DISABLED -6

// Opaque handles
typedef void* abi_framework_t;
typedef void* abi_database_t;
typedef void* abi_gpu_t;

// Options
typedef struct {
    bool enable_ai;
    bool enable_gpu;
    bool enable_database;
    bool enable_network;
    bool enable_web;
    bool enable_profiling;
} abi_options_t;

// SIMD capabilities
typedef struct {
    size_t vector_size;
    bool has_simd;
    int arch;
} abi_simd_caps_t;

// Database config
typedef struct {
    const char* name;
    size_t dimension;
    size_t initial_capacity;
} abi_database_config_t;

// Search result
typedef struct {
    uint64_t id;
    float score;
    const float* vector;
    size_t vector_len;
} abi_search_result_t;

// GPU config
typedef struct {
    int backend;
    int device_index;
} abi_gpu_config_t;

// GPU device info
typedef struct {
    char name[256];
    int backend;
    size_t total_memory;
    size_t free_memory;
    int compute_units;
} abi_gpu_device_info_t;

// Framework functions
extern abi_error_t abi_init(abi_framework_t* out_framework);
extern abi_error_t abi_init_with_options(const abi_options_t* options, abi_framework_t* out_framework);
extern void abi_shutdown(abi_framework_t framework);
extern const char* abi_version();
extern bool abi_is_feature_enabled(abi_framework_t framework, const char* feature);

// SIMD functions
extern void abi_simd_get_caps(abi_simd_caps_t* out_caps);
extern bool abi_simd_available();
extern void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);
extern float abi_simd_vector_dot(const float* a, const float* b, size_t len);
extern float abi_simd_vector_l2_norm(const float* v, size_t len);
extern float abi_simd_cosine_similarity(const float* a, const float* b, size_t len);
extern void abi_simd_matrix_multiply(const float* a, const float* b, float* result,
                                     size_t m, size_t n, size_t k);

// Database functions
extern abi_error_t abi_database_create(const abi_database_config_t* config, abi_database_t* out_db);
extern void abi_database_close(abi_database_t db);
extern abi_error_t abi_database_insert(abi_database_t db, uint64_t id,
                                       const float* vector, size_t vector_len);
extern abi_error_t abi_database_search(abi_database_t db, const float* query,
                                       size_t query_len, size_t k,
                                       abi_search_result_t* out_results, size_t* out_count);
extern abi_error_t abi_database_delete(abi_database_t db, uint64_t id);
extern size_t abi_database_count(abi_database_t db);

// GPU functions
extern abi_error_t abi_gpu_init(const abi_gpu_config_t* config, abi_gpu_t* out_gpu);
extern void abi_gpu_shutdown(abi_gpu_t gpu);
extern bool abi_gpu_is_available();
extern abi_error_t abi_gpu_list_devices(abi_gpu_device_info_t* out_devices,
                                        size_t max_devices, size_t* out_count);
extern abi_error_t abi_gpu_matrix_multiply(abi_gpu_t gpu,
                                           const float* a, const float* b, float* result,
                                           size_t m, size_t n, size_t k);
extern abi_error_t abi_gpu_vector_add(abi_gpu_t gpu,
                                      const float* a, const float* b, float* result, size_t len);
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// Error types
var (
	ErrInitFailed      = errors.New("initialization failed")
	ErrNotInitialized  = errors.New("framework not initialized")
	ErrOutOfMemory     = errors.New("out of memory")
	ErrInvalidArgument = errors.New("invalid argument")
	ErrFeatureDisabled = errors.New("feature disabled")
	ErrUnknown         = errors.New("unknown error")
)

// convertError converts a C error code to a Go error
func convertError(code C.abi_error_t) error {
	switch code {
	case C.ABI_OK:
		return nil
	case C.ABI_ERROR_INIT_FAILED:
		return ErrInitFailed
	case C.ABI_ERROR_NOT_INITIALIZED:
		return ErrNotInitialized
	case C.ABI_ERROR_OUT_OF_MEMORY:
		return ErrOutOfMemory
	case C.ABI_ERROR_INVALID_ARGUMENT:
		return ErrInvalidArgument
	case C.ABI_ERROR_FEATURE_DISABLED:
		return ErrFeatureDisabled
	default:
		return fmt.Errorf("%w: code %d", ErrUnknown, code)
	}
}

// Architecture represents the CPU architecture for SIMD operations
type Architecture int

const (
	ArchGeneric Architecture = iota
	ArchX86_64
	ArchAarch64
	ArchWasm
)

func (a Architecture) String() string {
	switch a {
	case ArchX86_64:
		return "x86_64"
	case ArchAarch64:
		return "aarch64"
	case ArchWasm:
		return "wasm"
	default:
		return "generic"
	}
}

// SIMDCapabilities contains information about SIMD support
type SIMDCapabilities struct {
	VectorSize int
	HasSIMD    bool
	Arch       Architecture
}

// GetSIMDCapabilities returns the SIMD capabilities of the current platform
func GetSIMDCapabilities() SIMDCapabilities {
	var caps C.abi_simd_caps_t
	C.abi_simd_get_caps(&caps)
	return SIMDCapabilities{
		VectorSize: int(caps.vector_size),
		HasSIMD:    bool(caps.has_simd),
		Arch:       Architecture(caps.arch),
	}
}

// SIMDAvailable returns true if SIMD operations are available
func SIMDAvailable() bool {
	return bool(C.abi_simd_available())
}

// Version returns the ABI Framework version string
func Version() string {
	return C.GoString(C.abi_version())
}

// Config holds framework configuration options
type Config struct {
	EnableAI        bool
	EnableGPU       bool
	EnableDatabase  bool
	EnableNetwork   bool
	EnableWeb       bool
	EnableProfiling bool
}

// DefaultConfig returns a configuration with all features enabled
func DefaultConfig() Config {
	return Config{
		EnableAI:        true,
		EnableGPU:       true,
		EnableDatabase:  true,
		EnableNetwork:   true,
		EnableWeb:       true,
		EnableProfiling: true,
	}
}

// MinimalConfig returns a configuration with only essential features
func MinimalConfig() Config {
	return Config{
		EnableAI:        false,
		EnableGPU:       false,
		EnableDatabase:  false,
		EnableNetwork:   false,
		EnableWeb:       false,
		EnableProfiling: false,
	}
}

// Framework represents an initialized ABI Framework instance
type Framework struct {
	handle C.abi_framework_t
}

// NewFramework creates a new Framework instance with the given configuration
func NewFramework(config Config) (*Framework, error) {
	opts := C.abi_options_t{
		enable_ai:        C.bool(config.EnableAI),
		enable_gpu:       C.bool(config.EnableGPU),
		enable_database:  C.bool(config.EnableDatabase),
		enable_network:   C.bool(config.EnableNetwork),
		enable_web:       C.bool(config.EnableWeb),
		enable_profiling: C.bool(config.EnableProfiling),
	}

	var handle C.abi_framework_t
	code := C.abi_init_with_options(&opts, &handle)
	if err := convertError(code); err != nil {
		return nil, err
	}

	return &Framework{handle: handle}, nil
}

// Close shuts down the framework and releases resources
func (f *Framework) Close() {
	if f.handle != nil {
		C.abi_shutdown(f.handle)
		f.handle = nil
	}
}

// IsFeatureEnabled checks if a specific feature is enabled
func (f *Framework) IsFeatureEnabled(feature string) bool {
	cFeature := C.CString(feature)
	defer C.free(unsafe.Pointer(cFeature))
	return bool(C.abi_is_feature_enabled(f.handle, cFeature))
}
