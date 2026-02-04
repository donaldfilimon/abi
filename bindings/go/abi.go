// Package abi provides Go bindings for the ABI Framework.
//
// The ABI Framework is a high-performance vector database and AI inference library
// written in Zig. These bindings use cgo to interface with the C shared library.
//
// # Requirements
//
// Build the shared library first:
//
//	zig build lib
//
// Then set the library path:
//
//	export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH  # Linux
//	export DYLD_LIBRARY_PATH=$PWD/zig-out/lib:$DYLD_LIBRARY_PATH  # macOS
//
// # Example
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/donaldfilimon/abi/bindings/go"
//	)
//
//	func main() {
//		framework, err := abi.Init()
//		if err != nil {
//			panic(err)
//		}
//		defer framework.Shutdown()
//
//		fmt.Println("ABI Version:", abi.Version())
//
//		db, err := framework.CreateDB(128)
//		if err != nil {
//			panic(err)
//		}
//		defer db.Destroy()
//
//		db.Insert(1, []float32{1.0, 0.0, ...})
//		results, _ := db.Search([]float32{1.0, 0.0, ...}, 10)
//	}
package abi

/*
#cgo LDFLAGS: -labi
#cgo CFLAGS: -I${SRCDIR}/../c/include

#include <stdlib.h>
#include <stdint.h>

// Forward declarations matching abi.h
typedef void* AbiHandle;

typedef enum {
    ABI_SUCCESS = 0,
    ABI_ERROR_UNKNOWN = 1,
    ABI_ERROR_INVALID_ARGUMENT = 2,
    ABI_ERROR_OUT_OF_MEMORY = 3,
    ABI_ERROR_INITIALIZATION_FAILED = 4,
    ABI_ERROR_NOT_INITIALIZED = 5,
} AbiStatus;

// Core functions
extern AbiHandle abi_init(void);
extern void abi_shutdown(AbiHandle handle);
extern const char* abi_version(void);

// Database functions
extern AbiStatus abi_db_create(AbiHandle handle, uint32_t dimension, AbiHandle* db_out);
extern AbiStatus abi_db_insert(AbiHandle db, uint64_t id, const float* vector, size_t vector_len);
extern AbiStatus abi_db_search(AbiHandle db, const float* vector, size_t vector_len, uint32_t k, uint64_t* ids_out, float* scores_out);
extern void abi_db_destroy(AbiHandle db);
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Status codes returned by ABI functions.
type Status int

const (
	StatusSuccess              Status = 0
	StatusErrorUnknown         Status = 1
	StatusErrorInvalidArgument Status = 2
	StatusErrorOutOfMemory     Status = 3
	StatusErrorInitFailed      Status = 4
	StatusErrorNotInitialized  Status = 5
)

// Error types for ABI operations.
var (
	ErrUnknown         = errors.New("abi: unknown error")
	ErrInvalidArgument = errors.New("abi: invalid argument")
	ErrOutOfMemory     = errors.New("abi: out of memory")
	ErrInitFailed      = errors.New("abi: initialization failed")
	ErrNotInitialized  = errors.New("abi: not initialized")
)

// statusToError converts a C status code to a Go error.
func statusToError(status C.AbiStatus) error {
	switch Status(status) {
	case StatusSuccess:
		return nil
	case StatusErrorUnknown:
		return ErrUnknown
	case StatusErrorInvalidArgument:
		return ErrInvalidArgument
	case StatusErrorOutOfMemory:
		return ErrOutOfMemory
	case StatusErrorInitFailed:
		return ErrInitFailed
	case StatusErrorNotInitialized:
		return ErrNotInitialized
	default:
		return ErrUnknown
	}
}

// Framework represents the ABI framework instance.
type Framework struct {
	handle C.AbiHandle
}

// Init initializes the ABI framework and returns a Framework instance.
func Init() (*Framework, error) {
	handle := C.abi_init()
	if handle == nil {
		return nil, ErrInitFailed
	}
	return &Framework{handle: handle}, nil
}

// Shutdown releases all resources associated with the framework.
func (f *Framework) Shutdown() {
	if f.handle != nil {
		C.abi_shutdown(f.handle)
		f.handle = nil
	}
}

// Version returns the ABI framework version string.
func Version() string {
	return C.GoString(C.abi_version())
}

// CreateDB creates a new vector database with the specified dimension.
func (f *Framework) CreateDB(dimension uint32) (*VectorDatabase, error) {
	if f.handle == nil {
		return nil, ErrNotInitialized
	}

	var dbHandle C.AbiHandle
	status := C.abi_db_create(f.handle, C.uint32_t(dimension), &dbHandle)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	return &VectorDatabase{
		handle:    dbHandle,
		dimension: dimension,
	}, nil
}

// VectorDatabase represents a vector database instance.
type VectorDatabase struct {
	handle    C.AbiHandle
	dimension uint32
}

// Dimension returns the vector dimension of this database.
func (db *VectorDatabase) Dimension() uint32 {
	return db.dimension
}

// Insert adds a vector with the given ID to the database.
func (db *VectorDatabase) Insert(id uint64, vector []float32) error {
	if db.handle == nil {
		return ErrNotInitialized
	}
	if len(vector) != int(db.dimension) {
		return ErrInvalidArgument
	}

	status := C.abi_db_insert(
		db.handle,
		C.uint64_t(id),
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.size_t(len(vector)),
	)
	return statusToError(status)
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID    uint64
	Score float32
}

// Search finds the k most similar vectors to the query vector.
func (db *VectorDatabase) Search(query []float32, k uint32) ([]SearchResult, error) {
	if db.handle == nil {
		return nil, ErrNotInitialized
	}
	if len(query) != int(db.dimension) {
		return nil, ErrInvalidArgument
	}
	if k == 0 {
		return nil, ErrInvalidArgument
	}

	// Allocate output buffers
	ids := make([]uint64, k)
	scores := make([]float32, k)

	status := C.abi_db_search(
		db.handle,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(len(query)),
		C.uint32_t(k),
		(*C.uint64_t)(unsafe.Pointer(&ids[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}

	// Convert to results
	results := make([]SearchResult, k)
	for i := uint32(0); i < k; i++ {
		results[i] = SearchResult{
			ID:    ids[i],
			Score: scores[i],
		}
	}

	return results, nil
}

// Destroy releases all resources associated with the database.
func (db *VectorDatabase) Destroy() {
	if db.handle != nil {
		C.abi_db_destroy(db.handle)
		db.handle = nil
	}
}
