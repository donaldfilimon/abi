package abi

/*
#include <stdlib.h>
#include <stdint.h>

typedef int abi_error_t;
typedef void* abi_database_t;

typedef struct {
    const char* name;
    size_t dimension;
    size_t initial_capacity;
} abi_database_config_t;

typedef struct {
    uint64_t id;
    float score;
    const float* vector;
    size_t vector_len;
} abi_search_result_t;

extern abi_error_t abi_database_create(const abi_database_config_t* config, abi_database_t* out_db);
extern void abi_database_close(abi_database_t db);
extern abi_error_t abi_database_insert(abi_database_t db, uint64_t id,
                                       const float* vector, size_t vector_len);
extern abi_error_t abi_database_search(abi_database_t db, const float* query,
                                       size_t query_len, size_t k,
                                       abi_search_result_t* out_results, size_t* out_count);
extern abi_error_t abi_database_delete(abi_database_t db, uint64_t id);
extern size_t abi_database_count(abi_database_t db);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// SearchResult represents a single result from a similarity search
type SearchResult struct {
	ID     uint64
	Score  float32
	Vector []float32 // May be nil if vector wasn't returned
}

// DatabaseConfig contains configuration for creating a vector database
type DatabaseConfig struct {
	Name            string
	Dimension       int
	InitialCapacity int
}

// NewDatabaseConfig creates a new configuration with sensible defaults
func NewDatabaseConfig(name string, dimension int) DatabaseConfig {
	return DatabaseConfig{
		Name:            name,
		Dimension:       dimension,
		InitialCapacity: 1000,
	}
}

// WithCapacity sets the initial capacity
func (c DatabaseConfig) WithCapacity(capacity int) DatabaseConfig {
	c.InitialCapacity = capacity
	return c
}

// VectorDatabase provides high-performance vector similarity search
type VectorDatabase struct {
	handle    C.abi_database_t
	dimension int
}

// NewVectorDatabase creates a new vector database with the given name and dimension
func NewVectorDatabase(name string, dimension int) (*VectorDatabase, error) {
	return NewVectorDatabaseWithConfig(NewDatabaseConfig(name, dimension))
}

// NewVectorDatabaseWithConfig creates a new vector database with custom configuration
func NewVectorDatabaseWithConfig(config DatabaseConfig) (*VectorDatabase, error) {
	cName := C.CString(config.Name)
	defer C.free(unsafe.Pointer(cName))

	cConfig := C.abi_database_config_t{
		name:             cName,
		dimension:        C.size_t(config.Dimension),
		initial_capacity: C.size_t(config.InitialCapacity),
	}

	var handle C.abi_database_t
	code := C.abi_database_create(&cConfig, &handle)
	if err := convertError(code); err != nil {
		return nil, fmt.Errorf("failed to create database: %w", err)
	}

	return &VectorDatabase{
		handle:    handle,
		dimension: config.Dimension,
	}, nil
}

// Close closes the database and releases resources
func (db *VectorDatabase) Close() {
	if db.handle != nil {
		C.abi_database_close(db.handle)
		db.handle = nil
	}
}

// Dimension returns the vector dimension for this database
func (db *VectorDatabase) Dimension() int {
	return db.dimension
}

// Len returns the number of vectors in the database
func (db *VectorDatabase) Len() int {
	return int(C.abi_database_count(db.handle))
}

// IsEmpty returns true if the database contains no vectors
func (db *VectorDatabase) IsEmpty() bool {
	return db.Len() == 0
}

// Insert adds a vector to the database with the given ID
func (db *VectorDatabase) Insert(id uint64, vector []float32) error {
	if len(vector) != db.dimension {
		return fmt.Errorf("%w: vector dimension %d doesn't match database dimension %d",
			ErrInvalidArgument, len(vector), db.dimension)
	}

	code := C.abi_database_insert(
		db.handle,
		C.uint64_t(id),
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.size_t(len(vector)),
	)
	return convertError(code)
}

// InsertBatch inserts multiple vectors at once
// Returns the number of successfully inserted vectors
func (db *VectorDatabase) InsertBatch(items []struct {
	ID     uint64
	Vector []float32
}) (int, error) {
	count := 0
	for _, item := range items {
		if err := db.Insert(item.ID, item.Vector); err == nil {
			count++
		}
	}
	return count, nil
}

// Search finds the k most similar vectors to the query
func (db *VectorDatabase) Search(query []float32, k int) ([]SearchResult, error) {
	if len(query) != db.dimension {
		return nil, fmt.Errorf("%w: query dimension %d doesn't match database dimension %d",
			ErrInvalidArgument, len(query), db.dimension)
	}

	results := make([]C.abi_search_result_t, k)
	var count C.size_t

	code := C.abi_database_search(
		db.handle,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(len(query)),
		C.size_t(k),
		(*C.abi_search_result_t)(unsafe.Pointer(&results[0])),
		&count,
	)
	if err := convertError(code); err != nil {
		return nil, err
	}

	goResults := make([]SearchResult, int(count))
	for i := 0; i < int(count); i++ {
		r := results[i]
		goResults[i] = SearchResult{
			ID:    uint64(r.id),
			Score: float32(r.score),
		}
		// Copy vector if present
		if r.vector != nil && r.vector_len > 0 {
			vec := make([]float32, int(r.vector_len))
			copy(vec, (*[1 << 30]float32)(unsafe.Pointer(r.vector))[:r.vector_len:r.vector_len])
			goResults[i].Vector = vec
		}
	}

	return goResults, nil
}

// Delete removes a vector by ID
func (db *VectorDatabase) Delete(id uint64) error {
	code := C.abi_database_delete(db.handle, C.uint64_t(id))
	return convertError(code)
}
