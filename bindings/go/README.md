# ABI Framework Go Bindings

Go bindings for the ABI Framework using cgo, providing high-performance vector database operations.

## Requirements

- Go 1.21+
- ABI shared library (`libabi.dylib` on macOS, `libabi.so` on Linux)
- C compiler (for cgo)

## Building the Shared Library

From the ABI repository root:

```bash
# Build the shared library
zig build lib

# The library will be in zig-out/lib/
```

## Setting Library Path

Before running Go code, set the library path:

```bash
# macOS
export DYLD_LIBRARY_PATH=$PWD/zig-out/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH
```

## Quick Start

```go
package main

import (
    "fmt"
    abi "github.com/donaldfilimon/abi/bindings/go"
)

func main() {
    // Initialize framework
    framework, err := abi.Init()
    if err != nil {
        panic(err)
    }
    defer framework.Shutdown()

    fmt.Println("ABI Version:", abi.Version())

    // Create a vector database with 128-dimensional vectors
    db, err := framework.CreateDB(128)
    if err != nil {
        panic(err)
    }
    defer db.Destroy()

    // Insert vectors
    db.Insert(1, []float32{1.0, 0.0, /* ... 128 elements */})
    db.Insert(2, []float32{0.0, 1.0, /* ... 128 elements */})

    // Search for similar vectors
    results, err := db.Search([]float32{1.0, 0.0, /* ... */}, 10)
    if err != nil {
        panic(err)
    }

    for _, r := range results {
        fmt.Printf("ID: %d, Score: %f\n", r.ID, r.Score)
    }
}
```

## API Reference

### Package Functions

```go
// Version returns the ABI framework version string.
func Version() string

// Init initializes the ABI framework and returns a Framework instance.
func Init() (*Framework, error)
```

### Framework

```go
type Framework struct {
    // ...
}

// Shutdown releases all resources associated with the framework.
func (f *Framework) Shutdown()

// CreateDB creates a new vector database with the specified dimension.
func (f *Framework) CreateDB(dimension uint32) (*VectorDatabase, error)
```

### VectorDatabase

```go
type VectorDatabase struct {
    // ...
}

// Dimension returns the vector dimension of this database.
func (db *VectorDatabase) Dimension() uint32

// Insert adds a vector with the given ID to the database.
func (db *VectorDatabase) Insert(id uint64, vector []float32) error

// Search finds the k most similar vectors to the query vector.
func (db *VectorDatabase) Search(query []float32, k uint32) ([]SearchResult, error)

// Destroy releases all resources associated with the database.
func (db *VectorDatabase) Destroy()
```

### Context-Aware Methods

The bindings provide context-aware versions of operations for cancellation and timeout support:

```go
import (
    "context"
    "time"
)

// Search with timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()
results, err := db.SearchWithContext(ctx, query, 10)
if err == context.DeadlineExceeded {
    // Handle timeout
}

// Insert with cancellation
ctx, cancel := context.WithCancel(context.Background())
err := db.InsertWithContext(ctx, id, vector)

// Batch insert with context (checks context between each insert)
ids := []uint64{1, 2, 3}
vectors := [][]float32{...}
inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
if err == context.Canceled {
    fmt.Printf("Cancelled after inserting %d vectors\n", inserted)
}
```

**Note:** CGO calls cannot be truly interrupted mid-execution. Context checking happens before and after each CGO call. For batch operations, use `InsertBatchWithContext` which checks the context between each insert for better cancellation granularity.

### SearchResult

```go
type SearchResult struct {
    ID    uint64
    Score float32
}
```

### Error Types

```go
var (
    ErrUnknown         = errors.New("abi: unknown error")
    ErrInvalidArgument = errors.New("abi: invalid argument")
    ErrOutOfMemory     = errors.New("abi: out of memory")
    ErrInitFailed      = errors.New("abi: initialization failed")
    ErrNotInitialized  = errors.New("abi: not initialized")
)
```

## Running Tests

```bash
# From the ABI repository root
zig build lib

# Run tests (macOS)
CGO_LDFLAGS="-L$PWD/zig-out/lib" DYLD_LIBRARY_PATH=$PWD/zig-out/lib \
    go test -v ./bindings/go/

# Run tests (Linux)
CGO_LDFLAGS="-L$PWD/zig-out/lib" LD_LIBRARY_PATH=$PWD/zig-out/lib \
    go test -v ./bindings/go/

# Run benchmarks
CGO_LDFLAGS="-L$PWD/zig-out/lib" DYLD_LIBRARY_PATH=$PWD/zig-out/lib \
    go test -bench=. ./bindings/go/
```

## Running Examples

```bash
# From the ABI repository root
zig build lib

# macOS
CGO_LDFLAGS="-L$PWD/zig-out/lib" DYLD_LIBRARY_PATH=$PWD/zig-out/lib \
    go run ./bindings/go/examples/vector_search/main.go

# Linux
CGO_LDFLAGS="-L$PWD/zig-out/lib" LD_LIBRARY_PATH=$PWD/zig-out/lib \
    go run ./bindings/go/examples/vector_search/main.go
```

## Thread Safety

The ABI Framework is designed for concurrent use:

- Multiple goroutines can safely perform searches concurrently
- Insert operations should be serialized or use appropriate synchronization
- Each `VectorDatabase` instance maintains its own state

## Performance

Typical performance on modern hardware (128-dimensional vectors, 1000 vectors):

- Insert: ~20,000 vectors/sec
- Search (k=10): ~4,000 queries/sec

## License

MIT License - see the main ABI repository for details.
