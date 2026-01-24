# ABI Framework Go Bindings

Go bindings for the ABI Framework using cgo.

## Features

- **SIMD Operations**: High-performance vector math with automatic SIMD acceleration
- **Vector Database**: HNSW-indexed similarity search
- **GPU Acceleration**: Multi-backend support (CUDA, Vulkan, Metal, WebGPU)

## Installation

```bash
go get github.com/donaldfilimon/abi-go
```

## Prerequisites

The bindings require the ABI native library to be built:

```bash
# Build the native library first
cd $PROJECT_ROOT
zig build

# Set library path (Linux/macOS)
export LD_LIBRARY_PATH=$PROJECT_ROOT/zig-out/lib:$LD_LIBRARY_PATH

# Or on macOS
export DYLD_LIBRARY_PATH=$PROJECT_ROOT/zig-out/lib:$DYLD_LIBRARY_PATH
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    abi "github.com/donaldfilimon/abi-go"
)

func main() {
    // Vector operations (no initialization needed)
    a := []float32{1.0, 2.0, 3.0, 4.0}
    b := []float32{4.0, 3.0, 2.0, 1.0}

    similarity := abi.CosineSimilarity(a, b)
    fmt.Printf("Cosine similarity: %.4f\n", similarity)

    dot := abi.DotProduct(a, b)
    fmt.Printf("Dot product: %.1f\n", dot)

    // Framework initialization (requires native library)
    fw, err := abi.NewFramework(abi.DefaultConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer fw.Close()

    fmt.Printf("AI enabled: %v\n", fw.IsFeatureEnabled("ai"))
}
```

## Modules

### SIMD Operations

```go
import abi "github.com/donaldfilimon/abi-go"

a := []float32{1.0, 2.0, 3.0, 4.0}
b := []float32{4.0, 3.0, 2.0, 1.0}

// Basic operations
sum := abi.Add(a, b)
diff := abi.Subtract(a, b)
scaled := abi.Scale(a, 2.0)

// Vector math
dot := abi.DotProduct(a, b)
norm := abi.L2Norm(a)
normalized := abi.Normalize(a)
similarity := abi.CosineSimilarity(a, b)
distance := abi.EuclideanDistance(a, b)

// Matrix multiplication (row-major order)
matA := []float32{1.0, 2.0, 3.0, 4.0} // 2x2
matB := []float32{5.0, 6.0, 7.0, 8.0} // 2x2
result := abi.MatrixMultiply(matA, matB, 2, 2, 2)

// Check capabilities
caps := abi.GetSIMDCapabilities()
fmt.Printf("SIMD: %v, Vector size: %d, Arch: %s\n",
    caps.HasSIMD, caps.VectorSize, caps.Arch)
```

### Vector Database

```go
import abi "github.com/donaldfilimon/abi-go"

// Create a database
db, err := abi.NewVectorDatabase("embeddings", 384)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Or with custom configuration
config := abi.NewDatabaseConfig("embeddings", 384).
    WithCapacity(10000)
db, err := abi.NewVectorDatabaseWithConfig(config)

// Insert vectors
err = db.Insert(1, embedding1)
err = db.Insert(2, embedding2)

// Search for similar vectors
results, err := db.Search(queryVector, 10)
for _, r := range results {
    fmt.Printf("ID: %d, Score: %.4f\n", r.ID, r.Score)
}

// Delete a vector
err = db.Delete(1)

// Get stats
fmt.Printf("Database has %d vectors\n", db.Len())
```

### GPU Acceleration

```go
import abi "github.com/donaldfilimon/abi-go"

// Check availability
if !abi.GPUAvailable() {
    fmt.Println("No GPU available")
    return
}

// List devices
devices, _ := abi.ListGPUDevices()
for _, d := range devices {
    fmt.Printf("%s (%s) - %d MB\n",
        d.Name, d.Backend, d.TotalMemory/1024/1024)
}

// Create context with auto-detected backend
gpu, err := abi.NewGPUContext(abi.BackendAuto)
if err != nil {
    log.Fatal(err)
}
defer gpu.Close()

// Or specify backend
config := abi.DefaultGPUConfig().
    WithBackend(abi.BackendCUDA).
    WithDevice(0)
gpu, err := abi.NewGPUContextWithConfig(config)

// GPU-accelerated operations
result, err := gpu.MatrixMultiply(a, b, m, n, k)
sum, err := gpu.VectorAdd(a, b)
```

## Error Handling

```go
import (
    "errors"
    abi "github.com/donaldfilimon/abi-go"
)

db, err := abi.NewVectorDatabase("test", 384)
if err != nil {
    switch {
    case errors.Is(err, abi.ErrNotInitialized):
        fmt.Println("Framework not initialized")
    case errors.Is(err, abi.ErrFeatureDisabled):
        fmt.Println("Database feature is disabled")
    case errors.Is(err, abi.ErrInvalidArgument):
        fmt.Println("Invalid argument")
    default:
        fmt.Printf("Error: %v\n", err)
    }
}
```

## Thread Safety

- `Framework` is **not** safe for concurrent use from multiple goroutines
- `VectorDatabase` is **not** safe for concurrent use
- `GPUContext` is **not** safe for concurrent use

For concurrent access, use sync.Mutex or create separate instances per goroutine.

## Building

```bash
# Ensure native library is built
cd $PROJECT_ROOT
zig build

# Build Go code
cd bindings/go
go build ./...

# Run tests
go test ./...

# Run example
go run example/main.go
```

## License

Same license as the ABI Framework (see repository root).
