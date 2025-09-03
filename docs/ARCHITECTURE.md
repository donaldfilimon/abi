# WDBX Architecture Guide

## Overview

WDBX (Vector Database for AI) is a high-performance, modular vector database designed for AI applications. It provides efficient storage, indexing, and retrieval of high-dimensional vector embeddings with a focus on performance, reliability, and ease of use.

## Architecture Principles

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Performance**: SIMD optimizations and efficient algorithms
3. **Flexibility**: Pluggable components for storage, indexing, and APIs
4. **Type Safety**: Leveraging Zig's compile-time features
5. **Simplicity**: Clean, understandable code structure

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌──────────────┐   │
│  │   CLI   │  │   HTTP   │  │  TCP  │  │  WebSocket   │   │
│  └────┬────┘  └────┬─────┘  └───┬───┘  └──────┬───────┘   │
│       │            │             │              │            │
│       └────────────┴─────────────┴──────────────┘           │
│                           │                                  │
├───────────────────────────┼─────────────────────────────────┤
│                     Core Database                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Database Manager                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────────┐   │   │
│  │  │ Config  │  │  Stats  │  │  Thread Safety   │   │   │
│  │  └─────────┘  └─────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌───────────────┬────────┴────────┬────────────────────┐  │
│  │    Vector     │     Index       │     Storage        │  │
│  │  Operations   │    Backend      │     Backend        │  │
│  │               │                 │                    │  │
│  │ ┌───────────┐ │ ┌────────────┐ │ ┌───────────────┐ │  │
│  │ │   SIMD    │ │ │    HNSW    │ │ │     File      │ │  │
│  │ │ Distance  │ │ │    Flat    │ │ │    Memory     │ │  │
│  │ │  Metrics  │ │ │    IVF     │ │ │     MMap      │ │  │
│  │ └───────────┘ │ └────────────┘ │ └───────────────┘ │  │
│  └───────────────┴─────────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Database Manager (`core/database.zig`)

The central component that coordinates all database operations:

- **Initialization**: Sets up storage and index backends
- **Vector Operations**: Add, update, delete vectors
- **Search**: Coordinate searches through index backend
- **Transaction Management**: Ensures data consistency
- **Statistics**: Tracks performance metrics

```zig
pub const Database = struct {
    allocator: std.mem.Allocator,
    config: DatabaseConfig,
    storage_backend: storage.StorageBackend,
    index_backend: index.IndexBackend,
    vector_ops: vector.VectorOps,
    // ...
};
```

### 2. Vector Operations (`core/vector/`)

Optimized vector operations with automatic SIMD detection:

- **Distance Metrics**: Euclidean, Cosine, Dot Product, Manhattan
- **SIMD Acceleration**: AVX2 (x86_64), NEON (ARM)
- **Vector Arithmetic**: Add, subtract, scale, normalize
- **Memory Alignment**: Optimal cache utilization

### 3. Index Backend (`core/index/`)

Pluggable indexing algorithms:

#### Flat Index
- Brute-force search
- 100% recall
- Best for small datasets (<10K vectors)

#### HNSW Index
- Hierarchical Navigable Small World
- Approximate nearest neighbor search
- Scalable to millions of vectors
- Configurable accuracy/speed trade-off

#### Future Indexes
- IVF (Inverted File): Clustering-based index
- LSH (Locality Sensitive Hashing): Hash-based index

### 4. Storage Backend (`core/storage/`)

Flexible storage options:

#### File Storage
- Binary format with efficient I/O
- Page-aligned operations
- Append-only writes for durability
- Defragmentation support

#### Memory Storage
- In-memory storage for testing
- Fast operations
- No persistence

#### Future Storage
- Memory-mapped files
- Distributed storage
- Cloud storage adapters

### 5. API Layer (`api/`)

Multiple interfaces for different use cases:

#### CLI (`api/cli/`)
- Interactive and batch modes
- Multiple output formats
- Scripting support

#### HTTP Server (`api/http/`)
- RESTful API
- JSON request/response
- JWT authentication
- WebSocket support

#### TCP Server (`api/tcp/`)
- Binary protocol
- Low latency
- Persistent connections

## Data Flow

### Write Path

1. **API Request**: Vector data received through API
2. **Validation**: Dimension and format checking
3. **Storage Write**: Append vector to storage backend
4. **Index Update**: Add vector to index structure
5. **Response**: Return vector ID to client

### Read Path

1. **API Request**: Query vector received
2. **Index Search**: Find nearest neighbors
3. **Storage Read**: Fetch vector data for results
4. **Post-processing**: Apply filters, limits
5. **Response**: Return search results

## Performance Optimizations

### SIMD Acceleration

Automatic detection and use of SIMD instructions:

```zig
// Automatic SIMD selection
pub fn init(allocator: std.mem.Allocator, use_simd: bool) !Self {
    const simd_ops = if (use_simd and simd.isSupported()) 
        try simd.SimdOps.init(allocator)
    else 
        null;
    // ...
}
```

### Memory Management

- Arena allocators for temporary data
- Memory pools for frequent allocations
- Aligned allocations for SIMD
- Zero-copy operations where possible

### Concurrency

- Thread-safe database operations
- Lock-free data structures
- Parallel index construction
- Concurrent searches

## Configuration

### Database Configuration

```zig
pub const DatabaseConfig = struct {
    dimensions: u32,
    index_type: index.IndexType = .hnsw,
    storage_type: storage.StorageType = .file,
    distance_metric: vector.DistanceMetric = .euclidean,
    enable_simd: bool = true,
    hnsw_m: u32 = 16,
    hnsw_ef_construction: u32 = 200,
    // ...
};
```

### API Configuration

```zig
pub const ApiConfig = struct {
    enable_auth: bool = true,
    enable_rate_limit: bool = true,
    rate_limit_rpm: u32 = 1000,
    enable_metrics: bool = true,
    // ...
};
```

## Error Handling

Comprehensive error handling with context:

```zig
pub const ErrorSet = error{
    InvalidFileFormat,
    CorruptedData,
    InvalidDimensions,
    // ...
};

pub const ErrorContext = struct {
    error_type: ErrorSet,
    message: []const u8,
    file: []const u8,
    line: u32,
    timestamp: i64,
};
```

## Testing Strategy

### Unit Tests
- Per-module tests
- Mock dependencies
- Property-based testing

### Integration Tests
- End-to-end scenarios
- Multi-component interactions
- Performance regression tests

### Benchmarks
- Micro-benchmarks for hot paths
- Macro-benchmarks for real workloads
- Comparison with other databases

## Deployment

### Single Node
- Simple file-based deployment
- Systemd service configuration
- Docker container support

### High Availability
- Read replicas
- Failover support
- Backup strategies

### Monitoring
- Prometheus metrics
- Health endpoints
- Performance dashboards

## Future Directions

1. **GPU Acceleration**: CUDA/ROCm support for large-scale operations
2. **Distributed Mode**: Sharding and replication
3. **Advanced Indexes**: Graph-based and learned indexes
4. **Query Optimization**: Cost-based optimizer
5. **Multi-modal Support**: Text, image, and audio embeddings