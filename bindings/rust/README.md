# ABI Framework Rust Bindings

Rust bindings for the ABI Framework, providing safe wrappers around the native C library.

## Features

- **SIMD Operations**: High-performance vector math with automatic SIMD acceleration
- **Vector Database**: HNSW-indexed similarity search
- **GPU Acceleration**: Multi-backend support (CUDA, Vulkan, Metal, WebGPU)
- **AI Integration**: Agent system and LLM inference (requires `ai` feature)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
abi-framework = { path = "../bindings/rust" }
```

Or with specific features:

```toml
[dependencies]
abi-framework = { path = "../bindings/rust", features = ["full"] }
```

## Quick Start

```rust
use abi::{simd, Framework, Config};

fn main() -> abi::Result<()> {
    // Vector operations (no framework initialization needed)
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![4.0, 3.0, 2.0, 1.0];

    let similarity = simd::cosine_similarity(&a, &b);
    println!("Cosine similarity: {:.4}", similarity);

    let dot = simd::dot_product(&a, &b);
    println!("Dot product: {}", dot);

    // Framework initialization (requires native library)
    let framework = Framework::new(Config::default())?;
    println!("AI enabled: {}", framework.is_feature_enabled("ai"));

    Ok(())
}
```

## Modules

### SIMD (`abi::simd`)

SIMD-accelerated vector operations with automatic fallback to scalar code:

```rust
use abi::simd;

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![4.0, 3.0, 2.0, 1.0];

// Basic operations
let sum = simd::add(&a, &b);
let diff = simd::subtract(&a, &b);
let scaled = simd::scale(&a, 2.0);

// Vector math
let dot = simd::dot_product(&a, &b);
let norm = simd::l2_norm(&a);
let normalized = simd::normalize(&a);
let similarity = simd::cosine_similarity(&a, &b);
let distance = simd::euclidean_distance(&a, &b);

// Matrix multiplication (row-major order)
let mat_a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
let mat_b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
let result = simd::matrix_multiply(&mat_a, &mat_b, 2, 2, 2);
```

### Vector Database (`abi::database`)

High-performance vector database with HNSW indexing:

```rust
use abi::database::{VectorDatabase, VectorDatabaseBuilder};

// Create a database
let mut db = VectorDatabase::new("embeddings", 384)?;

// Or use the builder
let mut db = VectorDatabaseBuilder::new("embeddings", 384)
    .initial_capacity(10000)
    .build()?;

// Insert vectors
db.insert(1, &embedding1)?;
db.insert(2, &embedding2)?;

// Batch insert
db.insert_batch(&[
    (3, &embedding3),
    (4, &embedding4),
])?;

// Search for similar vectors
let results = db.search(&query_vector, 10)?;
for result in results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
}

// Delete a vector
db.delete(1)?;
```

### GPU Acceleration (`abi::gpu`)

GPU-accelerated matrix operations:

```rust
use abi::gpu::{GpuContext, Backend, Config};

// Check availability
if !abi::gpu::is_available() {
    println!("No GPU available");
    return Ok(());
}

// List devices
for device in abi::gpu::list_devices()? {
    println!("{} ({:?}) - {} MB",
        device.name,
        device.backend,
        device.total_memory / 1024 / 1024
    );
}

// Create context with auto-detected backend
let gpu = GpuContext::new(Backend::Auto)?;

// Or specify backend and device
let gpu = GpuContext::with_config(Config::with_backend(Backend::Cuda).device(0))?;

// GPU-accelerated operations
let result = gpu.matrix_multiply(&a, &b, m, n, k)?;
let sum = gpu.vector_add(&a, &b)?;
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | Yes | SIMD vector operations |
| `database` | Yes | Vector database |
| `gpu` | No | GPU acceleration |
| `ai` | No | AI agent system |
| `full` | No | All features |

## Building

The bindings require the ABI native library to be built and available:

```bash
# Build the native library first
cd ../..
zig build

# Then build the Rust bindings
cd bindings/rust
cargo build
```

For development without the native library, the SIMD module provides pure Rust fallbacks.

## Examples

Run the basic example:

```bash
cargo run --example basic
```

Run the vector database example:

```bash
cargo run --example vector_db --features database
```

## Thread Safety

- `Framework` is `Send` (can be moved between threads)
- `VectorDatabase` is `Send` (can be moved between threads)
- `GpuContext` is `Send` (can be moved between threads)

For concurrent access, wrap in `Arc<Mutex<T>>` or use separate instances per thread.

## Error Handling

All fallible operations return `abi::Result<T>`:

```rust
use abi::Error;

match framework.some_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(Error::NotInitialized) => println!("Framework not initialized"),
    Err(Error::FeatureDisabled(f)) => println!("Feature {} is disabled", f),
    Err(Error::InvalidArgument(msg)) => println!("Invalid argument: {}", msg),
    Err(e) => println!("Other error: {}", e),
}
```

## License

Same license as the ABI Framework (see repository root).
