# ABI Rust Bindings

Safe Rust bindings for the ABI Framework, providing:

- SIMD-accelerated vector operations
- Vector database for similarity search
- GPU acceleration
- AI agent capabilities

## Prerequisites

1. Build the ABI C library:
   ```bash
   cd bindings/c
   zig build
   ```

2. Set library path:
   ```bash
   # macOS
   export DYLD_LIBRARY_PATH=$PWD/../c/zig-out/lib:$DYLD_LIBRARY_PATH

   # Linux
   export LD_LIBRARY_PATH=$PWD/../c/zig-out/lib:$LD_LIBRARY_PATH
   ```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
abi = { path = "path/to/abi/bindings/rust" }
```

### Basic Example

```rust
use abi::{Framework, Simd, VectorDatabase};

fn main() -> Result<(), abi::Error> {
    // Initialize the framework
    let framework = Framework::new()?;
    println!("ABI version: {}", framework.version());

    // SIMD vector operations
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let dot = Simd::dot_product(&a, &b);
    println!("Dot product: {}", dot);

    // Vector database
    let db = VectorDatabase::new("test", 128)?;
    db.insert(1, &vec![0.1; 128])?;

    let results = db.search(&vec![0.1; 128], 10)?;
    for result in results {
        println!("ID: {}, Score: {}", result.id, result.score);
    }

    Ok(())
}
```

### SIMD Operations

```rust
use abi::Simd;

// Check SIMD availability
if Simd::is_available() {
    let caps = Simd::capabilities();
    println!("Best SIMD: {}", caps.best_level());
}

// Vector operations
let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];

let sum = Simd::add(&a, &b);
let dot = Simd::dot_product(&a, &b);
let norm = Simd::l2_norm(&a);
let similarity = Simd::cosine_similarity(&a, &b);
let distance = Simd::euclidean_distance(&a, &b);
let normalized = Simd::normalize(&a);
```

### GPU Acceleration

```rust
use abi::{Gpu, gpu::Backend};

if Gpu::is_available() {
    let gpu = Gpu::new()?;
    println!("Backend: {}", gpu.backend_name());
}

// Or with specific config
let config = abi::gpu::GpuConfig {
    backend: Backend::Vulkan,
    device_index: 0,
    enable_profiling: true,
};
let gpu = Gpu::with_config(config)?;
```

### AI Agents

```rust
use abi::{Framework, Agent};

let framework = Framework::new()?;
let agent = Agent::new(&framework, "assistant")?;

let response = agent.chat("Hello!")?;
println!("Agent: {}", response);

agent.clear_history()?;
```

## Features

- `simd` (default): SIMD operations
- `gpu`: GPU acceleration
- `database`: Vector database
- `agent`: AI agent capabilities

## Building

```bash
cargo build
cargo test
cargo bench
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
