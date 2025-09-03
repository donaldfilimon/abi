# WDBX-AI Vector Database

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/wdbx/wdbx-ai)
[![Zig](https://img.shields.io/badge/zig-0.15.1-orange.svg)](https://ziglang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

WDBX-AI is a high-performance vector database written in Zig, designed for AI and machine learning applications. It features HNSW indexing, SIMD optimizations, and a comprehensive plugin system.

## Features

- 🚀 **High Performance**: SIMD-optimized vector operations
- 🔍 **Fast Search**: HNSW (Hierarchical Navigable Small World) indexing
- 🧠 **AI Integration**: Built-in neural network support
- 🔌 **Extensible**: Plugin system for custom functionality
- 🛡️ **Production Ready**: Comprehensive error handling and logging
- 📊 **Monitoring**: Built-in performance profiling and metrics
- 🔧 **Developer Friendly**: Clean API and extensive documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wdbx/wdbx-ai.git
cd wdbx-ai

# Build the project
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Install
zig build install
```

### Basic Usage

```zig
const std = @import("std");
const wdbx = @import("wdbx");

pub fn main() !void {
    // Initialize
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create database
    const db = try wdbx.database.create(allocator, "vectors.db");
    defer db.deinit();
    
    // Insert vectors
    const vector = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const id = try db.writeRecord(&vector);
    
    // Search similar vectors
    const results = try db.search(&vector, 10, .{
        .metric = .cosine,
    });
    
    for (results) |result| {
        std.debug.print("ID: {}, Score: {d}\n", .{ result.id, result.score });
    }
}
```

## Architecture

WDBX-AI is organized into several modules:

```
wdbx-ai/
├── src/
│   ├── core/          # Core utilities (memory, threading, logging)
│   ├── database/      # Database engine and storage
│   ├── simd/          # SIMD-optimized operations
│   ├── ai/            # Neural network and ML features
│   ├── plugins/       # Plugin system
│   └── wdbx/          # CLI and HTTP server
├── tests/             # Test suite
├── benchmarks/        # Performance benchmarks
├── examples/          # Example applications
└── docs/              # Documentation
```

## Performance

WDBX-AI is designed for maximum performance:

- **SIMD Optimization**: Automatic detection and use of AVX, SSE, and NEON instructions
- **Parallel Processing**: Built-in thread pool for concurrent operations
- **Memory Efficiency**: Custom allocators for different use cases
- **Zero-Copy Design**: Minimal data copying for optimal throughput

### Benchmarks

On a typical system (Intel i7, 16GB RAM):

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Insert | 1M vectors/sec | < 1μs |
| Search (HNSW) | 10K queries/sec | < 100μs |
| Batch Insert | 5M vectors/sec | - |
| Range Query | 50K queries/sec | < 20μs |

## Configuration

### Environment Variables

```bash
export WDBX_DATABASE_PATH=/path/to/db
export WDBX_CACHE_SIZE=1073741824  # 1GB
export WDBX_LOG_LEVEL=info
export WDBX_THREAD_COUNT=8
```

### Configuration File

Create a `config.toml` file:

```toml
[database]
path = "wdbx.db"
cache_size = 1073741824
enable_compression = true
checkpoint_interval = 1000

[server]
host = "0.0.0.0"
port = 8080
max_connections = 100

[ai]
model_path = "models/"
embedding_dimensions = 768

[performance]
thread_count = 0  # 0 = auto-detect
use_simd = true
batch_size = 1000
```

## CLI Usage

WDBX-AI includes a comprehensive CLI:

```bash
# Start interactive REPL
wdbx

# Execute commands
wdbx create mydatabase
wdbx insert mydatabase vector.json
wdbx search mydatabase query.json --top-k 10
wdbx info mydatabase

# Start HTTP server
wdbx serve --port 8080

# Benchmark
wdbx benchmark --duration 60
```

## HTTP API

Start the HTTP server:

```bash
wdbx serve --port 8080
```

### Endpoints

```bash
# Health check
GET /health

# Insert vector
POST /vectors
{
  "id": "vec1",
  "vector": [0.1, 0.2, 0.3],
  "metadata": {"category": "example"}
}

# Search vectors
POST /search
{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 10,
  "metric": "cosine"
}

# Get vector by ID
GET /vectors/{id}

# Delete vector
DELETE /vectors/{id}

# Database statistics
GET /stats
```

## Plugin Development

Create custom plugins to extend WDBX-AI:

```zig
const std = @import("std");
const wdbx = @import("wdbx");

pub fn init(allocator: std.mem.Allocator) !void {
    // Plugin initialization
}

pub fn process(data: []const u8) ![]u8 {
    // Custom processing logic
    return data;
}

pub fn deinit() void {
    // Cleanup
}

// Export plugin interface
pub const plugin = wdbx.plugins.Interface{
    .name = "custom_processor",
    .version = "1.0.0",
    .init = init,
    .deinit = deinit,
    .functions = &[_]wdbx.plugins.Function{
        .{ .name = "process", .ptr = process },
    },
};
```

## Advanced Features

### Neural Network Integration

```zig
// Create and train a neural network
var network = try wdbx.ai.NeuralNetwork.init(allocator, .{
    .input_size = 784,
    .hidden_sizes = &[_]usize{ 128, 64 },
    .output_size = 10,
    .activation = .relu,
});
defer network.deinit();

try network.train(training_data, .{
    .epochs = 100,
    .batch_size = 32,
    .learning_rate = 0.001,
});

// Use for inference
const output = try network.forward(input);
```

### Custom Indexing

```zig
// Implement custom index
const CustomIndex = struct {
    // ... implementation
};

// Register with database
try db.registerIndex("custom", CustomIndex);
```

### Streaming Operations

```zig
// Stream large datasets
var stream = try db.createStream();
defer stream.close();

while (try stream.next()) |batch| {
    // Process batch of vectors
}
```

## Monitoring and Debugging

### Performance Monitoring

```zig
// Enable performance monitoring
var monitor = try wdbx.performance.Monitor.init(allocator);
defer monitor.deinit();

// Record metrics
try monitor.recordLatency("query", latency_ms);

// Get statistics
const stats = monitor.getStats();
```

### Debug Logging

```zig
// Enable debug logging
var logger = try wdbx.core.logging.Logger.init(allocator, .{
    .level = .trace,
    .output = .{ .file = "debug.log" },
});
```

### Memory Tracking

```zig
// Track memory usage
var tracker = wdbx.memory_tracker.TrackedAllocator.init(allocator);
defer {
    const stats = tracker.getStats();
    std.log.info("Peak memory: {} bytes", .{stats.peak_usage});
    tracker.deinit();
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install Zig 0.15.1
# See https://ziglang.org/download/

# Clone repository
git clone https://github.com/wdbx/wdbx-ai.git
cd wdbx-ai

# Run tests
zig build test

# Run benchmarks
zig build bench

# Format code
zig build fmt

# Generate documentation
zig build docs
```

## License

WDBX-AI is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/wdbx/wdbx-ai/issues)
- 💬 [Discussions](https://github.com/wdbx/wdbx-ai/discussions)
- 📧 Email: support@wdbx.ai

## Acknowledgments

WDBX-AI builds upon excellent work from the Zig community and incorporates ideas from:
- Facebook's Faiss
- Spotify's Annoy
- Google's ScaNN
- The Zig standard library

## Roadmap

- [ ] Distributed clustering support
- [ ] GPU acceleration
- [ ] Additional index types (LSH, IVF)
- [ ] Language bindings (Python, Rust, Go)
- [ ] Cloud-native deployment options
- [ ] Advanced query optimization
- [ ] Real-time index updates

---

Made with ❤️ by the WDBX team