# ABI Framework API Reference

The ABI framework provides a comprehensive set of APIs for AI/ML applications, GPU acceleration, web services, and database operations.

## Table of Contents

- [Core Framework](#core-framework)
- [AI/ML APIs](#aiml-apis)
- [GPU Acceleration](#gpu-acceleration)
- [Web Services](#web-services)
- [Database Operations](#database-operations)
- [Python Bindings](#python-bindings)

## Core Framework

### Framework Initialization

```zig
const abi = @import("abi");

// Initialize framework with default options
var framework = try abi.init(allocator, abi.FrameworkOptions{});
defer abi.shutdown(&framework);

// Initialize with custom options
var framework = try abi.init(allocator, abi.FrameworkOptions{
    .enable_gpu = true,
    .enable_ai = true,
    .enable_web = true,
    .enable_database = true,
});
defer abi.shutdown(&framework);
```

### FrameworkOptions

```zig
pub const FrameworkOptions = struct {
    enable_gpu: bool = true,
    enable_ai: bool = true,
    enable_web: bool = true,
    enable_database: bool = true,
    log_level: std.log.Level = .info,
    max_memory_mb: usize = 1024,
};
```

## AI/ML APIs

### Transformer Models

```zig
const transformer = framework.ai.?.transformer;

// Create transformer configuration
const config = transformer.TransformerConfig{
    .vocab_size = 30000,
    .d_model = 512,
    .n_heads = 8,
    .n_layers = 6,
    .max_seq_len = 1024,
};

// Initialize transformer
var model = try transformer.Transformer.init(allocator, config);
defer model.deinit();

// Forward pass
const input_ids = [_]u32{1, 2, 3, 4}; // Token IDs
const output = try model.forward(&input_ids, .{});
```

#### TransformerConfig

```zig
pub const TransformerConfig = struct {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    max_seq_len: usize,
    dropout: f32 = 0.1,
    activation: Activation = .gelu,
};
```

### Vector Database

```zig
const database = framework.database.?;

// Create vector database
var db = try database.VectorDatabase.init(allocator, .{
    .dimensions = 512,
    .index_type = .hnsw,
    .metric = .cosine,
});
defer db.deinit();

// Insert vectors
const id = try db.insert(&embedding_vector);

// Search for similar vectors
const results = try db.search(&query_vector, .{.top_k = 10});
defer allocator.free(results);
```

### Reinforcement Learning

```zig
const rl = framework.ai.?.reinforcement_learning;

// Create Q-learning agent
var agent = try rl.QAgent.init(allocator, .{
    .state_size = 100,
    .action_count = 4,
    .learning_rate = 0.01,
    .gamma = 0.99,
});
defer agent.deinit();

// Select action
const action = agent.selectAction(&current_state);

// Update Q-values
try agent.update(&current_state, action, reward, &next_state);
```

## GPU Acceleration

### Unified Accelerator Interface

```zig
const accelerator = framework.shared.platform.accelerator;

// Create best available accelerator
var accel = accelerator.createBestAccelerator(allocator);
defer accel.deinit();

// Allocate GPU memory
const buffer = try accel.alloc(f32, 1024);
defer accel.free(buffer);

// Copy data to GPU
try accel.copyToDevice(buffer, &host_data);

// Launch compute kernel
try accel.launchKernel("vector_add", &.{buffer_a, buffer_b, result}, .{64, 1, 1});
```

### Vulkan Backend

```zig
const vulkan = framework.gpu.?.vulkan;

// Initialize Vulkan renderer
var renderer = try vulkan.VulkanRenderer.init(allocator);
defer renderer.deinit();

try renderer.initialize();

// Vector operations
try renderer.vectorAdd(allocator, &a, &b, &result);
try renderer.matrixMultiply(allocator, &a, &b, &result, m, n, k);
```

### SIMD-Optimized Vector Search

```zig
const vec_search = framework.gpu.?.vector_search;

// Initialize GPU-accelerated vector search
var searcher = vec_search.VectorSearchGPU.init(allocator, &accel, dimensions);
defer searcher.deinit();

// Insert vectors
const id = try searcher.insert(&embedding);

// Search (uses SIMD acceleration automatically)
const neighbors = try searcher.search(&query, top_k);
defer allocator.free(neighbors);
```

## Web Services

### HTTP Server

```zig
const web = framework.web.?;

// Create HTTP server
var server = try web.EnhancedWebServer.init(allocator, .{
    .port = 8080,
    .enable_ssl = false,
    .max_connections = 1000,
});
defer server.deinit();

// Add routes
try server.router.get("/api/health", healthHandler);
try server.router.post("/api/predict", predictHandler);

// Start server
try server.start();
```

### Authentication & JWT

```zig
const auth = server.auth_manager;

// Generate JWT token
const token = try auth.generateToken(allocator, "user_id");
defer allocator.free(token);

// Validate token
const is_valid = try auth.validateToken(token);
if (is_valid) {
    const user_id = try auth.getUserFromToken(token);
    defer allocator.free(user_id);
    // Use authenticated user_id
}
```

### WebSocket Support

```zig
// WebSocket handler
fn websocketHandler(ctx: *web.RequestContext) !void {
    var ws = try ctx.upgradeToWebSocket();
    defer ws.close();

    while (try ws.receive()) |message| {
        // Handle WebSocket message
        try ws.send(.{.text = "Echo: " ++ message.text});
    }
}
```

## Database Operations

### PostgreSQL Integration

```zig
const db = framework.database.?;

// Connect to database
var conn = try db.connect(.{
    .host = "localhost",
    .port = 5432,
    .database = "abi_db",
    .user = "abi_user",
    .password = "password",
});
defer conn.close();

// Execute query
const result = try conn.query("SELECT * FROM vectors WHERE id = $1", .{vector_id});
defer result.deinit();

// Prepared statements
const stmt = try conn.prepare("INSERT INTO vectors (data) VALUES ($1)");
defer stmt.deinit();

try stmt.execute(&.{&vector_data});
```

### Redis Caching

```zig
const cache = framework.database.?.redis;

// Connect to Redis
var client = try cache.connect("localhost:6379");
defer client.close();

// Cache operations
try client.set("key", "value", .{ .ttl = 3600 });
const value = try client.get(allocator, "key");
defer allocator.free(value);
```

## Python Bindings

### Installation

```bash
pip install abi-framework
```

### Basic Usage

```python
import abi

# Create transformer model
model = abi.Transformer({
    'vocab_size': 30000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6
})

# Process input
input_ids = [1, 2, 3, 4, 5]
output = model.forward(input_ids)

# Vector database
db = abi.VectorDatabase(dimensions=512)
db.insert([0.1, 0.2, ...])  # 512-dimensional vector
results = db.search(query_vector, top_k=10)

# Reinforcement learning
agent = abi.ReinforcementLearning("q_learning",
    state_size=100, action_count=4)
```

### Advanced Features

```python
# GPU acceleration
with abi.gpu_context():
    model = abi.Transformer(config)
    # Model runs on GPU automatically

# Custom training loops
optimizer = abi.Adam(learning_rate=0.001)
for batch in training_data:
    loss = model.train_step(batch, optimizer)

# Model serialization
model.save("model.bin")
model.load("model.bin")
```

## Error Handling

All ABI APIs return errors using Zig's error union types:

```zig
const result = api_call() catch |err| {
    switch (err) {
        error.OutOfMemory => // Handle memory errors
        error.InvalidInput => // Handle validation errors
        error.GPUUnavailable => // Handle GPU errors
        else => // Handle other errors
    }
    return err;
};
```

## Memory Management

ABI follows Zig's ownership model:

- Functions that allocate memory return slices that must be freed by the caller
- Use `defer allocator.free(result)` to ensure cleanup
- Framework components have `deinit()` methods for proper cleanup
- Memory is managed per-allocator instance

## Performance Considerations

### GPU Acceleration
- Enable GPU features at build time: `zig build -Denable-gpu=true`
- Use unified accelerator interface for cross-platform GPU support
- Batch operations for optimal GPU utilization

### Memory Optimization
- Use arena allocators for temporary allocations
- Pre-allocate buffers when possible
- Monitor memory usage with built-in metrics

### Concurrent Processing
- Framework components are thread-safe where documented
- Use async/await for I/O operations
- Consider connection pooling for database operations

## Configuration

### Build-time Options

```bash
# Full feature build
zig build -Doptimize=ReleaseFast

# Minimal build
zig build -Denable-gpu=false -Denable-ai=false

# GPU-specific builds
zig build -Dgpu-vulkan=true -Dgpu-cuda=false
```

### Runtime Configuration

```zig
const config = abi.FrameworkOptions{
    .enable_gpu = true,
    .enable_ai = true,
    .enable_web = true,
    .enable_database = true,
    .log_level = .debug,
    .max_memory_mb = 2048,
};
```

## Troubleshooting

### Common Issues

1. **Build failures**: Ensure Zig 0.16+ is installed and all dependencies are available
2. **GPU not detected**: Check GPU drivers and ensure appropriate backend is enabled
3. **Memory issues**: Monitor memory usage and consider increasing limits
4. **Performance problems**: Enable GPU acceleration and check for bottlenecks

### Debugging

```zig
// Enable debug logging
const config = abi.FrameworkOptions{
    .log_level = .debug,
};

// Use debug builds for development
zig build -Doptimize=Debug
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and API design patterns.