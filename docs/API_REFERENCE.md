# WDBX-AI API Reference

## Table of Contents

1. [Core Module](#core-module)
2. [Database Module](#database-module)
3. [SIMD Module](#simd-module)
4. [AI Module](#ai-module)
5. [Plugin System](#plugin-system)
6. [Error Handling](#error-handling)
7. [Configuration](#configuration)
8. [Utilities](#utilities)

## Core Module

The core module provides fundamental types and utilities used throughout WDBX-AI.

### Import

```zig
const core = @import("wdbx").core;
```

### Memory Management

#### Allocators

```zig
// Memory pool for fixed-size allocations
const pool = try core.allocators.MemoryPool.init(allocator, block_size, block_count);
defer pool.deinit();

const memory = pool.alloc() orelse return error.OutOfMemory;
defer pool.free(memory);

// Stack allocator for temporary allocations
var buffer: [1024]u8 = undefined;
var stack = core.allocators.StackAllocator.init(&buffer);
const stack_allocator = stack.allocator();
```

#### Collections

```zig
// Thread-safe ring buffer
var ring = try core.collections.RingBuffer.init(allocator, capacity);
defer ring.deinit();

try ring.write(data);
const bytes_read = try ring.read(buffer);

// Lock-free concurrent queue
var queue = try core.collections.ConcurrentQueue(i32).init(allocator);
defer queue.deinit();

try queue.enqueue(42);
const value = queue.dequeue();

// Priority queue
var pq = core.collections.PriorityQueue(Task, Context, compareFn).init(allocator, context);
defer pq.deinit();

try pq.push(task);
const highest_priority = pq.pop();
```

### Logging

```zig
// Initialize logger
var logger = try core.logging.Logger.init(allocator, .{
    .level = .info,
    .output = .{ .file = "app.log" },
    .use_color = true,
});
defer logger.deinit();

// Log messages
logger.info(@src(), "Application started", .{});
logger.warn(@src(), "Low memory: {d} bytes remaining", .{bytes_free});
logger.err(@src(), "Failed to connect: {s}", .{error_message});

// Scoped logger
const db_logger = core.logging.ScopedLogger.init(&logger, "database");
db_logger.debug(@src(), "Executing query: {s}", .{query});

// Performance logging
var perf = core.logging.PerfLogger.begin(&logger, "expensive_operation");
// ... do work ...
perf.checkpoint("halfway");
// ... more work ...
perf.end();
```

### Configuration

```zig
// Load configuration
var config = try core.config.Config.load(allocator, "config.toml");
defer config.deinit();

// Get values
const port = config.getInt("server.port") orelse 8080;
const debug = config.getBool("app.debug") orelse false;
const db_path = config.getString("database.path") orelse "data.db";

// Set values
try config.setString("app.version", "2.0.0");
try config.setInt("cache.size", 1024 * 1024);

// Save configuration
try config.save("config.toml");
```

### Threading

```zig
// Thread pool
var pool = try core.threading.ThreadPool.init(allocator, null); // null = auto-detect CPU count
defer pool.deinit();

// Submit tasks
try pool.submit(processItem, &item);
pool.wait(); // Wait for all tasks to complete

// Parallel operations
try core.threading.parallelFor(Item, items, &pool, processItem);

const results = try core.threading.parallelMap(
    i32, f32, allocator, numbers, &pool, 
    struct {
        fn convert(n: i32) f32 {
            return @as(f32, @floatFromInt(n)) * 2.0;
        }
    }.convert
);

// Channel for communication
var channel = try core.threading.Channel(Message).init(allocator, 100);
defer channel.deinit();

try channel.send(message);
const received = try channel.receive();
```

### Time Utilities

```zig
// High-resolution timer
var timer = core.time.Timer.start();
const elapsed_ns = timer.read();
const elapsed_ms = core.time.Duration.fromNanos(elapsed_ns).toMillis();

// Duration calculations
const timeout = core.time.Duration.fromSeconds(30);
const remaining = timeout.sub(elapsed);

// Rate limiting
var limiter = core.time.RateLimiter.init(100, 10); // 100/sec, burst of 10
if (limiter.allow(1)) {
    // Process request
}

// Deadlines
const deadline = core.time.Deadline.fromNow(core.time.Duration.fromSeconds(5));
while (!deadline.hasExpired()) {
    // Do work with timeout
}
```

## Database Module

The database module provides vector storage and retrieval with HNSW indexing.

### Import

```zig
const database = @import("wdbx").database;
```

### Basic Operations

```zig
// Create database
const db = try database.create(allocator, "vectors.db");
defer db.deinit();

// Write records
const record_id = try db.writeRecord(data);

// Read records
if (try db.readRecord(record_id)) |record| {
    std.debug.print("Record {d}: {} bytes\n", .{ record.id, record.data.len });
}

// Delete records
const deleted = try db.deleteRecord(record_id);

// Compact database
try db.compact();
```

### Batch Operations

```zig
// Batch insert
try database.BatchOps.insertBatch(db, records);

// Batch read
const results = try database.BatchOps.readBatch(db, record_ids, allocator);
defer allocator.free(results);
```

### Vector Operations

```zig
// Vector similarity metrics
const distance = database.Metric.cosine.distance(vector_a, vector_b);

// Search options
const options = database.SearchOptions{
    .metric = .euclidean,
    .filter = struct {
        fn filter(metadata: std.StringHashMap(database.MetadataValue)) bool {
            const category = metadata.get("category") orelse return false;
            return switch (category) {
                .string => |s| std.mem.eql(u8, s, "images"),
                else => false,
            };
        }
    }.filter,
};
```

### Import/Export

```zig
// Export to JSON
const writer = std.io.getStdOut().writer();
try database.ImportExport.exportJson(db, writer, record_ids);

// Import from JSON (not yet implemented)
// try database.ImportExport.importJson(db, reader);
```

## SIMD Module

The SIMD module provides optimized vector operations.

### Import

```zig
const simd = @import("wdbx").simd;
```

### Vector Operations

```zig
// SIMD vector operations
const result = simd.VectorOps.dotProduct(vec_a, vec_b);
const distance = simd.VectorOps.euclideanDistance(vec_a, vec_b);

// Check SIMD support
if (simd.Vector.isSimdAvailable(8)) {
    // Use 8-wide SIMD operations
}

// Matrix operations
var matrix = try simd.MatrixOps.create(allocator, rows, cols);
defer matrix.deinit();

try simd.MatrixOps.multiply(result, matrix_a, matrix_b);
```

## AI Module

The AI module provides neural network and embedding capabilities.

### Import

```zig
const ai = @import("wdbx").ai;
```

### Neural Networks

```zig
// Create neural network
var network = try ai.NeuralNetwork.init(allocator, .{
    .input_size = 784,
    .hidden_sizes = &[_]usize{ 128, 64 },
    .output_size = 10,
    .activation = .relu,
});
defer network.deinit();

// Forward pass
const output = try network.forward(input);

// Training
try network.train(training_data, .{
    .epochs = 100,
    .batch_size = 32,
    .learning_rate = 0.001,
});
```

### Embeddings

```zig
// Create embedding generator
var embedder = try ai.EmbeddingGenerator.init(allocator, .{
    .model_path = "models/embeddings.bin",
    .dimensions = 768,
});
defer embedder.deinit();

// Generate embeddings
const embedding = try embedder.embed(text);
```

## Plugin System

The plugin system allows extending WDBX-AI functionality.

### Import

```zig
const plugins = @import("wdbx").plugins;
```

### Using Plugins

```zig
// Load plugin
var plugin = try plugins.load(allocator, "plugins/custom_indexer.so");
defer plugin.deinit();

// Call plugin function
const result = try plugin.call("process", .{input_data});

// Register plugin
try plugins.registry.register("custom_indexer", plugin);
```

## Error Handling

WDBX-AI provides comprehensive error handling utilities.

### Import

```zig
const errors = @import("wdbx").errors;
```

### Error Types

```zig
// Use error types
return errors.WdbxError.DatabaseNotFound;

// Create error info
const info = errors.ErrorInfo.init(
    errors.WdbxError.DimensionMismatch,
    "Expected 768 dimensions, got 512"
)
.withContext("during vector insertion")
.withSource(@src());

// Result with error info
fn findVector(id: []const u8) errors.ResultWithInfo(Vector) {
    const vector = db.get(id) orelse {
        return .{ .err = errors.ErrorInfo.init(
            errors.WdbxError.RecordNotFound,
            "Vector not found"
        )};
    };
    return .{ .ok = vector };
}
```

### Error Recovery

```zig
// Retry with backoff
const result = try errors.RecoveryStrategy.retry_with_backoff.execute(
    DatabaseRecord,
    fetchFromDatabase,
    .{
        .max_retries = 5,
        .initial_delay_ms = 100,
        .backoff_factor = 2.0,
    }
);

// Error aggregation
var aggregator = errors.ErrorAggregator.init(allocator);
defer aggregator.deinit();

// Collect errors
for (files) |file| {
    processFile(file) catch |err| {
        try aggregator.addError(err, file);
    };
}

if (aggregator.hasErrors()) {
    std.log.err("{}", .{aggregator});
}
```

## Configuration

### Environment Variables

```bash
WDBX_DATABASE_PATH=/path/to/db
WDBX_CACHE_SIZE=1073741824  # 1GB
WDBX_LOG_LEVEL=debug
WDBX_THREAD_COUNT=8
```

### Configuration File (config.toml)

```toml
[database]
path = "wdbx.db"
cache_size = 1073741824
enable_compression = true

[server]
host = "0.0.0.0"
port = 8080
max_connections = 100

[ai]
model_path = "models/"
embedding_dimensions = 768
batch_size = 32

[performance]
thread_count = 0  # 0 = auto-detect
use_simd = true
```

## Utilities

### Memory Tracking

```zig
const tracker = @import("wdbx").memory_tracker;

// Track allocations
var tracked_allocator = tracker.TrackedAllocator.init(base_allocator);
defer tracked_allocator.deinit();

// Get statistics
const stats = tracked_allocator.getStats();
std.log.info("Peak memory usage: {} bytes", .{stats.peak_usage});
```

### Performance Monitoring

```zig
const perf = @import("wdbx").performance;

// Create monitor
var monitor = try perf.Monitor.init(allocator);
defer monitor.deinit();

// Record metrics
try monitor.recordLatency("query", latency_ms);
try monitor.recordThroughput("inserts", items_per_second);

// Get statistics
const stats = monitor.getStats("query");
std.log.info("Average query latency: {d:.2}ms", .{stats.avg});
```

## Example: Complete Application

```zig
const std = @import("std");
const wdbx = @import("wdbx");

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize WDBX
    try wdbx.init(allocator);
    defer wdbx.deinit();
    
    // Load configuration
    var config = try wdbx.core.config.Config.load(allocator, "config.toml");
    defer config.deinit();
    
    // Create logger
    var logger = try wdbx.core.logging.Logger.init(allocator, .{
        .level = .info,
        .output = .stderr,
    });
    defer logger.deinit();
    
    // Create database
    const db = try wdbx.database.create(allocator, "vectors.db");
    defer db.deinit();
    
    // Create thread pool
    var pool = try wdbx.core.threading.ThreadPool.init(allocator, null);
    defer pool.deinit();
    
    // Process data
    logger.info(@src(), "Starting vector processing", .{});
    
    const vectors = try loadVectors(allocator);
    defer allocator.free(vectors);
    
    // Insert vectors in parallel
    try wdbx.core.threading.parallelFor(
        Vector,
        vectors,
        &pool,
        struct {
            fn process(v: *Vector) void {
                db.writeRecord(v.data) catch |err| {
                    std.log.err("Failed to insert vector: {}", .{err});
                };
            }
        }.process
    );
    
    logger.info(@src(), "Processed {d} vectors", .{vectors.len});
}
```

## Best Practices

1. **Always use proper error handling**
   ```zig
   const result = operation() catch |err| {
       logger.err(@src(), "Operation failed: {}", .{err});
       return err;
   };
   ```

2. **Defer cleanup immediately after resource allocation**
   ```zig
   const resource = try allocate();
   defer resource.deinit();
   ```

3. **Use appropriate allocators for different use cases**
   - Arena allocator for batch operations
   - Pool allocator for fixed-size allocations
   - Stack allocator for temporary data

4. **Enable SIMD when available**
   ```zig
   if (wdbx.simd.Vector.isSimdAvailable(8)) {
       // Use SIMD operations
   }
   ```

5. **Use thread pools for CPU-intensive operations**
   ```zig
   var pool = try wdbx.core.threading.ThreadPool.init(allocator, null);
   defer pool.deinit();
   ```

6. **Monitor performance in production**
   ```zig
   var monitor = try wdbx.performance.Monitor.init(allocator);
   defer monitor.deinit();
   ```

7. **Configure appropriate cache sizes**
   ```toml
   [database]
   cache_size = 1073741824  # 1GB for large datasets
   ```

8. **Use batch operations when possible**
   ```zig
   try wdbx.database.BatchOps.insertBatch(db, records);
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase cache size
   - Use streaming operations
   - Enable compression

2. **Slow Queries**
   - Check index configuration
   - Optimize vector dimensions
   - Use appropriate similarity metrics

3. **High CPU Usage**
   - Limit thread pool size
   - Enable rate limiting
   - Profile hot paths

4. **Database Corruption**
   - Run integrity checks
   - Enable checksums
   - Use atomic operations

### Debug Mode

Enable debug logging:
```zig
var logger = try wdbx.core.logging.Logger.init(allocator, .{
    .level = .trace,
    .include_location = true,
});
```

### Performance Profiling

```zig
var profiler = try wdbx.performance.Profiler.init(allocator);
defer profiler.deinit();

profiler.begin("operation");
// ... do work ...
profiler.end("operation");

const report = try profiler.generateReport();
std.log.info("{s}", .{report});
```