# ABI Framework – Agent Guidelines

## Build/Test Commands

### Core Build Commands
```bash
zig build                                    # Build library + CLI (all features enabled by default)
zig build test                              # Run all tests across unit/integration/cross-platform
zig build run                               # Run CLI
zig build check                             # Aggregate validation (format + build + analysis)
zig build docs                              # Generate documentation
```

### Feature-Specific Builds
```bash
# Build with specific feature combinations
zig build -Denable-gpu=true -Denable-ai=true -Denable-web=true -Denable-database=true
zig build -Denable-gpu=false -Denable-ai=true  # AI only
zig build -Denable-database=true -Denable-web=false  # Database only

# Optimization levels
zig build -Doptimize=Debug         # Default: includes debug info
zig build -Doptimize=ReleaseFast   # Maximum performance, minimal safety
zig build -Doptimize=ReleaseSafe   # Performance with safety checks
zig build -Doptimize=ReleaseSmall  # Minimize binary size
```

### Running Individual Tests
```bash
# Unit tests
zig test tests/unit/test_build.zig
zig test tests/unit/test_database.zig
zig test tests/unit/test_gpu.zig
zig test tests/unit/test_ai.zig

# Integration tests
zig test tests/integration/database_ops_test.zig
zig test tests/integration/framework_lifecycle_test.zig
# Note: Some integration tests may have import path issues and need fixing

# Cross-platform tests
zig test tests/cross-platform/linux.zig
zig test tests/cross-platform/windows.zig
zig test tests/cross-platform/macos.zig

# Performance benchmarks
zig test tests/benchmarks/
```

### Development Tools
```bash
zig fmt .                    # Format all Zig files
zig fmt --check .           # Check formatting without modifying files
zig build -Denable-tracy=true  # Build with Tracy profiler integration
```

## Code Style Guidelines

### Formatting
- **Indentation**: 4 spaces, no tabs; lines under 100 characters; `zig fmt` required
- **Braces**: Zig standard placement (opening brace on same line)
- **Spacing**: Consistent spacing around operators and keywords

### Naming Conventions
```zig
// Types: PascalCase
pub const NeuralNetwork = struct { ... };
pub const VectorDatabase = struct { ... };
pub const ErrorContext = struct { ... };

// Functions: snake_case
pub fn create_database(allocator: Allocator) !*Database { ... }
pub fn initialize_neural_network(config: Config) !void { ... }
pub fn process_batch_data(data: []const f32) !Result { ... }

// Variables: snake_case
const max_connection_pool_size = 1000;
var current_allocator: Allocator = undefined;

// Constants: UPPER_SNAKE_CASE
const MAX_CONNECTION_POOL_SIZE = 1000;
const DEFAULT_TIMEOUT_MS = 5000;
const API_VERSION = "0.2.0";

// Files: snake_case
// ✓ allocators.zig, feature_manager.zig, vector_search.zig
// ✗ Allocators.zig, FeatureManager.zig, VectorSearch.zig
```

### Documentation Standards
```zig
//! Module-level documentation at the top of every file
//! This module provides comprehensive error handling utilities
//! for the ABI framework.

/// Calculate the squared Euclidean distance between two vectors.
/// This function uses SIMD optimizations when available for maximum performance.
/// Vectors must have the same length.
///
/// # Parameters
/// - `a`: First vector as a slice of f32 values
/// - `b`: Second vector as a slice of f32 values
///
/// # Returns
/// Squared Euclidean distance as f32
///
/// # Errors
/// - `DimensionMismatch`: When vectors have different lengths
/// - `OutOfMemory`: When allocation fails
///
/// # Example
/// ```zig
/// const distance = try distance_squared(&[3]f32{1, 2, 3}, &[3]f32{4, 5, 6});
/// // distance ≈ 27.0
/// ```
pub fn distance_squared(a: []const f32, b: []const f32) !f32 {
    // Implementation...
}
```

### Import Patterns
```zig
// Group std imports first, then local imports
const std = @import("std");
const testing = std.testing;
const mem = std.mem;
const fs = std.fs;

// Within framework modules: relative imports
const types = @import("../core/types.zig");
const errors = @import("../core/errors.zig");
const allocators = @import("../core/allocators.zig");

// In tests: import the framework module
const abi = @import("abi"); // Available in tests via build.zig
const framework = abi.framework;
const features = abi.features;

// Explicit imports only - avoid usingnamespace
// ✓ const ArrayList = std.ArrayList;
// ✗ usingnamespace std;

// ✓ Framework usage in tests
// const db = abi.database;
// const ai = abi.ai;
```

## Error Handling Patterns

### Error Types and Enums
```zig
// Define specific error types for each module
pub const DatabaseError = error{
    InvalidDimension,
    BufferTooSmall,
    CorruptedData,
    ConnectionFailed,
    TimeoutExceeded,
};

// Framework-wide error set
pub const Error = error{
    InvalidConfig,
    InvalidParameter,
    NotFound,
    AlreadyExists,
    Timeout,
    RateLimited,
    PermissionDenied,
    Unavailable,
    InternalError,
    OutOfMemory,
    NotImplemented,
    FeatureDisabled,
};
```

### Error Handling and Propagation
```zig
// Use ! return types for fallible functions
pub fn createDatabase(path: []const u8, allocator: Allocator) !*Database {
    // Validate parameters
    if (path.len == 0) {
        return DatabaseError.InvalidParameter;
    }

    // Use errdefer for cleanup on error
    var file = try fs.cwd().openFile(path, .{});
    errdefer file.close();

    var db = try allocator.create(Database);
    errdefer allocator.destroy(db);

    // Initialize database...
    try db.init(allocator);

    return db;
}

// Handle errors at appropriate levels
pub fn processRequest(req: Request) !Response {
    const db = try getDatabase();
    defer db.close();

    // Try an operation, handle specific errors
    const result = db.query(req.query) catch |err| switch (err) {
        DatabaseError.TimeoutExceeded => return error.RequestTimeout,
        DatabaseError.ConnectionFailed => return error.ServiceUnavailable,
        else => return err,
    };

    return Response{ .data = result };
}
```

### Error Context and Rich Diagnostics
```zig
pub const ErrorContext = struct {
    code: types.ErrorCode,
    message: []const u8,
    file: []const u8,
    line: u32,
    timestamp: i64,

    pub fn format(self: ErrorContext, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator,
            "Error {d} at {s}:{d} ({d}): {s}",
            .{ self.code, self.file, self.line, self.timestamp, self.message }
        );
    }
};
```

## Testing Organization

### Test Structure
- **Unit tests**: `tests/unit/` - individual component tests
- **Integration tests**: `tests/integration/` - system interaction tests
- **Cross-platform tests**: `tests/cross-platform/` - platform-specific validation
- **Benchmarks**: `tests/benchmarks/` - performance tests

### Test Patterns
```zig
// In test files, import the framework correctly
const std = @import("std");
const testing = std.testing;
const abi = @import("abi"); // Import framework via build.zig

// Test blocks at file end
test "feature: basic functionality" {
    const allocator = testing.allocator;

    // Setup
    var instance = try createInstance(allocator);
    defer instance.deinit();

    // Test execution
    const result = try instance.performOperation("test");

    // Assertions
    try testing.expectEqualStrings("expected", result);
}

test "feature: error handling" {
    const allocator = testing.allocator;

    // Test error conditions
    const result = createInstance(allocator);
    try testing.expectError(error.InvalidInput, result);
}

test "feature: memory safety" {
    const allocator = testing.allocator;

    // Test memory management
    var instance = try createInstance(allocator);
    defer instance.deinit();

    // Verify no memory leaks
    const stats = allocator.getStats();
    try testing.expectEqual(@as(usize, 0), stats.active_allocations);
}

test "performance: within baseline" {
    const allocator = testing.allocator;

    // Measure performance
    const start_time = std.time.nanoTimestamp();
    try performOperation(allocator);
    const end_time = std.time.nanoTimestamp();

    const duration_ns = @as(u64, @intCast(end_time - start_time));
    const duration_ms = duration_ns / 1_000_000;

    // Assert performance within acceptable range
    try testing.expectLessThan(duration_ms, MAX_ALLOWED_TIME_MS);
}
```

### Test Coverage Requirements
- **New Features**: 100% test coverage required
- **Bug Fixes**: Include regression tests
- **Performance Changes**: Include benchmark tests
- **API Changes**: Include integration tests
- **Error Conditions**: Test all error paths
- **Memory Safety**: Verify no leaks in all tests

## Memory Management Guidelines

### Resource Cleanup
```zig
// Always use defer for cleanup
pub fn createNeuralNetwork(allocator: Allocator) !*NeuralNetwork {
    var network = try allocator.create(NeuralNetwork);
    errdefer allocator.destroy(network);

    network.weights = try allocator.alloc(f32, 1000);
    errdefer allocator.free(network.weights);

    network.biases = try allocator.alloc(f32, 100);
    errdefer allocator.free(network.biases);

    // Initialize network...
    try network.init();

    return network;
}

// Use arena allocators for temporary data
pub fn processBatch(data: []const u8, allocator: Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    // Use temp_allocator for temporary allocations
    const parsed = try parseData(data, temp_allocator);
    const processed = try applyTransformations(parsed, temp_allocator);

    // Write results using main allocator
    try writeResults(processed, allocator);
}
```

### Allocator Patterns
- **General Purpose Allocator (GPA)**: For general allocation with leak detection in tests
- **Arena Allocators**: For temporary allocations with automatic cleanup
- **Fixed Buffer Allocators**: For known-size allocations
- **Explicit Allocators**: Always pass allocators as parameters

## Architecture and Module Organization

### Framework Structure
```
lib/
├── core/              # Core utilities (allocators, collections, diagnostics, errors, io, types)
├── features/          # Feature modules (ai, connectors, database, gpu, monitoring, web)
├── framework/         # Framework orchestration (catalog, config, feature_manager, runtime, state)
└── shared/            # Shared utilities (logging, observability, platform, simd, utils)
```

### Feature-Based Architecture
- **Compile-time feature selection** via build options
- **Explicit module exports** (no `usingnamespace`)
- **Clean separation** between core, features, and framework layers
- **Modular design** with clear boundaries

### Zig 0.16 Patterns
- **VTable allocators**: Use `vtable.*` allocators for polymorphic types
- **Async I/O patterns**: Leverage Zig's async/await for I/O operations
- **SIMD operations**: Use `@Vector` types and operations where beneficial
- **Comptime features**: Maximize compile-time computation and validation

## Advanced Patterns and Best Practices

### Plugin System Implementation
```zig
pub const PluginLoader = struct {
    loaded_libraries: std.ArrayListUnmanaged(LoadedLibrary),
    plugin_paths: std.ArrayListUnmanaged([]u8),

    pub fn addPluginPath(self: *PluginLoader, path: []const u8) !void {
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        try self.plugin_paths.append(self.allocator, owned_path);
    }

    pub fn loadPlugin(self: *PluginLoader, path: []const u8) !*const PluginInterface {
        // Safe plugin loading with cleanup on error
        errdefer self.unloadLibrary(handle) catch {};
        // Implementation...
    }
};
```

### Advanced SIMD with Performance Monitoring
```zig
pub const PerformanceMonitor = struct {
    operation_count: std.atomic.Value(u64),
    simd_usage_count: std.atomic.Value(u64),

    pub fn recordOperation(self: *PerformanceMonitor, duration_ns: u64, used_simd: bool) void {
        _ = self.operation_count.fetchAdd(1, .monotonic);
        if (used_simd) {
            _ = self.simd_usage_count.fetchAdd(1, .monotonic);
        }
    }
};

pub fn vectorLeakyRelu(data: []f32, slope: f32) void {
    const zero_vec = @as(FloatVector, @splat(@as(f32, 0.0)));
    const slope_vec = @as(FloatVector, @splat(slope));

    var i: usize = 0;
    while (i < data.len) : (i += SIMD_WIDTH) {
        const vec = loadVector(data[i..@min(i + SIMD_WIDTH, data.len)]);
        const mask = vec < zero_vec;
        const leaky = vec * slope_vec;
        const blended = @select(f32, mask, leaky, vec);
        storeVector(blended, data[i..@min(i + SIMD_WIDTH, data.len)]);
    }
}
```

### Performance Profiling Infrastructure
```zig
pub const CallRecord = struct {
    function_name: []const u8,
    file: []const u8,
    line: u32,
    entry_time: u64,
    exit_time: u64,
    depth: u32,
    parent_id: ?u64,
    call_id: u64,
    thread_id: std.Thread.Id,

    pub fn duration(self: CallRecord) u64 {
        return self.exit_time - self.entry_time;
    }
};

pub const PerformanceProfiler = struct {
    call_records: std.ArrayList(CallRecord),
    current_depth: std.atomic.Value(u32),

    pub fn startCall(self: *PerformanceProfiler, function_name: []const u8) !u64 {
        const call_id = self.nextCallId();
        const record = CallRecord{
            .function_name = try self.allocator.dupe(u8, function_name),
            .entry_time = std.time.nanoTimestamp(),
            // ... other fields
        };
        try self.call_records.append(record);
        return call_id;
    }
};
```

### Error Context Chaining
```zig
pub const ErrorContext = struct {
    code: types.ErrorCode,
    message: []const u8,
    file: []const u8,
    line: u32,
    timestamp: i64,
    cause: ?*const ErrorContext = null,

    pub fn withCause(self: ErrorContext, cause: *const ErrorContext) ErrorContext {
        return ErrorContext{
            .code = self.code,
            .message = self.message,
            .file = self.file,
            .line = self.line,
            .timestamp = self.timestamp,
            .cause = cause,
        };
    }

    pub fn formatChain(self: ErrorContext, allocator: Allocator) ![]const u8 {
        var chain = std.ArrayList(u8).init(allocator);
        errdefer chain.deinit();

        var current: ?*const ErrorContext = &self;
        while (current) |ctx| {
            try chain.writer().print("Error {d}: {s}\n", .{ctx.code, ctx.message});
            current = ctx.cause;
        }

        return chain.toOwnedSlice();
    }
};
```

## Performance Guidelines

### SIMD Optimizations
```zig
// Use SIMD operations for vector computations
pub fn vector_add(a: []const f32, b: []const f32, result: []f32) void {
    // SIMD version (when available)
    const Vec4 = @Vector(4, f32);
    var i: usize = 0;
    while (i + 4 <= a.len) : (i += 4) {
        const va: Vec4 = a[i..][0..4].*;
        const vb: Vec4 = b[i..][0..4].*;
        result[i..][0..4].* = va + vb;
    }
    // Handle remaining elements...
}
```

### Memory Layout Optimization
- **Cache alignment**: Align data structures for optimal cache performance
- **Prefetching**: Use prefetch hints for predictable access patterns
- **Contiguous allocation**: Minimize pointer chasing and cache misses

### Performance Testing
```zig
// Include performance benchmarks for critical paths
test "performance: matrix multiplication" {
    const allocator = testing.allocator;

    // Setup test data
    const size = 1000;
    var a = try allocator.alloc(f32, size * size);
    defer allocator.free(a);
    // ... initialize matrices

    // Time the operation
    const start = std.time.nanoTimestamp();
    try matrixMultiply(a, b, result);
    const end = std.time.nanoTimestamp();

    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;

    // Assert performance requirements
    try testing.expectLessThan(duration_ms, 100.0); // Must complete within 100ms

    // Log for regression tracking
    std.debug.print("Matrix multiplication ({d}x{d}): {d:.2}ms\n", .{size, size, duration_ms});
}
```

### Performance Targets and Benchmarks

The ABI framework establishes specific performance targets to ensure optimal operation across different use cases. All performance benchmarks must meet these targets for commits to be accepted.

#### Core Performance Targets

##### Vector Operations
- **Vector Addition (1M elements)**: < 1.0ms on modern hardware
- **Vector Dot Product (1K dimensions)**: < 0.1ms per operation
- **Matrix Multiplication (256x256)**: < 50ms on CPU, < 10ms with GPU acceleration
- **Vector Database Search (10K vectors, 384D)**: < 5ms for top-10 results

##### Neural Network Operations
- **Forward Pass (1000 neurons, 1 layer)**: < 2ms
- **Training Step (batch size 32)**: < 10ms per step
- **Embedding Generation (512 tokens)**: < 50ms

##### Database Operations
- **Vector Insertion (384D)**: < 0.5ms per vector
- **Similarity Search (10K vectors)**: < 1ms for top-K retrieval
- **Batch Insert (1000 vectors)**: < 100ms

##### Web Server Performance
- **Request Latency (simple endpoint)**: < 1ms
- **Concurrent Requests (100 clients)**: Maintain < 10ms p95 latency
- **WebSocket Message Round-trip**: < 5ms

##### Plugin System
- **Plugin Load Time**: < 50ms per plugin
- **Plugin Execution Overhead**: < 0.1ms per call
- **Memory Overhead**: < 1MB per loaded plugin

#### Benchmarking Procedures

##### Running Benchmarks

```bash
# Run all benchmarks
zig build test -- tests/benchmarks/

# Run specific benchmark suite
zig test tests/benchmarks/ai_benchmarks.zig

# Run with performance profiling
zig build -Dtracy=true test -- tests/benchmarks/

# Generate benchmark reports
zig build test -- tests/benchmarks/ 2>&1 | tee benchmark_results.txt
```

##### Benchmark Categories

###### Micro-benchmarks (`tests/benchmarks/`)
- **SIMD Operations**: Test vector math performance
- **Memory Operations**: Test allocation/deallocation speed
- **System Calls**: Test I/O performance

###### Integration Benchmarks (`tests/integration/`)
- **End-to-End AI Pipeline**: Complete ML workflow performance
- **Database Operations**: Full CRUD with indexing
- **Network Operations**: Web server and API performance

###### Performance Regression Tests
- **Historical Comparison**: Compare against previous releases
- **Memory Leak Detection**: Run with address sanitizer
- **CPU Usage Monitoring**: Track per-operation CPU consumption

##### Benchmark Implementation Guidelines

```zig
const std = @import("std");
const testing = std.testing;

/// Standard benchmark structure
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u32,
    total_time_ns: u64,
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    memory_used_bytes: usize,

    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        const avg_ms = @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000_000.0;
        const min_ms = @as(f64, @floatFromInt(self.min_time_ns)) / 1_000_000.0;
        const max_ms = @as(f64, @floatFromInt(self.max_time_ns)) / 1_000_000.0;
        const memory_kb = self.memory_used_bytes / 1024;

        try writer.print("{s}: {d:.3}ms avg ({d:.3}ms min, {d:.3}ms max) - {d}KB mem\n",
            .{self.name, avg_ms, min_ms, max_ms, memory_kb});
    }
};

/// Benchmark runner with statistical analysis
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayList(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator) BenchmarkRunner {
        return .{
            .allocator = allocator,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        self.results.deinit();
    }

    /// Run benchmark with multiple iterations and statistical analysis
    pub fn runBenchmark(
        self: *BenchmarkRunner,
        name: []const u8,
        comptime benchmark_fn: fn() anyerror!void,
        iterations: u32,
    ) !void {
        var times = try std.ArrayList(u64).initCapacity(self.allocator, iterations);
        defer times.deinit();

        var memory_usage: usize = 0;

        // Warm-up run
        try benchmark_fn();

        // Timed runs
        for (0..iterations) |_| {
            const start = std.time.nanoTimestamp();
            try benchmark_fn();
            const end = std.time.nanoTimestamp();
            try times.append(@intCast(end - start));

            // Track memory usage (simplified)
            memory_usage = @max(memory_usage, 1024 * 1024); // Placeholder
        }

        // Calculate statistics
        var total: u64 = 0;
        var min_time = times.items[0];
        var max_time = times.items[0];

        for (times.items) |t| {
            total += t;
            min_time = @min(min_time, t);
            max_time = @max(max_time, t);
        }

        const avg_time = total / iterations;

        try self.results.append(.{
            .name = try self.allocator.dupe(u8, name),
            .iterations = iterations,
            .total_time_ns = total,
            .avg_time_ns = avg_time,
            .min_time_ns = min_time,
            .max_time_ns = max_time,
            .memory_used_bytes = memory_usage,
        });
    }

    pub fn printReport(self: BenchmarkRunner) void {
        std.debug.print("\n=== Benchmark Results ===\n", .{});
        for (self.results.items) |result| {
            std.debug.print("{}", .{result});
        }
        std.debug.print("========================\n", .{});
    }
};
```

##### Performance Regression Detection

```zig
/// Performance regression test
test "performance: no regression in vector operations" {
    var runner = BenchmarkRunner.init(testing.allocator);
    defer runner.deinit();

    // Run current implementation
    try runner.runBenchmark("vector_add_1M", benchmarkVectorAdd1M, 100);

    const result = runner.results.items[0];

    // Check against performance targets
    const target_ns: u64 = 1_000_000; // 1ms in nanoseconds
    try testing.expectLessThan(result.avg_time_ns, target_ns);

    // Log for monitoring
    std.debug.print("Vector add performance: {}ns avg\n", .{result.avg_time_ns});

    // In CI, compare against baseline stored in repository
    // if (std.fs.cwd().openFile("performance_baseline.json", .{})) |baseline_file| {
    //     // Compare and fail if regression detected
    // }
}

fn benchmarkVectorAdd1M() !void {
    const size = 1_000_000;
    var a = try testing.allocator.alloc(f32, size);
    defer testing.allocator.free(a);
    var b = try testing.allocator.alloc(f32, size);
    defer testing.allocator.free(b);
    var result = try testing.allocator.alloc(f32, size);
    defer testing.allocator.free(result);

    // Initialize test data
    for (a, 0..) |*val, i| val.* = @floatFromInt(i);
    for (b, 0..) |*val, i| val.* = @floatFromInt(i * 2);

    // Perform operation
    for (a, b, result) |x, y, *r| {
        r.* = x + y;
    }
}
```

##### Memory Performance Benchmarks

```zig
test "memory: allocation performance" {
    const allocator = testing.allocator;

    const start = std.time.nanoTimestamp();

    // Allocate and free many small objects
    for (0..10000) |_| {
        const data = try allocator.alloc(u8, 1024);
        defer allocator.free(data);
        // Touch memory to ensure allocation
        data[0] = 1;
    }

    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;

    // Should complete within reasonable time
    try testing.expectLessThan(duration_ms, 100.0); // 100ms for 10K allocations

    std.debug.print("Memory allocation benchmark: {d:.2}ms\n", .{duration_ms});
}
```

#### Benchmark Maintenance

- **Baseline Updates**: Update performance baselines quarterly
- **Hardware Consistency**: Run benchmarks on standardized hardware configurations
- **Statistical Analysis**: Use statistical methods to account for system variability
- **Continuous Monitoring**: Track performance trends over time
- **Failure Investigation**: Investigate and document any performance regressions

## Development Workflow

### Daily Development
1. **Format code**: `zig fmt .`
2. **Build and test**: `zig build test`
3. **Run specific tests**: `zig test tests/unit/test_build.zig`
4. **Verify performance**: Run relevant benchmarks
5. **Check documentation**: `zig build docs`

### Before Committing
1. **All tests pass**: `zig build test`
2. **Code formatted**: `zig fmt --check .`
3. **No memory leaks**: Test with GPA leak detection
4. **Performance maintained**: Run performance benchmarks
5. **Documentation updated**: All public APIs documented

### Feature Development
1. **Create feature branch**: `git checkout -b feature/your-feature-name`
2. **Implement with tests**: Write tests alongside implementation
3. **Performance testing**: Include benchmarks for performance-critical code
4. **Documentation**: Update docs and examples
5. **Code review**: Ensure style guidelines are followed

## CI/CD Pipeline

The ABI framework uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline ensures code quality, runs comprehensive tests, and automates releases.

### GitHub Actions Workflows

Create the following workflow files in `.github/workflows/`:

#### CI Pipeline (`ci.yml`)
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        zig-version: ['0.13.0']

    steps:
    - uses: actions/checkout@v4

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: ${{ matrix.zig-version }}

    - name: Verify Zig version
      run: zig version

    - name: Check formatting
      run: zig fmt --check .

    - name: Build library
      run: zig build

    - name: Run unit tests
      run: zig build test

    - name: Run integration tests
      run: zig test tests/integration/

    - name: Build CLI
      run: zig build -Denable-gpu=false

    - name: Test CLI
      run: zig build run -- --help

  performance:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
    - uses: actions/checkout@v4

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: '0.13.0'

    - name: Run performance benchmarks
      run: zig build test -- tests/benchmarks/

    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json

  security:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: '0.13.0'

    - name: Run security scan
      run: |
        # Check for common security issues
        zig build -Doptimize=ReleaseSafe

    - name: Memory safety check
      run: |
        # Run tests with leak detection
        zig build test -Doptimize=Debug

  docs:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: '0.13.0'

    - name: Generate documentation
      run: zig build docs

    - name: Deploy docs to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/
```

#### Release Pipeline (`release.yml`)
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: '0.13.0'

    - name: Get version
      id: get_version
      run: |
        VERSION=${GITHUB_REF#refs/tags/v}
        echo "version=$VERSION" >> $GITHUB_OUTPUT

    - name: Build release binaries
      run: |
        # Build for multiple platforms
        zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux
        zig build -Doptimize=ReleaseFast -Dtarget=x86_64-windows
        zig build -Doptimize=ReleaseFast -Dtarget=x86_64-macos
        zig build -Doptimize=ReleaseFast -Dtarget=aarch64-linux

    - name: Create release archives
      run: |
        mkdir release
        tar -czf release/abi-${{ steps.get_version.outputs.version }}-linux-x86_64.tar.gz zig-out/bin/abi
        zip release/abi-${{ steps.get_version.outputs.version }}-windows-x86_64.zip zig-out/bin/abi.exe
        # Add other platforms...

    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          release/*.tar.gz
          release/*.zip
        generate_release_notes: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  publish:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && contains(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/checkout@v4

    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      run: |
        docker build -t ghcr.io/${{ github.repository }}:${{ github.ref_name }} .
        docker push ghcr.io/${{ github.repository }}:${{ github.ref_name }}

        # Also tag as latest for main releases
        if [[ ${{ github.ref_name }} =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
          docker tag ghcr.io/${{ github.repository }}:${{ github.ref_name }} ghcr.io/${{ github.repository }}:latest
          docker push ghcr.io/${{ github.repository }}:latest
        fi
```

#### CodeQL Security Scan (`codeql.yml`)
```yaml
name: "CodeQL"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Mondays

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix:
        language: [ 'cpp', 'javascript' ]  # Zig not yet supported, use C++ analysis

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}

    - name: Setup Zig
      uses: mlugg/setup-zig@v1
      with:
        version: '0.13.0'

    - name: Build project
      run: zig build

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
```

### Local CI Simulation

To test CI locally before pushing:

```bash
# Check formatting
zig fmt --check .

# Build and test
zig build test

# Run specific test suites
zig test tests/unit/
zig test tests/integration/

# Performance testing
zig build test -- tests/benchmarks/

# Documentation generation
zig build docs
```

### Release Process

1. **Version bumping**: Update version in `build.zig` and commit
2. **Create release branch**: `git checkout -b release/v1.2.3`
3. **Update changelog**: Document changes in `CHANGELOG.md`
4. **Create annotated tag**: `git tag -a v1.2.3 -m "Release version 1.2.3"`
5. **Push tag**: `git push origin v1.2.3`
6. **CI/CD handles the rest**: GitHub Actions builds releases and publishes

### Docker Integration

For containerized deployments:

```dockerfile
# Dockerfile
FROM ziglang/zig:0.13.0 as builder

WORKDIR /app
COPY . .

# Build the application
RUN zig build -Doptimize=ReleaseFast

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/zig-out/bin/abi /usr/local/bin/abi

EXPOSE 8080
CMD ["abi", "serve"]
```

Build and run:
```bash
docker build -t abi-framework .
docker run -p 8080:8080 abi-framework
```

## Security Considerations

- **No secrets in code**: Never commit API keys, passwords, or tokens
- **Input validation**: Always validate external inputs
- **Resource limits**: Implement rate limiting and resource bounds
- **Memory safety**: Leverage Zig's compile-time guarantees
- **Error handling**: Don't leak sensitive information in error messages

## Practical Examples

This section provides concrete code examples demonstrating the patterns and conventions documented above.

### Complete Feature Implementation Example

```zig
//! Example: Custom AI Feature Implementation
//! Demonstrates framework integration, error handling, and testing

const std = @import("std");
const abi = @import("abi");

/// Custom AI Processor implementing the framework patterns
pub const CustomAIProcessor = struct {
    allocator: std.mem.Allocator,
    config: Config,
    neural_network: ?*abi.ai.NeuralNetwork = null,

    pub const Config = struct {
        model_path: []const u8,
        max_batch_size: u32 = 32,
        enable_gpu: bool = true,
        memory_limit_mb: u32 = 1024,
    };

    /// Initialize with proper error handling and resource management
    pub fn init(allocator: std.mem.Allocator, config: Config) !*CustomAIProcessor {
        // Validate configuration
        if (config.max_batch_size == 0) {
            return abi.core.Error.InvalidParameter;
        }

        const self = try allocator.create(CustomAIProcessor);
        errdefer allocator.destroy(self);

        // Duplicate config strings to avoid lifetime issues
        const model_path = try allocator.dupe(u8, config.model_path);
        errdefer allocator.free(model_path);

        self.* = .{
            .allocator = allocator,
            .config = .{
                .model_path = model_path,
                .max_batch_size = config.max_batch_size,
                .enable_gpu = config.enable_gpu,
                .memory_limit_mb = config.memory_limit_mb,
            },
            .neural_network = null,
        };

        // Initialize neural network with framework integration
        try self.initializeNeuralNetwork();
        errdefer self.deinitNeuralNetwork();

        return self;
    }

    /// Clean shutdown with proper resource cleanup
    pub fn deinit(self: *CustomAIProcessor) void {
        self.deinitNeuralNetwork();
        self.allocator.free(self.config.model_path);
        self.allocator.destroy(self);
    }

    /// Process input with SIMD optimization and error handling
    pub fn processBatch(self: *CustomAIProcessor, inputs: []const []const f32) ![][]f32 {
        if (inputs.len > self.config.max_batch_size) {
            return abi.core.Error.InvalidParameter;
        }

        var results = try std.ArrayList([]f32).initCapacity(self.allocator, inputs.len);
        defer {
            for (results.items) |result| self.allocator.free(result);
            results.deinit();
        }

        // Use SIMD operations for performance
        for (inputs) |input| {
            const output = try self.processSingleInput(input);
            try results.append(output);
        }

        return results.toOwnedSlice();
    }

    /// Internal method with detailed error context
    fn processSingleInput(self: *CustomAIProcessor, input: []const f32) ![]f32 {
        if (self.neural_network == null) {
            return abi.core.Error.NotInitialized;
        }

        // Allocate output buffer
        const output = try self.allocator.alloc(f32, self.getOutputSize());
        errdefer self.allocator.free(output);

        // Forward pass with error handling
        self.neural_network.?.forward(input, output) catch |err| {
            // Add context to error
            std.log.err("Neural network forward pass failed: {}", .{err});
            return err;
        };

        return output;
    }

    fn initializeNeuralNetwork(self: *CustomAIProcessor) !void {
        // Check if GPU is available and enabled
        const gpu_available = abi.gpu.isAvailable();
        const use_gpu = self.config.enable_gpu and gpu_available;

        // Create neural network configuration
        const config = abi.ai.NeuralNetworkConfig{
            .input_size = self.getInputSize(),
            .hidden_sizes = &[_]u32{256, 128, 64},
            .output_size = self.getOutputSize(),
            .activation = .relu,
            .use_gpu = use_gpu,
        };

        self.neural_network = try abi.ai.createNeuralNetwork(self.allocator, config);
    }

    fn deinitNeuralNetwork(self: *CustomAIProcessor) void {
        if (self.neural_network) |nn| {
            abi.ai.destroyNeuralNetwork(nn);
            self.neural_network = null;
        }
    }

    fn getInputSize(self: *CustomAIProcessor) u32 {
        _ = self;
        return 784; // Example: 28x28 image flattened
    }

    fn getOutputSize(self: *CustomAIProcessor) u32 {
        _ = self;
        return 10; // Example: 10 classes
    }
};

/// Comprehensive test suite following testing patterns
test "CustomAIProcessor: initialization and basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test configuration
    const config = CustomAIProcessor.Config{
        .model_path = "/tmp/test_model",
        .max_batch_size = 16,
        .enable_gpu = false, // Disable for testing
    };

    // Initialize processor
    var processor = try CustomAIProcessor.init(allocator, config);
    defer processor.deinit();

    // Test basic functionality
    const input = try allocator.alloc(f32, processor.getInputSize());
    defer allocator.free(input);

    // Fill with test data
    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }

    // Process single input
    const output = try processor.processSingleInput(input);
    defer allocator.free(output);

    try std.testing.expectEqual(@as(usize, processor.getOutputSize()), output.len);

    // Verify output is reasonable (not all zeros)
    var has_nonzero = false;
    for (output) |v| {
        if (v != 0.0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "CustomAIProcessor: batch processing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = CustomAIProcessor.Config{
        .model_path = "/tmp/test_model",
        .max_batch_size = 8,
    };

    var processor = try CustomAIProcessor.init(allocator, config);
    defer processor.deinit();

    // Create batch of inputs
    const batch_size = 4;
    var inputs = try allocator.alloc([]f32, batch_size);
    defer {
        for (inputs) |input| allocator.free(input);
        allocator.free(inputs);
    }

    for (inputs, 0..) |*input, i| {
        input.* = try allocator.alloc(f32, processor.getInputSize());
        // Fill with different patterns for each input
        for (input.*, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt((i + j) % 10)) / 10.0;
        }
    }

    // Process batch
    const outputs = try processor.processBatch(inputs);
    defer {
        for (outputs) |output| allocator.free(output);
        allocator.free(outputs);
    }

    try std.testing.expectEqual(@as(usize, batch_size), outputs.len);
    for (outputs) |output| {
        try std.testing.expectEqual(@as(usize, processor.getOutputSize()), output.len);
    }
}

test "CustomAIProcessor: error handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test invalid configuration
    const invalid_config = CustomAIProcessor.Config{
        .model_path = "",
        .max_batch_size = 0, // Invalid
    };

    const result = CustomAIProcessor.init(allocator, invalid_config);
    try std.testing.expectError(abi.core.Error.InvalidParameter, result);
}
```

### Database Integration Example

```zig
//! Example: Vector Database Integration
//! Shows proper database usage patterns with error handling

const std = @import("std");
const abi = @import("abi");

/// Document processing service using vector database
pub const DocumentProcessor = struct {
    allocator: std.mem.Allocator,
    database: *abi.database.VectorDatabase,
    embedding_processor: *CustomAIProcessor,

    pub fn init(allocator: std.mem.Allocator) !*DocumentProcessor {
        const self = try allocator.create(DocumentProcessor);
        errdefer allocator.destroy(self);

        // Initialize vector database
        const db_config = abi.database.VectorDatabaseConfig{
            .dimension = 384, // Embedding dimension
            .max_vectors = 10000,
            .distance_metric = .cosine,
        };

        self.database = try abi.database.createVectorDatabase(allocator, db_config);
        errdefer abi.database.destroyVectorDatabase(self.database);

        // Initialize embedding processor
        const ai_config = CustomAIProcessor.Config{
            .model_path = "models/embedding_model",
            .max_batch_size = 32,
        };

        self.embedding_processor = try CustomAIProcessor.init(allocator, ai_config);
        errdefer self.embedding_processor.deinit();

        self.* = .{
            .allocator = allocator,
            .database = self.database,
            .embedding_processor = self.embedding_processor,
        };

        return self;
    }

    pub fn deinit(self: *DocumentProcessor) void {
        self.embedding_processor.deinit();
        abi.database.destroyVectorDatabase(self.database);
        self.allocator.destroy(self);
    }

    /// Index document with embedding generation
    pub fn indexDocument(self: *DocumentProcessor, doc_id: []const u8, content: []const u8) !u32 {
        // Generate embedding for content
        const embedding = try self.generateEmbedding(content);
        defer self.allocator.free(embedding);

        // Store in database
        const id = try abi.database.insertVector(self.database, embedding, doc_id);
        return id;
    }

    /// Search for similar documents
    pub fn searchDocuments(self: *DocumentProcessor, query: []const u8, limit: u32) ![]abi.database.SearchResult {
        // Generate query embedding
        const query_embedding = try self.generateEmbedding(query);
        defer self.allocator.free(query_embedding);

        // Search database
        return try abi.database.searchVectors(self.database, query_embedding, limit);
    }

    fn generateEmbedding(self: *DocumentProcessor, text: []const u8) ![]f32 {
        // Tokenize text (simplified)
        const tokens = try self.tokenizeText(text);
        defer self.allocator.free(tokens);

        // Convert to numerical input (simplified)
        const input = try self.tokensToInput(tokens);
        defer self.allocator.free(input);

        // Generate embedding
        const embedding = try self.embedding_processor.processSingleInput(input);
        return embedding;
    }

    fn tokenizeText(self: *DocumentProcessor, text: []const u8) ![]u32 {
        // Simplified tokenization - split by spaces
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        var iter = std.mem.split(u8, text, " ");
        while (iter.next()) |word| {
            // Simple hash-based tokenization (for demo)
            const token = std.hash.Wyhash.hash(0, word);
            try tokens.append(@intCast(token & 0xFFFF)); // Limit token range
        }

        return tokens.toOwnedSlice();
    }

    fn tokensToInput(self: *DocumentProcessor, tokens: []u32) ![]f32 {
        // Convert tokens to fixed-size input vector
        const input_size = self.embedding_processor.getInputSize();
        const input = try self.allocator.alloc(f32, input_size);

        // Simple token averaging (for demo)
        const avg_token = if (tokens.len > 0)
            @as(f32, @floatFromInt(std.mem.reduce(.Add, u32, tokens) orelse 0)) / @as(f32, @floatFromInt(tokens.len))
        else
            0.0;

        for (input) |*v| {
            v.* = avg_token / 65536.0; // Normalize
        }

        return input;
    }
};
```

### Plugin Development Example

```zig
//! Example: Custom Plugin Implementation
//! Demonstrates plugin interface, lifecycle, and integration

const std = @import("std");
const abi = @import("abi");

/// Custom logging plugin that integrates with the framework
pub const CustomLoggingPlugin = struct {
    const PluginInterface = abi.plugins.PluginInterface;
    const PluginConfig = abi.plugins.PluginConfig;
    const PluginContext = abi.plugins.PluginContext;

    allocator: std.mem.Allocator,
    context: ?*PluginContext = null,
    log_file: ?std.fs.File = null,
    config: Config,

    pub const Config = struct {
        log_path: []const u8 = "logs/plugin.log",
        max_file_size_mb: u32 = 100,
        enable_console_output: bool = true,
    };

    // Plugin interface implementation
    pub const interface = PluginInterface{
        .name = "custom_logging",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .description = "Advanced logging plugin with file and console output",

        .initialize = initialize,
        .shutdown = shutdown,
        .getCapabilities = getCapabilities,
        .execute = execute,
    };

    fn initialize(ctx: *PluginContext, config: *const PluginConfig) callconv(.C) abi.plugins.PluginError!void {
        const self = @as(*CustomLoggingPlugin, @ptrCast(@alignCast(ctx.plugin_data.?)));

        // Parse configuration
        const log_path = config.getParameter("log_path") orelse "logs/plugin.log";
        const max_size = std.fmt.parseInt(u32, config.getParameter("max_file_size_mb") orelse "100", 10) catch 100;
        const console_output = std.mem.eql(u8, config.getParameter("enable_console_output") orelse "true", "true");

        self.config = .{
            .log_path = try self.allocator.dupe(u8, log_path),
            .max_file_size_mb = max_size,
            .enable_console_output = console_output,
        };

        // Create log directory if needed
        const log_dir = std.fs.path.dirname(log_path) orelse "";
        if (log_dir.len > 0) {
            try std.fs.cwd().makePath(log_dir);
        }

        // Open log file
        self.log_file = try std.fs.cwd().createFile(log_path, .{ .truncate = false, .read = false });
        try self.log_file.?.seekFromEnd(0);

        // Log initialization
        try self.log(.info, "Custom logging plugin initialized", .{});
    }

    fn shutdown(ctx: *PluginContext) callconv(.C) void {
        const self = @as(*CustomLoggingPlugin, @ptrCast(@alignCast(ctx.plugin_data.?)));

        if (self.log_file) |file| {
            self.log(.info, "Custom logging plugin shutting down", .{}) catch {};
            file.close();
        }

        self.allocator.free(self.config.log_path);
    }

    fn getCapabilities(ctx: *PluginContext) callconv(.C) abi.plugins.PluginCapabilities {
        _ = ctx;
        return .{
            .supported_operations = &[_]abi.plugins.OperationType{
                .logging,
                .monitoring,
            },
        };
    }

    fn execute(ctx: *PluginContext, operation: abi.plugins.OperationType, params: *const abi.plugins.OperationParams) callconv(.C) abi.plugins.PluginError!void {
        const self = @as(*CustomLoggingPlugin, @ptrCast(@alignCast(ctx.plugin_data.?)));

        switch (operation) {
            .logging => {
                const level = std.meta.stringToEnum(LogLevel, params.getString("level") orelse "info") orelse .info;
                const message = params.getString("message") orelse "";
                try self.log(level, message, .{});
            },
            else => return abi.plugins.PluginError.UnsupportedOperation,
        }
    }

    // Plugin-specific methods
    pub fn log(self: *CustomLoggingPlugin, level: LogLevel, comptime format: []const u8, args: anytype) !void {
        const timestamp = std.time.timestamp();
        const level_str = switch (level) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .error => "ERROR",
        };

        // Format log message
        const message = try std.fmt.allocPrint(self.allocator, "[{d}] {s}: " ++ format ++ "\n", .{timestamp, level_str} ++ args);
        defer self.allocator.free(message);

        // Write to file
        if (self.log_file) |file| {
            _ = try file.write(message);
            try file.sync();
        }

        // Write to console if enabled
        if (self.config.enable_console_output) {
            std.debug.print("{s}", .{message});
        }
    }

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        error,
    };
};

// Plugin registration function
pub export fn abi_plugin_init(allocator: std.mem.Allocator) callconv(.C) *const PluginInterface {
    // Create plugin instance
    const plugin = allocator.create(CustomLoggingPlugin) catch @panic("Failed to create plugin");
    plugin.* = .{
        .allocator = allocator,
    };

    return &CustomLoggingPlugin.interface;
}

pub export fn abi_plugin_deinit(plugin: *const PluginInterface, allocator: std.mem.Allocator) callconv(.C) void {
    _ = plugin;
    // Note: In a real implementation, you'd track the plugin instance
    // This is simplified for the example
    _ = allocator;
}
```

### Web Server Integration Example

```zig
//! Example: REST API Server with Framework Integration
//! Shows web server setup, routing, and middleware patterns

const std = @import("std");
const abi = @import("abi");

/// REST API server for the AI processing service
pub const APIServer = struct {
    allocator: std.mem.Allocator,
    server: *abi.web.EnhancedWebServer,
    processor: *DocumentProcessor,

    pub fn init(allocator: std.mem.Allocator, processor: *DocumentProcessor) !*APIServer {
        const self = try allocator.create(APIServer);
        errdefer allocator.destroy(self);

        // Configure web server
        const server_config = abi.web.WebServerConfig{
            .port = 8080,
            .host = "0.0.0.0",
            .enable_cors = true,
            .enable_compression = true,
            .max_connections = 1000,
            .request_timeout_ms = 30000,
        };

        self.server = try abi.web.createWebServer(allocator, server_config);
        errdefer abi.web.destroyWebServer(self.server);

        self.* = .{
            .allocator = allocator,
            .server = self.server,
            .processor = processor,
        };

        // Register routes
        try self.registerRoutes();

        return self;
    }

    pub fn deinit(self: *APIServer) void {
        abi.web.destroyWebServer(self.server);
        self.allocator.destroy(self);
    }

    pub fn start(self: *APIServer) !void {
        std.log.info("Starting API server on http://localhost:8080", .{});
        try abi.web.startWebServer(self.server);
    }

    fn registerRoutes(self: *APIServer) !void {
        // Health check endpoint
        try abi.web.addRoute(self.server, .GET, "/health", healthCheckHandler, self);

        // Document management endpoints
        try abi.web.addRoute(self.server, .POST, "/documents", indexDocumentHandler, self);
        try abi.web.addRoute(self.server, .GET, "/documents/search", searchDocumentsHandler, self);

        // API documentation
        try abi.web.addRoute(self.server, .GET, "/docs", apiDocsHandler, self);
    }

    fn healthCheckHandler(ctx: *abi.web.RequestContext) !void {
        const self = @as(*APIServer, @ptrCast(@alignCast(ctx.user_data.?)));

        const response = abi.web.JsonResponse{
            .status = "healthy",
            .timestamp = std.time.timestamp(),
            .version = "1.0.0",
        };

        try ctx.json(response, .{});
    }

    fn indexDocumentHandler(ctx: *abi.web.RequestContext) !void {
        const self = @as(*APIServer, @ptrCast(@alignCast(ctx.user_data.?)));

        // Parse JSON request
        const request = try ctx.parseJson(struct {
            id: []const u8,
            content: []const u8,
        });

        // Index document
        const doc_id = try self.processor.indexDocument(request.id, request.content);

        const response = abi.web.JsonResponse{
            .document_id = doc_id,
            .status = "indexed",
        };

        try ctx.json(response, .{ .status_code = 201 });
    }

    fn searchDocumentsHandler(ctx: *abi.web.RequestContext) !void {
        const self = @as(*APIServer, @ptrCast(@alignCast(ctx.user_data.?)));

        // Get query parameter
        const query = ctx.getQueryParam("q") orelse return error.MissingQueryParameter;
        const limit_str = ctx.getQueryParam("limit") orelse "10";
        const limit = std.fmt.parseInt(u32, limit_str, 10) catch 10;

        // Search documents
        const results = try self.processor.searchDocuments(query, limit);
        defer self.allocator.free(results);

        try ctx.json(.{ .results = results }, .{});
    }

    fn apiDocsHandler(ctx: *abi.web.RequestContext) !void {
        const docs = struct {
            pub const api_docs = .{
                .title = "ABI AI Processing API",
                .version = "1.0.0",
                .endpoints = .{
                    .{
                        .method = "GET",
                        .path = "/health",
                        .description = "Health check endpoint",
                    },
                    .{
                        .method = "POST",
                        .path = "/documents",
                        .description = "Index a document",
                        .body = .{
                            .id = "string",
                            .content = "string",
                        },
                    },
                    .{
                        .method = "GET",
                        .path = "/documents/search",
                        .description = "Search documents",
                        .query_params = .{
                            .q = "search query",
                            .limit = "maximum results (optional, default: 10)",
                        },
                    },
                },
            };
        };

        try ctx.json(docs.api_docs, .{});
    }
};

// Server startup example
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize document processor
    var processor = try DocumentProcessor.init(allocator);
    defer processor.deinit();

    // Initialize API server
    var server = try APIServer.init(allocator, processor);
    defer server.deinit();

    // Start server
    try server.start();
}
```

These examples demonstrate how to apply the patterns documented in this guide to build real features within the ABI framework. Each example includes proper error handling, resource management, testing, and integration with the framework's architecture.

## API Reference

This section provides a comprehensive reference for the ABI framework's public APIs. The framework is organized into logical modules with clear separation of concerns.

### Core Framework APIs

#### Framework Initialization

```zig
const abi = @import("abi");

// Initialize the framework
var framework = try abi.init(allocator, abi.FrameworkOptions{
    .max_plugins = 128,
    .auto_discover_plugins = true,
    .enable_gpu = true,
    .enable_ai = true,
});
defer abi.shutdown(&framework);

// Framework is now ready for use
try framework.start();
```

**FrameworkOptions**:
- `max_plugins: u32` - Maximum number of plugins to load (default: 128)
- `auto_discover_plugins: bool` - Automatically discover plugins in search paths (default: true)
- `enable_gpu: bool` - Enable GPU acceleration features (default: true)
- `enable_ai: bool` - Enable AI/ML features (default: true)
- `enable_database: bool` - Enable database features (default: true)
- `enable_web: bool` - Enable web server features (default: true)

#### Framework Methods

```zig
// Check if a feature is enabled
const ai_enabled = framework.isFeatureEnabled(.ai);

// Get framework status
const status = framework.getStatus();

// Get loaded plugins count
const plugin_count = framework.getPluginCount();

// Load a specific plugin
try framework.loadPlugin("path/to/plugin.so");

// Unload a plugin
try framework.unloadPlugin("plugin_name");
```

### AI Module APIs

#### Neural Networks

```zig
const ai = abi.ai;

// Create a neural network
const network = try ai.createNeuralNetwork(allocator, ai.NeuralNetworkConfig{
    .input_size = 784,
    .hidden_sizes = &[_]u32{ 256, 128 },
    .output_size = 10,
    .activation = .relu,
    .use_gpu = true,
});

// Forward pass
const input = try allocator.alloc(f32, 784);
defer allocator.free(input);
// ... fill input data ...
const output = try network.forward(input, allocator);
defer allocator.free(output);

// Training
try network.train(input, &target, learning_rate);

// Cleanup
ai.destroyNeuralNetwork(network);
```

**NeuralNetworkConfig**:
- `input_size: u32` - Number of input neurons
- `hidden_sizes: []const u32` - Sizes of hidden layers
- `output_size: u32` - Number of output neurons
- `activation: ActivationType` - Activation function (.relu, .sigmoid, .tanh)
- `learning_rate: f32` - Learning rate for training (default: 0.01)
- `use_gpu: bool` - Use GPU acceleration if available

#### Agent System

```zig
// Create an enhanced agent
const agent = try ai.createEnhancedAgent(allocator, ai.AgentConfig{
    .name = "my_agent",
    .persona = "helpful_assistant",
    .max_memory_size = 1024 * 1024, // 1MB
    .model_path = "models/agent_model.bin",
});

// Generate response
const response = try agent.generateResponse("Hello, how are you?", allocator);
defer allocator.free(response);

// Add to memory
try agent.addToMemory("User asked about weather", allocator);

// Get memory count
const mem_count = agent.getMemoryCount();

// Cleanup
ai.destroyEnhancedAgent(agent);
```

### Database Module APIs

#### Vector Database

```zig
const db = abi.database;

// Initialize vector database
const vector_db = try db.createVectorDatabase(allocator, db.VectorDatabaseConfig{
    .dimension = 384, // Embedding dimension
    .max_vectors = 100000,
    .distance_metric = .cosine,
    .index_type = .hnsw,
});

// Insert vectors
const vector = try allocator.alloc(f32, 384);
defer allocator.free(vector);
// ... fill vector data ...
const id = try db.insertVector(vector_db, vector, "document_id");

// Search similar vectors
const query = try allocator.alloc(f32, 384);
defer allocator.free(query);
// ... fill query data ...
const results = try db.searchVectors(vector_db, query, 10, allocator);
defer allocator.free(results);

// Get vector by ID
const retrieved = try db.getVector(vector_db, id, allocator);
defer allocator.free(retrieved);

// Cleanup
db.destroyVectorDatabase(vector_db);
```

**VectorDatabaseConfig**:
- `dimension: u32` - Vector dimension (e.g., 384 for BERT embeddings)
- `max_vectors: u32` - Maximum number of vectors to store
- `distance_metric: DistanceMetric` - Distance calculation method (.cosine, .euclidean, .dot_product)
- `index_type: IndexType` - Indexing algorithm (.hnsw, .flat, .ivf)

#### Search Result Structure

```zig
pub const SearchResult = struct {
    id: u32,              // Vector ID
    score: f32,           // Similarity score
    metadata: []const u8, // Associated metadata
};
```

### GPU Module APIs

#### GPU Backend

```zig
const gpu = abi.gpu;

// Initialize GPU backend
const backend = try gpu.createBackend(allocator, gpu.BackendConfig{
    .max_batch_size = 1024,
    .memory_limit = 1024 * 1024 * 1024, // 1GB
    .debug_validation = false,
    .power_preference = .high_performance,
});

// Check GPU availability
const has_gpu = gpu.isAvailable();

// Get GPU memory info
const mem_info = backend.getMemoryInfo();

// Allocate GPU buffer
const buffer = try backend.allocateBuffer(f32, 1024 * 1024);

// Copy data to GPU
try backend.copyToGPU(host_data, buffer, 1024 * 1024);

// Execute kernel
try backend.executeKernel("vector_add", .{ buffer_a, buffer_b, result }, 1024);

// Copy result back
try backend.copyFromGPU(result, host_result, 1024 * 1024);

// Cleanup
gpu.destroyBackend(backend);
```

### Web Module APIs

#### Web Server

```zig
const web = abi.web;

// Create web server
const server = try web.createServer(allocator, web.ServerConfig{
    .port = 8080,
    .host = "0.0.0.0",
    .enable_cors = true,
    .enable_compression = true,
    .max_connections = 1000,
    .request_timeout_ms = 30000,
});

// Register routes
try web.addRoute(server, .GET, "/api/health", healthHandler, user_data);
try web.addRoute(server, .POST, "/api/data", dataHandler, user_data);

// WebSocket support
try web.addWebSocketRoute(server, "/ws", wsHandler, user_data);

// Start server
try web.startServer(server);

// Server runs until stopped
web.stopServer(server);

// Cleanup
web.destroyServer(server);
```

**ServerConfig**:
- `port: u16` - Port to listen on
- `host: []const u8` - Host address to bind to
- `enable_cors: bool` - Enable CORS headers
- `enable_compression: bool` - Enable response compression
- `enable_websocket: bool` - Enable WebSocket support
- `max_connections: u32` - Maximum concurrent connections
- `request_timeout_ms: u32` - Request timeout in milliseconds

#### Request Handler Signature

```zig
pub fn requestHandler(ctx: *web.RequestContext) !void {
    // Access request data
    const method = ctx.request.method;
    const path = ctx.request.path;
    const headers = ctx.request.headers;
    const body = ctx.request.body;

    // Get query parameters
    const param = ctx.getQueryParam("key") orelse "default";

    // Send response
    try ctx.json(.{ .status = "ok", .data = "response" }, .{});

    // Or send custom response
    try ctx.send(200, "text/plain", "Hello World!");
}
```

### Plugin System APIs

#### Plugin Registry

```zig
const plugins = abi.plugins;

// Initialize plugin system
var registry = try plugins.init(allocator);
defer registry.deinit();

// Add plugin search path
try registry.addPluginPath("/usr/local/lib/plugins");

// Load plugin from file
const plugin = try registry.loadPlugin("my_plugin.so");
defer registry.unloadPlugin(plugin);

// Call plugin function
const result = try plugin.call("process_data", .{ input_data });

// List loaded plugins
const plugin_list = try registry.listPlugins(allocator);
defer allocator.free(plugin_list);
```

#### Plugin Development

```zig
// Plugin interface implementation
pub const interface = plugins.PluginInterface{
    .name = "my_plugin",
    .version = .{ .major = 1, .minor = 0, .patch = 0 },
    .description = "My custom plugin",

    .initialize = initialize,
    .shutdown = shutdown,
    .getCapabilities = getCapabilities,
    .execute = execute,
};

// Required callback functions
fn initialize(ctx: *plugins.PluginContext, config: *const plugins.PluginConfig) !void {
    // Plugin initialization logic
}

fn shutdown(ctx: *plugins.PluginContext) void {
    // Plugin cleanup logic
}

fn getCapabilities(ctx: *plugins.PluginContext) plugins.PluginCapabilities {
    return .{
        .supported_operations = &.{ .custom_processing },
    };
}

fn execute(ctx: *plugins.PluginContext, op: plugins.OperationType, params: *const plugins.OperationParams) !void {
    // Plugin execution logic
}
```

### Error Types

#### Core Errors

```zig
pub const Error = error{
    InvalidConfig,
    InvalidParameter,
    NotFound,
    AlreadyExists,
    Timeout,
    RateLimited,
    PermissionDenied,
    Unavailable,
    InternalError,
    OutOfMemory,
    NotImplemented,
    FeatureDisabled,
};
```

#### Module-Specific Errors

- **AI Errors**: `InvalidModel`, `TrainingFailed`, `InferenceError`
- **Database Errors**: `ConnectionFailed`, `InvalidQuery`, `IndexError`
- **GPU Errors**: `DeviceNotFound`, `OutOfMemory`, `KernelError`
- **Web Errors**: `InvalidRequest`, `ServerError`, `ConnectionFailed`
- **Plugin Errors**: `LoadFailed`, `IncompatibleVersion`, `ExecutionFailed`

### Type Definitions

#### Common Types

```zig
// Memory management
pub const Allocator = std.mem.Allocator;

// Time and duration
pub const Timestamp = i64;  // Unix timestamp in milliseconds
pub const Duration = u64;   // Duration in nanoseconds

// Data structures
pub const String = []const u8;
pub const Buffer = []u8;
pub const Vector = []f32;
pub const Matrix = []const []const f32;
```

#### Configuration Types

```zig
// Global configuration
pub const Config = struct {
    log_level: enum { debug, info, warn, error } = .info,
    max_memory_mb: u32 = 1024,
    temp_dir: []const u8 = "/tmp",
    plugin_search_paths: []const []const u8 = &.{"./plugins"},
};

// Feature flags
pub const FeatureFlags = packed struct {
    ai: bool = true,
    gpu: bool = true,
    database: bool = true,
    web: bool = true,
    monitoring: bool = true,
};
```

### Constants

#### Version Information

```zig
pub const VERSION = struct {
    pub const MAJOR = 0;
    pub const MINOR = 1;
    pub const PATCH = 0;

    pub fn string() []const u8 {
        return "0.1.0";
    }
};
```

#### Default Values

```zig
// Network defaults
pub const DEFAULT_PORT = 8080;
pub const DEFAULT_HOST = "127.0.0.1";
pub const DEFAULT_TIMEOUT_MS = 30000;

// Database defaults
pub const DEFAULT_VECTOR_DIMENSION = 384;
pub const DEFAULT_MAX_VECTORS = 100000;
pub const DEFAULT_DISTANCE_METRIC = .cosine;

// AI defaults
pub const DEFAULT_LEARNING_RATE = 0.01;
pub const DEFAULT_BATCH_SIZE = 32;
pub const DEFAULT_EPOCHS = 100;
```

### Utility Functions

#### Memory Management

```zig
// Safe allocation with error handling
pub fn safeAlloc(comptime T: type, allocator: Allocator, count: usize) ![]T {
    return allocator.alloc(T, count) catch |err| {
        log.err("Failed to allocate {} items of type {}: {}", .{count, @typeName(T), err});
        return err;
    };
}

// Safe deallocation
pub fn safeFree(allocator: Allocator, memory: anytype) void {
    if (memory.len > 0) {
        allocator.free(memory);
    }
}
```

#### Logging

```zig
const log = abi.logging;

// Log levels
log.debug("Debug message: {}", .{value});
log.info("Info message");
log.warn("Warning: {}", .{condition});
log.err("Error: {}", .{err});

// Structured logging
log.info("Operation completed", .{
    .operation = "data_processing",
    .duration_ms = 150,
    .items_processed = 1000,
});
```

#### Performance Monitoring

```zig
const metrics = abi.observability.metrics;

// Record timing
const timer = metrics.startTimer("operation_name");
defer timer.stop();

// Record counter
metrics.increment("requests_total", .{ .method = "GET", .status = "200" });

// Record gauge
metrics.set("memory_usage_mb", current_memory_mb);
```

### Build System Integration

#### Feature Flags

```bash
# Build with specific features enabled
zig build -Denable-gpu=true -Denable-ai=true -Denable-web=true

# Build with all features disabled
zig build -Denable-gpu=false -Denable-ai=false -Denable-web=false

# Build with optimizations
zig build -Doptimize=ReleaseFast
```

#### Cross-Compilation

```bash
# Build for different targets
zig build -Dtarget=x86_64-linux
zig build -Dtarget=x86_64-windows
zig build -Dtarget=aarch64-linux
```

## Versioning and Release Process

The ABI framework follows semantic versioning and maintains detailed changelogs for all releases.

### Versioning Scheme

The framework uses [Semantic Versioning](https://semver.org/) with the format `MAJOR.MINOR.PATCH`:

- **MAJOR**: Breaking changes that require code updates
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes and small improvements

#### Version Information

```zig
pub const VERSION = struct {
    pub const MAJOR = 0;
    pub const MINOR = 1;
    pub const PATCH = 0;

    pub fn string() []const u8 {
        return "0.1.0";
    }

    pub fn isCompatible(major: u32, minor: u32) bool {
        return major == MAJOR and minor <= MINOR;
    }
};
```

### Changelog Format

All changes are documented in `CHANGELOG.md` with the following format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New feature descriptions

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related changes

## [0.1.0] - 2024-01-15

### Added
- Initial release of ABI framework
- Core AI and ML functionality
- Vector database support
- GPU acceleration
- Plugin system
- Web server capabilities
```

### Release Process

#### 1. Pre-Release Preparation

**Update version numbers:**
```bash
# Update version in build.zig
const build_options = b.addOptions();
build_options.addOption([]const u8, "package_version", "0.2.0");

# Update version in source code
pub const VERSION = struct {
    pub const MAJOR = 0;
    pub const MINOR = 2;
    pub const PATCH = 0;
};
```

**Update changelog:**
```markdown
## [Unreleased]

### Added
- New features for this release

## [0.2.0] - 2024-02-XX

### Added
- Feature descriptions
- API changes
- Breaking changes (if any)
```

#### 2. Testing and Validation

**Run full test suite:**
```bash
# Run all tests
zig build test

# Run integration tests
zig test tests/integration/

# Run performance benchmarks
zig build test -- tests/benchmarks/

# Cross-platform testing
zig build test -Dtarget=x86_64-linux
zig build test -Dtarget=x86_64-windows
zig build test -Dtarget=x86_64-macos
```

**Verify documentation:**
```bash
# Generate docs
zig build docs

# Check for build warnings
zig build -Doptimize=ReleaseSafe
```

#### 3. Create Release Branch

```bash
# Create release branch
git checkout -b release/v0.2.0

# Update changelog with release date
## [0.2.0] - 2024-02-15

# Commit changes
git add CHANGELOG.md build.zig
git commit -m "Release version 0.2.0"

# Push to remote
git push origin release/v0.2.0
```

#### 4. Create GitHub Release

**Create annotated tag:**
```bash
# Create signed tag with release notes
git tag -a v0.2.0 -m "Release version 0.2.0

## What's New
- New features...
- Bug fixes...
- Performance improvements...

## Breaking Changes
- List any breaking changes

## Migration Guide
- Instructions for upgrading"
```

**Push tag:**
```bash
git push origin v0.2.0
```

#### 5. GitHub Actions Release

The CI/CD pipeline automatically:

1. **Builds release binaries** for multiple platforms
2. **Creates GitHub release** with pre-built binaries
3. **Generates release notes** from changelog
4. **Publishes Docker images** to GitHub Container Registry

**Release artifacts:**
- `abi-linux-x86_64` - Linux x86_64 binary
- `abi-windows-x86_64.exe` - Windows x86_64 executable
- `abi-macos-x86_64` - macOS x86_64 binary
- `abi-linux-aarch64` - Linux ARM64 binary

#### 6. Post-Release Tasks

**Update documentation:**
```bash
# Merge release branch back to main
git checkout main
git merge release/v0.2.0

# Update version to next development version
## [Unreleased] - Development

# Push changes
git push origin main
```

**Announce release:**
- Update project website
- Send release announcements
- Update package managers (if applicable)

### Release Checklist

#### Pre-Release
- [ ] All tests pass
- [ ] Performance benchmarks meet targets
- [ ] Documentation is up to date
- [ ] Changelog is complete
- [ ] Version numbers updated
- [ ] Breaking changes documented

#### Release
- [ ] Release branch created
- [ ] Tag created and pushed
- [ ] GitHub Actions builds successfully
- [ ] Release binaries uploaded
- [ ] Docker images published

#### Post-Release
- [ ] Release branch merged to main
- [ ] Development version bumped
- [ ] Announcements sent
- [ ] Issues and PRs updated

### Maintenance Releases

For patch releases (bug fixes only):

1. **Create hotfix branch** from the last release tag
2. **Apply fixes** and update changelog
3. **Test thoroughly** - no new features allowed
4. **Create patch tag** and release
5. **Merge back** to main and develop branches

### Deprecation Policy

1. **Mark as deprecated** in minor releases with warnings
2. **Remove in next major release** with migration guide
3. **Maintain compatibility** for one major version cycle

### Security Releases

For security fixes:

1. **Immediate patch release** without waiting for feature complete
2. **Private disclosure** process for responsible disclosure
3. **Security advisories** published on GitHub
4. **Automated dependency updates** for security patches

## Troubleshooting Guide

This section covers common issues and solutions when working with the ABI framework.

### Build Issues

#### "error: import of file outside module path"

**Problem**: Getting import path errors when running tests or building.

**Solutions**:
1. **Use the build system**: Always run tests through `zig build test` rather than `zig test` directly
2. **Check module imports**: Ensure you're importing the `abi` module, not individual files
3. **Verify build.zig**: Make sure your build.zig properly imports the abi module

```bash
# Correct way to run tests
zig build test

# Incorrect - will fail with import errors
zig test tests/integration/comprehensive_test_suite.zig
```

#### "error: declarations are not allowed between container fields"

**Problem**: Zig compilation error about declarations between struct fields.

**Solutions**:
1. **Move all fields to top**: In Zig structs, all fields must be declared before any functions or other declarations
2. **Check struct layout**: Ensure proper ordering: `const/fields` → functions

```zig
// Correct
pub const MyStruct = struct {
    // All fields first
    field1: u32,
    field2: []const u8,

    // Then functions
    pub fn init() MyStruct { ... }
};

// Incorrect
pub const MyStruct = struct {
    field1: u32,
    pub fn init() MyStruct { ... }  // ERROR: function before all fields
    field2: []const u8,
};
```

### Runtime Issues

#### Memory Leaks in Tests

**Problem**: Tests pass but memory leak detection fails.

**Solutions**:
1. **Use defer for cleanup**: Always use `defer` for resource cleanup
2. **Check allocation patterns**: Ensure all allocations are properly freed
3. **Use testing.allocator**: Use `testing.allocator` for test allocations

```zig
test "memory leak free" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator;

    var data = try allocator.alloc(u8, 1024);
    defer allocator.free(data); // Always defer cleanup

    // Test logic here
    try testing.expect(data.len == 1024);
}
```

#### Plugin Loading Failures

**Problem**: Plugins fail to load or initialize.

**Solutions**:
1. **Check plugin path**: Ensure plugin files are in the correct directory
2. **Verify plugin interface**: Make sure plugin exports the required interface
3. **Check dependencies**: Ensure all required dependencies are available

```zig
// Plugin must export this interface
pub export fn abi_plugin_init(allocator: std.mem.Allocator) callconv(.C) *const PluginInterface {
    // Implementation
}

pub export fn abi_plugin_deinit(plugin: *const PluginInterface, allocator: std.mem.Allocator) callconv(.C) void {
    // Cleanup
}
```

### Performance Issues

#### Slow Vector Operations

**Problem**: Vector operations are slower than expected.

**Solutions**:
1. **Enable SIMD**: Ensure SIMD optimizations are enabled
2. **Check CPU features**: Verify the target CPU supports required SIMD instructions
3. **Use appropriate data types**: Use aligned data structures for better performance

```zig
// Enable SIMD optimizations
const Vec4 = @Vector(4, f32);

// Ensure data is properly aligned
var data align(16) = [_]f32{1.0, 2.0, 3.0, 4.0};
```

#### GPU Acceleration Not Working

**Problem**: GPU features are enabled but not accelerating.

**Solutions**:
1. **Check GPU availability**: Verify GPU is detected by the framework
2. **Enable GPU features**: Ensure `-Denable-gpu=true` is set during build
3. **Verify drivers**: Check that appropriate GPU drivers are installed

```zig
// Check GPU availability
const has_gpu = abi.gpu.isAvailable();
if (!has_gpu) {
    // Fallback to CPU implementation
}
```

### Database Issues

#### Vector Search Performance

**Problem**: Vector searches are slow or returning incorrect results.

**Solutions**:
1. **Check index type**: Use appropriate index type for your use case (HNSW for speed, flat for accuracy)
2. **Verify dimensions**: Ensure all vectors have the correct dimensions
3. **Monitor memory usage**: Large datasets may require more memory

```zig
const config = abi.database.VectorDatabaseConfig{
    .dimension = 384,
    .max_vectors = 100000,
    .distance_metric = .cosine,
    .index_type = .hnsw, // Good for performance
};
```

#### Connection Failures

**Problem**: Database connections fail or timeout.

**Solutions**:
1. **Check connection parameters**: Verify host, port, and credentials
2. **Increase timeouts**: Adjust connection and query timeouts
3. **Monitor connections**: Ensure connection pooling is working correctly

### Web Server Issues

#### Request Handling Errors

**Problem**: Web server fails to handle requests properly.

**Solutions**:
1. **Check route registration**: Ensure routes are registered before starting server
2. **Verify middleware**: Make sure middleware is properly configured
3. **Check CORS settings**: Ensure CORS is configured for cross-origin requests

```zig
// Register routes before starting server
try abi.web.addRoute(server, .GET, "/api/health", healthHandler, null);
try abi.web.startServer(server);
```

#### WebSocket Connection Issues

**Problem**: WebSocket connections fail or disconnect unexpectedly.

**Solutions**:
1. **Enable WebSocket support**: Set `enable_websocket = true` in server config
2. **Check protocols**: Ensure client and server agree on WebSocket protocol
3. **Monitor connection limits**: Check if connection limits are being exceeded

### Testing Issues

#### Integration Tests Failing

**Problem**: Integration tests pass individually but fail in CI.

**Solutions**:
1. **Check dependencies**: Ensure all test dependencies are available
2. **Isolate tests**: Run tests individually to identify conflicts
3. **Check resource usage**: Some tests may require exclusive access to resources

```bash
# Run tests individually
zig build test -- tests/integration/framework_lifecycle_test.zig
zig build test -- tests/integration/database_ops_test.zig
```

#### Benchmark Instability

**Problem**: Performance benchmarks show inconsistent results.

**Solutions**:
1. **Warm up the system**: Run benchmarks multiple times and discard initial runs
2. **Control environment**: Minimize background processes during benchmarking
3. **Use statistical analysis**: Run benchmarks multiple times and use averages

```zig
// Warm up before benchmarking
for (0..5) |_| {
    _ = try benchmarkFunction(); // Discard results
}

// Now run actual benchmarks
for (0..100) |_| {
    try runBenchmark();
}
```

### Development Environment Issues

#### Zig Version Mismatches

**Problem**: Code works locally but fails in CI due to Zig version differences.

**Solutions**:
1. **Check .zigversion**: Ensure all environments use the same Zig version
2. **Use version ranges**: Specify minimum supported Zig versions
3. **Test across versions**: Run CI with multiple Zig versions

```bash
# Check Zig version
zig version

# Should match .zigversion file
cat .zigversion
```

#### Dependency Issues

**Problem**: External dependencies are not available or incompatible.

**Solutions**:
1. **Check build.zig.zon**: Verify dependency versions
2. **Update dependencies**: Use `zig fetch` to update to latest compatible versions
3. **Vendor dependencies**: Consider vendoring critical dependencies

```bash
# Update dependencies
zig fetch

# Clean and rebuild
rm -rf zig-cache zig-out
zig build
```

### Deployment Issues

#### Docker Build Failures

**Problem**: Docker builds fail in CI but work locally.

**Solutions**:
1. **Check base image**: Ensure Docker image has required dependencies
2. **Multi-stage builds**: Use appropriate base images for build vs runtime
3. **Cache optimization**: Optimize Docker layer caching

```dockerfile
# Multi-stage build example
FROM ziglang/zig:0.13.0 as builder
WORKDIR /app
COPY . .
RUN zig build -Doptimize=ReleaseFast

FROM debian:bookworm-slim
COPY --from=builder /app/zig-out/bin/abi /usr/local/bin/abi
CMD ["abi", "serve"]
```

#### GitHub Actions Issues

**Problem**: CI workflows fail intermittently.

**Solutions**:
1. **Check resource limits**: Ensure workflows don't exceed GitHub's limits
2. **Add retry logic**: Use actions that support automatic retries
3. **Monitor artifacts**: Check that artifacts are properly uploaded/downloaded

```yaml
# Add retry logic to flaky steps
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: zig build test
```

### Getting Help

If you encounter an issue not covered here:

1. **Check existing issues**: Search GitHub issues for similar problems
2. **Run diagnostics**: Use `zig build --verbose` for detailed build information
3. **Collect logs**: Gather relevant log output and system information
4. **Create minimal reproduction**: Isolate the issue in a minimal test case
5. **File an issue**: Open a GitHub issue with complete information

---
Follow these conventions to maintain code quality, performance, and consistency across the ABI framework.</content>
<parameter name="filePath">AGENTS.md