# API Reference

Comprehensive API documentation for the Abi AI Framework with usage examples.

## Table of Contents

- [Core Framework](#core-framework)
- [AI & Machine Learning](#ai--machine-learning)
- [Performance & Acceleration](#performance--acceleration)
- [Monitoring & Profiling](#monitoring--profiling)
- [Developer Tools](#developer-tools)
- [Examples](#examples)

## Core Framework

### `root.zig` - Framework Initialization

Framework initialization and configuration.

#### `Config` - Framework Configuration

```zig
const Config = struct {
    memory: MemoryConfig,
    performance: PerformanceConfig,
    ai: AIConfig,
    network: NetworkConfig,
    logging: LoggingConfig,
    security: SecurityConfig,
};
```

#### `Context` - Framework Context

```zig
const Context = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config) !*Context
    pub fn deinit(self: *Context) void
    pub fn global() !*Context
};
```

#### Usage Example

```zig
const abi = @import("abi");

var context = try abi.Context.init(std.heap.page_allocator, .{});
defer context.deinit();

const global_ctx = try abi.Context.global();
```

## AI & Machine Learning

### `ai/mod.zig` - AI Agent System

#### `Agent` - AI Agent

```zig
pub const Agent = struct {
    config: AgentConfig,
    context: *Context,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, persona: PersonaType) !*Agent
    pub fn deinit(self: *Agent) void
    pub fn generate(self: *Agent, prompt: []const u8, options: GenerationOptions) !GenerationResult
    pub fn setPersona(self: *Agent, persona: PersonaType) void
    pub fn clearHistory(self: *Agent) void
    pub fn getSystemPrompt(self: *Agent) []const u8
};
```

#### `PersonaType` - AI Personas

```zig
pub const PersonaType = enum {
    adaptive, creative, analytical, technical,
    conversational, educational, professional, casual,

    pub fn getDescription(self: PersonaType) []const u8
    pub fn getSystemPrompt(self: PersonaType) []const u8
};
```

#### `GenerationOptions` - Text Generation Options

```zig
pub const GenerationOptions = struct {
    stream_callback: ?*const fn ([]const u8) void = null,
    max_tokens: ?usize = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    stop_sequences: ?[][]const u8 = null,
    enable_safety: ?bool = null,
    system_prompt: ?[]const u8 = null,
};
```

#### `GenerationResult` - AI Response

```zig
pub const GenerationResult = struct {
    content: []const u8,
    usage: ?UsageStats,
    finish_reason: FinishReason,
    model: []const u8,
};
```

#### Usage Example

```zig
const ai = @import("ai");

var agent = try ai.Agent.init(allocator, .creative);
defer agent.deinit();

const result = try agent.generate("Write a creative story about AI", .{
    .max_tokens = 500,
    .temperature = 0.8,
});
defer allocator.free(result.content);

std.debug.print("AI Response: {s}\n", .{result.content});
agent.setPersona(.technical);
```

### `neural.zig` - Neural Networks

#### `NeuralNetwork` - Neural Network

```zig
pub const NeuralNetwork = struct {
    layers: std.array_list.Managed(*Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*NeuralNetwork
    pub fn deinit(self: *NeuralNetwork) void
    pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void
    pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32
    pub fn trainStep(self: *NeuralNetwork, input: []const f32, target: []const f32, learning_rate: f32) !f32
    pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork
};
```

#### `Layer` - Network Layer

```zig
pub const Layer = struct {
    layer_type: LayerType,
    weights: []f32,
    biases: []f32,
    activation: Activation,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !*Layer
    pub fn deinit(self: *Layer) void
    pub fn forward(self: *Layer, input: []const f32, allocator: std.mem.Allocator) ![]f32
    pub fn backward(self: *Layer, input: []const f32, output: []const f32, output_gradient: []const f32, learning_rate: f32, allocator: std.mem.Allocator) ![]f32
};
```

#### `LayerConfig` - Layer Configuration

```zig
pub const LayerConfig = struct {
    layer_type: LayerType,
    input_size: usize,
    output_size: usize,
    activation: Activation = .ReLU,
    dropout_rate: f32 = 0.0,
    weight_init_scale: f32 = 1.0,
};
```

#### `LayerType` - Layer Types

```zig
pub const LayerType = enum {
    Dense,
    Embedding,
    Dropout,
    Normalization,
    Activation,
};
```

#### `Activation` - Activation Functions

```zig
pub const Activation = enum {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    None,

    pub fn apply(self: Activation, x: f32) f32
    pub fn derivative(self: Activation, x: f32) f32
};
```

#### Usage Example

```zig
const neural = @import("neural");

// Create a simple neural network
var network = try neural.NeuralNetwork.init(allocator);
defer network.deinit();

// Add layers
try network.addLayer(.{
    .layer_type = .Dense,
    .input_size = 784,
    .output_size = 128,
    .activation = .ReLU,
});

try network.addLayer(.{
    .layer_type = .Dense,
    .input_size = 128,
    .output_size = 10,
    .activation = .Softmax,
});

// Forward pass
const input = [_]f32{ /* 784 float values */ };
const output = try network.forward(&input);
defer allocator.free(output);

// Training
const target = [_]f32{ /* 10 target values */ };
const loss = try network.trainStep(&input, &target, 0.01);
```

## Performance & Acceleration

### `simd/mod.zig` - SIMD Operations

#### Vector Operations

```zig
/// SIMD configuration
pub const SIMDConfig = struct {
    width_f32: comptime_int,
    width_f64: comptime_int,
    width_i32: comptime_int,
    max_width: comptime_int,
    arch_name: []const u8,
};

/// Global SIMD configuration
pub const simd_config: SIMDConfig = // Architecture-specific configuration

/// Text processing operations
pub const text = struct {
    pub fn countByte(haystack: []const u8, needle: u8) usize
    pub fn findByte(haystack: []const u8, needle: u8) ?usize
    pub fn toLowerAscii(dst: []u8, src: []const u8) void
};

/// Vector operations
pub const vector = struct {
    pub fn dotProduct(a: []const f32, b: []const f32) f32
    pub fn distance(a: []const f32, b: []const f32) f32
    pub fn normalize(vector: []f32) void
    pub fn add(a: []const f32, b: []const f32, result: []f32) void
    pub fn scale(vector: []const f32, scalar: f32, result: []f32) void
};

/// Matrix operations
pub const matrix = struct {
    pub fn multiply(a: []const f32, b: []const f32, result: []f32, m: u32, n: u32, k: u32) !void
    pub fn transpose(input: []const f32, output: []f32, rows: usize, cols: usize) void
};
```

#### Usage Example

```zig
const simd = @import("simd");

// Check SIMD configuration
std.debug.print("SIMD config: {s}, f32 width: {d}\n",
               .{simd.config.arch_name, simd.config.width_f32});

// Text processing with SIMD
const text = "Hello, World! SIMD is fast!";
const count = simd.text.countByte(text, 'l'); // SIMD-accelerated
const pos = simd.text.findByte(text, 'W');     // SIMD-accelerated

// Vector operations
const a = [_]f32{1.0, 2.0, 3.0, 4.0};
const b = [_]f32{5.0, 6.0, 7.0, 8.0};
var result: [4]f32 = undefined;

const dot_product = vector.dotProduct(&a, &b);
const distance = vector.distance(&a, &b);
vector.add(&a, &b, &result);
vector.normalize(&result);
```

## Monitoring & Profiling

### `memory_tracker.zig` - Memory Tracking

#### `MemoryProfiler` - Memory Profiler

```zig
pub const MemoryProfiler = struct {
    config: MemoryProfilerConfig,
    allocator: std.mem.Allocator,
    stats: MemoryStats,

    pub fn init(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !*MemoryProfiler
    pub fn deinit(self: *MemoryProfiler) void
    pub fn recordAllocation(self: *MemoryProfiler, size: usize, alignment: u29, file: []const u8, line: u32, function: []const u8) !u64
    pub fn recordDeallocation(self: *MemoryProfiler, id: u64) void
    pub fn getStats(self: *MemoryProfiler) MemoryStats
    pub fn generateReport(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]u8
};
```

#### `TrackedAllocator` - Memory Tracking Allocator

```zig
pub const TrackedAllocator = struct {
    parent_allocator: std.mem.Allocator,
    profiler: *MemoryProfiler,

    pub fn init(parent_allocator: std.mem.Allocator, profiler: *MemoryProfiler) TrackedAllocator
    pub fn allocator(self: *TrackedAllocator) std.mem.Allocator
};
```

#### Usage Example

```zig
const memory_tracker = @import("memory_tracker");

// Initialize memory profiler
var profiler = try memory_tracker.MemoryProfiler.init(allocator, memory_tracker.utils.developmentConfig());
defer profiler.deinit();

// Create tracked allocator
var tracked_alloc = memory_tracker.TrackedAllocator.init(std.heap.page_allocator, &profiler);
const track_alloc = tracked_alloc.allocator();

// Use tracked allocator
const data = try track_alloc.alloc(u8, 1024);
defer track_alloc.free(data);

// Generate memory report
const report = try profiler.generateReport(allocator);
defer allocator.free(report);
```

### `performance_profiler.zig` - Performance Profiling

#### `PerformanceProfiler` - Performance Profiler

```zig
pub const PerformanceProfiler = struct {
    config: ProfilingConfig,
    function_profilers: std.StringHashMap(FunctionProfiler),

    pub fn init(allocator: std.mem.Allocator, config: ProfilingConfig) !*PerformanceProfiler
    pub fn startSession(self: *PerformanceProfiler, session_name: []const u8) !void
    pub fn endSession(self: *PerformanceProfiler) ![]u8
    pub fn startFunctionCall(self: *PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) !u64
    pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void
};
```

#### `FunctionProfiler` - Function-Level Profiling

```zig
pub const FunctionProfiler = struct {
    function_name: []const u8,
    file_name: []const u8,
    line_number: u32,
    call_count: u64,
    total_time: u64,
    min_time: u64,
    max_time: u64,
    average_time: f64,

    pub fn enter(self: *FunctionProfiler) u64
    pub fn exit(self: *FunctionProfiler, entry_time: u64) void
};
```

#### Usage Example

```zig
const performance_profiler = @import("performance_profiler");

// Initialize performance profiler
var profiler = try performance_profiler.PerformanceProfiler.init(allocator, performance_profiler.utils.developmentConfig());
defer profiler.deinit();

// Start profiling session
try profiler.startSession("my_operation");

// Your code here (will be profiled)
const result = try performExpensiveOperation();

// End profiling and get report
const report = try profiler.endSession();
defer allocator.free(report);
```

### `benchmarking.zig` - Benchmarking Framework

#### `Benchmark` - Individual Benchmark

```zig
pub const Benchmark = struct {
    name: []const u8,
    function: *const BenchmarkFn,
    description: []const u8,

    pub fn run(self: *Benchmark, allocator: std.mem.Allocator, config: BenchmarkConfig) !BenchmarkResult
};
```

#### `BenchmarkSuite` - Benchmark Suite

```zig
pub const BenchmarkSuite = struct {
    name: []const u8,
    benchmarks: std.array_list.Managed(Benchmark),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, config: BenchmarkConfig) !*BenchmarkSuite
    pub fn run(self: *BenchmarkSuite, allocator: std.mem.Allocator) ![]BenchmarkResult
};
```

#### Usage Example

```zig
const benchmarking = @import("benchmarking");

// Create benchmark function
fn myBenchmarkFunction(runner: *benchmarking.BenchmarkRunner, allocator: std.mem.Allocator) !void {
    // Benchmark implementation
    const data = try allocator.alloc(u8, 1000);
    defer allocator.free(data);
    // ... perform operations to benchmark
}

// Create benchmark
const benchmark = benchmarking.Benchmark.init(
    "my_benchmark",
    myBenchmarkFunction,
    "Test benchmark for my operation"
);

// Run benchmark
const config = benchmarking.utils.standardConfig();
const result = try benchmark.run(allocator, config);

std.debug.print("Benchmark result: {d:.2} ns/op\n", .{result.timing.mean});
```

## Developer Tools

### `database.zig` - Vector Database

#### `Db` - Vector Database

```zig
pub const Db = struct {
    pub fn init(self: *Db, dimension: usize) !void
    pub fn addEmbedding(self: *Db, vector: []const f32) !RowId
    pub fn search(self: *Db, query: []const f32, k: usize, allocator: std.mem.Allocator) ![]SearchResult
    pub fn getStats(self: *Db) DatabaseStats
    pub fn optimize(self: *Db) !void
    pub fn close(self: *Db) void
};
```

#### Usage Example

```zig
const database = @import("database");

// Open or create database
var db = try database.Db.open("vectors.wdbx", true);
defer db.close();

// Initialize with embedding dimension
try db.init(384);

// Add vectors
const embedding = [_]f32{0.1, 0.2, 0.3, /* ... 384 values */};
const row_id = try db.addEmbedding(&embedding);

// Search for similar vectors
const query = [_]f32{0.15, 0.25, 0.35, /* ... 384 values */};
const results = try db.search(&query, 10, allocator);
defer allocator.free(results);
```

## Examples

### Complete Application

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Initialize framework with memory tracking
    var context = try abi.Context.init(std.heap.page_allocator, .{
        .enable_memory_tracking = true,
        .enable_performance_profiling = true,
    });
    defer context.deinit();

    // Initialize AI agent
    const ai = abi.features.ai;
    var agent = try ai.agent.Agent.init(std.heap.page_allocator, .{ .name = "Creative", .persona = .creative });
    defer agent.deinit();

    // Initialize neural network
    var network = try ai.neural.NeuralNetwork.init(std.heap.page_allocator, .{});
    defer network.deinit();

    // Add layers
    try network.addLayer(.{
        .layer_type = .Dense,
        .input_size = 10,
        .output_size = 5,
        .activation = .ReLU,
    });

    // Initialize memory profiler
    var memory_profiler = try abi.memory_tracker.MemoryProfiler.init(
        std.heap.page_allocator,
        abi.memory_tracker.utils.developmentConfig()
    );
    defer memory_profiler.deinit();

    // Initialize performance profiler
    var perf_profiler = try abi.performance_profiler.PerformanceProfiler.init(
        std.heap.page_allocator,
        abi.performance_profiler.utils.developmentConfig()
    );
    defer perf_profiler.deinit();

    // Start profiling session
    try perf_profiler.startSession("ai_workflow");

    // AI interaction
    const prompt = "Explain neural networks in simple terms";
    const ai_result = try agent.generate(prompt, .{});
    defer std.heap.page_allocator.free(ai_result.content);

    // Neural network computation
    const input = [_]f32{1.0, 0.5, -0.5, 0.2, 0.8, 0.1, -0.3, 0.9, 0.4, -0.7};
    const nn_output = try network.forward(&input);
    defer std.heap.page_allocator.free(nn_output);

    // Generate performance report
    const perf_report = try perf_profiler.endSession();
    defer std.heap.page_allocator.free(perf_report);

    // Generate memory report
    const mem_report = try memory_profiler.generateReport(std.heap.page_allocator);
    defer std.heap.page_allocator.free(mem_report);

    // Output results
    std.debug.print("AI Response: {s}\n", .{ai_result.content});
    std.debug.print("Neural Network Output: {any}\n", .{nn_output});
    std.debug.print("Performance Report:\n{s}\n", .{perf_report});
    std.debug.print("Memory Report:\n{s}\n", .{mem_report});
}
```

### Memory-Safe Neural Network Training

```zig
const std = @import("std");
const abi = @import("abi");

pub fn trainModel(allocator: std.mem.Allocator, data_path: []const u8) !void {
    // Initialize memory profiler
    var memory_profiler = try abi.memory_tracker.MemoryProfiler.init(
        allocator,
        abi.memory_tracker.utils.developmentConfig()
    );
    defer memory_profiler.deinit();

    // Create tracked allocator
    var tracked_alloc = abi.memory_tracker.TrackedAllocator.init(allocator, &memory_profiler);
    const track_alloc = tracked_alloc.allocator();

    // Initialize neural network with tracked allocator
    var network = try abi.neural.NeuralNetwork.init(track_alloc);
    defer network.deinit();

    // Add layers
    try network.addLayer(.{
        .layer_type = .Dense,
        .input_size = 784,
        .output_size = 128,
        .activation = .ReLU,
    });

    try network.addLayer(.{
        .layer_type = .Dense,
        .input_size = 128,
        .output_size = 10,
        .activation = .Softmax,
    });

    // Training loop with memory safety
    const epochs = 10;
    for (0..epochs) |epoch| {
        // Load training data (simplified)
        const input = try track_alloc.alloc(f32, 784);
        defer track_alloc.free(input);

        const target = try track_alloc.alloc(f32, 10);
        defer track_alloc.free(target);

        // Fill with training data
        // ... (data loading logic)

        // Train step
        const loss = try network.trainStep(input, target, 0.01);

        // Log progress with memory info
        const mem_stats = memory_profiler.getStats();
        std.debug.print("Epoch {d}: Loss = {d:.4}, Memory = {d} bytes\n",
                       .{epoch + 1, loss, mem_stats.currentUsage()});
    }

    // Generate final memory report
    const report = try memory_profiler.generateReport(allocator);
    defer allocator.free(report);

    std.debug.print("Training completed. Memory Report:\n{s}\n", .{report});
}
```

This API reference covers the main components and usage patterns. For more detailed information about specific functions and advanced usage, see the inline documentation in the source code or run `zig doc` on the individual modules.

