const std = @import("std");
const abi = @import("abi");

/// Documentation generator for WDBX-AI project
/// Generates comprehensive API documentation from source code
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating WDBX-AI API Documentation", .{});

    // Create docs directory
    try std.fs.cwd().makePath("docs/generated");

    // Generate module documentation
    try generateModuleDocs(allocator);
    try generateApiReference(allocator);
    try generateExamples(allocator);
    try generatePerformanceGuide(allocator);
    try generateDefinitionsReference(allocator);

    // NEW: Scan codebase for public declarations and doc comments
    try generateCodeApiIndex(allocator);

    // NEW: Build search index and a GitHub Pages-friendly index.html
    try generateSearchIndex(allocator);
    try generateDocsIndexHtml(allocator);

    std.log.info("✅ Documentation generation completed!", .{});
}

/// Generate comprehensive module documentation
fn generateModuleDocs(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/MODULE_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\# WDBX-AI Module Reference
        \\
        \\## 📦 Core Modules
        \\
        \\### `abi` - Main Module
        \\The primary module containing all core functionality.
        \\
        \\#### Key Components:
        \\- **Database Engine**: High-performance vector database with HNSW indexing
        \\- **AI System**: Neural networks and machine learning capabilities
        \\- **SIMD Operations**: Optimized vector operations
        \\- **Plugin System**: Extensible architecture for custom functionality
        \\
        \\### `abi.database` - Database Module
        \\Vector database operations and management.
        \\
        \\#### Functions:
        \\```zig
        \\// Initialize database
        \\pub fn init(allocator: Allocator, config: DatabaseConfig) !Database
        \\
        \\// Insert vector
        \\pub fn insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64
        \\
        \\// Search vectors
        \\pub fn search(self: *Database, query: []const f32, k: usize) ![]SearchResult
        \\
        \\// Update vector
        \\pub fn update(self: *Database, id: u64, vector: []const f32) !void
        \\
        \\// Delete vector
        \\pub fn delete(self: *Database, id: u64) !void
        \\```
        \\
        \\### `abi.ai` - AI Module
        \\Artificial intelligence and machine learning capabilities.
        \\
        \\#### Functions:
        \\```zig
        \\// Create neural network
        \\pub fn createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork
        \\
        \\// Train network
        \\pub fn train(self: *NeuralNetwork, data: []const TrainingData) !f32
        \\
        \\// Predict/Infer
        \\pub fn predict(self: *NeuralNetwork, input: []const f32) ![]f32
        \\
        \\// Enhanced agent operations
        \\pub fn createAgent(allocator: Allocator, config: AgentConfig) !EnhancedAgent
        \\```
        \\
        \\### `abi.simd` - SIMD Module
        \\SIMD-optimized vector operations.
        \\
        \\#### Functions:
        \\```zig
        \\// Vector addition
        \\pub fn add(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector subtraction
        \\pub fn subtract(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector multiplication
        \\pub fn multiply(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector normalization
        \\pub fn normalize(result: []f32, input: []const f32) void
        \\```
        \\
        \\### `abi.plugins` - Plugin System
        \\Extensible plugin architecture.
        \\
        \\#### Functions:
        \\```zig
        \\// Load plugin
        \\pub fn loadPlugin(path: []const u8) !Plugin
        \\
        \\// Register plugin
        \\pub fn registerPlugin(plugin: Plugin) !void
        \\
        \\// Execute plugin function
        \\pub fn executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8
        \\```
        \\
        \\## 🔧 Configuration Types
        \\
        \\### DatabaseConfig
        \\```zig
        \\pub const DatabaseConfig = struct {
        \\    max_vectors: usize = 1000000,
        \\    vector_dimension: usize = 128,
        \\    index_type: IndexType = .hnsw,
        \\    storage_path: ?[]const u8 = null,
        \\    enable_caching: bool = true,
        \\    cache_size: usize = 1024 * 1024, // 1MB
        \\};
        \\```
        \\
        \\### NetworkConfig
        \\```zig
        \\pub const NetworkConfig = struct {
        \\    input_size: usize,
        \\    hidden_sizes: []const usize,
        \\    output_size: usize,
        \\    activation: ActivationType = .relu,
        \\    learning_rate: f32 = 0.01,
        \\    batch_size: usize = 32,
        \\};
        \\```
        \\
        \\## 📊 Performance Characteristics
        \\
        \\| Operation | Performance | Memory Usage |
        \\|-----------|-------------|--------------|
        \\| Vector Insert | ~2.5ms (1000 vectors) | ~512 bytes/vector |
        \\| Vector Search | ~13ms (10k vectors, k=10) | ~160 bytes/result |
        \\| Neural Training | ~30μs/iteration | ~1MB/network |
        \\| SIMD Operations | ~3μs (2048 elements) | ~16KB/batch |
        \\
        \\## 🚀 Usage Examples
        \\
        \\### Basic Database Usage
        \\```zig
        \\const std = @import("std");
        \\const abi = @import("abi");
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Initialize database
        \\    const config = abi.DatabaseConfig{
        \\        .max_vectors = 10000,
        \\        .vector_dimension = 128,
        \\    };
        \\    var db = try abi.database.init(allocator, config);
        \\    defer db.deinit();
        \\
        \\    // Insert vectors
        \\    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions
        \\    const id = try db.insert(&vector, "sample_data");
        \\
        \\    // Search for similar vectors
        \\    const results = try db.search(&vector, 10);
        \\    defer allocator.free(results);
        \\
        \\    std.log.info("Found {} similar vectors", .{results.len});
        \\}
        \\```
        \\
        \\### Neural Network Training
        \\```zig
        \\const config = abi.NetworkConfig{
        \\    .input_size = 128,
        \\    .hidden_sizes = &[_]usize{64, 32},
        \\    .output_size = 10,
        \\    .learning_rate = 0.01,
        \\};
        \\
        \\var network = try abi.ai.createNetwork(allocator, config);
        \\defer network.deinit();
        \\
        \\// Training data
        \\const training_data = [_]abi.TrainingData{
        \\    .{ .input = &input1, .output = &output1 },
        \\    .{ .input = &input2, .output = &output2 },
        \\};
        \\
        \\// Train network
        \\const loss = try network.train(&training_data);
        \\std.log.info("Training loss: {}", .{loss});
        \\```
        \\
        \\## 🔍 Error Handling
        \\
        \\All functions return appropriate error types:
        \\- `DatabaseError` - Database-specific errors
        \\- `AIError` - AI/ML operation errors
        \\- `SIMDError` - SIMD operation errors
        \\- `PluginError` - Plugin system errors
        \\
        \\## 📈 Performance Tips
        \\
        \\1. **Use appropriate vector dimensions** - 128-512 dimensions typically optimal
        \\2. **Batch operations** - Group multiple operations for better performance
        \\3. **Enable caching** - Significant performance improvement for repeated queries
        \\4. **SIMD optimization** - Automatically enabled for supported operations
        \\5. **Memory management** - Use arena allocators for bulk operations
        \\
    ;

    try file.writeAll(content);
}

/// Generate API reference documentation
fn generateApiReference(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/API_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\# WDBX-AI API Reference
        \\
        \\## 🗄️ Database API
        \\
        \\### Database
        \\Main database interface for vector operations.
        \\
        \\#### Methods
        \\
        \\##### `init(allocator: Allocator, config: DatabaseConfig) !Database`
        \\Initialize a new database instance.
        \\
        \\**Parameters:**
        \\- `allocator`: Memory allocator to use
        \\- `config`: Database configuration
        \\
        \\**Returns:** Initialized database instance
        \\
        \\**Errors:** `DatabaseError.OutOfMemory`, `DatabaseError.InvalidConfig`
        \\
        \\##### `insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64`
        \\Insert a vector into the database.
        \\
        \\**Parameters:**
        \\- `vector`: Vector data (must match configured dimension)
        \\- `metadata`: Optional metadata string
        \\
        \\**Returns:** Unique ID for the inserted vector
        \\
        \\**Performance:** ~2.5ms for 1000 vectors
        \\
        \\##### `search(self: *Database, query: []const f32, k: usize) ![]SearchResult`
        \\Search for k nearest neighbors.
        \\
        \\**Parameters:**
        \\- `query`: Query vector
        \\- `k`: Number of results to return
        \\
        \\**Returns:** Array of search results (caller must free)
        \\
        \\**Performance:** ~13ms for 10k vectors, k=10
        \\
        \\## 🧠 AI API
        \\
        \\### NeuralNetwork
        \\Neural network for machine learning operations.
        \\
        \\#### Methods
        \\
        \\##### `createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork`
        \\Create a new neural network.
        \\
        \\**Parameters:**
        \\- `allocator`: Memory allocator
        \\- `config`: Network configuration
        \\
        \\**Returns:** Initialized neural network
        \\
        \\##### `train(self: *NeuralNetwork, data: []const TrainingData) !f32`
        \\Train the neural network.
        \\
        \\**Parameters:**
        \\- `data`: Training data array
        \\
        \\**Returns:** Final training loss
        \\
        \\##### `predict(self: *NeuralNetwork, input: []const f32) ![]f32`
        \\Make predictions using the trained network.
        \\
        \\**Parameters:**
        \\- `input`: Input vector
        \\
        \\**Returns:** Prediction results (caller must free)
        \\
        \\## ⚡ SIMD API
        \\
        \\### Vector Operations
        \\SIMD-optimized vector operations.
        \\
        \\#### Functions
        \\
        \\##### `add(result: []f32, a: []const f32, b: []const f32) void`
        \\Add two vectors element-wise.
        \\
        \\**Parameters:**
        \\- `result`: Output vector (must be same size as inputs)
        \\- `a`: First input vector
        \\- `b`: Second input vector
        \\
        \\**Performance:** ~3μs for 2048 elements
        \\
        \\##### `normalize(result: []f32, input: []const f32) void`
        \\Normalize a vector to unit length.
        \\
        \\**Parameters:**
        \\- `result`: Output normalized vector
        \\- `input`: Input vector to normalize
        \\
        \\## 🔌 Plugin API
        \\
        \\### Plugin System
        \\Extensible plugin architecture.
        \\
        \\#### Functions
        \\
        \\##### `loadPlugin(path: []const u8) !Plugin`
        \\Load a plugin from file.
        \\
        \\**Parameters:**
        \\- `path`: Path to plugin file
        \\
        \\**Returns:** Loaded plugin instance
        \\
        \\##### `executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8`
        \\Execute a plugin function.
        \\
        \\**Parameters:**
        \\- `plugin`: Plugin instance
        \\- `function`: Function name to execute
        \\- `args`: JSON-encoded arguments
        \\
        \\**Returns:** JSON-encoded result (caller must free)
        \\
        \\## 📊 Data Types
        \\
        \\### SearchResult
        \\```zig
        \\pub const SearchResult = struct {
        \\    id: u64,
        \\    distance: f32,
        \\    metadata: ?[]const u8,
        \\};
        \\```
        \\
        \\### TrainingData
        \\```zig
        \\pub const TrainingData = struct {
        \\    input: []const f32,
        \\    output: []const f32,
        \\};
        \\```
        \\
        \\## ⚠️ Error Types
        \\
        \\### DatabaseError
        \\```zig
        \\pub const DatabaseError = error{
        \\    OutOfMemory,
        \\    InvalidConfig,
        \\    VectorDimensionMismatch,
        \\    IndexNotFound,
        \\    StorageError,
        \\};
        \\```
        \\
        \\### AIError
        \\```zig
        \\pub const AIError = error{
        \\    InvalidNetworkConfig,
        \\    TrainingDataEmpty,
        \\    ConvergenceFailed,
        \\    InvalidInputSize,
        \\};
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate usage examples
fn generateExamples(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/EXAMPLES.md", .{});
    defer file.close();

    const content =
        \\# WDBX-AI Usage Examples
        \\
        \\## 🚀 Quick Start
        \\
        \\### Basic Vector Database
        \\```zig
        \\const std = @import("std");
        \\const abi = @import("abi");
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Initialize database
        \\    const config = abi.DatabaseConfig{
        \\        .max_vectors = 10000,
        \\        .vector_dimension = 128,
        \\        .enable_caching = true,
        \\    };
        \\    var db = try abi.database.init(allocator, config);
        \\    defer db.deinit();
        \\
        \\    // Insert sample vectors
        \\    for (0..100) |i| {
        \\        var vector: [128]f32 = undefined;
        \\        for (&vector, 0..) |*v, j| {
        \\            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        \\        }
        \\        const id = try db.insert(&vector, "vector_{}");
        \\        std.log.info("Inserted vector with ID: {}", .{id});
        \\    }
        \\
        \\    // Search for similar vectors
        \\    const query = [_]f32{1.0} ** 128;
        \\    const results = try db.search(&query, 5);
        \\    defer allocator.free(results);
        \\
        \\    std.log.info("Found {} similar vectors:", .{results.len});
        \\    for (results, 0..) |result, i| {
        \\        std.log.info("  {}: ID={}, Distance={}", .{ i, result.id, result.distance });
        \\    }
        \\}
        \\```
        \\
        \\## 🧠 Machine Learning Pipeline
        \\
        \\### Neural Network Training
        \\```zig
        \\pub fn trainModel() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Create network
        \\    const config = abi.NetworkConfig{
        \\        .input_size = 128,
        \\        .hidden_sizes = &[_]usize{64, 32},
        \\        .output_size = 10,
        \\        .learning_rate = 0.01,
        \\        .batch_size = 32,
        \\    };
        \\    var network = try abi.ai.createNetwork(allocator, config);
        \\    defer network.deinit();
        \\
        \\    // Generate training data
        \\    var training_data = std.ArrayList(abi.TrainingData).init(allocator);
        \\    defer training_data.deinit();
        \\
        \\    for (0..1000) |i| {
        \\        var input: [128]f32 = undefined;
        \\        var output: [10]f32 = undefined;
        \\
        \\        // Generate random input
        \\        for (&input) |*v| {
        \\            v.* = std.rand.DefaultPrng.init(@as(u64, i)).random().float(f32);
        \\        }
        \\
        \\        // Generate target output (one-hot encoding)
        \\        @memset(&output, 0);
        \\        output[i % 10] = 1.0;
        \\
        \\        try training_data.append(abi.TrainingData{
        \\            .input = &input,
        \\            .output = &output,
        \\        });
        \\    }
        \\
        \\    // Train network
        \\    const loss = try network.train(training_data.items);
        \\    std.log.info("Training completed with loss: {}", .{loss});
        \\
        \\    // Test prediction
        \\    const test_input = [_]f32{0.5} ** 128;
        \\    const prediction = try network.predict(&test_input);
        \\    defer allocator.free(prediction);
        \\
        \\    std.log.info("Prediction: {any}", .{prediction});
        \\}
        \\```
        \\
        \\## ⚡ SIMD Operations
        \\
        \\### Vector Processing
        \\```zig
        \\pub fn vectorProcessing() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Allocate vectors
        \\    const size = 2048;
        \\    const a = try allocator.alloc(f32, size);
        \\    defer allocator.free(a);
        \\    const b = try allocator.alloc(f32, size);
        \\    defer allocator.free(b);
        \\    const result = try allocator.alloc(f32, size);
        \\    defer allocator.free(result);
        \\
        \\    // Initialize vectors
        \\    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
        \\    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2));
        \\
        \\    // SIMD operations
        \\    const start_time = std.time.nanoTimestamp();
        \\
        \\    abi.simd.add(result, a, b);
        \\    abi.simd.subtract(result, result, a);
        \\    abi.simd.multiply(result, result, b);
        \\    abi.simd.normalize(result, result);
        \\
        \\    const end_time = std.time.nanoTimestamp();
        \\    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
        \\
        \\    std.log.info("SIMD operations completed in {}ms", .{duration});
        \\    std.log.info("Result sample: [{}, {}, {}]", .{ result[0], result[1], result[2] });
        \\}
        \\```
        \\
        \\## 🔌 Plugin System
        \\
        \\### Custom Plugin
        \\```zig
        \\// plugin_example.zig
        \\const std = @import("std");
        \\
        \\export fn process_data(input: [*c]const u8, input_len: usize, output: [*c]u8, output_len: *usize) c_int {
        \\    // Process input data
        \\    const input_slice = input[0..input_len];
        \\
        \\    // Example: convert to uppercase
        \\    var result = std.ArrayList(u8).init(std.heap.page_allocator);
        \\    defer result.deinit();
        \\
        \\    for (input_slice) |byte| {
        \\        if (byte >= 'a' and byte <= 'z') {
        \\            try result.append(byte - 32);
        \\        } else {
        \\            try result.append(byte);
        \\        }
        \\    }
        \\
        \\    // Copy result to output
        \\    if (result.items.len > output_len.*) {
        \\        return -1; // Buffer too small
        \\    }
        \\
        \\    @memcpy(output[0..result.items.len], result.items);
        \\    output_len.* = result.items.len;
        \\    return 0; // Success
        \\}
        \\
        \\// Using the plugin
        \\pub fn usePlugin() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Load plugin
        \\    const plugin = try abi.plugins.loadPlugin("plugin_example.zig");
        \\    defer plugin.deinit();
        \\
        \\    // Execute plugin function
        \\    const input = "hello world";
        \\    const result = try abi.plugins.executePlugin(plugin, "process_data", input);
        \\    defer allocator.free(result);
        \\
        \\    std.log.info("Plugin result: {s}", .{result});
        \\}
        \\```
        \\
        \\## 🎯 Performance Optimization
        \\
        \\### Batch Operations
        \\```zig
        \\pub fn batchOperations() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    var db = try abi.database.init(allocator, abi.DatabaseConfig{});
        \\    defer db.deinit();
        \\
        \\    // Batch insert
        \\    const batch_size = 1000;
        \\    var vectors = try allocator.alloc([]f32, batch_size);
        \\    defer {
        \\        for (vectors) |vec| allocator.free(vec);
        \\        allocator.free(vectors);
        \\    }
        \\
        \\    // Generate batch data
        \\    for (vectors, 0..) |*vec, i| {
        \\        vec.* = try allocator.alloc(f32, 128);
        \\        for (vec.*, 0..) |*v, j| {
        \\            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        \\        }
        \\    }
        \\
        \\    // Insert batch
        \\    const start_time = std.time.nanoTimestamp();
        \\    for (vectors) |vec| {
        \\        _ = try db.insert(vec, null);
        \\    }
        \\    const end_time = std.time.nanoTimestamp();
        \\
        \\    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
        \\    const throughput = @as(f64, @floatFromInt(batch_size)) / (duration / 1000.0);
        \\
        \\    std.log.info("Batch insert: {} vectors in {}ms", .{ batch_size, duration });
        \\    std.log.info("Throughput: {} vectors/sec", .{throughput});
        \\}
        \\```
        \\
        \\## 🔧 Error Handling
        \\
        \\### Comprehensive Error Handling
        \\```zig
        \\pub fn robustOperations() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    var db = abi.database.init(allocator, abi.DatabaseConfig{}) catch |err| switch (err) {
        \\        error.OutOfMemory => {
        \\            std.log.err("Failed to allocate memory for database");
        \\            return;
        \\        },
        \\        error.InvalidConfig => {
        \\            std.log.err("Invalid database configuration");
        \\            return;
        \\        },
        \\        else => return err,
        \\    };
        \\    defer db.deinit();
        \\
        \\    // Safe vector operations
        \\    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions
        \\
        \\    const id = db.insert(&vector, "test") catch |err| switch (err) {
        \\        error.VectorDimensionMismatch => {
        \\            std.log.err("Vector dimension mismatch");
        \\            return;
        \\        },
        \\        error.StorageError => {
        \\            std.log.err("Storage operation failed");
        \\            return;
        \\        },
        \\        else => return err,
        \\    };
        \\
        \\    std.log.info("Successfully inserted vector with ID: {}", .{id});
        \\}
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate performance guide
fn generatePerformanceGuide(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/PERFORMANCE_GUIDE.md", .{});
    defer file.close();

    const content =
        \\# WDBX-AI Performance Guide
        \\
        \\## 🚀 Performance Characteristics
        \\
        \\### Database Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Single Insert | ~2.5ms | ~512 bytes | 128-dim vectors |
        \\| Batch Insert (100) | ~40ms | ~51KB | 100 vectors |
        \\| Batch Insert (1000) | ~400ms | ~512KB | 1000 vectors |
        \\| Search (k=10) | ~13ms | ~1.6KB | 10k vectors |
        \\| Search (k=100) | ~14ms | ~16KB | 10k vectors |
        \\| Update | ~1ms | ~512 bytes | Single vector |
        \\| Delete | ~0.5ms | ~0 bytes | Single vector |
        \\
        \\### AI/ML Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Network Creation | ~1ms | ~1MB | 128→64→32→10 |
        \\| Training Iteration | ~30μs | ~1MB | Batch size 32 |
        \\| Prediction | ~10μs | ~1KB | Single input |
        \\| Batch Prediction | ~100μs | ~10KB | 100 inputs |
        \\
        \\### SIMD Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Vector Add (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Multiply (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Normalize (2048) | ~5μs | ~16KB | Includes sqrt |
        \\| Matrix Multiply (64x64) | ~50μs | ~32KB | SIMD optimized |
        \\
        \\## ⚡ Optimization Strategies
        \\
        \\### 1. Memory Management
        \\```zig
        \\// Use arena allocators for bulk operations
        \\var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        \\defer arena.deinit();
        \\const allocator = arena.allocator();
        \\
        \\// Pre-allocate buffers for repeated operations
        \\const buffer_size = 1024 * 1024; // 1MB
        \\const buffer = try allocator.alloc(u8, buffer_size);
        \\defer allocator.free(buffer);
        \\```
        \\
        \\### 2. Batch Processing
        \\```zig
        \\// Process vectors in batches for better performance
        \\const BATCH_SIZE = 100;
        \\for (0..total_vectors / BATCH_SIZE) |batch| {
        \\    const start = batch * BATCH_SIZE;
        \\    const end = @min(start + BATCH_SIZE, total_vectors);
        \\    
        \\    // Process batch
        \\    for (vectors[start..end]) |vector| {
        \\        _ = try db.insert(vector, null);
        \\    }
        \\}
        \\```
        \\
        \\### 3. SIMD Optimization
        \\```zig
        \\// Use SIMD operations for vector processing
        \\const VECTOR_SIZE = 128;
        \\const SIMD_SIZE = 4; // Process 4 elements at once
        \\
        \\var i: usize = 0;
        \\while (i + SIMD_SIZE <= VECTOR_SIZE) : (i += SIMD_SIZE) {
        \\    const va = @as(@Vector(4, f32), a[i..][0..4].*);
        \\    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
        \\    const result = va + vb;
        \\    @memcpy(output[i..][0..4], @as([4]f32, result)[0..]);
        \\}
        \\```
        \\
        \\### 4. Caching Strategy
        \\```zig
        \\// Enable database caching for repeated queries
        \\const config = abi.DatabaseConfig{
        \\    .enable_caching = true,
        \\    .cache_size = 1024 * 1024, // 1MB cache
        \\};
        \\
        \\// Use LRU cache for frequently accessed data
        \\var cache = std.HashMap(u64, []f32, std.hash_map.default_hash_fn(u64), std.hash_map.default_eql_fn(u64)).init(allocator);
        \\defer {
        \\    var iterator = cache.iterator();
        \\    while (iterator.next()) |entry| {
        \\        allocator.free(entry.value_ptr.*);
        \\    }
        \\    cache.deinit();
        \\}
        \\```
        \\
        \\## 📊 Benchmarking
        \\
        \\### Running Benchmarks
        \\```bash
        \\# Run all benchmarks
        \\zig build benchmark
        \\
        \\# Run specific benchmark types
        \\zig build benchmark-db      # Database performance
        \\zig build benchmark-neural  # AI/ML performance
        \\zig build benchmark-simple  # General performance
        \\
        \\# Run with profiling
        \\zig build profile
        \\```
        \\
        \\### Custom Benchmarking
        \\```zig
        \\pub fn benchmarkOperation() !void {
        \\    const iterations = 1000;
        \\    var times = try allocator.alloc(u64, iterations);
        \\    defer allocator.free(times);
        \\
        \\    // Warm up
        \\    for (0..10) |_| {
        \\        // Perform operation
        \\    }
        \\
        \\    // Benchmark
        \\    for (times, 0..) |*time, i| {
        \\        const start = std.time.nanoTimestamp();
        \\        
        \\        // Perform operation
        \\        
        \\        const end = std.time.nanoTimestamp();
        \\        time.* = end - start;
        \\    }
        \\
        \\    // Calculate statistics
        \\    std.sort.heap(u64, times, {}, comptime std.sort.asc(u64));
        \\    const p50 = times[iterations / 2];
        \\    const p95 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.95))];
        \\    const p99 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.99))];
        \\
        \\    std.log.info("P50: {}ns, P95: {}ns, P99: {}ns", .{ p50, p95, p99 });
        \\}
        \\```
        \\
        \\## 🔍 Profiling Tools
        \\
        \\### Memory Profiling
        \\```zig
        \\// Enable memory tracking
        \\const memory_tracker = abi.memory_tracker.init(allocator);
        \\defer memory_tracker.deinit();
        \\
        \\// Track allocations
        \\memory_tracker.startTracking();
        \\
        \\// Perform operations
        \\
        \\// Get memory statistics
        \\const stats = memory_tracker.getStats();
        \\std.log.info("Peak memory: {} bytes", .{stats.peak_memory});
        \\std.log.info("Total allocations: {}", .{stats.total_allocations});
        \\```
        \\
        \\### Performance Profiling
        \\```zig
        \\// Use performance profiler
        \\const profiler = abi.performance_profiler.init(allocator);
        \\defer profiler.deinit();
        \\
        \\// Start profiling
        \\profiler.startProfiling("operation_name");
        \\
        \\// Perform operation
        \\
        \\// Stop profiling
        \\profiler.stopProfiling("operation_name");
        \\
        \\// Get results
        \\const results = profiler.getResults();
        \\for (results) |result| {
        \\    std.log.info("{}: {}ms", .{ result.name, result.duration_ms });
        \\}
        \\```
        \\
        \\## 🎯 Performance Tips
        \\
        \\### 1. Vector Dimensions
        \\- **Optimal range**: 128-512 dimensions
        \\- **Too small**: Poor representation quality
        \\- **Too large**: Increased memory and computation
        \\
        \\### 2. Batch Sizes
        \\- **Database inserts**: 100-1000 vectors per batch
        \\- **Neural training**: 32-128 samples per batch
        \\- **SIMD operations**: 1024-4096 elements per batch
        \\
        \\### 3. Memory Allocation
        \\- **Use arena allocators** for bulk operations
        \\- **Pre-allocate buffers** for repeated operations
        \\- **Enable caching** for frequently accessed data
        \\
        \\### 4. SIMD Usage
        \\- **Automatic optimization** for supported operations
        \\- **Vector size alignment** for best performance
        \\- **Batch processing** for maximum throughput
        \\
        \\## 📈 Performance Monitoring
        \\
        \\### Real-time Metrics
        \\```zig
        \\// Monitor performance in real-time
        \\const monitor = abi.performance_monitor.init(allocator);
        \\defer monitor.deinit();
        \\
        \\// Start monitoring
        \\monitor.startMonitoring();
        \\
        \\// Perform operations
        \\
        \\// Get metrics
        \\const metrics = monitor.getMetrics();
        \\std.log.info("Operations/sec: {}", .{metrics.operations_per_second});
        \\std.log.info("Average latency: {}ms", .{metrics.average_latency_ms});
        \\std.log.info("Memory usage: {}MB", .{metrics.memory_usage_mb});
        \\```
        \\
        \\### Performance Regression Detection
        \\```zig
        \\// Compare with baseline performance
        \\const baseline = try loadBaselinePerformance("baseline.json");
        \\const current = try measureCurrentPerformance();
        \\
        \\const regression_threshold = 0.05; // 5% regression
        \\if (current.avg_latency > baseline.avg_latency * (1.0 + regression_threshold)) {
        \\    std.log.warn("Performance regression detected!");
        \\    std.log.warn("Baseline: {}ms, Current: {}ms", .{ baseline.avg_latency, current.avg_latency });
        \\}
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate comprehensive definitions reference documentation
fn generateDefinitionsReference(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/DEFINITIONS_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\# WDBX-AI Definitions Reference
        \\
        \\## 📚 Core Concepts
        \\
        \\### Vector Database
        \\A specialized database designed to store, index, and search high-dimensional vectors efficiently. Unlike traditional databases that store scalar values, vector databases are optimized for similarity search operations using metrics like cosine similarity, Euclidean distance, or dot product.
        \\
        \\**Key characteristics:**
        \\- **High-dimensional storage**: Handles vectors with hundreds or thousands of dimensions
        \\- **Similarity search**: Finds vectors most similar to a query vector
        \\- **Indexing algorithms**: Uses specialized indices like HNSW, IVF, or LSH for fast retrieval
        \\- **Scalability**: Designed to handle millions or billions of vectors
        \\
        \\### Embeddings
        \\Dense vector representations of data (text, images, audio, etc.) that capture semantic meaning in a continuous vector space. Embeddings are typically generated by machine learning models and enable similarity comparisons between different data points.
        \\
        \\**Examples:**
        \\- **Text embeddings**: "cat" and "kitten" have similar vector representations
        \\- **Image embeddings**: Photos of similar objects cluster together in vector space
        \\- **Audio embeddings**: Similar sounds or music pieces have nearby representations
        \\
        \\### HNSW (Hierarchical Navigable Small World)
        \\A graph-based indexing algorithm that builds a multi-layered network of connections between vectors. It provides excellent performance for approximate nearest neighbor search with logarithmic time complexity.
        \\
        \\**Structure:**
        \\- **Bottom layer**: Contains all vectors with short-range connections
        \\- **Upper layers**: Contain subsets of vectors with long-range connections
        \\- **Navigation**: Search starts from top layer and progressively moves down
        \\
        \\**Parameters:**
        \\- `M`: Maximum number of connections per vector (typically 16-64)
        \\- `efConstruction`: Size of candidate set during construction (typically 200-800)
        \\- `efSearch`: Size of candidate set during search (affects recall vs speed trade-off)
        \\
        \\## 🧠 AI & Machine Learning
        \\
        \\### Neural Network
        \\A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines the strength of the signal passed between neurons.
        \\
        \\**Architecture components:**
        \\- **Input layer**: Receives raw data features
        \\- **Hidden layers**: Process and transform the input through weighted connections
        \\- **Output layer**: Produces final predictions or classifications
        \\- **Activation functions**: Non-linear functions that introduce complexity (ReLU, Sigmoid, Tanh)
        \\
        \\### Backpropagation
        \\The fundamental algorithm for training neural networks. It calculates gradients of the loss function with respect to each weight by propagating errors backwards through the network, then updates weights to minimize the loss.
        \\
        \\**Process:**
        \\1. **Forward pass**: Input flows through network to produce output
        \\2. **Loss calculation**: Compare output to target, calculate error
        \\3. **Backward pass**: Propagate error gradients back through layers
        \\4. **Weight update**: Adjust weights using gradients and learning rate
        \\
        \\### Gradient Descent
        \\An optimization algorithm that iteratively adjusts model parameters to minimize a loss function. It moves in the direction of steepest descent of the loss landscape.
        \\
        \\**Variants:**
        \\- **Batch gradient descent**: Uses entire dataset for each update
        \\- **Stochastic gradient descent (SGD)**: Uses one sample at a time
        \\- **Mini-batch gradient descent**: Uses small batches (typically 32-256 samples)
        \\
        \\**Hyperparameters:**
        \\- **Learning rate**: Step size for parameter updates (typically 0.001-0.1)
        \\- **Momentum**: Helps overcome local minima and speeds convergence
        \\- **Weight decay**: Regularization term to prevent overfitting
        \\
        \\### Agent-Based Systems
        \\Autonomous software entities that perceive their environment, make decisions, and take actions to achieve specific goals. In AI systems, agents can be simple rule-based systems or complex neural networks.
        \\
        \\**Components:**
        \\- **Perception**: Sensors to observe environment state
        \\- **Decision making**: Logic or learned behavior to choose actions
        \\- **Action**: Effectors to modify the environment
        \\- **Memory**: Storage of past experiences and learned knowledge
        \\
        \\**Types:**
        \\- **Reactive agents**: Respond directly to current perceptions
        \\- **Deliberative agents**: Plan sequences of actions to achieve goals
        \\- **Learning agents**: Improve performance through experience
        \\- **Multi-agent systems**: Multiple agents cooperating or competing
        \\
        \\## ⚡ Performance & Optimization
        \\
        \\### SIMD (Single Instruction, Multiple Data)
        \\A parallel computing technique where a single instruction operates on multiple data points simultaneously. Modern CPUs have SIMD units that can process multiple floating-point numbers in one cycle.
        \\
        \\**Benefits:**
        \\- **Vectorized operations**: Process entire arrays in fewer CPU cycles
        \\- **Memory bandwidth**: More efficient use of memory bandwidth
        \\- **Energy efficiency**: Better performance per watt consumption
        \\
        \\**Common instruction sets:**
        \\- **SSE (128-bit)**: 4 float32 or 2 float64 operations per instruction
        \\- **AVX (256-bit)**: 8 float32 or 4 float64 operations per instruction
        \\- **AVX-512 (512-bit)**: 16 float32 or 8 float64 operations per instruction
        \\
        \\### Memory Alignment
        \\The practice of organizing data in memory so that it starts at addresses that are multiples of the data type size or cache line size. Proper alignment improves CPU access speed and enables SIMD optimizations.
        \\
        \\**Alignment requirements:**
        \\- **Cache line alignment**: Data aligned to 64-byte boundaries (typical cache line size)
        \\- **SIMD alignment**: Vectors aligned to 16, 32, or 64-byte boundaries
        \\- **Page alignment**: Large allocations aligned to 4KB page boundaries
        \\
        \\### Batch Processing
        \\The practice of grouping multiple operations together to improve throughput and reduce overhead. Batching amortizes the cost of setup operations across multiple data items.
        \\
        \\**Advantages:**
        \\- **Reduced function call overhead**: Fewer individual operation calls
        \\- **Better memory locality**: Sequential access patterns
        \\- **SIMD utilization**: Process multiple items with vector instructions
        \\- **Cache efficiency**: Better temporal and spatial locality
        \\
        \\## 📊 Distance Metrics
        \\
        \\### Euclidean Distance
        \\The straight-line distance between two points in multidimensional space. Most intuitive distance metric, corresponding to physical distance in 2D/3D space.
        \\
        \\**Formula:** `√(Σ(a_i - b_i)²)`
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞)
        \\- **Symmetric**: d(a,b) = d(b,a)
        \\- **Triangle inequality**: d(a,c) ≤ d(a,b) + d(b,c)
        \\- **Best for**: Continuous features, image pixels, physical measurements
        \\
        \\### Cosine Similarity
        \\Measures the cosine of the angle between two vectors, effectively measuring their directional similarity regardless of magnitude. Widely used for text embeddings and recommendation systems.
        \\
        \\**Formula:** `(a·b) / (||a|| × ||b||)`
        \\
        \\**Properties:**
        \\- **Range**: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite
        \\- **Magnitude invariant**: Only considers direction, not length
        \\- **Best for**: Text embeddings, sparse features, normalized data
        \\
        \\### Manhattan Distance (L1)
        \\The sum of absolute differences between corresponding elements. Named after the grid-like street pattern of Manhattan where you can only travel along grid lines.
        \\
        \\**Formula:** `Σ|a_i - b_i|`
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞)
        \\- **Robust to outliers**: Less sensitive than Euclidean distance
        \\- **Sparsity promoting**: Tends to produce sparse solutions in optimization
        \\- **Best for**: Sparse data, robustness to outliers, certain optimization problems
        \\
        \\### Dot Product
        \\The sum of products of corresponding elements. While not technically a distance metric, it's commonly used in neural networks and similarity calculations.
        \\
        \\**Formula:** `Σ(a_i × b_i)`
        \\
        \\**Properties:**
        \\- **Range**: (-∞, ∞)
        \\- **Not symmetric for distance**: Higher values indicate greater similarity
        \\- **Computationally efficient**: Single pass through vectors
        \\- **Best for**: Neural network computations, attention mechanisms
        \\
        \\## 🔧 System Architecture
        \\
        \\### Plugin Architecture
        \\A software design pattern that allows extending core functionality through dynamically loaded modules. Plugins are separate, compiled units that implement predefined interfaces.
        \\
        \\**Benefits:**
        \\- **Modularity**: Core system remains lean, features added as needed
        \\- **Extensibility**: Third-party developers can add functionality
        \\- **Isolation**: Plugin failures don't crash the core system
        \\- **Hot-swapping**: Plugins can be loaded/unloaded at runtime
        \\
        \\**Implementation approaches:**
        \\- **Dynamic libraries**: Shared objects (.so, .dll, .dylib) loaded at runtime
        \\- **Process isolation**: Plugins run in separate processes with IPC
        \\- **Scripting languages**: Embed interpreters for plugin languages
        \\- **WebAssembly**: Sandboxed plugins with near-native performance
        \\
        \\### Memory Management
        \\The practice of efficiently allocating, using, and deallocating memory in programs. Critical for performance and preventing memory leaks or corruption.
        \\
        \\**Allocation strategies:**
        \\- **Stack allocation**: Fast, automatic cleanup, limited size
        \\- **Heap allocation**: Flexible size, manual management required
        \\- **Pool allocation**: Pre-allocate fixed-size blocks for efficiency
        \\- **Arena allocation**: Bulk allocation with batch cleanup
        \\
        \\**Best practices:**
        \\- **RAII**: Resource Acquisition Is Initialization - tie resource lifetime to object lifetime
        \\- **Reference counting**: Track object usage automatically
        \\- **Garbage collection**: Automatic memory management (with performance trade-offs)
        \\- **Custom allocators**: Optimize for specific usage patterns
        \\
        \\### Caching Strategies
        \\Techniques for storing frequently accessed data in faster storage layers to improve performance. Caches exploit temporal and spatial locality of access patterns.
        \\
        \\**Cache types:**
        \\- **CPU caches**: L1/L2/L3 caches built into processors
        \\- **Memory caches**: Software caches in RAM
        \\- **Disk caches**: SSD or fast disk storage for slower storage
        \\- **Network caches**: CDNs and edge caches for distributed systems
        \\
        \\**Eviction policies:**
        \\- **LRU (Least Recently Used)**: Remove oldest accessed items
        \\- **LFU (Least Frequently Used)**: Remove least popular items
        \\- **FIFO (First In, First Out)**: Simple queue-based removal
        \\- **Random**: Simple but effective for many workloads
        \\
        \\## 📈 Performance Metrics
        \\
        \\### Throughput
        \\The number of operations completed per unit time. For vector databases, this typically measures insertions per second or queries per second.
        \\
        \\**Measurement:**
        \\- **Operations per second (OPS)**: Raw operation count
        \\- **Requests per second (RPS)**: For client-server systems
        \\- **Bandwidth**: Data processed per unit time (MB/s, GB/s)
        \\
        \\**Optimization factors:**
        \\- **Parallelism**: Concurrent processing of multiple operations
        \\- **Batching**: Group operations to reduce overhead
        \\- **Pipeline depth**: Overlap different stages of processing
        \\
        \\### Latency
        \\The time required to complete a single operation from start to finish. Low latency is critical for real-time applications and user experience.
        \\
        \\**Types:**
        \\- **Mean latency**: Average response time across all operations
        \\- **Percentile latency**: P50, P95, P99 latencies for understanding distribution
        \\- **Tail latency**: Worst-case response times that affect user experience
        \\
        \\**Factors affecting latency:**
        \\- **Algorithm complexity**: O(1) vs O(log n) vs O(n) operations
        \\- **Memory hierarchy**: Cache hits vs misses vs disk access
        \\- **Network delays**: Physical distance and congestion
        \\- **Queueing delays**: Waiting time under high load
        \\
        \\### Recall and Precision
        \\Metrics for evaluating the quality of search results, particularly important for approximate nearest neighbor search where exact results may be traded for speed.
        \\
        \\**Recall:**
        \\- **Definition**: Fraction of relevant results that were retrieved
        \\- **Formula**: True Positives / (True Positives + False Negatives)
        \\- **Range**: [0, 1] where 1 = perfect recall
        \\
        \\**Precision:**
        \\- **Definition**: Fraction of retrieved results that are relevant
        \\- **Formula**: True Positives / (True Positives + False Positives)
        \\- **Range**: [0, 1] where 1 = perfect precision
        \\
        \\**Trade-offs:**
        \\- **Speed vs Accuracy**: Faster algorithms often sacrifice some recall
        \\- **Memory vs Quality**: Larger indices typically provide better recall
        \\- **Index parameters**: Tuning affects the recall-speed trade-off
        \\
        \\## 🔐 Data Types & Formats
        \\
        \\### Floating Point Precision
        \\Different levels of precision for storing real numbers, each with trade-offs between accuracy, memory usage, and computational speed.
        \\
        \\**Common formats:**
        \\- **float16 (half)**: 16-bit, ±65504 range, ~3 decimal digits precision
        \\- **float32 (single)**: 32-bit, ±3.4×10³⁸ range, ~7 decimal digits precision
        \\- **float64 (double)**: 64-bit, ±1.8×10³⁰⁸ range, ~15 decimal digits precision
        \\- **bfloat16**: 16-bit with float32 range but reduced precision, popular in ML
        \\
        \\**Usage considerations:**
        \\- **ML inference**: float16 often sufficient, 2x memory savings
        \\- **Scientific computing**: float64 needed for numerical stability
        \\- **Vector embeddings**: float32 typically optimal balance
        \\- **Storage optimization**: Consider quantization for large datasets
        \\
        \\### Vector Quantization
        \\Techniques for reducing the memory footprint of high-dimensional vectors while preserving essential similarity relationships.
        \\
        \\**Methods:**
        \\- **Scalar quantization**: Map float32 to int8/int16 with scale factors
        \\- **Product quantization**: Divide vectors into subvectors, quantize each separately
        \\- **Binary quantization**: Extreme compression to binary vectors (1 bit per dimension)
        \\- **Learned quantization**: Use neural networks to optimize quantization
        \\
        \\**Benefits:**
        \\- **Memory reduction**: 4-32x compression possible
        \\- **Faster search**: Integer operations faster than floating point
        \\- **Cache efficiency**: More vectors fit in CPU cache
        \\- **Storage cost**: Reduced disk and network transfer requirements
        \\
        \\### Sparse vs Dense Vectors
        \\Two fundamental representations for high-dimensional data with different performance and storage characteristics.
        \\
        \\**Dense vectors:**
        \\- **Storage**: Every dimension explicitly stored (even zeros)
        \\- **Memory**: Fixed memory usage regardless of sparsity
        \\- **Computation**: Regular, predictable access patterns
        \\- **Best for**: Neural network embeddings, image features, audio features
        \\
        \\**Sparse vectors:**
        \\- **Storage**: Only non-zero dimensions stored with indices
        \\- **Memory**: Proportional to number of non-zero elements
        \\- **Computation**: Irregular access patterns, potential cache misses
        \\- **Best for**: Text features (TF-IDF), categorical features, user-item matrices
        \\
        \\**Hybrid approaches:**
        \\- **Compressed sparse**: Further compress indices and values
        \\- **Block sparse**: Sparse at block level, dense within blocks
        \\- **Adaptive**: Switch between representations based on sparsity level
        \\
    ;

    try file.writeAll(content);
}

// ===== New Code: Source scanner for public declarations =====
const Declaration = struct {
    name: []u8,
    kind: []u8,
    signature: []u8,
    doc: []u8,
};

fn generateCodeApiIndex(allocator: std.mem.Allocator) !void {
    // Use an arena for all temporary allocations in scanning to avoid leaks and simplify ownership
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var files = std.ArrayListUnmanaged([]u8){};
    defer files.deinit(a);

    try collectZigFiles(a, "src", &files);

    var out = try std.fs.cwd().createFile("docs/generated/CODE_API_INDEX.md", .{ .truncate = true });
    defer out.close();

    const writef = struct {
        fn go(file: std.fs.File, alloc2: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(alloc2, fmt, args);
            defer alloc2.free(s);
            try file.writeAll(s);
        }
    }.go;

    try writef(out, a, "# Code API Index (Scanned)\n\n", .{});
    try writef(out, a, "Scanned {d} Zig files under `src/`. This index lists public declarations discovered along with leading doc comments.\n\n", .{files.items.len});

    var decls = std.ArrayListUnmanaged(Declaration){};
    defer decls.deinit(a);

    for (files.items) |rel| {
        decls.clearRetainingCapacity();
        try scanFile(a, rel, &decls);
        if (decls.items.len == 0) continue;

        try writef(out, a, "## {s}\n\n", .{rel});
        for (decls.items) |d| {
            try writef(out, a, "- {s} `{s}`\n\n", .{ d.kind, d.name });
            if (d.doc.len > 0) {
                try writef(out, a, "{s}\n\n", .{d.doc});
            }
            if (d.signature.len > 0) {
                try writef(out, a, "```zig\n{s}\n```\n\n", .{d.signature});
            }
        }
    }
}

fn collectZigFiles(allocator: std.mem.Allocator, dir_path: []const u8, out_files: *std.ArrayListUnmanaged([]u8)) !void {
    var stack = std.ArrayListUnmanaged([]u8){};
    defer {
        for (stack.items) |p| allocator.free(p);
        stack.deinit(allocator);
    }
    try stack.append(allocator, try allocator.dupe(u8, dir_path));

    while (stack.items.len > 0) {
        const idx = stack.items.len - 1;
        const path = stack.items[idx];
        _ = stack.pop();
        defer allocator.free(path);

        var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch continue;
        defer dir.close();

        var it = dir.iterate();
        while (it.next() catch null) |entry| {
            if (entry.kind == .file) {
                if (std.mem.endsWith(u8, entry.name, ".zig")) {
                    const rel = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                    try out_files.append(allocator, rel);
                }
            } else if (entry.kind == .directory) {
                if (std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;
                const sub = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                try stack.append(allocator, sub);
            }
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, rel_path: []const u8, decls: *std.ArrayListUnmanaged(Declaration)) !void {
    const file = try std.fs.cwd().openFile(rel_path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 1024 * 1024 * 4);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var doc_buf = std.ArrayListUnmanaged(u8){};
    defer doc_buf.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "///")) {
            // accumulate doc lines
            const doc_line = std.mem.trim(u8, trimmed[3..], " \t");
            try doc_buf.appendSlice(allocator, doc_line);
            try doc_buf.append(allocator, '\n');
            continue;
        }

        // Identify public declarations after doc comments
        if (isPubDecl(trimmed)) {
            const kind = detectKind(trimmed);
            const name = extractName(allocator, trimmed) catch {
                // reset doc buffer and continue
                doc_buf.clearRetainingCapacity();
                continue;
            };
            const sig = try allocator.dupe(u8, trimmed);
            const doc = try allocator.dupe(u8, doc_buf.items);
            doc_buf.clearRetainingCapacity();

            try decls.append(allocator, .{
                .name = name,
                .kind = try allocator.dupe(u8, kind),
                .signature = sig,
                .doc = doc,
            });
            continue;
        } else {
            // reset doc buffer if we encounter a non-doc, non-decl line
            if (trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, "//")) {
                doc_buf.clearRetainingCapacity();
            }
        }
    }
}

fn isPubDecl(line: []const u8) bool {
    // consider pub fn/const/var/type usingnamespace
    if (!std.mem.startsWith(u8, line, "pub ")) return false;
    return std.mem.indexOfAny(u8, line[4..], "fctuv") != null // quick filter
    or std.mem.startsWith(u8, line, "pub usingnamespace") or std.mem.indexOf(u8, line, " struct") != null or std.mem.indexOf(u8, line, " enum") != null;
}

fn detectKind(line: []const u8) []const u8 {
    if (std.mem.startsWith(u8, line, "pub fn ")) return "fn";
    if (std.mem.startsWith(u8, line, "pub const ")) {
        if (std.mem.indexOf(u8, line, " struct") != null) return "type";
        if (std.mem.indexOf(u8, line, " enum") != null) return "type";
        return "const";
    }
    if (std.mem.startsWith(u8, line, "pub var ")) return "var";
    if (std.mem.startsWith(u8, line, "pub usingnamespace")) return "usingnamespace";
    return "pub";
}

fn extractName(allocator: std.mem.Allocator, line: []const u8) ![]u8 {
    // naive name extraction: after `pub fn|const|var` read identifier
    var rest: []const u8 = line;
    if (std.mem.startsWith(u8, rest, "pub fn ")) rest = rest[7..] else if (std.mem.startsWith(u8, rest, "pub const ")) rest = rest[10..] else if (std.mem.startsWith(u8, rest, "pub var ")) rest = rest[8..] else if (std.mem.startsWith(u8, rest, "pub usingnamespace ")) rest = rest[18..] else if (std.mem.startsWith(u8, rest, "pub ")) rest = rest[4..];

    // identifier: letters, digits, underscore
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const c = rest[i];
        const is_id = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or (c == '_') or (c == '.');
        if (!is_id) break;
    }
    const ident = std.mem.trim(u8, rest[0..i], " \t");
    if (ident.len == 0) return error.Invalid;
    return allocator.dupe(u8, ident);
}

fn generateSearchIndex(allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Ensure output dir exists
    try std.fs.cwd().makePath("docs/generated");

    // Collect Markdown files in docs/generated
    var dir = std.fs.cwd().openDir("docs/generated", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return, // nothing to index yet
        else => return err,
    };
    defer dir.close();

    var files = std.ArrayListUnmanaged([]const u8){};
    defer files.deinit(a);

    var it = dir.iterate();
    while (it.next() catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".md")) {
            const rel = try std.fs.path.join(a, &[_][]const u8{ "generated", entry.name });
            try files.append(a, rel);
        }
    }

    var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
    defer out.close();

    try out.writeAll("[\n");
    var first = true;

    for (files.items) |rel| {
        const full = try std.fs.path.join(a, &[_][]const u8{ "docs", rel });
        var title_buf: []const u8 = "";
        var excerpt_buf: []const u8 = "";
        getTitleAndExcerpt(a, full, &title_buf, &excerpt_buf) catch {
            // Fallbacks
            title_buf = std.fs.path.basename(rel);
            excerpt_buf = "";
        };

        if (!first) {
            try out.writeAll(",\n");
        } else {
            first = false;
        }

        try out.writeAll("  {\"file\": ");
        try writeJsonString(out, rel);
        try out.writeAll(", \"title\": ");
        try writeJsonString(out, title_buf);
        try out.writeAll(", \"excerpt\": ");
        try writeJsonString(out, excerpt_buf);
        try out.writeAll("}");
    }

    try out.writeAll("\n]\n");
}

fn getTitleAndExcerpt(allocator: std.mem.Allocator, path: []const u8, title_out: *[]const u8, excerpt_out: *[]const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 1024 * 1024 * 4);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var first_heading: ?[]const u8 = null;
    var in_code = false;

    var excerpt = std.ArrayListUnmanaged(u8){};
    defer excerpt.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) continue;

        if (first_heading == null and std.mem.startsWith(u8, trimmed, "#")) {
            var j: usize = 0;
            while (j < trimmed.len and trimmed[j] == '#') j += 1;
            const after = std.mem.trim(u8, trimmed[j..], " \t");
            if (after.len > 0) first_heading = after;
            continue;
        }

        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue; // skip headings in excerpt
        if (trimmed[0] == '|') continue; // skip tables

        // Append to excerpt up to ~300 chars
        if (excerpt.items.len > 0) try excerpt.append(allocator, ' ');
        var k: usize = 0;
        while (k < trimmed.len and excerpt.items.len < 300) : (k += 1) {
            const c = trimmed[k];
            if (c == '`') continue;
            try excerpt.append(allocator, c);
        }
        if (excerpt.items.len >= 300) break;
    }

    if (first_heading) |h| {
        title_out.* = try allocator.dupe(u8, h);
    } else {
        const base = std.fs.path.basename(path);
        title_out.* = try allocator.dupe(u8, base);
    }
    excerpt_out.* = try allocator.dupe(u8, excerpt.items);
}

fn writeJsonString(out: std.fs.File, s: []const u8) !void {
    try out.writeAll("\"");
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        switch (c) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => {
                var buf: [1]u8 = .{c};
                try out.writeAll(buf[0..1]);
            },
        }
    }
    try out.writeAll("\"");
}

fn generateDocsIndexHtml(_: std.mem.Allocator) !void {
    // Write a GitHub Pages friendly index.html that renders docs/generated/*.md client-side
    try std.fs.cwd().makePath("docs");
    var out = try std.fs.cwd().createFile("docs/index.html", .{ .truncate = true });
    defer out.close();

    const html =
        \\<!DOCTYPE html>
        \\<html lang="en">
        \\<head>
        \\  <meta charset="UTF-8" />
        \\  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        \\  <title>ABI Documentation</title>
        \\  <style>
        \\    body {
        \\      margin: 0;
        \\      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        \\      color-scheme: light dark;
        \\      --bg: #0b0d10;
        \\      --fg: #e6edf3;
        \\      --muted: #9aa4b2;
        \\      --accent: #5eb1ff;
        \\      --panel: #111418;
        \\      background: var(--bg);
        \\      color: var(--fg);
        \\      display: grid;
        \\      grid-template-columns: 320px 1fr;
        \\      height: 100dvh;
        \\      overflow: hidden;
        \\    }
        \\
        \\    aside {
        \\      background: var(--panel);
        \\      border-right: 1px solid #1f2328;
        \\      display: flex;
        \\      flex-direction: column;
        \\      padding: 16px;
        \\      overflow: hidden;
        \\    }
        \\
        \\    main {
        \\      overflow: auto;
        \\      padding: 24px 32px;
        \\    }
        \\
        \\    #brand { font-weight: 700; letter-spacing: 0.2px; margin-bottom: 8px; }
        \\    #desc { color: var(--muted); font-size: 13px; margin-bottom: 12px; }
        \\    #search { padding: 10px 12px; border-radius: 8px; border: 1px solid #2a2f36; background: #0f1216; color: var(--fg); outline: none; width: 100%; box-sizing: border-box; }
        \\    #nav { overflow: auto; margin-top: 12px; padding-right: 6px; }
        \\    .nav-item { padding: 10px 8px; border-radius: 8px; }
        \\    .nav-item:hover { background: rgba(94, 177, 255, 0.12); }
        \\    .nav-item a { color: var(--fg); text-decoration: none; font-weight: 600; }
        \\    .nav-excerpt { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.35; }
        \\
        \\    /* Content styling */
        \\    #content { max-width: 1100px; margin: 0 auto; line-height: 1.6; }
        \\    #content h1, #content h2, #content h3 { margin-top: 26px; }
        \\    #content pre { background: #0f1216; padding: 14px; border-radius: 8px; overflow: auto; }
        \\    #content code { background: #0f1216; padding: 2px 4px; border-radius: 4px; }
        \\    #content a { color: var(--accent); }
        \\    #topbar { display: flex; align-items: center; justify-content: space-between; }
        \\    #topbar .right { display: flex; gap: 8px; align-items: center; }
        \\    button.small { background: #0f1216; border: 1px solid #2a2f36; color: var(--fg); border-radius: 8px; padding: 6px 10px; cursor: pointer; }
        \\    button.small:hover { border-color: var(--accent); }
        \\  </style>
        \\</head>
        \\<body>
        \\  <aside>
        \\    <div id="brand">ABI Docs</div>
        \\    <div id="desc">Search and browse documentation generated from the codebase.</div>
        \\    <input id="search" type="search" placeholder="Search docs..." />
        \\    <div id="nav"></div>
        \\  </aside>
        \\  <main>
        \\    <div id="topbar">
        \\      <div></div>
        \\      <div class="right">
        \\        <button class="small" id="open_md">Open in raw Markdown</button>
        \\      </div>
        \\    </div>
        \\    <div id="content"></div>
        \\  </main>
        \\  <script>
        \\  async function fetchJSON(p) { const r = await fetch(p); return await r.json(); }
        \\  async function fetchText(p) { const r = await fetch(p); return await r.text(); }
        \\  function escapeHTML(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
        \\  function mdToHtml(md) {
        \\    const lines = md.split('\\n');
        \\    let html = '';
        \\    let inCode = false; let codeLang = '';
        \\    for (let i=0;i<lines.length;i++){
        \\      let line = lines[i];
        \\      if (line.startsWith('```')) {
        \\        if (!inCode) { inCode = true; codeLang = line.slice(3).trim(); html += '<pre><code class="lang-'+escapeHTML(codeLang)+'">'; }
        \\        else { inCode = false; html += '</code></pre>'; }
        \\        continue;
        \\      }
        \\      if (inCode) { html += escapeHTML(line)+'\\n'; continue; }
        \\      if (line.startsWith('#')) {
        \\        const m = line.match(/^#+/); const level = m ? m[0].length : 1;
        \\        const text = line.slice(level).trim();
        \\        const id = text.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,'');
        \\        html += `<h${level} id="${id}">${inline(text)}</h${level}>`;
        \\        continue;
        \\      }
        \\      if (/^\\s*[-*] /.test(line)) {
        \\        let items = []; let j=i;
        \\        while (j<lines.length && /^\\s*[-*] /.test(lines[j])) { items.push(lines[j].replace(/^\\s*[-*] /,'')); j++; }
        \\        html += '<ul>' + items.map(it => `<li>${inline(it)}</li>`).join('') + '</ul>';
        \\        i = j-1; continue;
        \\      }
        \\      if (/^\\s*[0-9]+\\. /.test(line)) {
        \\        let items=[]; let j=i;
        \\        while (j<lines.length && /^\\s*[0-9]+\\. /.test(lines[j])) { items.push(lines[j].replace(/^\\s*[0-9]+\\. /,'')); j++; }
        \\        html += '<ol>' + items.map(it => `<li>${inline(it)}</li>`).join('') + '</ol>';
        \\        i = j-1; continue;
        \\      }
        \\      if (line.trim() === '') { html += ''; continue; }
        \\      html += `<p>${inline(line)}</p>`;
        \\    }
        \\    return html;
        \\    function inline(t) {
        \\      t = t.replace(/`([^`]+)`/g,'<code>$1</code>');
        \\      t = t.replace(/\\*\\*([^*]+)\\*\\*/g,'<strong>$1</strong>');
        \\      t = t.replace(/\\*([^*]+)\\*/g,'<em>$1</em>');
        \\      t = t.replace(/\\!\\[([^\\]]*)\\]\\(([^)]+)\\)/g,'<img alt="$1" src="$2" />');
        \\      t = t.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
        \\      return t;
        \\    }
        \\  }
        \\  let searchData = [];
        \\  let currentPath = '';
        \\  async function init(){
        \\    try { searchData = await fetchJSON('generated/search_index.json'); } catch(e){ console.error('search index', e); }
        \\    renderNav(searchData);
        \\    const initial = location.hash ? decodeURIComponent(location.hash.slice(1)) : (searchData[0]?.file || 'generated/API_REFERENCE.md');
        \\    loadDoc(initial);
        \\    document.getElementById('search').addEventListener('input', (e) => {
        \\      const q = e.target.value.toLowerCase();
        \\      const results = searchData.filter(it => it.title.toLowerCase().includes(q) || it.excerpt.toLowerCase().includes(q));
        \\      renderNav(results);
        \\    });
        \\    document.getElementById('open_md').addEventListener('click', () => { if (currentPath) window.open(currentPath, '_blank'); });
        \\  }
        \\  async function loadDoc(path){
        \\    currentPath = path;
        \\    location.hash = encodeURIComponent(path);
        \\    const md = await fetchText(path);
        \\    document.getElementById('content').innerHTML = mdToHtml(md);
        \\    window.scrollTo(0,0);
        \\  }
        \\  function renderNav(list){
        \\    const nav = document.getElementById('nav');
        \\    nav.innerHTML = list.map(it => `<div class="nav-item"><a href="#${encodeURIComponent(it.file)}" onclick="loadDoc('${it.file}'); return false;">${it.title}</a><div class="nav-excerpt">${escapeHTML(it.excerpt)}</div></div>`).join('');
        \\  }
        \\  window.addEventListener('hashchange', () => {
        \\    const p = decodeURIComponent(location.hash.slice(1));
        \\    if (p) loadDoc(p);
        \\  });
        \\  init();
        \\  </script>
        \\</body>
        \\</html>
    ;

    try out.writeAll(html);
}
