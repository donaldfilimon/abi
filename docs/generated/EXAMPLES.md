# WDBX-AI Usage Examples

## 🚀 Quick Start

### Basic Vector Database
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize database
    const config = abi.DatabaseConfig{
        .max_vectors = 10000,
        .vector_dimension = 128,
        .enable_caching = true,
    };
    var db = try abi.database.init(allocator, config);
    defer db.deinit();

    // Insert sample vectors
    for (0..100) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }
        const id = try db.insert(&vector, "vector_{}");
        std.log.info("Inserted vector with ID: {}", .{id});
    }

    // Search for similar vectors
    const query = [_]f32{1.0} ** 128;
    const results = try db.search(&query, 5);
    defer allocator.free(results);

    std.log.info("Found {} similar vectors:", .{results.len});
    for (results, 0..) |result, i| {
        std.log.info("  {}: ID={}, Distance={}", .{ i, result.id, result.distance });
    }
}
```

## 🧠 Machine Learning Pipeline

### Neural Network Training
```zig
pub fn trainModel() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create network
    const config = abi.NetworkConfig{
        .input_size = 128,
        .hidden_sizes = &[_]usize{64, 32},
        .output_size = 10,
        .learning_rate = 0.01,
        .batch_size = 32,
    };
    var network = try abi.ai.createNetwork(allocator, config);
    defer network.deinit();

    // Generate training data
    var training_data = std.ArrayList(abi.TrainingData).init(allocator);
    defer training_data.deinit();

    for (0..1000) |i| {
        var input: [128]f32 = undefined;
        var output: [10]f32 = undefined;

        // Generate random input
        for (&input) |*v| {
            v.* = std.rand.DefaultPrng.init(@as(u64, i)).random().float(f32);
        }

        // Generate target output (one-hot encoding)
        @memset(&output, 0);
        output[i % 10] = 1.0;

        try training_data.append(abi.TrainingData{
            .input = &input,
            .output = &output,
        });
    }

    // Train network
    const loss = try network.train(training_data.items);
    std.log.info("Training completed with loss: {}", .{loss});

    // Test prediction
    const test_input = [_]f32{0.5} ** 128;
    const prediction = try network.predict(&test_input);
    defer allocator.free(prediction);

    std.log.info("Prediction: {any}", .{prediction});
}
```

## ⚡ SIMD Operations

### Vector Processing
```zig
pub fn vectorProcessing() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate vectors
    const size = 2048;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize vectors
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2));

    // SIMD operations
    const start_time = std.time.nanoTimestamp();

    abi.simd.add(result, a, b);
    abi.simd.subtract(result, result, a);
    abi.simd.multiply(result, result, b);
    abi.simd.normalize(result, result);

    const end_time = std.time.nanoTimestamp();
    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds

    std.log.info("SIMD operations completed in {}ms", .{duration});
    std.log.info("Result sample: [{}, {}, {}]", .{ result[0], result[1], result[2] });
}
```

## 🔌 Plugin System

### Custom Plugin
```zig
// plugin_example.zig
const std = @import("std");

export fn process_data(input: [*c]const u8, input_len: usize, output: [*c]u8, output_len: *usize) c_int {
    // Process input data
    const input_slice = input[0..input_len];

    // Example: convert to uppercase
    var result = std.ArrayList(u8).init(std.heap.page_allocator);
    defer result.deinit();

    for (input_slice) |byte| {
        if (byte >= 'a' and byte <= 'z') {
            try result.append(byte - 32);
        } else {
            try result.append(byte);
        }
    }

    // Copy result to output
    if (result.items.len > output_len.*) {
        return -1; // Buffer too small
    }

    @memcpy(output[0..result.items.len], result.items);
    output_len.* = result.items.len;
    return 0; // Success
}

// Using the plugin
pub fn usePlugin() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load plugin
    const plugin = try abi.plugins.loadPlugin("plugin_example.zig");
    defer plugin.deinit();

    // Execute plugin function
    const input = "hello world";
    const result = try abi.plugins.executePlugin(plugin, "process_data", input);
    defer allocator.free(result);

    std.log.info("Plugin result: {s}", .{result});
}
```

## 🎯 Performance Optimization

### Batch Operations
```zig
pub fn batchOperations() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var db = try abi.database.init(allocator, abi.DatabaseConfig{});
    defer db.deinit();

    // Batch insert
    const batch_size = 1000;
    var vectors = try allocator.alloc([]f32, batch_size);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }

    // Generate batch data
    for (vectors, 0..) |*vec, i| {
        vec.* = try allocator.alloc(f32, 128);
        for (vec.*, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }
    }

    // Insert batch
    const start_time = std.time.nanoTimestamp();
    for (vectors) |vec| {
        _ = try db.insert(vec, null);
    }
    const end_time = std.time.nanoTimestamp();

    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
    const throughput = @as(f64, @floatFromInt(batch_size)) / (duration / 1000.0);

    std.log.info("Batch insert: {} vectors in {}ms", .{ batch_size, duration });
    std.log.info("Throughput: {} vectors/sec", .{throughput});
}
```

## 🔧 Error Handling

### Comprehensive Error Handling
```zig
pub fn robustOperations() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var db = abi.database.init(allocator, abi.DatabaseConfig{}) catch |err| switch (err) {
        error.OutOfMemory => {
            std.log.err("Failed to allocate memory for database");
            return;
        },
        error.InvalidConfig => {
            std.log.err("Invalid database configuration");
            return;
        },
        else => return err,
    };
    defer db.deinit();

    // Safe vector operations
    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions

    const id = db.insert(&vector, "test") catch |err| switch (err) {
        error.VectorDimensionMismatch => {
            std.log.err("Vector dimension mismatch");
            return;
        },
        error.StorageError => {
            std.log.err("Storage operation failed");
            return;
        },
        else => return err,
    };

    std.log.info("Successfully inserted vector with ID: {}", .{id});
}
```
