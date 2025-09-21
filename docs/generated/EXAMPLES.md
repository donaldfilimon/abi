---
layout: documentation
title: "Examples & Tutorials"
description: "Practical examples and tutorials for using ABI effectively"
---

# ABI Usage Examples

## 🚀 Quick Start

### Basic Vector Database
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const database = abi.features.database;
    var db = try database.database.Db.open("vectors.wdbx", true);
    defer db.close();

    // Initialise storage
    try db.init(128);

    // Insert sample vectors
    for (0..100) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }
        const id = try db.addEmbedding(&vector);
        std.log.info("Stored vector with ID: {}", .{id});
    }

    // Search for similar vectors
    const query = [_]f32{1.0} ** 128;
    const results = try db.search(&query, 5, allocator);
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
pub fn buildModel() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const ai = abi.features.ai;
    var network = try ai.neural.NeuralNetwork.init(allocator, .{
        .learning_rate = 0.01,
        .batch_size = 32,
    });
    defer network.deinit();

    try network.addLayer(.{
        .type = .Dense,
        .input_size = 128,
        .output_size = 64,
        .activation = .ReLU,
    });
    try network.addLayer(.{
        .type = .Dense,
        .input_size = 64,
        .output_size = 10,
    });

    const sample = [_]f32{0.5} ** 128;
    const logits = try network.forward(&sample);
    defer allocator.free(logits);

    std.log.info("Forward pass complete with {} outputs", .{logits.len});
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
    var result = std.array_list.Managed(u8).init(std.heap.page_allocator);
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
    var framework = try abi.init(std.heap.page_allocator, .{
        .plugin_paths = &.{ "./plugins" },
        .auto_discover_plugins = true,
        .auto_register_plugins = true,
    });
    defer framework.deinit();

    try framework.refreshPlugins();

    const registry = framework.pluginRegistry();
    std.log.info("Loaded plugins: {d}", .{registry.getPluginCount()});
}
```

## 🎯 Performance Optimization

### Batch Operations
```zig
pub fn batchOperations() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const database = abi.features.database;
    var db = try database.database.Db.open("vectors.wdbx", true);
    defer db.close();
    try db.init(128);

    const batch_size = 512;
    var vectors = try allocator.alloc([128]f32, batch_size);
    defer allocator.free(vectors);

    for (vectors, 0..) |*vec, i| {
        for (vec.*, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }
    }

    const start_time = std.time.nanoTimestamp();
    for (vectors) |vec| {
        _ = try db.addEmbedding(&vec);
    }
    const end_time = std.time.nanoTimestamp();

    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const throughput = @as(f64, @floatFromInt(batch_size)) / (duration_ms / 1000.0);

    std.log.info("Batch insert: {} vectors in {d:.2}ms", .{ batch_size, duration_ms });
    std.log.info("Throughput: {d:.2} vectors/sec", .{throughput});
}
```

## 🔧 Error Handling

### Comprehensive Error Handling
```zig
pub fn robustOperations() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const database = abi.features.database;
    var db = database.database.Db.open("vectors.wdbx", true) catch |err| switch (err) {
        error.OutOfMemory => {
            std.log.err("Failed to allocate memory for database");
            return;
        },
        else => return err,
    };
    defer db.close();

    db.init(128) catch |err| switch (err) {
        database.database.Db.DbError.DimensionMismatch => {
            std.log.err("Invalid dimensionality");
            return;
        },
        else => return err,
    };

    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions

    const id = db.addEmbedding(&vector) catch |err| switch (err) {
        database.database.Db.DbError.DimensionMismatch => {
            std.log.err("Vector dimension mismatch");
            return;
        },
        else => return err,
    };

    std.log.info("Successfully inserted vector with ID: {}", .{id});
}
```
