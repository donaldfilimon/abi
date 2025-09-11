---
layout: default
title: Getting Started with ABI
---

# Getting Started with ABI

This guide will help you get started with the ABI framework, from installation to building your first AI application.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **CPU**: x86_64, ARM64, or RISC-V processor

### Software Dependencies

1. **Zig Compiler**: Version 0.15.1 or later
2. **Git**: For cloning the repository
3. **Vulkan SDK** (optional, for GPU acceleration):
   - Windows: Download from [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
   - Linux: `sudo apt install libvulkan-dev vulkan-tools`
   - macOS: `brew install molten-vk vulkan-headers`

## Installation

### Option 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/your-username/abi.git
cd abi

# Build the project
zig build

# Run tests to verify installation
zig build test
```

### Option 2: Download Release

Download the latest release from [GitHub Releases](https://github.com/your-username/abi/releases) and extract the archive.

## Your First ABI Application

Create a new Zig file called `hello_abi.zig`:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple vector database
    var db = try abi.Db.init(allocator, .{
        .dimensions = 3,
        .max_elements = 1000,
    });
    defer db.deinit();

    std.log.info("ABI Vector Database initialized!", .{});

    // Create some sample vectors
    const vectors = [_][3]f32{
        [_]f32{1.0, 0.0, 0.0},  // Red
        [_]f32{0.0, 1.0, 0.0},  // Green
        [_]f32{0.0, 0.0, 1.0},  // Blue
    };

    // Insert vectors into the database
    for (vectors, 0..) |vec, i| {
        const id = try std.fmt.allocPrint(allocator, "color_{}", .{i});
        defer allocator.free(id);
        try db.insert(&vec, id);
    }

    std.log.info("Inserted {} vectors", .{vectors.len});

    // Search for similar vectors
    const query = [_]f32{0.9, 0.1, 0.0};  // Reddish color
    const results = try db.search(&query, 2);

    std.log.info("Search results for reddish color:", .{});
    for (results.items) |result| {
        std.log.info("  ID: {s}, Distance: {d:.3}", .{
            result.id,
            result.distance
        });
    }
}
```

### Building and Running

```bash
# Build the application
zig build-exe hello_abi.zig

# Run it
./hello_abi
```

You should see output similar to:
```
info: ABI Vector Database initialized!
info: Inserted 3 vectors
info: Search results for reddish color:
info:   ID: color_0, Distance: 0.100
info:   ID: color_1, Distance: 1.345
```

## Understanding the Basics

### Vector Databases

ABI uses vector databases to store and search high-dimensional vectors efficiently. Key concepts:

- **Dimensions**: The number of features in each vector (e.g., 128 for text embeddings)
- **Similarity Search**: Finding vectors closest to a query vector
- **HNSW Algorithm**: Hierarchical Navigable Small World for fast approximate nearest neighbor search

### Memory Management

ABI uses Zig's allocator system for efficient memory management:

```zig
// General Purpose Allocator (good for development)
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Arena Allocator (good for short-lived allocations)
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const temp_allocator = arena.allocator();
```

### Error Handling

ABI follows Zig's error handling patterns:

```zig
const result = db.insert(&vector, "my_id") catch |err| {
    std.log.err("Failed to insert vector: {}", .{err});
    return err;
};
```

## Next Steps

### Explore More Features

1. **AI Components**: Try the neural network and embedding features
2. **GPU Acceleration**: Enable Vulkan for GPU-accelerated operations
3. **Performance Monitoring**: Use the built-in profiling tools
4. **Plugin System**: Extend ABI with custom functionality

### Learning Resources

- [API Reference](/api/) - Complete API documentation
- [Examples](/examples/) - Code samples and tutorials
- [Performance Guide](/performance/) - Optimization tips and benchmarks
- [Contributing Guide](/contributing/) - How to contribute to the project

### Example Applications

- **Recommendation System**: Build product recommendations using vector similarity
- **Image Search**: Search images by content using visual embeddings
- **Text Classification**: Classify text using transformer embeddings
- **Anomaly Detection**: Detect outliers in high-dimensional data

## Troubleshooting

### Common Issues

**Build fails with Vulkan errors:**
- Install Vulkan SDK for your platform
- Ensure Vulkan libraries are in your system PATH

**Out of memory errors:**
- Increase available RAM
- Use smaller batch sizes for operations
- Enable memory pooling in configuration

**Slow performance:**
- Enable SIMD optimizations: `zig build -Doptimize=ReleaseFast`
- Use GPU acceleration if available
- Profile your application: `zig build perf-ci`

### Getting Help

- [GitHub Issues](https://github.com/your-username/abi/issues) - Report bugs and request features
- [Discussions](https://github.com/your-username/abi/discussions) - Ask questions and get help
- [Documentation](/docs/) - Browse the full documentation

---

Ready to build something amazing with ABI? Check out our [examples](/examples/) for more inspiration!
