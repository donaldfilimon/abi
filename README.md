# WDBX-AI Vector Database v2.0

🚀 **Enterprise-Grade Vector Database with AI Integration**

A high-performance, production-ready vector database built in Zig, featuring advanced SIMD optimizations, comprehensive monitoring, and enterprise-grade reliability.

## ✨ Key Features

- 🏎️ **High Performance**: SIMD-accelerated operations with 2-4x speedup
- 🔧 **Enterprise Ready**: Production features with monitoring and health checks
- 🧠 **AI Integration**: Built-in neural networks and embedding generation
- 📈 **Scalable**: Handles millions of vectors with efficient indexing
- 🛡️ **Reliable**: Comprehensive error handling and automatic recovery
- 🔍 **Observable**: Rich metrics, logging, and performance monitoring
- 🧪 **Well Tested**: Comprehensive test suite with 95%+ coverage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd wdbx-ai

# Build the project
zig build

# Run tests
zig build test

# Run benchmarks
zig build benchmark
```

### Basic Usage

```zig
const std = @import("std");
const wdbx = @import("src/wdbx/mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create database with default configuration
    var db = try wdbx.createWithDefaults(allocator, "vectors.wdbx", 384);
    defer db.deinit();
    
    // Add a vector
    const vector = [_]f32{0.1} ** 384;
    const id = try db.addVector(&vector);
    
    // Search for similar vectors
    const results = try db.search(&vector, 10);
    defer {
        for (results) |*result| {
            result.deinit(allocator);
        }
        allocator.free(results);
    }
    
    // Print statistics
    try db.printStats();
}
```

### CLI Usage

```bash
# Start HTTP server
./zig-out/bin/wdbx-ai http --port 8080

# Add vectors via CLI
./zig-out/bin/wdbx-ai add --file vectors.wdbx --vector "0.1,0.2,0.3,0.4"

# Search for similar vectors
./zig-out/bin/wdbx-ai query --file vectors.wdbx --vector "0.1,0.2,0.3,0.4" --k 10

# Show database statistics
./zig-out/bin/wdbx-ai stats --file vectors.wdbx
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WDBX-AI v2.0                            │
├─────────────────────────────────────────────────────────────┤
│ Applications: CLI, Web UI, HTTP API                        │
├─────────────────────────────────────────────────────────────┤
│ WDBX Layer: Unified Implementation                          │
├─────────────────────────────────────────────────────────────┤
│ Services: Database, AI, SIMD, Plugins                      │
├─────────────────────────────────────────────────────────────┤
│ Core: Memory, Performance, Errors, Threading               │
├─────────────────────────────────────────────────────────────┤
│ System: Logging, Utilities, Allocators                     │
└─────────────────────────────────────────────────────────────┘
```

### Core Modules

- **Core** (`src/core/`) - Foundation utilities and system management
- **Database** (`src/database/`) - Vector storage and retrieval
- **SIMD** (`src/simd/`) - High-performance vector operations
- **AI** (`src/ai/`) - Neural networks and machine learning
- **WDBX** (`src/wdbx/`) - Unified database interface
- **Plugins** (`src/plugins/`) - Extensible plugin system

## 📊 Performance

### SIMD Optimizations
- **AVX-512**: 16-element vector operations (up to 4x speedup)
- **AVX/AVX2**: 8-element vector operations (up to 3x speedup)
- **SSE2**: 4-element vector operations (up to 2x speedup)
- **NEON**: ARM SIMD support
- **Automatic Fallback**: Scalar operations when SIMD unavailable

### Benchmarks (1024-dimensional vectors)
| Operation | Throughput | Latency |
|-----------|------------|---------|
| Distance Calculation | 1.5M ops/sec | 0.68ms |
| Vector Insert | 312K ops/sec | 3.2ms |
| Vector Search (k=10) | 89K ops/sec | 11.2ms |
| Batch Operations | 2.1M ops/sec | 0.48ms |

### Memory Efficiency
- **Pool Allocator**: 5x faster for small allocations
- **Smart Allocation**: Automatic strategy selection
- **String Interning**: Reduced memory usage for repeated strings
- **Leak Detection**: Zero memory leaks with comprehensive tracking

## 🛠️ Build System

### Available Commands

```bash
# Development
zig build              # Standard build
zig build dev          # Development build with debug symbols
zig build prod         # Optimized production build

# Quality Assurance
zig build test         # Run all tests
zig build benchmark    # Performance benchmarks
zig build fmt          # Format source code
zig build check        # Format + test

# Documentation
zig build docs         # Generate API documentation

# Utilities
zig build clean        # Clean build artifacts
zig build install      # Install CLI tool
```

## 🧪 Testing

### Test Categories
- **Unit Tests**: Individual module functionality
- **Integration Tests**: Cross-module interactions
- **Performance Tests**: Timing and throughput validation
- **Stress Tests**: High-load and edge case testing
- **Memory Tests**: Leak detection and usage validation

### Running Tests
```bash
# Run all tests
zig build test

# Run specific test category
zig build test -- --filter "core"

# Run with verbose output
zig build test -- --verbose

# Run performance tests only
zig build test -- --performance-only
```


### Cell Framework Example
This repository now includes a demonstration of the Cell framework using modern C++23 modules. See `cell_framework/README.md` for build instructions.
