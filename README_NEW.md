# Abi Framework

> **Modern, modular Zig framework for AI/ML experiments and production workloads**

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange.svg)](https://ziglang.org/builds/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-purple.svg)](CHANGELOG.md)

## 🎯 What is Abi?

Abi is an experimental framework that provides a curated set of feature modules for building high-performance AI/ML applications in Zig. It emphasizes:

- **🚀 Performance**: Zero-cost abstractions, SIMD optimizations, and minimal overhead
- **🔧 Modularity**: Composable features with compile-time selection
- **🛡️ Type Safety**: Leveraging Zig's compile-time guarantees
- **🧪 Testability**: Built with testing in mind from the ground up
- **📊 Observability**: Comprehensive monitoring and diagnostics

## ✨ Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **AI/ML** | Agent system, neural networks, transformers, RL | ✅ Production |
| **Vector Database** | High-performance vector storage and retrieval | ✅ Production |
| **GPU Acceleration** | Multi-backend GPU compute (CUDA, Vulkan, Metal) | 🔄 In Progress |
| **Web Server** | HTTP server and client | ✅ Production |
| **Monitoring** | Metrics, logging, and distributed tracing | ✅ Production |
| **Plugin System** | Dynamic plugin loading and management | 🔄 In Progress |

### New in 0.2.0

- ✅ **Modular Build System** - Feature flags for conditional compilation
- ✅ **I/O Abstraction Layer** - Testable, composable I/O operations
- ✅ **Comprehensive Error Handling** - Rich error context and diagnostics
- ✅ **Improved Testing** - Separate unit and integration test suites
- ✅ **Better Documentation** - Architecture guides and API references

## 🚀 Quick Start

### Prerequisites

- **Zig** `0.16.0-dev.254+6dd0270a1` or later
- A C++ compiler (for some dependencies)

### Installation

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
```

### Basic Usage

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the framework
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Create an AI agent
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(allocator, .{ .name = "Assistant" });
    defer agent.deinit();

    // Process a query
    const response = try agent.process("Hello, world!", allocator);
    defer allocator.free(@constCast(response));

    std.debug.print("Agent response: {s}\n", .{response});
}
```

### Building with Feature Flags

```bash
# Build with specific features
zig build -Denable-ai=true -Denable-gpu=true -Dgpu-cuda=true

# Build and run tests
zig build test              # Unit tests
zig build test-integration  # Integration tests
zig build test-all          # All tests

# Build examples
zig build examples          # All examples
zig build run-ai_demo       # Run specific example

# Build benchmarks
zig build bench
zig build run-bench

# Generate documentation
zig build docs
zig build docs-auto
```

## 📖 Documentation

### User Guides

- **[Getting Started](docs/guides/GETTING_STARTED.md)** - Your first Abi application
- **[Architecture](docs/ARCHITECTURE.md)** - System design and principles
- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples](examples/)** - Practical code examples

### Development

- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Redesign Plan](REDESIGN_PLAN.md)** - Framework redesign details
- **[Redesign Summary](docs/REDESIGN_SUMMARY.md)** - What's new in 0.2.0
- **[Modernization Status](MODERNIZATION_STATUS.md)** - Current progress

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│          Application Layer                   │
│        (CLI, User Code, Tools)              │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│          Framework Layer                     │
│    Runtime · Features · Plugins             │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│          Core Infrastructure                 │
│    I/O · Errors · Diagnostics · Types       │
└─────────────────────────────────────────────┘
```

### Module Organization

```
lib/
├── core/              # Core infrastructure
│   ├── io.zig         # I/O abstractions
│   ├── errors.zig     # Error definitions
│   ├── diagnostics.zig # Diagnostics system
│   └── ...
├── features/          # Feature modules
│   ├── ai/            # AI/ML capabilities
│   ├── database/      # Vector database
│   ├── gpu/           # GPU acceleration
│   └── ...
└── framework/         # Framework runtime
    ├── runtime.zig    # Lifecycle management
    └── ...
```

## 🔧 CLI Usage

The Abi CLI provides comprehensive access to all framework features:

```bash
# Show help
./zig-out/bin/abi --help

# Feature management
./zig-out/bin/abi features list
./zig-out/bin/abi features status

# AI operations
./zig-out/bin/abi agent run --name "MyAgent"
./zig-out/bin/abi agent list

# Database operations
./zig-out/bin/abi db create --name vectors
./zig-out/bin/abi db query --vector "..."

# GPU benchmarks
./zig-out/bin/abi gpu bench
./zig-out/bin/abi gpu info

# Version information
./zig-out/bin/abi version
```

## 🧪 Testing

### Running Tests

```bash
# Unit tests
zig build test

# Integration tests
zig build test-integration

# All tests
zig build test-all

# With coverage
zig build test -- --coverage
```

### Test Organization

```
tests/
├── unit/              # Unit tests (mirrors lib/)
├── integration/       # Integration tests
│   ├── ai_pipeline_test.zig
│   ├── database_ops_test.zig
│   └── framework_lifecycle_test.zig
└── fixtures/          # Test utilities
```

### Writing Tests

```zig
const std = @import("std");
const abi = @import("abi");
const testing = std.testing;

test "AI agent processes input correctly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);
    
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(allocator, .{ .name = "Test" });
    defer agent.deinit();
    
    const response = try agent.process("test", allocator);
    defer allocator.free(@constCast(response));
    
    try testing.expect(response.len > 0);
}
```

## 📊 Examples

### AI Agent

```zig
const abi = @import("abi");

var agent = try abi.ai.agent.Agent.init(allocator, .{
    .name = "Assistant",
    .max_retries = 3,
});
defer agent.deinit();

const response = try agent.process("Explain quantum computing", allocator);
defer allocator.free(@constCast(response));
```

### Vector Database

```zig
const db = abi.database;

var vector_db = try db.VectorDB.init(allocator, .{
    .dimension = 128,
    .metric = .cosine,
});
defer vector_db.deinit();

try vector_db.insert("doc1", embedding);
const results = try vector_db.search(query, 10);
```

### GPU Compute

```zig
const gpu = abi.gpu;

var backend = try gpu.selectBackend(allocator);
defer backend.deinit();

const kernel = try gpu.loadKernel("matrix_mul");
try backend.execute(kernel, .{ .a = a, .b = b, .result = result });
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/donaldfilimon/abi.git
cd abi

# Run tests
zig build test-all

# Format code
zig fmt .

# Build all examples
zig build examples
```

### Code Guidelines

- Follow Zig 0.16 best practices
- Add tests for new features
- Update documentation
- Use the provided error handling infrastructure
- Inject dependencies (especially I/O)

## 🗺️ Roadmap

### Current (v0.2.0)

- [x] Modular build system
- [x] I/O abstraction layer
- [x] Comprehensive error handling
- [x] Improved testing infrastructure

### Next (v0.3.0)

- [ ] Complete GPU backend implementations
- [ ] Advanced monitoring and tracing
- [ ] Plugin system v2
- [ ] Performance optimizations

### Future

- [ ] Distributed computing support
- [ ] Advanced ML model formats
- [ ] Production deployment guides
- [ ] Cloud provider integrations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Zig team for creating an amazing language
- All contributors to this project
- The AI/ML and systems programming communities

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/donaldfilimon/abi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/donaldfilimon/abi/discussions)
- **Documentation**: [docs/](docs/)

---

**Built with ❤️ using Zig 0.16**

*Last Updated: October 8, 2025*
