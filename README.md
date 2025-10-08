# ABI Framework

> Modern, modular Zig framework for AI/ML experiments and production workloads

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange.svg)](https://ziglang.org/builds/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-purple.svg)](CHANGELOG.md)

## 🎯 What is ABI?

ABI is an experimental framework that provides a curated set of feature modules for building high-performance AI/ML applications in Zig. It emphasizes:

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

## 🔧 CLI Usage

The ABI CLI provides comprehensive access to all framework features:

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
src/
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

### Feature Modules

The framework provides these major feature namespaces:

- `abi.ai` – AI agents, neural networks, and ML utilities
- `abi.database` – High-performance vector database with HTTP/CLI interfaces
- `abi.gpu` – Multi-backend GPU acceleration (CUDA, Vulkan, Metal)
- `abi.web` – HTTP server and client components
- `abi.monitoring` – Metrics, logging, and distributed tracing
- `abi.connectors` – Third-party integrations and adapters
- `abi.wdbx` – Compatibility namespace for database operations

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
├── unit/              # Unit tests (mirrors src/)
├── integration/       # Integration tests
│   ├── ai_pipeline_test.zig
│   ├── database_ops_test.zig
│   └── framework_lifecycle_test.zig
└── fixtures/          # Test utilities
```

## 📚 Documentation

### User Guides

- **[Getting Started](docs/guides/GETTING_STARTED.md)** - Your first ABI application
- **[Architecture](docs/ARCHITECTURE.md)** - System design and principles
- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples](examples/)** - Practical code examples

### Development

- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Redesign Plan](REDESIGN_PLAN.md)** - Framework redesign details
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade instructions

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