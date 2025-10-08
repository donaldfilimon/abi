# ABI Framework

> Experimental Zig framework that provides a bootstrap runtime and a curated set of feature modules for AI experiments.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0-orange.svg)](https://ziglang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Version-0.1.0a-purple.svg)](CHANGELOG.md)

## Project Status

`abi` is not a full-stack product yet. The current executable initializes the framework, emits a textual summary of the configured modules, and exits. The value of the repository lies in the reusable modules under `lib/` that you can import from your own applications.

The `0.1.0a` prerelease focuses on:

- providing consistent imports such as `@import("abi").ai` and `@import("abi").database`
- documenting the bootstrap CLI accurately
- establishing a truthful changelog for the initial prerelease
- capturing the broader modernization roadmap documented in [`REDESIGN_PLAN.md`](REDESIGN_PLAN.md)

## Quick Start

### Prerequisites

- **Zig** `0.16.0` (see `.zigversion` for the authoritative toolchain)
- A C++ compiler for Zig's build dependencies

### Clone and Build

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
zig build test
zig build docs   # generate API docs via tools/docs_generator
zig build tools  # build the aggregated tools CLI (abi-tools)
```

The default build produces `zig-out/bin/abi`. This executable implements a modern sub-command based CLI. Use `abi --help` to view available commands and `abi <subcommand> --help` for detailed usage.

```bash
# Show help (lists all sub-commands)
./zig-out/bin/abi help

# Run the benchmark suite
zig build bench -- --format=markdown --output=results

# Run developer tools entrypoint
zig build tools -- --help

# Example: list enabled features in JSON mode
./zig-out/bin/abi features list --json

# Build and run the tools CLI (aggregates utilities under src/tools)
zig build tools
./zig-out/bin/abi-tools --help

# Or run directly through the build system
zig build tools-run -- --help
```

Sample output:

```
ABI Framework bootstrap complete
• Features: ai, database, gpu, monitoring, web, connectors
• Plugins: discovery disabled (configure via abi.framework)
```

### Using the Library from Zig

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), .{});
    defer abi.shutdown(&framework);

    // Load the lightweight agent prototype.
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(gpa.allocator(), .{ .name = "QuickStart" });
    defer agent.deinit();

    const reply = try agent.process("Hello", gpa.allocator());
    defer gpa.allocator().free(@constCast(reply));
}
```

## Architecture

The framework is organized into clear, modular components:

```
abi/
├── lib/                    # Core library code
│   ├── core/              # Fundamental types and utilities
│   ├── features/          # Feature modules (ai, gpu, database, etc.)
│   ├── framework/         # Framework orchestration
│   ├── shared/           # Shared utilities
│   └── mod.zig           # Main library entry point
├── bin/                  # Executable entry points
├── examples/             # Usage examples
├── tests/               # Test suite
├── tools/               # Development and build tools
├── docs/               # Documentation
└── config/             # Configuration files
```

### Feature Modules

The top-level module re-exports the major feature namespaces for convenience:

- `abi.ai` – experimental agents and model helpers
- `abi.database` – WDBX vector database components and HTTP/CLI front-ends
- `abi.gpu` – GPU utilities (currently CPU-backed stubs)
- `abi.web` – minimal HTTP scaffolding used by the WDBX demo
- `abi.monitoring` – logging and metrics helpers shared across modules
- `abi.connectors` – placeholder for third-party integrations
- `abi.wdbx` – compatibility namespace exposing the database module and helpers
- `abi.VectorOps` – SIMD helpers re-exported from `abi.simd`

## Development

### Build System

The project uses a modern Zig build system with multiple targets:

```bash
# Build everything
zig build

# Build in release mode
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Run integration tests
zig build test-integration

# Run benchmarks
zig build bench

# Generate documentation
zig build docs

# Format code
zig build fmt

# Run linter
zig build lint

# Clean build artifacts
zig build clean
```

### Development Tools

```bash
# Setup development environment
./tools/dev/setup.sh

# Build with options
./tools/build/build.sh --release --test --docs

# Deploy to different environments
./tools/deploy/deploy.sh --environment production --package --docker
```

### Examples

```bash
# Basic usage example
zig run examples/basic-usage.zig

# Advanced features example
zig run examples/advanced-features.zig
```

## Documentation

- **[Getting Started Guide](docs/guides/getting-started.md)** – Quick start guide
- **[API Reference](docs/api/)** – Generated API documentation
- **[Examples](examples/)** – Working code examples
- **[Development Guide](docs/guides/development.md)** – Development workflow

## CLI Usage

The ABI CLI provides comprehensive command-line access to framework features:

```bash
# Framework management
abi framework status
abi framework start
abi framework stop

# Feature management
abi features list
abi features enable gpu monitoring
abi features disable ai

# AI operations (when enabled)
abi ai run --name "MyAgent" --message "Hello"

# Database operations (when enabled)
abi database insert --vec "1.0,2.0,3.0" --meta "test"
abi database search --vec "1.0,2.0,3.0" -k 5

# Output in JSON format
abi features list --json
abi framework status --json
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues and proposing changes.

### Development Workflow

- Format code with `zig fmt .`
- Run the full test suite with `zig build test`
- Use `zig build run` to execute the bootstrap binary under the debug configuration
- Check the [development guide](docs/guides/development.md) for detailed workflow

## License

MIT License – see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Roadmap

The redesign introduces a cleaner, more modular architecture that:

- ✅ **Separates concerns** – Clear distinction between library, executables, examples, and tools
- ✅ **Modernizes build system** – Improved build.zig with multiple targets and better caching
- ✅ **Streamlines documentation** – Organized docs structure with generated API references
- ✅ **Consolidates tools** – Organized scripts and tools into logical groups
- ✅ **Improves configuration** – Environment-specific configuration management
- 🔄 **Enhances testing** – Comprehensive test suite with unit, integration, and benchmark tests
- 🔄 **Expands examples** – More comprehensive examples demonstrating framework capabilities