# Abi AI Framework
> Modular Zig framework for composing ABI feature sets and managing plugin lifecycles.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev.254%2B6dd0270a1-orange.svg)](https://ziglang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Cross--platform-green.svg)]()

## 🚀 Features

### Core Capabilities
- **Framework bootstrap**: The `abi` executable initialises the framework and prints a runtime summary.
- **Feature toggles**: `abi.Framework` exposes bitset-backed toggles for major feature groups.
- **Plugin registry**: Shared plugin registry helpers handle discovery, loading and lifecycle calls.
- **Module re-exports**: `@import("abi")` surfaces organised submodules under `abi.features`, `abi.framework`, and `abi.shared`.

### Platform Support
- **Cross-platform**: Builds on Linux, macOS, and Windows with the Zig toolchain.
- **CLI-first**: Ships as a bootstrap executable; additional services are built on top of the API surface.

## Installation

### Prerequisites
- **Zig 0.16.0-dev.254+6dd0270a1** (see `.zigversion` and verify with `zig version`)
- Optional GPU drivers if you plan to explore the GPU feature modules.

### Compatibility

- **Current Toolchain**: Zig 0.16.0-dev.254+6dd0270a1. This repository's `.zigversion` file is the authoritative reference—match it exactly when installing Zig.
- **Legacy Notes**: Earlier Zig 0.15.x instructions are available in the Git history for teams maintaining older deployments.

### Quick Start
```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build -Doptimize=ReleaseFast
zig build test
zig build run
```

## 📁 Project Structure

```
abi/
├── src/
│   ├── features/      # Feature families re-exported under `abi.features`
│   ├── framework/     # Framework configuration, runtime, and feature toggles
│   ├── shared/        # Common utilities (logging, platform helpers, registry)
│   ├── examples/      # Library usage snippets
│   ├── tools/         # Internal utilities and maintenance commands
│   ├── main.zig       # Bootstrap executable entry point
│   └── mod.zig        # Public module surface
├── tests/             # Additional top-level tests
├── docs/              # Project documentation
├── benchmarks/        # Benchmark harnesses
└── scripts/           # Development scripts

## Quick Start

### Basic Usage
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var framework = try abi.init(allocator, .{});
    defer framework.deinit();

    // Write the bootstrap summary to stdout
    try framework.writeSummary(std.io.getStdOut().writer());
}
```

> `abi.init` allocates internal bookkeeping structures; always pair it with `Framework.deinit` (or `abi.shutdown`).

### Vector Database
```zig
const std = @import("std");
const abi = @import("abi");

// Access the database feature module through the re-exported API.
const Database = abi.features.database;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var db = try Database.database.Db.open("vectors.wdbx", true);
    defer db.close();

    const embedding = [_]f32{ 0.1, 0.2, 0.3 };
    try db.init(embedding.len);

    const results = try db.search(&embedding, 10, gpa.allocator());
    defer gpa.allocator().free(results);
}
```

## Modules

- **`abi.framework`**: High-level configuration, runtime orchestration, and feature toggles.
- **`abi.features.ai`**: AI-focused modules including agent pipelines and neural helpers.
- **`abi.features.database`**: WDBX storage engine, search helpers, and CLI bindings.
- **`abi.features.web`**: HTTP scaffolding and routing utilities exposed via the framework.
- **`abi.features.monitoring`**: Telemetry and instrumentation primitives.
- **`abi.features.gpu`**: GPU-oriented abstractions and compute helpers.
- **`abi.features.connectors`**: Integrations for external services and adapters.
- **`abi.shared`**: Common logging, platform, and plugin registry utilities.

## CLI

The bundled executable initialises the framework and prints a runtime summary. It is
primarily a bootstrap tool for validating feature toggles and plugin discovery logic.

```bash
# Build and run the bootstrap executable
zig build run

# Or invoke the binary directly after building
./zig-out/bin/abi
```

Sample output:

```
ABI Framework bootstrap complete
Features enabled (5):
  - AI/Agents: Conversation agents, training loops, and inference helpers
  - Vector Database: High-performance embedding and vector persistence layer
  - Web Services: HTTP servers, clients, and gateway orchestration
  - Monitoring: Instrumentation, telemetry, and health checks
  - SIMD Runtime: Runtime SIMD utilities and fast math operations
Plugin search paths: none configured
Registered plugins: 0
Discovered plugins awaiting load: 0
```

Use the executable as a reference implementation for invoking `abi.init` and interacting with
`abi.Framework`; custom applications are expected to build on the library API directly.

## API Surface

The `abi` module provides a curated export of framework primitives:

- `abi.init(allocator, options)` and `abi.shutdown` manage the primary `Framework` lifecycle.
- `abi.Framework` exposes feature toggles, plugin path helpers (`setPluginPaths`, `refreshPlugins`),
  and summary writers (`writeSummary`).
- `abi.framework` re-exports the full framework configuration and runtime modules when you need
  lower-level access.
- `abi.features.*` namespaces organise feature code such as `abi.features.ai`,
  `abi.features.database`, and `abi.features.monitoring`.

Refer to `src/mod.zig` for the complete list of re-exports and `src/framework/runtime.zig` for
runtime semantics.

## Testing

```bash
# Run all tests registered in build.zig (currently exercises src/main.zig)
zig build test

# Targeted test files
zig test tests/test_create.zig
zig test tests/cross-platform/linux.zig
zig test tests/cross-platform/macos.zig
zig test tests/cross-platform/windows.zig
```

The cross-platform tests gracefully skip when run on unsupported operating systems, so it's
safe to invoke the full list from any development environment.

**Quality Goals:**
- Maintain leak-free operation under `zig build test`.
- Keep feature and plugin API changes documented in `CHANGELOG.md`.
- Ensure new modules include unit tests or examples illustrating usage.

## Contributing

1. Fork and create feature branch
2. Run tests: `zig build test`
3. Check memory: `zig build test-memory`
4. Submit PR with tests

## License

MIT License - see [LICENSE](LICENSE)

## Links

- [Documentation](docs/)
- [Issues](https://github.com/donaldfilimon/abi/issues)
