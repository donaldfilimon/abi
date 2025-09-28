# Abi AI Framework
> Ultra-high-performance AI framework with GPU acceleration, lock-free concurrency, and platform-optimized implementations.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange.svg)](https://ziglang.org/builds/)
[![Docs](https://img.shields.io/badge/Docs-Latest-blue.svg)](https://donaldfilimon.github.io/abi/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Cross--platform-green.svg)]()

Abi couples a high-throughput runtime, GPU-accelerated AI primitives, and a production-ready vector database into a cohesive
toolkit for building latency-sensitive agents, analytics pipelines, and services that must scale from embedded targets to
cloud fleets.

---

## Table of Contents
- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Usage Examples](#usage-examples)
- [Vector Database](#vector-database)
- [Build & Configuration](#build--configuration)
- [Performance & Benchmarking](#performance--benchmarking)
- [Testing](#testing)
- [Documentation & Resources](#documentation--resources)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Abi is organized into three collaborating planes:
- **Runtime Plane** – Schedules workloads, manages lock-free concurrency primitives, and bridges CPU/GPU execution paths.
- **AI & Data Plane** – Hosts persona-driven agents, neural network training, and the WDBX vector database for embeddings.
- **Operations Plane** – Exposes CLI tooling, HTTP/WebSocket services, and observability pipelines for production readiness.

Each plane is implemented in Zig modules under `src/` and can be embedded into existing Zig applications or driven through the
bundled CLI.

---

## Key Capabilities
### Core Framework
- GPU acceleration via WebGPU with platform-specific fallbacks.
- SIMD-optimized text processing (3 GB/s+ throughput) and neural network primitives.
- Lock-free queues, wait-free messaging, and zero-copy allocators for low-latency pipelines.
- Production-grade HTTP/TCP servers with adaptive load-balancing and fault tolerance.

### AI & Tooling
- Multi-persona chat agents with OpenAI integration and persona presets.
- Complete neural network training stack (FFN, CNN, RNN, Transformer) supporting distributed execution modes.
- Plugin system for cross-platform dynamic loading (`.dll`/`.so`/`.dylib`) with type-safe Zig wrappers.
- Observability hooks: Prometheus metrics, performance profiling, memory tracking, and debug tooling.

### Platform Support
- Cross-platform support for Linux, macOS, Windows, and embedded targets.
- Deployment recipes covering Docker, Kubernetes, and standalone binaries.
- Legacy Zig 0.15.x notes are retained for teams maintaining older deployments.

---

## Installation
### Prerequisites
- **Zig 0.16.0-dev (master)** (see `.zigversion` to match the toolchain).
- GPU drivers (optional, required for CUDA/WebGPU backends).
- OpenAI API key (for hosted LLM/persona integrations).

### Clone & Build
```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build -Doptimize=ReleaseFast
```

### Compatibility Notes
- `.zigversion` pins the supported compiler. Align CI and local toolchains with that version.
- Historical instructions for Zig 0.16-dev experiments remain in the migration playbook for future workstreams.

---

## Quick Start
```bash
# Run tests registered in build.zig
zig build test

# Launch the default executable
zig build run

# Ensure formatting is up to date
zig build fmt

# Inspect the CLI helper
./zig-out/bin/abi --help

# Generate docs and benchmarks
zig build docs
zig build bench-all
```

Additional targets:
- `zig build bench-simd` – SIMD micro-benchmarks.
- `zig build cross-platform` – Build the supported target matrix.
- `zig build -Denable_heavy_tests=true test` – Include long-running suites.
- `zig build summary` – Format, document, and test in one step.

---

## Project Layout
```text
abi/
├── src/                # Core framework, AI modules, GPU backends
├── tests/              # Unit, integration, and platform-specific suites
├── docs/               # Manual + generated documentation portal
├── examples/           # Minimal runnable samples
├── tools/              # Developer utilities and automation helpers
└── zig-out/            # Build artifacts (generated)
```

---

## Usage Examples
### Bootstrap the Framework
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var agent = try abi.features.ai.agent.Agent.init(allocator, .{ .name = "QuickStart" });
    defer agent.deinit();

    const response = try agent.process("Hello!", allocator);
    defer allocator.free(@constCast(response));
}
```

### CLI Commands
```bash
abi help                         # Overview of available subcommands
abi chat --persona creative      # Interactive chat session
abi llm train data.csv           # Model training pipeline
abi wdbx http --host 0.0.0.0     # Start HTTP server for the vector DB
abi benchmark --memory-track     # Performance + memory benchmarks
```

### REST & WebSocket Endpoints
- `GET /health` – Service health probe.
- `GET /api/status` – Framework status snapshot.
- `POST /api/agent/query` – AI agent request (JSON payload).
- `POST /api/database/search` – Vector similarity search.
- `WebSocket /ws` – Bi-directional chat stream.

---

## Vector Database
The WDBX storage engine powers embeddings and similarity search while preserving allocator safety.

**Highlights**
- SIMD-accelerated insert/search paths validated at **2,777+ ops/sec**.
- K-nearest neighbor queries via CLI, HTTP REST, and WebSocket APIs.
- Metadata lifecycle helpers (e.g., `Db.freeResults`) for deterministic resource cleanup.

**Example**
```zig
const std = @import("std");
const abi = @import("abi");

var db = try abi.features.database.database.Db.open("vectors.wdbx", true);
defer db.close();

const allocator = std.heap.page_allocator;
const embedding = [_]f32{ 0.1, 0.2, 0.3 };
const matches = try db.search(&embedding, 10, allocator);
defer abi.features.database.database.Db.freeResults(matches, allocator);
```

---

## Build & Configuration
Tune features with Zig build options:
- `-Denable_cuda=true|false` – Toggle CUDA acceleration (default: true).
- `-Denable_spirv=true|false` – Compile Vulkan/SPIR-V shaders (default: true).
- `-Denable_wasm=true|false` – Emit WebAssembly artifacts (default: true).
- `-Denable-ansi=true|false` – Enable ANSI colour codes in CLI output (default: true).
- `-Dstrict-io=true|false` – Abort on first detected writer error (default: false).
- `-Dexperimental=true|false` – Opt into experimental feature gates (default: false).
- `-Dtarget=<triple>` – Cross-compile (e.g., `x86_64-linux-gnu`, `aarch64-macos`).
- `-Doptimize=Debug|ReleaseSafe|ReleaseFast|ReleaseSmall` – Build profile selection.
- `-Denable_heavy_tests=true` – Include long-running performance and HNSW suites.

Compile-time settings surface through the `options` module:
```zig
const options = @import("options");
std.log.info("CUDA enabled? {}", .{ options.enable_cuda });
```

---

## Performance & Benchmarking
| Component            | Performance Target           | Notes                                      |
|----------------------|------------------------------|--------------------------------------------|
| Text processing      | ≥3.2 GB/s SIMD throughput    | Alignment-safe, zero-copy message passing. |
| Vector operations    | ≥2,777 ops/sec               | Validated via WDBX micro-benchmarks.       |
| Neural inference     | <1 ms for 32×32 networks     | SIMD + GPU acceleration where available.   |
| Lock-free queue      | 10M ops/sec                  | Single producer/consumer benchmark.        |

Common workflows:
- `zig build bench-all` – Run the full benchmark suite.
- `zig build bench-simd` – Stress text/vector SIMD kernels.
- `abi benchmark --memory-track` – Runtime profiling with memory tracking enabled.

---

## Testing
```bash
# Run all tests registered in build.zig
zig build test

# Target specific suites
zig test tests/test_memory_management.zig
zig test tests/test_cli_integration.zig
zig test tests/cross-platform/linux.zig
zig test tests/cross-platform/macos.zig
zig test tests/cross-platform/windows.zig
```

Cross-platform suites gracefully skip on unsupported hosts. Debug builds enable leak detection by default; aim to keep
performance regressions under 5% across releases.

---

## Documentation & Resources
- [`docs/`](docs/) – Landing page for manuals, deployment guides, and generated references.
- [`docs/reports/engineering_status.md`](docs/reports/engineering_status.md) – Up-to-date framework snapshot, module map, and automation overview.
- [`docs/reports/cross_platform_testing.md`](docs/reports/cross_platform_testing.md) – Supported matrix, run commands, and generated smoke-test catalog.
- [`docs/PRODUCTION_DEPLOYMENT.md`](docs/PRODUCTION_DEPLOYMENT.md) – Detailed production rollout guide and environment checklists.
- [`docs/generated/API_REFERENCE/`](docs/generated/API_REFERENCE/) – Generated API reference updated via `zig build docs`.
- [`docs/generated/MODULE_REFERENCE/`](docs/generated/MODULE_REFERENCE/) – Architecture documentation by module.
- [`docs/generated/PERFORMANCE_GUIDE/`](docs/generated/PERFORMANCE_GUIDE/) – Optimization and benchmarking playbooks.
- [`docs/generated/DEFINITIONS_REFERENCE/`](docs/generated/DEFINITIONS_REFERENCE/) – Glossary of terminology and core concepts.
- [`CROSS_PLATFORM_TESTING_GUIDE.md`](CROSS_PLATFORM_TESTING_GUIDE.md) – Legacy consolidated reference retained for historical context.
- [`MIGRATION_REPORT.md`](MIGRATION_REPORT.md) – Release-to-release upgrade notes for long-lived deployments.
- CI generates fresh API docs via `zig build docs` and publishes them with GitHub Pages once the main branch passes.

---

## Contributing
Review [`AGENTS.md`](AGENTS.md) for contributor workflows and required checks before submitting changes.
1. Fork the repository and create a feature branch.
2. Run `zig build`, `zig build test`, and relevant benchmarks before submitting changes.
3. Update documentation when modifying public APIs or behavior.
4. Open a pull request with a clear summary, reproduction steps for fixes, and benchmark deltas when relevant.

**Need help?**
- Documentation portal: [`docs/`](docs/)
- GitHub Issues: <https://github.com/donaldfilimon/abi/issues>
- Community channels (Discord/email): see the documentation landing page.

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

⭐ **Star the project if Abi helps accelerate your AI workloads!**

