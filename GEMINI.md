# ABI Framework Context

## Project Overview

**ABI** is a modern Zig framework designed for modular AI services, vector search, and
systems tooling. It provides a comprehensive set of features including an AI agent
runtime, vector database helpers, high-performance compute runtime, GPU backends, and
distributed network compute capabilities.

**Key Technologies:**

- **Language:** Zig (0.16.x)
- **Architecture:** Modular, with a core runtime and optional feature stacks
  (AI, GPU, Database, Web, Monitoring).
- **Compute:** Work-stealing scheduler, support for various GPU backends
  (CUDA, Vulkan, Metal, WebGPU).

## Directory Structure

- `src/abi.zig`: Public API surface.
- `src/root.zig`: Root module entrypoint.
- `src/framework/`: Core runtime configuration and lifecycle management.
- `src/features/`: Feature-specific modules (AI, Compute, Connectors, Database,
  GPU, Monitoring, Web).
- `src/compute/`: High-performance compute runtime implementation.
- `tests/`: Integration and unit tests.
- `tools/`: Utility tools, including the CLI entrypoint (`tools/cli/main.zig`).
- `docs/`: Comprehensive documentation suite.
- `build.zig` & `build.zig.zon`: Build configuration and dependencies.

## Build and Run

### Prerequisites

- Zig 0.16.x

### Standard Commands

- **Build Project:**
  ```bash
  zig build
  ```
- **Run All Tests:**
  ```bash
  zig build test
  ```
- **Run Benchmarks:**
  ```bash
  zig build benchmark
  ```
- **Run CLI (if enabled):**
  ```bash
  zig build run -- --help
  ```
- **Format Code:**
  ```bash
  zig fmt .
  ```

### Feature Flags

Enable or disable specific features during build or test:

- `-Denable-ai` (default: `true`)
- `-Denable-gpu` (default: `true`)
- `-Denable-web` (default: `true`)
- `-Denable-database` (default: `true`)
- `-Denable-network` (default: `false`)
- `-Denable-profiling` (default: `false`)

**Example - Build with specific options:**

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true
```

**GPU Backend Selection:**

- `-Dgpu-cuda`: Enable CUDA backend
- `-Dgpu-vulkan`: Enable Vulkan backend
- `-Dgpu-metal`: Enable Metal backend
- `-Dgpu-webgpu`: Enable WebGPU backend

## Development Conventions

- **Style:** Adhere to standard Zig style guidelines.
  - Indentation: 4 spaces (no tabs).
  - Line Length: < 100 characters.
  - Naming: `PascalCase` for types, `snake_case` for functions/variables.
- **Code Structure:**
  - Use explicit imports (avoid `usingnamespace`).
  - Use `!` return types with specific error enums.
  - Utilize `defer` and `errdefer` for resource cleanup.
- **Testing:**
  - Unit tests should accompany new features.
  - Run `zig build test` to verify changes.
- **Environment Variables:**
  - Check `README.md` for a list of supported environment variables for
    connectors (OpenAI, Hugging Face, etc.).

## Key Files for Context

- `src/abi.zig`: Check this to understand what is exposed publicly.
- `build.zig`: Defines the build graph and available options.
- `src/compute/runtime/engine.zig`: Core logic for the compute engine.
