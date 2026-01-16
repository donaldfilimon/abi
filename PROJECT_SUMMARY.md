# Project Summary

## Overview

ABI is a modular shared compute framework built on Zig 0.16.x.
The codebase separates public API, compute runtime, feature stacks, and shared
utilities so CPU, GPU, and distributed workloads can evolve independently.

## Module layout

- `src/abi.zig` exposes the public API surface
- `src/root.zig` is the root module
- `src/compute/` contains runtime, concurrency, memory, GPU, and network layers
- `src/features/` contains feature stacks (ai, gpu, database, web, monitoring,
  connectors, network)
- `src/shared/` contains logging, observability, platform, SIMD, and utilities
- `docs/` contains comprehensive project documentation
- `tools/cli/main.zig` is the primary CLI entrypoint (`src/main.zig` fallback)

## Examples

- `examples/` directory contains example programs (hello, database, agent, compute, gpu, network)
- `benchmarks/main.zig` runs the benchmark suite

## Build and test commands

```bash
zig build
zig build test
zig build benchmark
zig build run -- --help
```

## Feature flags

Defaults:

- `-Denable-gpu=true`
- `-Denable-ai=true`
- `-Denable-web=true`
- `-Denable-database=true`
- `-Denable-network=true`
- `-Denable-profiling=true`

GPU backends:

- `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`
- `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`

Example:

```bash
zig build -Denable-network=true -Dgpu-vulkan=true
```

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
See [TODO.md](TODO.md) for the list of pending implementations.
*See [TODO.md](TODO.md) and [ROADMAP.md](ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
