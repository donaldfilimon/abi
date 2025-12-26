# Project Summary

## Overview
Abbey-Aviva-Abi is a modular shared compute framework built on Zig 0.16.x.
The codebase separates public API, compute runtime, feature stacks, and shared
utilities so CPU, GPU, and distributed workloads can evolve independently.

## Module layout
- `src/abi.zig` exposes the public API surface
- `src/root.zig` is the root module
- `src/compute/` contains runtime, concurrency, memory, GPU, and network layers
- `src/features/` contains feature stacks (ai, gpu, database, web, monitoring,
  connectors, network)
- `src/shared/` contains logging, observability, platform, SIMD, and utilities
- `tools/cli/main.zig` is the primary CLI entrypoint (`src/main.zig` fallback)

## Examples
- `src/demo.zig` provides a small end-to-end example
- `src/compute/runtime/benchmark_demo.zig` runs the benchmark suite

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
- `-Denable-network=false`
- `-Denable-profiling=false`

GPU backends:
- `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`
- `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`

Example:
```bash
zig build -Denable-network=true -Dgpu-vulkan=true
```
