---
title: Getting Started
description: Install, build, and run ABI in under 5 minutes
section: Start
order: 2
---

# Getting Started

This guide walks you from a fresh clone to a running ABI build with passing tests.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Zig** | `0.16.0-dev.2596+469bf6af0` or newer (pinned in `.zigversion`) |
| **Git** | Any recent version |
| **Shell** | Bash, Zsh, or Fish on Linux / macOS |
| **Optional** | GPU drivers (CUDA, Vulkan, or Metal) for hardware-accelerated compute |
| **Optional** | Docker for containerized deployment |

### Installing Zig

The recommended approach is [zvm](https://github.com/marler182/zvm) (Zig Version Manager):

```bash
# Install zvm, then:
zvm install master
zvm use master

# Verify the version matches .zigversion
zig version
# Expected: 0.16.0-dev.2596+469bf6af0 (or newer)
```

Alternatively, download a matching nightly build from [ziglang.org/download](https://ziglang.org/download/).

## Install

```bash
git clone https://github.com/your-org/abi.git
cd abi

# Build with default flags (all features enabled except mobile)
zig build
```

The first build compiles the framework, CLI, and all enabled feature modules. Subsequent
builds are incremental.

### Verify the build

```bash
# Print version
zig build run -- version

# Show system info and which features are compiled in
zig build run -- system-info
```

## Run Tests

ABI maintains two test suites with enforced baselines.

### Main test suite

```bash
zig build test --summary all
```

Expected: **1248 pass, 5 skip** (1253 total).

The main tests live in `src/services/tests/mod.zig` and exercise cross-module integration,
stress tests, chaos tests, and parity checks.

### Feature inline tests

```bash
zig build feature-tests --summary all
```

Expected: **1095 pass** (1095 total).

Feature tests are inline `test {}` blocks inside each module's source files, compiled
through `src/feature_test_root.zig`.

### Test a single file

```bash
zig test src/path/to/file.zig
```

### Filter tests by name

```bash
zig test src/services/tests/mod.zig --test-filter "pattern"
```

## Feature Flags

All features default to **enabled** except `mobile`. Toggle them at build time:

```bash
# Disable GPU, enable mobile
zig build -Denable-gpu=false -Denable-mobile=true

# AI-only build
zig build -Denable-ai=true -Denable-database=false -Denable-network=false

# Validate that all flag combinations compile
zig build validate-flags
```

Common flags:

| Flag | Default | Module |
|------|---------|--------|
| `-Denable-ai` | `true` | AI monolith + ai_core |
| `-Denable-llm` | `true` | Inference (LLM, embeddings, vision) |
| `-Denable-gpu` | `true` | GPU acceleration |
| `-Denable-database` | `true` | Vector database |
| `-Denable-network` | `true` | Distributed network |
| `-Denable-web` | `true` | Web/HTTP utilities |
| `-Denable-cloud` | `true` | Cloud provider adapters |
| `-Denable-mobile` | **`false`** | Mobile platform |
| `-Denable-profiling` | `true` | Observability (metrics, tracing) |
| `-Denable-training` | `true` | Training pipelines |
| `-Denable-reasoning` | `true` | Abbey reasoning engine |

GPU backend selection:

```bash
zig build -Dgpu-backend=metal          # macOS
zig build -Dgpu-backend=cuda           # NVIDIA
zig build -Dgpu-backend=vulkan         # Cross-platform
zig build -Dgpu-backend=simulated      # Software fallback (always available)
```

See [Configuration](configuration.md) for the full flag and environment variable reference.

## Run Examples

ABI ships with 32 examples covering all major features.

```bash
# Build all examples
zig build examples

# Run a specific example
zig build run-hello
zig build run-gpu
```

## Full Validation

Before committing changes, run the complete local gate:

```bash
# Format + tests + feature tests + flag validation + CLI smoke tests
zig build full-check

# Or the extended version (adds examples, benchmarks, WASM check)
zig build verify-all
```

## Next Steps

- [Architecture](architecture.md) -- understand the module hierarchy and comptime gating
- [Configuration](configuration.md) -- all build flags and environment variables
- [CLI](cli.md) -- 28 commands for AI, GPU, database, and system management
