---
title: Getting Started
description: First build, first test, first example with ABI
section: Start
order: 3
---

# Getting Started

This guide takes you from a working toolchain to your first ABI build, test run, and
example in under five minutes. If you have not installed Zig yet, start with the
[Installation](installation.html) page.

## Prerequisites

Make sure your Zig version matches the pinned version:

```bash
zig version
# Expected: 0.16.0-dev.2611+f996d2866 or newer

cat .zigversion
# Should match
```

If these do not match, see [Installation](installation.html) for setup instructions.

## First Build

From the repository root:

```bash
zig build
```

This compiles the framework with all default features enabled (everything except
`mobile`). The first build takes a minute or two; subsequent builds are incremental.

Verify the build succeeded:

```bash
# Print version
zig build run -- version

# Show system info and which features are compiled in
zig build run -- system-info
```

## First Test Run

ABI maintains two test suites with enforced baselines.

### Main tests

```bash
zig build test --summary all
```

Expected: **1270 pass, 5 skip** (1275 total).

### Feature tests

```bash
zig build feature-tests --summary all
```

Expected: **1534 pass** (1534 total).

### Test a single file

```bash
zig test src/path/to/file.zig
```

### Filter tests by name

```bash
zig test src/services/tests/mod.zig --test-filter "pattern"
```

## First Example

ABI ships with 36 examples covering all major features. Build and run one:

```bash
# Build all examples
zig build examples

# Run the hello-world example
zig build run-hello
```

Or write your own minimal program:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework with default config
    var fw = try abi.initDefault(allocator);
    defer fw.deinit();

    std.debug.print("ABI v{s} running\n", .{abi.version()});
    std.debug.print("State: {t}\n", .{fw.getState()});
}
```

## Feature Flags

All features default to **enabled** except `mobile`. Toggle them at build time:

```bash
# Disable GPU, enable mobile
zig build -Denable-gpu=false -Denable-mobile=true

# AI-only build
zig build -Denable-ai=true -Denable-database=false -Denable-network=false

# Validate that all 34 flag combinations compile
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

See [Configuration](configuration.html) for the full 21-flag reference and all
environment variables.

## CLI Overview

ABI provides 28 CLI commands plus 8 aliases:

```bash
# List all commands
zig build run -- --help

# AI commands
zig build run -- llm chat
zig build run -- agent
zig build run -- embed

# GPU commands
zig build run -- gpu backends
zig build run -- gpu-dashboard

# Database commands
zig build run -- db stats
zig build run -- db query --text "hello" --top-k 5

# System commands
zig build run -- system-info
zig build run -- status
```

See [CLI](cli.html) for the complete command reference.

## Full Validation

Before committing changes, run the complete local gate:

```bash
# Format + tests + feature tests + flag validation + CLI smoke tests
zig build full-check

# Or the extended version (adds examples, benchmarks, WASM check, and Ralph gate)
zig build verify-all
```

## Next Steps

- [Architecture](architecture.html) -- understand the module hierarchy and comptime gating
- [Configuration](configuration.html) -- all build flags and environment variables
- [Framework Lifecycle](framework.html) -- deep dive into initialization and state management
- [CLI](cli.html) -- 28 commands for AI, GPU, database, and system management
