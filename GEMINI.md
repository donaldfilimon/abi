---
title: "GEMINI"
tags: [ai, agents, gemini]
---
# GEMINI.md
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Gemini-Agent_Guide-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini Guide"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
</p>

This file gives the Gemini agent guidance on interacting with the ABI framework. It mirrors the structure of `AGENTS.md` and `CLAUDE.md` but is tuned for Gemini‑like prompts.

## Quick Start for Gemini

```bash
zig build                             # Build the framework
zig build test --summary all          # Run all tests (regression)
zig fmt .                             # Format after edits
zig build run -- --help               # CLI help

# Example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
```

Typical Gemini usage:

```bash
zig build run -- llm generate "Hello" --max 60
```

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

### Architecture

The codebase uses a domain-driven modular structure with unified configuration and compile-time feature gating:

```
src/
├── abi.zig              # Public API entry point
├── framework.zig        # Framework orchestration
├── config/              # Domain-specific configs (gpu, ai, database, etc.)
├── ai/                  # AI module (core, implementation, gpu_interface)
├── connectors/          # External API connectors
├── database/            # Vector database (WDBX)
├── gpu/                 # GPU acceleration (backends)
├── ha/                  # High Availability
├── network/             # Distributed compute and Raft
├── observability/       # Metrics, tracing, monitoring
├── registry/            # Plugin registry (lifecycle, plugins)
├── runtime/             # Task engine
├── shared/              # Utils and helpers
├── tasks/               # Task management (persistence, querying, lifecycle)
└── web/                 # Web/HTTP utilities
```

## Zig 0.16 Patterns (CRITICAL)

> See `docs/migration/zig-0.16-migration.md` for comprehensive examples.

### I/O Backend Initialization

Zig 0.16 requires explicit I/O backend initialization for file and network operations.

```zig
// Initialize once, use for all file/network operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,  // .empty for library, .init() for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

// File read
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);
```

### Other Changes

```zig
// Error/enum formatting: use {t} instead of @errorName()/@tagName()
std.debug.print("Error: {t}", .{err});

// ArrayListUnmanaged: use .empty not .init()
var list = std.ArrayListUnmanaged(u8).empty;

// Timing: use Timer.start() not Instant.now()
var timer = std.time.Timer.start() catch return error.TimerFailed;

// HTTP server: use .interface for reader/writer
var server: std.http.Server = .init(
    &connection_reader.interface,
    &connection_writer.interface,
);
```

## Feature Management

Feature flags live in `build.zig` and correlate with `build_options`:

```zig
const build_options = struct{
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    // ... add others as needed
};
```

**Stub-Real Sync:** Changes to `src/<feature>/mod.zig` must be propagated to its `stub.zig`. The CI parity checker will report mismatches.

## Gotchas for Gemini
* **Stub‑Real Sync** – Changes to `src/<feature>/mod.zig` must be propagated to its `stub.zig`.
* **GPU Backends** – Use `-Dgpu-backend=auto` or `-Dgpu-backend=cuda,vulkan` (comma‑separated).
* **WASM Compatibility** – `database`, `network`, and `gpu` modules auto‑disable on WASM.
* **Compile‑time Flags** – `-D` options modify the compiled binary. `--enable-…` only toggles runtime behaviour.
* **Format Specifiers** – Always use `{t}` for printing errors and enums.

★ *Tip:* After modifications run `zig fmt .` followed by `zig build test --summary all` to verify nothing broke.