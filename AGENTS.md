# AGENTS.md

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

> **Codebase Status:** Synced with repository as of 2026-01-23.
> **Core Mandates:** See [PROMPT.md](PROMPT.md) for strict requirements and KPIs.

## Quick Start for AI Agents

```bash
# Essential commands
zig build                              # Build the project
zig build test --summary all           # Run all tests (regression)
zig fmt .                              # Format code after edits
zig build run -- --help                # CLI help

# Testing specific files
zig test src/specific_test.zig                            # Single test file
zig test src/module/test_file.zig -fno-sanitize-c        # Without C sanitation
zig test src/module/test_file.zig --reference-trace=5    # With debug refs

# Benchmarking (Required for perf changes)
zig build bench-competitive            # Run performance baseline
zig build benchmarks                  # Complete benchmark suite
```

## Critical Rules (Measurable)

### 1. Performance
- **Latency:** Dispatch overhead must remain < 50µs.
- **Throughput:** Kernels must achieve > 80% theoretical peak bandwidth.
- **Verification:** Run `zig build bench-competitive` before and after critical changes.

### 2. Quality
- **Coverage:** New modules must have unit tests. Maintain > 90% statement coverage.
- **Leaks:** Zero memory leaks allowed. Use `GeneralPurposeAllocator` in tests.
- **Formatting:** Zero `zig fmt` diffs allowed.

### 3. Architecture
- **VTable:** Use `src/gpu/interface.zig` VTable pattern for all new backends.
- **Unified Buffer:** Use `src/gpu/unified_buffer.zig` for all memory management.
- **std.gpu:** Prioritize Zig 0.16 native GPU features over C ABI bindings where possible.

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

### Architecture

```
src/
├── abi.zig              # Public API entry point
├── framework.zig        # Framework orchestration
├── config/              # Domain-specific configs (gpu, ai, database, etc.)
├── gpu/                 # Unified GPU API (std.gpu + Backends)
│   ├── std_gpu.zig      # Zig 0.16 Native Interface
│   ├── backends/        # VTable Implementations (Cuda/Vulkan/etc)
│   └── dispatcher.zig   # Kernel Selection Logic
├── ai/                  # AI module (core, implementation, gpu_interface)
├── database/            # Vector database (WDBX)
├── network/             # Distributed compute and Raft
├── observability/       # Metrics, tracing, monitoring
├── runtime/             # Task engine, Scheduler, Memory
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

### Error Formatting

```zig
// Use {t} for errors and enums
std.debug.print("Error: {t}", .{err});
```

## Common Workflows

### Adding a New Backend
1. Implement the VTable interface in `src/gpu/backends/<name>_vtable.zig`.
2. Register in `src/gpu/backend_factory.zig`.
3. Add build flag in `build.zig`.
4. Add to `src/gpu/interface.zig` BackendType enum.

### adding a New Kernel
1. Define in `src/gpu/dsl/mod.zig` if using DSL.
2. Or implement in `src/gpu/kernels/` for specific optimized versions.
3. Update `src/gpu/dispatcher.zig` to route to the new kernel.

## Testing & Debugging
```bash
zig build test --summary all
zig test src/runtime/engine/engine.zig
```

★ *Tip:* Run `zig fmt .` after any edit and immediately verify with `zig build test --summary all`.