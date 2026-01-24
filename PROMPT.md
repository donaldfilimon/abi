---
title: "PROMPT"
tags: ["requirements", "standards", "kpi", "architecture"]
---
# ABI Framework: Engineering Requirements & Standards

> **Status:** Synced with repository as of 2026-01-24.
> **Standard:** Zig 0.16 (Master Branch) with `std.gpu`.

## 1. Core Mandates

All contributions must adhere to these engineering standards. Deviations require explicit justification.

### 1.1 Technology Stack
*   **Language:** Zig 0.16.0-dev (Master). Use strict types, explicit allocators, and `std.ArrayListUnmanaged`.
*   **GPU Compute:** Prioritize `std.gpu` (Zig 0.16 native) for kernels. Fallback to VTable implementations (CUDA/Vulkan/Metal) only when features are missing.
*   **Memory Management:** 
    *   Use `std.mem.Allocator` passed explicitly to init functions.
    *   Use `std.heap.ArenaAllocator` for short-lived scopes.
    *   Zero global mutable state unless protected by `std.Thread.Mutex`.

### 1.2 Cross-Platform Targets
Code must compile and run on the following tiers:

| Tier | Platforms | Requirements |
|------|-----------|--------------|
| **Tier 1 (Desktop)** | Windows (MSVC/GNU), Linux (Gnu/Musl), macOS (ARM64/x86_64) | Full feature parity, high performance. |
| **Tier 2 (Mobile)** | Android (aarch64), iOS (aarch64) | `std.gpu` compute, minimized binary size, battery-aware scheduling. |
| **Tier 3 (Web)** | WASM32 (WASI/Emscripten) | No threads (async/await simulation), WebGPU backend, limited FS access. |

### 1.3 Measurable Outcomes (KPIs)
Changes must meet these measurable thresholds:

*   **Performance:**
    *   Kernel Latency: < 50µs dispatch overhead.
    *   Throughput: > 80% theoretical peak memory bandwidth (measured via `bench-competitive`).
    *   Startup Time: < 100ms to first token/kernel.
*   **Quality:**
    *   Test Coverage: > 90% statement coverage for core modules (`runtime`, `gpu`, `ai`).
    *   Zero Leaks: `GeneralPurposeAllocator` must report no leaks on shutdown.
    *   Linting: Zero `zig fmt` diffs.

## 2. Architecture & Patterns

### 2.1 Modular VTable Pattern
Backends (GPU, Database, Network) must use the **Interface VTable** pattern for runtime polymorphism without generic bloat.

```zig
pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    
    pub const VTable = struct {
        launchKernel: *const fn (*anyopaque, Config, Args) Error!void,
        // ...
    };
    
    pub fn launchKernel(self: Backend, ...) !void {
        return self.vtable.launchKernel(self.ptr, ...);
    }
};
```

### 2.2 Unified GPU API
*   **Buffer:** `UnifiedBuffer` handles host/device sync automatically (`dirty` flags).
*   **Stream:** All operations must take an optional `*Stream` for async execution.
*   **Fallback:** `Dispatcher` must automatically downgrade: `std.gpu` → `Backend VTable` → `SIMD CPU` → `Scalar CPU`.

## 3. Build System (`build.zig`)

### 3.1 Standard Flags
Use these flags to control the build graph. Do not introduce custom ad-hoc flags.

| Flag | Values | Description |
|------|--------|-------------|
| `-Dgpu-backend` | `auto`, `cuda`, `vulkan`, `metal`, `webgpu`, `stdgpu` | Comma-separated list of enabled backends. |
| `-Denable-ai` | `true`/`false` | Include LLM and training modules. |
| `-Denable-mobile` | `true`/`false` | Enable mobile-specific optimizations (strip debug, small buffer sizes). |
| `-Doptimize` | `Debug`, `ReleaseFast`, `ReleaseSafe`, `ReleaseSmall` | Standard Zig optimization modes. |

### 3.2 Mobile & Web Builds
*   **Android:** `zig build -Dtarget=aarch64-linux-android -Denable-mobile=true`
*   **iOS:** `zig build -Dtarget=aarch64-macos-none -Denable-mobile=true` (simulated as macOS for now)
*   **WASM:** `zig build -Dtarget=wasm32-wasi -Dgpu-backend=webgpu`

## 4. Workflows

### 4.1 Development
1.  **Format:** `zig fmt .`
2.  **Test:** `zig build test --summary all`
3.  **Benchmark:** `zig build bench-competitive`
4.  **Verify:** `zig build full-check`

### 4.2 Contribution Checklist
- [ ] Requirements: Does this match Tier 1/2/3 targets?
- [ ] Performance: Did you run benchmarks? Regression?
- [ ] Tests: Added unit tests?
- [ ] Docs: Updated `API_REFERENCE.md`?

## 5. Directory Structure (Canonical)

```
src/
├── abi.zig              # Public Facade
├── framework.zig        # Lifecycle Manager
├── config/              # Configuration Structs
├── gpu/                 # Unified GPU API (std.gpu + Backends)
│   ├── std_gpu.zig      # Zig 0.16 Native Interface
│   ├── backends/        # VTable Implementations (Cuda/Vulkan/etc)
│   └── dispatcher.zig   # Kernel Selection Logic
├── runtime/             # Engine, Scheduler, Memory
├── observability/       # Metrics, Tracing
└── shared/              # Cross-module Utilities
```

## 6. Terminology

*   **VTable:** Virtual Method Table for runtime polymorphism.
*   **SIMD:** Single Instruction, Multiple Data (CPU vectorization).
*   **Kernel:** A compute shader function executed on the GPU.
*   **Dispatcher:** The component selecting the best kernel/device.
*   **Unified Buffer:** Memory accessible by both Host (CPU) and Device (GPU).