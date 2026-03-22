# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion`. On macOS 26.4+ (Darwin 25.x), stock prebuilt Zig may be linker-blocked — use a host-built Zig matching `.zigversion` prepended to `PATH`:
```bash
export PATH="$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin:$PATH"
hash -r
```

## Build Commands

```bash
zig build                          # Build static library (default)
zig build test --summary all       # Run tests
zig build check                    # Lint + test + stub parity (full gate)
zig build lint                     # Check formatting (read-only)
zig build fix                      # Auto-format in place
zig build check-parity             # Verify mod/stub declaration parity
zig build lib                      # Build static library artifact
```

Do NOT run `zig fmt .` at the repo root — use `zig build fix` which scopes to `src/` and `build.zig`.

### Feature Flags

All features default to enabled except `feat-mobile` (false). Disable with `-Dfeat-<name>=false`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

The build.zig is self-contained (~170 lines) with all feature flags defined inline. No external build modules.

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle, feature catalog
- `src/features/` — 19 comptime-gated feature directories (gpu, ai, database, network, web, etc.)
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord, etc.)
- `src/tasks/` — Task management, async job queues
- `src/protocols/` — Protocol implementations: mcp/, lsp/, acp/, ha/
- `src/inference/` — ML inference: engine, scheduler, sampler, paged KV cache
- `src/core/database/` — Vector database implementation (consumed by features/database/ facade)
- `test/` — All test files (separate from production source)

### The Mod/Stub Pattern

Every feature under `src/features/<name>/` follows a contract:
- `mod.zig` — Real implementation
- `stub.zig` — API-compatible no-ops (same public surface, zero-cost when disabled)
- `types.zig` — Shared types used by both mod and stub

In `src/root.zig`, each feature uses comptime selection:
```zig
pub const gpu = if (build_options.feat_gpu) @import("features/gpu/mod.zig") else @import("features/gpu/stub.zig");
```

When modifying a feature's public API, **both `mod.zig` and `stub.zig` must be updated in sync**. Run `zig build check-parity` to verify. The parity checker lives at `src/feature_parity_tests.zig`.

### Build Options

The `build_options` module provides these fields (all `bool` unless noted):
- Feature flags: `feat_gpu`, `feat_ai`, `feat_database`, `feat_network`, `feat_profiling`, `feat_web`, `feat_pages`, `feat_analytics`, `feat_cloud`, `feat_auth`, `feat_messaging`, `feat_cache`, `feat_storage`, `feat_search`, `feat_mobile`, `feat_gateway`, `feat_benchmarks`, `feat_compute`, `feat_documents`, `feat_desktop`
- AI sub-features: `feat_llm`, `feat_training`, `feat_vision`, `feat_explore`, `feat_reasoning`
- Protocols: `feat_lsp`, `feat_mcp`
- GPU backends: `gpu_metal`, `gpu_cuda`, `gpu_vulkan`, `gpu_webgpu`, `gpu_opengl`, `gpu_opengles`, `gpu_webgl2`, `gpu_stdgpu`, `gpu_fpga`, `gpu_tpu`
- `package_version` (`[]const u8`)

## Key Conventions

- The public surface is `abi.<domain>` (e.g., `abi.gpu`, `abi.ai`, `abi.database`). Use `src/root.zig` as the single source of truth for what's exported.
- Struct field renames: grep for `.field_name` (with leading dot) to catch anonymous struct literals that won't match `StructName{` searches.
- `src/core/feature_catalog.zig` is the canonical source of truth for feature metadata.
- `src/core/stub_helpers.zig` provides `StubFeature`, `StubContext`, and `StubContextWithConfig` — reuse these in stubs instead of defining custom lifecycle boilerplate.
