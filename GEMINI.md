# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion`. Use `tools/zigup.sh --status` to auto-install the correct version, or `tools/zigup.sh --link` to symlink zig + zls into `~/.local/bin`.

## Build Commands

```bash
./build.sh                         # Build (macOS 26.4+ auto-relinks with Apple ld)
./build.sh test --summary all      # Run tests via wrapper (macOS 26.4+)
zig build                          # Build static library (Linux / older macOS)
zig build test --summary all       # Run tests (src/ + test/)
zig build check                    # Lint + test + stub parity (full gate)
zig build lint                     # Check formatting (read-only)
zig build fix                      # Auto-format in place
zig build check-parity             # Verify mod/stub declaration parity
zig build cross-check              # Verify cross-compilation (linux, wasi, x86_64)
zig build lib                      # Build static library artifact
zig build mcp                      # Build MCP stdio server (zig-out/bin/abi-mcp)
zig build cli                      # Build ABI CLI binary (zig-out/bin/abi)
zig build doctor                   # Report build configuration and diagnostics
```

Do NOT run `zig fmt .` at the repo root — use `zig build fix` which scopes to `src/` and `build.zig`.

On macOS 26.4+ (Darwin 25.x), stock Zig's LLD linker cannot link binaries. Use `./build.sh` which auto-relinks with Apple's native linker. On Linux / older macOS, `zig build` works directly.

### Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui`. Disable with `-Dfeat-<name>=false`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
```

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle, feature catalog
- `src/features/` — 20 feature directories (30 features total including AI sub-features)
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord, etc.)
- `src/protocols/` — Protocol implementations: mcp/, lsp/, acp/, ha/
- `src/inference/` — ML inference: engine, scheduler, sampler, paged KV cache
- `src/core/database/` — Vector database implementation
- `test/` — Integration tests via `test/mod.zig` (uses `@import("abi")`)

### The Mod/Stub Pattern

Every feature under `src/features/<name>/` follows a contract:
- `mod.zig` — Real implementation
- `stub.zig` — API-compatible no-ops (same public surface, zero-cost when disabled)
- `types.zig` — Shared types used by both mod and stub

In `src/root.zig`, each feature uses comptime selection:
```zig
pub const gpu = if (build_options.feat_gpu) @import("features/gpu/mod.zig") else @import("features/gpu/stub.zig");
```

When modifying a feature's public API, **both `mod.zig` and `stub.zig` must be updated in sync**. Run `zig build check-parity` to verify.

### Build Options

The `build_options` module provides these fields (all `bool` unless noted):
- Feature flags: `feat_gpu`, `feat_ai`, `feat_database`, `feat_network`, `feat_observability`, `feat_web`, `feat_pages`, `feat_analytics`, `feat_cloud`, `feat_auth`, `feat_messaging`, `feat_cache`, `feat_storage`, `feat_search`, `feat_mobile`, `feat_gateway`, `feat_benchmarks`, `feat_compute`, `feat_documents`, `feat_desktop`, `feat_tui`
- AI sub-features: `feat_llm`, `feat_training`, `feat_vision`, `feat_explore`, `feat_reasoning`
- Protocols: `feat_lsp`, `feat_mcp`
- GPU backends: `gpu_metal`, `gpu_cuda`, `gpu_vulkan`, `gpu_webgpu`, `gpu_opengl`, `gpu_opengles`, `gpu_webgl2`, `gpu_stdgpu`, `gpu_fpga`, `gpu_tpu`
- `package_version` (`[]const u8`)

### Test Architecture

Two test suites run under `zig build test`:
1. **Unit tests** (`src/root.zig`) — `refAllDecls` walks the module tree, running `test` blocks in every `.zig` file under `src/`.
2. **Integration tests** (`test/mod.zig`) — imports `@import("abi")` as an external consumer.

Both suites link the same platform frameworks (macOS: System, IOKit, Accelerate, Metal, objc).

## Import Rules

- **Within `src/`**: use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside `src/` — causes circular import error.
- **From `test/`**: use `@import("abi")` and `@import("build_options")` — wired by build.zig.
- **Cross-feature imports**: never import another feature's `mod.zig` directly. Use comptime gate: `const obs = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16).

## Key Conventions

- Public surface is `abi.<domain>` (e.g., `abi.gpu`, `abi.ai`). `src/root.zig` is the source of truth.
- Struct field renames: grep for `.field_name` (with leading dot) to catch anonymous struct literals.
- `src/core/feature_catalog.zig` is the canonical source of truth for feature metadata.
- `src/core/stub_helpers.zig` provides `StubFeature`, `StubContext`, and `StubContextWithConfig` — reuse in stubs.

## Zig 0.16 Gotchas

- `ArrayListUnmanaged` init: use `.empty` not `.{}` (struct fields changed)
- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- `var` vs `const`: compiler enforces const for never-mutated locals
- Function pointers: can call through `*const fn` directly without dereferencing
- `zig fmt .` from root: don't — use `zig build fix` to avoid vendored fixtures
