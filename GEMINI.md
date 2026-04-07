# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion`. Use `tools/zigly --status` to auto-install the correct version, or `tools/zigly --link` to symlink zig + zls into `~/.local/bin`.

## Quick Reference

| Command | Description |
|---------|-------------|
| `./build.sh` | Build (macOS 26.4+) |
| `zig build test --summary all` | All tests |
| `zig build test -- --test-filter "pattern"` | Single test |
| `zig build lint` | Check formatting |
| `zig build fix` | Auto-format in place |
| `zig build check` | Full gate (lint + test + parity) |

**Do NOT run `zig fmt .`** — use `zig build fix` which scopes to `src/`, `build.zig`, `build/`, and `test/`.

### Running Single Tests

```bash
# Run a specific test by name pattern
zig build test --summary all -- --test-filter "test_name_pattern"

# On macOS 26.4+:
./build.sh test --summary all -- --test-filter "test_name_pattern"
```

### Test Lanes

```bash
zig build messaging-tests agents-tests orchestration-tests
zig build gateway-tests inference-tests secrets-tests pitr-tests
```

27 focused test lanes exist. Run `zig build test --summary all` for full suite.

**Known pre-existing failures**: inference engine connector backend tests (2), auth integration tests (1 failure, 3 leaks).

### Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui`. Disable with `-Dfeat-<name>=false`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
```

## Architecture

### Architectural Rules (Massive Update Guidelines)
1. **Direct Domain API**: All domain logic must be exposed via top-level direct APIs (`abi.<domain>`). Legacy `abi.features` nesting is strictly prohibited.
2. **Comptime Feature Catalog Loops**: Use Zig's zero-overhead `inline for` loops driven by `src/core/feature_catalog.zig` for all framework initialization, shutdown, and registry logic. Do not hardcode lists.
3. **Unified Errors**: All modules must use the unified error handling framework defined in `src/core/errors.zig` instead of declaring localized, ad-hoc error sets.
4. **WDBX Integration**: Features (especially database/cache) should adhere to Liquid Glass memory models and utilize `LiquidGlassMemory` schemas where appropriate.

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle, feature catalog
- `src/features/` — 21 feature directories (60 features total including AI sub-features and protocols)
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
- Protocols: `feat_lsp`, `feat_mcp`, `feat_acp`, `feat_ha`
- GPU backends: `gpu_metal`, `gpu_cuda`, `gpu_vulkan`, `gpu_webgpu`, `gpu_opengl`, `gpu_opengles`, `gpu_webgl2`, `gpu_stdgpu`, `gpu_fpga`, `gpu_tpu`
- `package_version` (`[]const u8`)

### Test Architecture

Two test suites run under `zig build test`:
1. **Unit tests** (`src/root.zig`) — `refAllDecls` walks the module tree, running `test` blocks in every `.zig` file under `src/`.
2. **Integration tests** (`test/mod.zig`) — imports `@import("abi")` as an external consumer.

Both suites link the same platform frameworks (macOS: System, IOKit, Accelerate, Metal, objc).

## Critical Rules

1. **Never use `@import("abi")` from `src/`** — causes circular import
2. **Cross-feature imports**: use comptime gates `if (build_options.feat_X) mod else stub`
3. **Mod/stub parity**: update both together, run `zig build check-parity`
4. Use `.empty` not `.{}` for `ArrayListUnmanaged`/`HashMapUnmanaged` init
5. Use `foundation.time.unixMs()` not `std.time.milliTimestamp`
6. Use `foundation.sync.Mutex` not `std.Thread.Mutex`
7. On macOS 26.4+, use `./build.sh` not `zig build`
8. All path imports need explicit `.zig` extensions

## Code Style

### Naming
- camelCase: functions/methods
- PascalCase: types/structs/enums
- SCREAMING_SNAKE_CASE: constants
- snake_case: enum variants

### Error Handling
- `@compileError` — compile-time contract violations only
- `@panic` — unrecoverable invariant violations; never in library code (`src/`), only in CLI entry points and tests
- `unreachable` — provably impossible branches verified at comptime
- Error unions (`!`) — all runtime failure paths; prefer `error.FeatureDisabled` in stubs

### Memory & Ownership
- Always pair allocation/deallocation with `defer`
- String literals in structs with `deinit()` → always `allocator.dupe()`

## Import Rules

- **Within `src/`**: use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside `src/` — causes circular import error.
- **From `test/`**: use `@import("abi")` and `@import("build_options")` — wired by build.zig.
- **Cross-feature imports**: never import another feature's `mod.zig` directly. Use comptime gate: `const obs = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16).

## Zig 0.16 Gotchas

- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- `var` vs `const`: compiler enforces const for never-mutated locals
- Entry points use `pub fn main(init: std.process.Init) !void`
- `std.mem.trimRight` renamed to `std.mem.trimEnd`
- Platform-gated externs: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`
- `foundation.time.timestampSec()` is monotonic — use `clock_gettime(.REALTIME)` for persisted data
