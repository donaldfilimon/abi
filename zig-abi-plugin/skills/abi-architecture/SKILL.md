---
name: abi-architecture
description: Use when working on ABI framework architecture, adding features, modifying the build system, or understanding the comptime feature gating pattern. Trigger when user mentions "feature module", "mod.zig", "stub.zig", "feature flag", "build system", "feature catalog", "comptime gating", "BuildOptions", "test_discovery", or asks about how ABI features are structured.
---

# ABI Architecture Guide

## Comptime Feature Gating (Critical Pattern)

Every feature lives in `src/features/<name>/` with two files:
- **`mod.zig`** — real implementation (used when enabled)
- **`stub.zig`** — disabled fallback (used when `-Dfeat-<name>=false`)

The build system selects mod vs stub via `build_options.feat_<name>`. **When changing ANY public function in `mod.zig`, you MUST update `stub.zig` with the same signature.** Mismatches break disabled-flag builds.

### Stub Return Patterns
| Return Type | Stub Returns |
|-------------|-------------|
| `!T` (error union) | `error.FeatureDisabled` |
| `?T` (optional) | `null` |
| `void` | `{}` (empty body) |
| `[]T` (slice) | `&[0]T{}` or `&.{}` |
| `bool` | `false` |
| numeric | `0` |

### Adding a New Feature (8 Steps)

1. Create `src/features/<name>/mod.zig` (implementation)
2. Mirror every `pub fn` signature in `src/features/<name>/stub.zig`
3. Add `feat_<name>` flag to `build/options.zig` (`BuildOptions` + `CanonicalFlags`)
4. Register in `src/core/feature_catalog.zig` (Feature enum + `all` array)
5. Add test entries to `build/test_discovery.zig` manifest
6. Add flag combo rows to `build/flags.zig` if needed
7. Wire up in `src/abi.zig` (comptime feature selection)
8. Validate: `zig build validate-flags` then `zig build full-check`

## Build System Layout

`build.zig` imports modular components from `build/`:

| File | Purpose |
|------|---------|
| `options.zig` | `BuildOptions` struct, flag reading, comptime validation |
| `flags.zig` | `FlagCombo` validation matrix (39 combos; mobile gap pending) |
| `modules.zig` | Module creation helpers |
| `test_discovery.zig` | Single source of truth for feature test manifest |
| `cli_smoke_runner.zig` | CLI smoke test descriptors |
| `gpu.zig` / `gpu_policy.zig` | GPU backend selection |
| `link.zig` | Metal framework linking, Darwin SDK detection |
| `cel.zig` | CEL toolchain integration |
| `targets.zig` | Cross-compilation targets |
| `wasm.zig` / `mobile.zig` | Platform-specific build support |

## Import Rules

- External consumers: `@import("abi")`
- Feature modules in `src/features/`: **relative imports only** — never `@import("abi")`
- **Named modules** (registered in `build.zig`): Use `@import("wdbx")`, `@import("build_options")`, etc. — NOT relative paths like `@import("../../wdbx/wdbx.zig")`
- Enforced by `zig build check-imports`
- Files in `build/test_discovery.zig` must compile standalone with `zig test <file> -fno-emit-bin`
- Cross-directory `@import("../../")` breaks standalone compilation — inline small deps instead

### Module Conflict Rule (Critical)
A Zig file can only belong to ONE module. If `src/wdbx/wdbx.zig` is the root of the named `wdbx` module, importing it via relative path from `src/features/` claims it for the `abi` module too — causing a compile error. Always use the named import: `@import("wdbx")`.

## Feature Flag Conventions

- Prefix: `feat_<name>` (NOT `enable_<name>`)
- All default to `true`
- Canonical source: `build/options.zig` (`BuildOptions` struct)
- Feature metadata: `src/core/feature_catalog.zig`
- Internal flags: `feat_explore`, `feat_vision` (derived from `feat_ai`)
- GPU backend: `-Dgpu-backend=auto|cuda|vulkan|metal`

## Key Module Map

| Public Path | Entry File | Purpose |
|-------------|-----------|---------|
| `abi.App` / `abi.AppBuilder` | `src/abi.zig` | Primary runtime |
| `abi.wdbx` | `src/wdbx/wdbx.zig` | Vector database |
| `abi.wdbx.dist` | `src/wdbx/dist/mod.zig` | Distributed coordinator |
| `abi.features.ai` | `src/features/ai/mod.zig` | LLM, agents, training |
| `abi.features.database` | `src/features/database/mod.zig` | Semantic store, HNSW |
| `abi.personas` | (via abi.zig) | Multi-persona system |
| `abi.inference_engine` | (via abi.zig) | Token generation |
| `abi.server` | (via abi.zig) | REST API + OpenAI compat |

## Test Structure

Two test roots:
1. **Main tests** (`zig build test`): `src/services/tests/mod.zig` → ~1290 pass
2. **Feature tests** (`zig build feature-tests`): `build/test_discovery.zig` manifest → ~2836 pass

After adding tests: run `zig build update-baseline`.
After modifying CLI commands: run `zig build refresh-cli-registry`.
