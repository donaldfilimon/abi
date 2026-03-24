# AGENTS.md

Guidance for AI coding agents working in this repository.

## Quick Reference

- **Language**: Zig 0.16 (pinned in `.zigversion`)
- **Build**: `zig build` (Linux) or `./build.sh` (macOS 26.4+)
- **Test**: `zig build test --summary all` or `./build.sh test --summary all`
- **Lint**: `zig build lint` / **Fix**: `zig build fix`
- **Parity check**: `zig build check-parity`
- **Validation gates**: `zig build feature-tests`, `mcp-tests`, `gateway-tests`, `inference-tests`, `cli-tests`, `tui-tests`, `typecheck` (compile-only), `validate-flags`, `full-check`, `verify-all`
- **Full gate**: `zig build check` (lint + test + parity)
- **CLI**: `zig build cli` produces `zig-out/bin/abi`
- **MCP server**: `zig build mcp` produces `zig-out/bin/abi-mcp`

## Critical Rules

1. Never use `@import("abi")` from within `src/` — causes circular import error.
2. Cross-feature imports must use comptime gates: `if (build_options.feat_X) mod else stub`.
3. Use `.empty` not `.{}` for `ArrayListUnmanaged` / `HashMapUnmanaged` init (Zig 0.16).
4. Both `mod.zig` and `stub.zig` must be updated together — run `zig build check-parity`.
5. Use `foundation.time.unixMs()` not `std.time.milliTimestamp` (removed in 0.16).
6. Use `foundation.sync.Mutex` not `std.Thread.Mutex` (may be unavailable).
7. Never run `zig fmt .` at root — use `zig build fix` (scoped to `src/`, `build.zig`, `build/`, and `test/`).
8. All path imports require explicit `.zig` extensions.
9. `var` vs `const`: compiler enforces const for never-mutated locals.
10. On macOS 26.4+, use `./build.sh` instead of `zig build` (LLD linker issue).

## Architecture

- **Entrypoint**: `src/root.zig` re-exports all domains as `abi.<domain>`
- **Features**: 20 directories under `src/features/`, 32 features total (including AI sub-features)
- **Mod/Stub pattern**: each feature has `mod.zig` (real), `stub.zig` (no-op), `types.zig` (shared)
- **Comptime gating** in `root.zig`: `if (build_options.feat_gpu) mod else stub`
- **Core**: `src/core/` (config, errors, registry, feature catalog)
- **Foundation**: `src/foundation/` (logging, security, time, SIMD, sync)
- **Runtime**: `src/runtime/` (task scheduling, event loops)
- **Connectors**: `src/connectors/` (OpenAI, Anthropic, Discord, etc.)
- **Protocols**: `src/protocols/` (mcp/, lsp/, acp/, ha/) — all comptime-gated via `feat_mcp`, `feat_lsp`, `feat_acp`, `feat_ha`
- **Inference**: `src/inference/` (multi-backend ML engine)
- **Feature catalog**: `src/core/feature_catalog.zig` (canonical feature metadata)
- **Stub helpers**: `src/core/stub_helpers.zig` (reuse `StubFeature`, `StubContext` in stubs)

## Testing

- **Unit tests**: `src/root.zig` uses `refAllDecls` to walk all `test` blocks in `src/`
- **Integration tests**: `test/mod.zig` imports `@import("abi")` as external consumer
- Add new integration tests by importing them from `test/mod.zig`
- Both suites link macOS frameworks: System, IOKit, Accelerate, Metal, objc

## Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui`. Disable with `-Dfeat-<name>=false`. GPU backends: `-Dgpu-backend=metal` or `-Dgpu-backend=cuda,vulkan`.

## Import Rules

- **Within `src/`**: relative imports only (`@import("../../foundation/mod.zig")`)
- **From `test/`**: use `@import("abi")` and `@import("build_options")`
- **Cross-feature**: comptime gate, never import another feature's `mod.zig` directly
