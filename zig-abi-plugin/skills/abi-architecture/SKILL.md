# ABI Architecture & Module Guidelines

## Framework Overview

ABI is a modular Zig 0.16 framework designed for distributed AI systems and high-performance vector operations. It utilizes a feature-gate architecture controlled via `build.zig` flags.

## Module Structure

- **`src/root.zig`**: Canonical public library interface (exported as the `abi` module).
- **`src/core/`**: Non-optional framework components.
  - **`src/core/database/`**: Unified vector database (WDBX V3, exported as `wdbx` module).
  - **`src/core/framework/`**: Application lifecycle and registry.
- **`src/features/`**: Opt-in modular functionality (AI, GPU, Web, Cloud).
- **`tools/`**: Internal maintenance tools and executable roots.
  - **`tools/cli/`**: The `abi` CLI.
  - **`tools/server/`**: The REST API server.

## Import Rules

- External consumers: `@import("abi")`
- Feature modules in `src/features/`: **relative imports only** — never `@import("abi")`.
- **Named modules** (registered in `build.zig`): Use `@import("wdbx")`, `@import("build_options")`, etc. — NOT relative paths like `@import("../../core/database/wdbx.zig")`.
- Enforced by `zig build check-imports`.
- Files in `build/test_discovery.zig` must compile standalone with `zig test <file> -fno-emit-bin`.
- Cross-directory `@import("../../")` breaks standalone compilation — inline small deps instead.

### Module Conflict Rule (Critical)
A Zig file can only belong to ONE module. If `src/core/database/wdbx.zig` is the root of the named `wdbx` module, importing it via relative path from `src/features/` claims it for the `abi` module too — causing a compile error. Always use the named import: `@import("wdbx")`.

## Feature Flag Conventions

- Prefix: `feat_<name>` (NOT `enable_<name>`)
- All default to `true`.
- Flags are exported via the `build_options` module.

## Key Module Map

| Public Path | Entry File | Purpose |
|-------------|-----------|---------|
| `abi.App` / `abi.AppBuilder` | `src/root.zig` | Primary runtime |
| `abi.database` | `src/core/database/wdbx.zig` | Unified vector engine |
| `abi.database.dist` | `src/core/database/distributed/mod.zig` | Distributed coordinator |
| `abi.features.ai` | `src/features/ai/mod.zig` | LLM, agents, training |
| `abi.features.profiles` | `src/features/ai/profiles/mod.zig` | Behavior interaction |
| `abi.inference_engine` | `src/inference/engine.zig` | Token generation |
| `abi.server` | `tools/server/main.zig` | REST API |

## Performance & SIMD

- Use `abi.simd` for portable hardware acceleration.
- Prefer `std.Io` patterns for non-blocking I/O.
- Target `native` by default, but validate for `wasm32-wasi`.
