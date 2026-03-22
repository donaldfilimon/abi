# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zig 0.16 framework for AI services, vector search, and GPU compute. Pinned to `0.16.0-dev.2962+08416b44f` (`.zigversion`). Package entrypoint: `src/root.zig`, exposed as `@import("abi")`.

## Commands

See [docs/guides/zig-validation.md](docs/guides/zig-validation.md) for the full command reference, Darwin 25+ workarounds, Zig 0.16 API changes, benchmarks, and CLI quick reference.

Key commands:

```bash
zig build test --summary all          # primary tests
zig build full-check                  # pre-commit gate
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/  # format check (always works)
```

## Architecture

- `src/root.zig` — public package root; all `abi.<domain>` namespaces defined here
- `src/features/<name>/` — 20 comptime-gated imports across 19 feature directories, each with `mod.zig` + `stub.zig` + `types.zig` (`pages` is a sub-feature of `observability`)
- `src/services/` — Non-gated services: connectors, LSP, MCP, runtime, security, platform
- `src/services/shared/` — Shared foundations exposed as `abi.foundation` (logging, security, time/SIMD)
- `src/core/` — Config, feature catalog, registry (incl. plugin system), `stub_context.zig`
- `src/inference/` — ML inference: engine, scheduler, sampler (pluggable top-p/top-k strategies), paged KV cache
- `build/` — Modular build system:
  - `options.zig` — 27 `feat_*` flag definitions (`CanonicalFlags`)
  - `flags.zig` — 60-combo validation matrix
  - `modules.zig` — Module creation, `wireAbiImports()`
  - `module_catalog.zig` — Gendocs module registry (35 entries), feature test manifest (179 entries)
  - `link.zig` — Platform linking (macOS, Linux, Windows, BSD, Android, illumos, Haiku)
  - `test_discovery.zig` — Unified `abi` module test root
  - `cli_tests.zig` — CLI smoke (~53 vectors) and exhaustive integration tests
- `tools/cli/` — CLI commands and registry (`tools/cli/registry/`)
- `tools/gendocs/` — Documentation generator (edits go here, not in generated `docs/api/`)
- `bindings/` — C and WASM language bindings (C bindings include plugin registry API)
- `lang/` — High-level language bindings (Swift, Kotlin); wraps `bindings/c/`
- `tests/integration/` — Integration test matrix manifest and preflight diagnostics
- `examples/` — 36 standalone programs demonstrating API usage across all feature domains
- `zig-abi-plugin/` — Claude Code plugin: smart build routing, stub-sync validation, Zig 0.16 pattern checks, feature scaffolding (`/zig-abi:build`, `/zig-abi:check`, `/zig-abi:new-feature`)

### Public API surface

Top-level namespaces exported by `src/root.zig` (see source for full list):

- **Core:** `abi.config`, `abi.errors`, `abi.registry`, `abi.framework`
- **Services (non-gated):** `abi.foundation`, `abi.runtime`, `abi.platform`, `abi.connectors`, `abi.tasks`, `abi.mcp`, `abi.lsp`, `abi.acp`, `abi.ha`, `abi.inference`
- **Features (comptime-gated, 20 total):** `abi.gpu`, `abi.ai`, `abi.database`, `abi.network`, `abi.observability`, `abi.web`, `abi.pages`, `abi.analytics`, `abi.cloud`, `abi.auth`, `abi.messaging`, `abi.cache`, `abi.storage`, `abi.search`, `abi.mobile`, `abi.gateway`, `abi.benchmarks`, `abi.compute`, `abi.documents`, `abi.desktop`
- **Convenience:** `abi.App`, `abi.AppBuilder`, `abi.Gpu`, `abi.GpuBackend`, `abi.appBuilder()`, `abi.version()`, `abi.feature_catalog`

### Feature gating in root.zig

Features use comptime conditional imports:
```zig
pub const ai = if (build_options.feat_ai) @import("features/ai/mod.zig") else @import("features/ai/stub.zig");
```

### mod/stub contract

`stub.zig` must match `mod.zig` public signatures exactly. Shared types go in `types.zig` — both mod and stub import from it. Use `StubFeature`/`StubFeatureNoConfig` from `core/stub_context.zig` for common stub boilerplate. Sub-module stubs not needed, but CLI-accessed sub-modules must be re-exported from both mod and stub. Parity is enforced at compile time by `zig build check-stub-parity` (see `src/feature_parity_tests.zig`).

### foundation namespace

`abi.foundation` is **not** a separate named module. It's `src/services/shared/mod.zig` re-exported via `src/root.zig`. Files within the `abi` module reach it via relative paths (e.g., `@import("../../services/shared/mod.zig")`). External modules access it through `@import("abi").foundation`. `wireAbiImports(module, build_opts)` adds only `build_options` as a named import — foundation is wired through the normal `abi` module import graph, not as a separate named module.

## Import Rules (Critical)

- **External code** (CLI, tests with separate roots): use `@import("abi")`
- **Within `src/`** (part of the `abi` module): use **relative imports only**. `@import("abi")` from within the module causes a circular "no module named 'abi' available within module 'abi'" error.
- **Cross-feature imports**: never import another feature's `mod.zig` directly (bypasses the gate). Use `build_options` conditional: `const obs = if (build_options.feat_profiling) @import("../../observability/mod.zig") else @import("../../observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16)
- **Single-module file ownership**: every `.zig` file belongs to exactly one named module
- **Test roots**: `src/services/tests/mod.zig` is a separate test root (see Testing Patterns for details). Its child files must use `@import("abi")`, not relative imports
- **Build options stub**: when adding new `feat_*` flags to `build/options.zig`, also update `tools/cli/tests/build_options_stub.zig` to match

## Conventions

See [AGENTS.md](AGENTS.md) for the full contributor workflow contract, coding style, and commit conventions.

Key rules:
- `zig fmt` only — never `zig fmt .` from root (walks vendored fixtures that intentionally contain invalid code)
- `lower_snake_case` functions/files, `PascalCase` types/error sets
- Conventional commits (`fix:`, `feat:`, `docs:`, `chore:`, `style:`, `refactor:`), atomic scope
- Explicit error sets, propagate with `try`

## Testing Patterns

- **Unit tests**: inline `test` blocks within source files, run via `zig build test --summary all`
- **Feature tests**: `zig build feature-tests --summary all` exercises all 20 comptime-gated imports
- **Integration tests**: `tests/integration/` with manifest and preflight diagnostics
- **Full check**: `full-check` runs typecheck + test + check-docs + check-cli + validate-flags + check-feature-catalog
- **Test helpers**: `src/services/tests/helpers.zig` provides `TestAllocator` (leak detection), platform-aware skip (`skipIfNoGpu`), vector utilities, temp dir management
- **Test roots**: `src/services/tests/mod.zig` is a separate test root with named imports injected by `build.zig`. Its child files must use `@import("abi")`, not relative imports — switching to relative `src/root.zig` imports creates duplicate module ownership (`abi` and `root`) during `zig build test`. Test discovery uses `test {}` blocks with force-references (`_ = abi.ai.llm.io;`) to include submodule tests in the runner

## Workflow

See [AGENTS.md](AGENTS.md) for the full workflow and [docs/guides/zig-validation.md](docs/guides/zig-validation.md) for the verification gate table.

1. Review `tasks/lessons.md` at session start
2. Plan multi-file changes in `tasks/todo.md`
3. Run strongest available verification gate before completing
4. Update `stub.zig` when changing `mod.zig` signatures
5. Update `tasks/lessons.md` after corrections
6. Update `src/core/feature_catalog.zig` and regenerate artifacts before updating feature count references elsewhere
7. Version pin changes: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically

## Plugin System

External modules register at runtime via `abi.registry.plugin.PluginRegistry`. Plugins declare capabilities (`ai_provider`, `connector`, `storage_backend`, `gpu_backend`, etc.) and follow a `registered → loading → active → unloading` lifecycle. C bindings available in `bindings/c/include/abi.h` (`abi_plugin_register`, etc.). New C binding exports follow the opaque handle pattern: `FooHandle = opaque {}`, `FooWrapper` struct, `export fn` with integer return codes.

## Raft Consensus

`src/features/network/raft.zig` implements Raft with pre-vote protocol (prevents disruptive elections from partitioned nodes) and partition tolerance (leader steps down on quorum loss). Fault injection via `FaultInjector` for testing. Gated by `feat-network`.

## Markdown / .gitignore

`*.md` is globally ignored. Tracked markdown files require explicit `!/path.md` entries in `.gitignore`. When adding new `.md` files: create the file, add allowlist entry, verify with `git status`.

## Common Pitfalls & Troubleshooting

| Symptom / Mistake | Prevention & Fix |
|-------------------|-----------------|
| Undefined symbols (`_malloc_size`, etc.) on macOS | Stock Zig LLD fails on Darwin 25+. Use pinned Zig matching `.zigversion` on PATH, or format-check only (`zig fmt --check ...`) |
| "no module named 'abi' available within module 'abi'" | Never use `@import("abi")` inside `src/` — use relative imports. Only external code (CLI, tests) uses `@import("abi")` |
| "missing struct field: items" on unmanaged collections | Use `.empty` not `.{}` for `ArrayListUnmanaged`/`AutoHashMapUnmanaged` (Zig 0.16 change) |
| Stub compilation fails when feature disabled | `stub.zig` must match `mod.zig` signatures. Run `zig build test -Dfeat-<name>=false --summary all` to verify |
| CLI command can't find sub-module | CLI-accessed sub-modules must be re-exported from both `mod.zig` AND `stub.zig` |
| Cross-feature compile fails when feature disabled | Never import another feature's `mod.zig` directly — use conditional import with `build_options` |
| Import "file not found" after adding `.zig` suffix | Verify target file exists first — the real fix may be a gated import or different path |
| Hooks rewrite imports/ordering destructively | Restore with `git checkout HEAD -- <file>`. Use atomic `sed -i '' + git add` for edits that must survive hooks |
| `zig fmt .` fails on vendored code | Always specify explicit paths: `zig fmt build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` |
| Gendocs/registry out of sync | Update `src/core/feature_catalog.zig` first, then `zig build gendocs` + `zig build refresh-cli-registry` |
| Cross-directory imports in `build/` or `tools/` fail | Files in `build/` must compile standalone. Tool-side modules can't reach `../../build/*.zig` — pass shared metadata as named module imports from `build.zig` |
| Wrong Zig toolchain on Darwin 25+ | Use pinned Zig matching `.zigversion` on PATH. Format checks always work as fallback |

## References

- [AGENTS.md](AGENTS.md) — Workflow contract and verification gates
- [docs/guides/zig-validation.md](docs/guides/zig-validation.md) — Zig toolchain, commands, API changes, benchmarks
- [tasks/lessons.md](tasks/lessons.md) — Correction log (prevents repeat mistakes)
- [docs/PATTERNS.md](docs/PATTERNS.md) — Zig 0.16 codebase patterns
- [docs/STRUCTURE.md](docs/STRUCTURE.md) — Full directory tree reference
- [docs/guides/integration-environment.md](docs/guides/integration-environment.md) — CI/local environment contract
- [docs/plans/index.md](docs/plans/index.md) — Active execution plans and status
