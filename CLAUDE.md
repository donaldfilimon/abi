# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion`. The zig version manager auto-downloads the correct version:

```bash
tools/zigup.sh --status    # Print zig path (auto-install if missing)
tools/zigup.sh --link      # Symlink zig + zls into ~/.local/bin
# Also: --install, --unlink, --update, --check, --clean
```

Cross-compilation helper:
```bash
tools/crossbuild.sh        # Cross-compile for linux, wasi, x86_64 targets
```

Build Zig from source (Codeberg mirror, requires `brew install llvm`):
```bash
tools/compile_zig_codeberg.sh  # Compile Zig from master via Codeberg mirror
```

Auto-update checker:
```bash
tools/auto_update.sh       # Check and apply updates for zig + zls
```

Cache location: `~/.cache/abi-zig/<version>/bin/{zig,zls}`

To make zig and zls available globally, run `tools/zigup.sh --link` which symlinks them into `~/.local/bin`. Ensure `~/.local/bin` is on your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Build Commands

```bash
./build.sh                         # Build (macOS 26.4+ auto-relinks with Apple ld)
./build.sh --link lib              # Build and symlink zig+zls to ~/.local/bin
./build.sh test --summary all      # Run tests via wrapper (macOS 26.4+)
zig build                          # Build static library (Linux / older macOS)
zig build test --summary all       # Run tests (src/ + test/)
zig build check                    # Lint + test + stub parity (full gate)
zig build lint                     # Check formatting (read-only)
zig build fix                      # Auto-format in place
zig build check-parity             # Verify mod/stub declaration parity
zig build feature-tests            # Run feature integration and parity tests
zig build cli-tests                # Run CLI tests
zig build tui-tests                # Run TUI tests
zig build typecheck                # Compile-only validation for the current/selected target
zig build validate-flags           # Validate feature flags
zig build full-check               # Run full check
zig build verify-all               # Verify all components
zig build cross-check              # Verify cross-compilation (linux, wasi, x86_64)
zig build lib                      # Build static library artifact
zig build mcp                      # Build MCP stdio server (zig-out/bin/abi-mcp)
zig build cli                      # Build ABI CLI binary (zig-out/bin/abi)
zig build doctor                   # Report build configuration and diagnostics
```

Do NOT run `zig fmt .` at the repo root — use `zig build fix` which scopes to `src/` and `build.zig`.

### CLI Commands

Build with `zig build cli` (or `./build.sh cli`). Binary: `zig-out/bin/abi`.

```bash
abi                    # Smart status (feature count, enabled/disabled tags)
abi version            # Version and build info
abi doctor             # Build config report (all feature flags + GPU backends)
abi features           # List all 30 features from catalog with [+]/[-] status
abi platform           # Platform detection (OS, arch, CPU, GPU backends)
abi connectors         # List 16 LLM provider connectors with env vars
abi info               # Framework architecture summary
abi chat <message...>  # Route through multi-persona pipeline
abi db <subcommand>    # Vector database (add, query, stats, diagnostics, optimize, backup, restore, serve)
abi serve              # Start ACP HTTP server (default 127.0.0.1:8080)
abi acp serve          # Same as above (explicit ACP prefix)
abi dashboard          # Interactive TUI (requires -Dfeat-tui=true)
abi help               # Full help reference
```

On macOS 26.4+ (Darwin 25.x), stock prebuilt Zig's LLD linker cannot link binaries. Use `./build.sh` which auto-relinks with Apple's native linker. This applies to **all** build steps including tests: `./build.sh test --summary all`. On Linux / older macOS, `zig build` works directly.

### Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui` (both false). Disable with `-Dfeat-<name>=false`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

The build.zig is self-contained with all feature flags defined inline. No external build modules.

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle, feature catalog
- `src/features/` — 20 feature directories under src/features/ (30 features total including AI sub-features in the catalog)
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord, etc.)
- `src/tasks/` — Task management, async job queues
- `src/protocols/` — Protocol implementations: mcp/, lsp/, acp/, ha/
- `src/inference/` — ML inference: engine, scheduler, sampler, paged KV cache
- `src/core/database/` — Vector database implementation (consumed by features/database/ facade)
- `src/main.zig` — CLI entry point (builds as `abi` binary)
- `src/mcp_main.zig` — MCP stdio server entry point (builds as `abi-mcp` binary)
- `src/ffi.zig` — C-ABI FFI endpoints for linking as a static library (`libabi.a`)
- `test/` — Integration tests via `test/mod.zig` (uses `@import("abi")`, separate from unit tests in `src/`)

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

Note: `pages` is nested under `src/features/observability/pages/` (not its own top-level feature dir), but is gated by `feat_pages` independently from `feat_profiling`.

The mod/stub pattern also applies to protocols: `mcp` and `lsp` are comptime-gated via `feat_mcp` and `feat_lsp` in `root.zig`, with stubs at `src/protocols/{mcp,lsp}/stub.zig`.

Empty `struct {}` sub-module stubs are acceptable when the important types are re-exported at the stub's top level. Only expand sub-module stubs when external code accesses types through the sub-module namespace.

### Convenience Aliases in root.zig

- `abi.meta.package_version` / `abi.meta.version()` — version string from build options
- `abi.meta.features` — re-exports `src/core/feature_catalog.zig`
- `abi.app.App` / `abi.app.AppBuilder` / `abi.app.builder(allocator)` — framework lifecycle wrappers around `abi.framework`

### Build Options

The `build_options` module provides these fields (all `bool` unless noted):
- Feature flags: `feat_gpu`, `feat_ai`, `feat_database`, `feat_network`, `feat_profiling`, `feat_web`, `feat_pages`, `feat_analytics`, `feat_cloud`, `feat_auth`, `feat_messaging`, `feat_cache`, `feat_storage`, `feat_search`, `feat_mobile`, `feat_gateway`, `feat_benchmarks`, `feat_compute`, `feat_documents`, `feat_desktop`, `feat_tui`
- AI sub-features: `feat_llm`, `feat_training`, `feat_vision`, `feat_explore`, `feat_reasoning` (all require parent `feat_ai`; disabling `feat_ai` disables all sub-features)
- Protocols: `feat_lsp`, `feat_mcp`
- GPU backends: `gpu_metal`, `gpu_cuda`, `gpu_vulkan`, `gpu_webgpu`, `gpu_opengl`, `gpu_opengles`, `gpu_webgl2`, `gpu_stdgpu`, `gpu_fpga`, `gpu_tpu`
- `package_version` (`[]const u8`)

### GPU Backend Status

| Backend | Status | Notes |
|---------|--------|-------|
| Metal | Functional | macOS only, MPS acceleration, full compute pipeline |
| CUDA | Functional | NVIDIA GPUs, dynamic library loading |
| Vulkan | Functional | Cross-platform, full pipeline/descriptor management |
| stdgpu | Functional | CPU-based SPIR-V emulation (default, headless-safe) |
| WebGPU | Partial | API structure present, dynamic library loading |
| OpenGL | Partial | Compute shaders (GL 4.3+), 35+ function pointers |
| OpenGL ES | Partial | Mobile/embedded (GLES 3.1+) |
| WebGL2 | Stub | No compute shader support — returns error on all ops |
| DirectML | Stub | Windows-only, minimal implementation |
| FPGA | Stub | Simulation mode only, kernel modules not wired |

### Test Architecture

Two test suites run under `zig build test`:
1. **Unit tests** (`src/root.zig`) — `refAllDecls` walks the entire module tree, running `test` blocks in every reachable `.zig` file under `src/`.
2. **Integration tests** (`test/mod.zig`) — imports `@import("abi")` as an external consumer. Add new integration test files by importing them from `test/mod.zig`.

Both suites link the same platform frameworks (macOS: System, IOKit, Accelerate, Metal, objc).

To add a new integration test:
1. Create `test/integration/<name>_test.zig`
2. Import it from `test/mod.zig` (e.g., `const foo_tests = @import("integration/foo_test.zig");`)
3. Use `@import("abi")` and `@import("build_options")` — never relative imports from `test/`

### MCP Server

`zig build mcp` produces `zig-out/bin/abi-mcp`, a JSON-RPC 2.0 stdio server exposing database and ZLS tools for Claude Desktop, Cursor, etc. Entry point: `src/mcp_main.zig`.

### Multi-Persona Pipeline (Abbey-Aviva-Abi)

The full pipeline is wired end-to-end in `src/features/ai/persona/router.zig`:
```
User Input → Abi Analysis (sentiment + policy + rules)
  → AdaptiveModulator (EMA user preference learning)
  → Routing Decision (single / parallel / consensus)
  → Persona Execution (Abbey / Aviva / Abi)
  → Constitution Validation (6 principles)
  → WDBX Memory Storage (cryptographic block chain)
  → Response
```

Key files: `persona/router.zig` (orchestration), `persona/memory.zig` (WDBX storage), `abi/mod.zig` (routing), `modulation.zig` (preference learning), `constitution/mod.zig` (ethical enforcement).

### Inference Engine

Multi-backend engine (`src/inference/engine.zig`) supports:
- `demo` — synthetic text for testing (default)
- `connector` — delegates to external LLM providers (OpenAI, Anthropic, Ollama, etc.)
- `local` — built-in transformer forward pass (integration point for GGUF loading)

### Specification

`docs/spec/ABBEY-SPEC.md` — comprehensive mega spec covering architecture, personas, behavioral model, math foundations, ethics, benchmarks, implementation status, and visual assets.

## Import Rules

- **Within `src/`**: use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside the module — causes circular "no module named 'abi'" error.
- **From `test/`**: use `@import("abi")` and `@import("build_options")` — these are wired as named module imports by build.zig.
- **Cross-feature imports**: never import another feature's `mod.zig` directly (bypasses the comptime gate). Use conditional: `const obs = if (build_options.feat_profiling) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16).

## Key Conventions

- The public surface is `abi.<domain>` (e.g., `abi.gpu`, `abi.ai`, `abi.database`). Use `src/root.zig` as the single source of truth for what's exported.
- Struct field renames: grep for `.field_name` (with leading dot) to catch anonymous struct literals that won't match `StructName{` searches.
- `src/core/feature_catalog.zig` is the canonical source of truth for feature metadata.
- `src/core/stub_helpers.zig` provides `StubFeature`, `StubContext`, and `StubContextWithConfig` — reuse these in stubs instead of defining custom lifecycle boilerplate.
- Integration tests in `test/` must use public API accessors (e.g., `manager.getStatus()`) not direct struct field access. This preserves the consumer-API boundary and thread-safety contract.

### Error Handling Convention

- `@compileError` — compile-time contract violations only (e.g., `target_contract.zig` policy enforcement)
- `@panic` — unrecoverable invariant violations; never in library code (`src/`), only in CLI entry points (`src/main.zig`) and tests
- `unreachable` — provably impossible branches where the compiler can verify exhaustiveness at comptime
- Error unions — all runtime failure paths in library code; prefer `error.FeatureDisabled` in stubs

## Zig 0.16 Gotchas

- `ArrayListUnmanaged` init: use `.empty` not `.{}` (struct fields changed)
- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- `var` vs `const`: compiler enforces const for never-mutated locals
- Function pointers: can call through `*const fn` directly without dereferencing
- Entry points use `pub fn main(init: std.process.Init) !void` (not the older `pub fn main() !void`). Access args via `init.minimal.args`, allocator via `init.gpa` or `init.arena`.
- `zig fmt .` from root: don't — use `zig build fix` to avoid vendored fixtures
- IO operations: use `std.Io.Threaded` + `std.Io.Dir.cwd()` pattern (not the removed `std.fs.cwd()`)
- `extern` declarations in platform-gated structs: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`, not just OS. Otherwise symbols leak into feature-disabled builds (ref: `accelerate.zig` fix).
- `foundation.time.timestampSec()` is monotonic from process start — returns 0 in the first second. Use `std.posix.system.clock_gettime(.REALTIME, ...)` for wall-clock timestamps in persisted data.

## Skill Overrides

- **brainstorming**: For this Zig codebase, skip the full brainstorming workflow for: single-file bug fixes, stub parity fixes, import path updates, and Zig 0.16 migration changes. Use brainstorming only for new features, new modules, or architectural changes.
- **writing-skills / skill-creator**: For this project, keep skills concise. Follow the patterns in `.claude/skills/` as examples of well-scoped project skills.
