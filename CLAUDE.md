# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime. The package entrypoint is `src/root.zig`, exposed as `@import("abi")`.

Zig version is pinned in `.zigversion` (currently `0.16.0-dev.2984+cb7d2b056`). The zig version manager auto-downloads the correct version:

```bash
tools/zigup.sh --status    # Print zig path (auto-install if missing)
tools/zigup.sh --link      # Symlink zig + zls into ~/.local/bin
tools/zigup.sh --bootstrap # One-command project setup (install, link, verify)
tools/zigup.sh --doctor    # Toolchain health check (versions, PATH, platform)
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

`zigup.sh` also detects system-installed zig from zvm (`~/.zvm/bin/zig`) or brew, preferring those over its own cache when the version matches `.zigversion`. If zvm is installed, `zvm install master` provides both zig and zls.

To make zig and zls available globally, run `tools/zigup.sh --link` which symlinks them into `~/.local/bin`. Ensure `~/.local/bin` is on your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Quick Verify (fresh clone)

```bash
tools/zigup.sh --bootstrap         # Install zig + zls, symlink, verify
./build.sh test -Dfeat-gpu=false --summary all  # macOS 26.4+
# or: zig build test --summary all               # Linux / older macOS
```

### Claude Code Plugin

This repository includes a Claude Code plugin at `zig-abi-plugin/` providing:
- Build troubleshooting skill for linker errors and platform issues
- Zig 0.16 patterns skill for API guidance
- Feature scaffolding skill for new modules
- Pre/post tool hooks for validation

Install with: `claude --plugin-dir zig-abi-plugin`

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
zig build mcp-tests                # Run MCP integration tests
zig build cli-tests                # Run CLI tests
zig build tui-tests                # Run TUI tests
# Focused feature test lanes (27 total — each runs unit + integration tests):
zig build acp-tests                # ACP protocol tests
zig build agents-tests             # Agents tests
zig build auth-tests               # Auth tests
zig build cache-tests              # Cache tests
zig build cloud-tests              # Cloud tests
zig build compute-tests            # Compute tests
zig build connectors-tests         # Connectors tests
zig build database-tests           # Database tests
zig build desktop-tests            # Desktop tests
zig build documents-tests          # Documents tests
zig build gateway-tests            # Gateway tests
zig build gpu-tests                # GPU tests
zig build ha-tests                 # HA protocol tests
zig build inference-tests          # Inference tests
zig build lsp-tests                # LSP protocol tests
zig build messaging-tests          # Messaging tests
zig build multi-agent-tests        # Multi-agent tests
zig build network-tests            # Network tests
zig build observability-tests      # Observability tests
zig build orchestration-tests      # Orchestration tests
zig build pipeline-tests           # Pipeline DSL tests
zig build pitr-tests               # PITR tests
zig build search-tests             # Search tests
zig build secrets-tests            # Secrets tests
zig build storage-tests            # Storage tests
zig build tasks-tests              # Tasks tests
zig build web-tests                # Web tests
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

Do NOT run `zig fmt .` at the repo root — use `zig build fix` which scopes to `src/`, `build.zig`, `build/`, and `test/`.

### CLI Commands

Build with `zig build cli` (or `./build.sh cli`). Binary: `zig-out/bin/abi`.

```bash
abi                    # Smart status (feature count, enabled/disabled tags)
abi version            # Version and build info
abi doctor             # Build config report (all feature flags + GPU backends)
abi features           # List all 35 features from catalog with [+]/[-] status
abi platform           # Platform detection (OS, arch, CPU, GPU backends)
abi connectors         # List 12 LLM provider connectors with env vars
abi info               # Framework architecture summary
abi chat <message...>  # Route through multi-profile pipeline
abi db <subcommand>    # Vector database (add, query, stats, diagnostics, optimize, backup, restore, serve)
abi serve              # Start ACP HTTP server (default 127.0.0.1:8080)
abi acp serve          # Same as above (explicit ACP prefix)
abi dashboard          # Interactive TUI (requires -Dfeat-tui=true)
abi help               # Full help reference
```

`./build.sh` is a macOS 26.4+ (Darwin 25.x) wrapper that patches Zig's LLD linker incompatibility with the macOS SDK. It passes all arguments through to `zig build` (e.g., `./build.sh test --summary all`, `./build.sh -Dfeat-gpu=false`). The `--link` flag additionally symlinks zig+zls to `~/.local/bin`. On Linux / older macOS, `zig build` works directly.

**Why `build.sh` is required (not optional) on macOS 26.4+:** Zig compiles `build.zig` into a "build runner" binary *before* `build()` runs, using its internal LLD. On Darwin 25.x, LLD can't resolve system symbols from SDK `.tbd` files (arm64e vs arm64 mismatch). Our `use_lld = false` settings only affect artifacts created inside `build()`, not the build runner itself. This is a Zig compiler limitation — `build.sh` works around it by relinking the build runner with Apple's `/usr/bin/ld`.

### Running Single Tests

```bash
# Run a specific test by name pattern
zig build test --summary all -- --test-filter "test_name_pattern"

# On macOS 26.4+:
./build.sh test --summary all -- --test-filter "test_name_pattern"
```

Focused test lanes (e.g., `zig build messaging-tests`) are listed in Build Commands above.

### Feature Flags

All features default to enabled except `feat-mobile` and `feat-tui` (both false). Disable with `-Dfeat-<name>=false`:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
```

The build system is split across `build.zig` (root) and `build/` helpers:
- `build/flags.zig` — `FeatureFlags` struct, `hasBackend()`, `addAllBuildOptions()`
- `build/cross.zig` — cross-compilation targets (typecheck, cross-check steps)
- `build/linking.zig` — `linkDarwinArtifact()` for macOS framework linking
- `build/validation.zig` — test, parity, feature-test, and MCP-test step wiring. Uses `addFeatureTestLane()` helper for all 26 feature-specific test steps

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle, feature catalog
- `src/features/` — 20 feature directories under src/features/ (35 features total including AI sub-features in the catalog)
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord, etc.) — comptime-gated by `feat_connectors`
- `src/tasks/` — Task management, async job queues — comptime-gated by `feat_tasks`
- `src/protocols/` — Protocol implementations: mcp/, lsp/, acp/, ha/
- `src/inference/` — ML inference: engine, scheduler, sampler, paged KV cache — comptime-gated by `feat_inference`
- `src/core/database/` — Vector database implementation (consumed by features/database/ facade)
- `src/main.zig` — CLI entry point (builds as `abi` binary)
- `src/mcp_main.zig` — MCP stdio server entry point (builds as `abi-mcp` binary)
- `src/ffi.zig` — C-ABI FFI endpoints for linking as a static library (`libabi.a`)
- `build/` — Build helpers: flags, cross-compilation, linking, validation (imported by `build.zig`)
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

Note: `pages` is nested under `src/features/observability/pages/` (not its own top-level feature dir), but is gated by `feat_pages` independently from `feat_observability`.

The mod/stub pattern also applies to protocols: `mcp`, `lsp`, `acp`, and `ha` are comptime-gated via `feat_mcp`, `feat_lsp`, `feat_acp`, and `feat_ha` in `root.zig`, with stubs at `src/protocols/{mcp,lsp,acp,ha}/stub.zig`.

Empty `struct {}` sub-module stubs are acceptable when the important types are re-exported at the stub's top level. Only expand sub-module stubs when external code accesses types through the sub-module namespace.

### AI Sub-Features

The `ai` feature (`src/features/ai/`) contains 33+ sub-directories organized by group:
- **Inference:** `llm/`, `embeddings/`, `vision/`, `models/`, `streaming/`
- **Reasoning:** `abbey/`, `aviva/`, `abi/`, `constitution/`, `eval/`, `reasoning/`
- **Agents:** `agents/`, `tools/`, `multi_agent/`, `coordination/`, `orchestration/`
- **Learning:** `training/`, `memory/`, `federated/`
- **Support:** `templates/`, `prompts/`, `documents/`, `profiles/`, `context_engine/`
- **Pipeline:** `pipeline/` (composable prompt DSL with WDBX-backed steps)
- **Standalone:** `modulation.zig` (EMA preference learning), `self_improve.zig`, `profile/` (router pipeline)

### Convenience Aliases in root.zig

- `abi.meta.package_version` / `abi.meta.version()` — version string from build options
- `abi.meta.features` — re-exports `src/core/feature_catalog.zig`
- `abi.App` / `abi.AppBuilder` / `abi.appBuilder(allocator)` — framework lifecycle (shorthand for `abi.framework.Framework` etc.)
- `abi.version()` — shorthand for `abi.meta.version()`

### Build Options

The `build_options` module provides these fields (all `bool` unless noted):
- Feature flags: `feat_gpu`, `feat_ai`, `feat_database`, `feat_network`, `feat_observability`, `feat_web`, `feat_pages`, `feat_analytics`, `feat_cloud`, `feat_auth`, `feat_messaging`, `feat_cache`, `feat_storage`, `feat_search`, `feat_mobile`, `feat_gateway`, `feat_benchmarks`, `feat_compute`, `feat_documents`, `feat_desktop`, `feat_tui`, `feat_connectors`, `feat_tasks`, `feat_inference`
- AI sub-features: `feat_llm`, `feat_training`, `feat_vision`, `feat_explore`, `feat_reasoning` (all require parent `feat_ai`; disabling `feat_ai` disables all sub-features)
- Protocols: `feat_lsp`, `feat_mcp`, `feat_acp`, `feat_ha`
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
| TPU | Stub | Simulation mode only, tensor core abstraction |

### Test Architecture

Two test suites run under `zig build test`:
1. **Unit tests** (`src/root.zig`) — `refAllDecls` walks the entire module tree, running `test` blocks in every reachable `.zig` file under `src/`.
2. **Integration tests** (`test/mod.zig`) — imports `@import("abi")` as an external consumer. Add new integration test files by importing them from `test/mod.zig`.

Both suites link the same platform frameworks (macOS: System, IOKit, Accelerate, Metal, objc).

To add a new integration test:
1. Create `test/integration/<name>_test.zig`
2. Import it from `test/mod.zig` (e.g., `const foo_tests = @import("integration/foo_test.zig");`)
3. Use `@import("abi")` and `@import("build_options")` — never relative imports from `test/`

#### Focused Test Lanes

27 `src/*_mod_test.zig` files (acp, agents, auth, cache, cloud, compute, connectors, database, desktop, documents, gateway, gpu, ha, inference, lsp, messaging, multi_agent, network, observability, orchestration, pipeline, pitr, search, secrets, storage, tasks, web) are **test anchor** files. They sit at `src/` root so relative imports like `@import("features/messaging/mod.zig")` resolve correctly. Each anchor imports the feature's module and test file, then `refAllDecls` walks them. Corresponding `test/*_mod.zig` files (e.g., `test/messaging_mod.zig`) serve as integration test entry points for the same lane.

`build/validation.zig` wires each pair into a focused build step via `addModuleTests()` (unit, from `src/`) and `addIntegrationTests()` (integration, from `test/`). Both are combined under a single step like `zig build messaging-tests`.

To add a new focused test lane:
1. Create `src/<name>_mod_test.zig` — import the feature module and its tests, use `refAllDecls`
2. Create `test/<name>_mod.zig` — import integration tests via `@import("abi")`
3. Wire both in `build/validation.zig` following the existing pattern (unit + integration → named step)
4. Add the new step name to the `Steps` struct and return it from `addSteps()`
5. Document the `zig build <name>-tests` command in the Build Commands section above

### MCP Server

`zig build mcp` produces `zig-out/bin/abi-mcp`, a JSON-RPC 2.0 stdio server exposing database and ZLS tools for Claude Desktop, Cursor, etc. Entry point: `src/mcp_main.zig`.

### Multi-Profile Pipeline (Abbey-Aviva-Abi)

The full pipeline is wired end-to-end in `src/features/ai/profile/router.zig`:
```
User Input → Abi Analysis (sentiment + policy + rules)
  → AdaptiveModulator (EMA user preference learning)
  → Routing Decision (single / parallel / consensus)
  → Profile Execution (Abbey / Aviva / Abi)
  → Constitution Validation (6 principles)
  → WDBX Memory Storage (cryptographic block chain)
  → Response
```

Key files: `profile/router.zig` (orchestration), `profile/memory.zig` (WDBX storage), `abi/mod.zig` (routing), `modulation.zig` (preference learning), `constitution/mod.zig` (ethical enforcement).

### Abbey Dynamic Model (Pipeline DSL)

The pipeline DSL (`src/features/ai/pipeline/`) provides a composable, chainable alternative to the procedural router pipeline. Each step is a typed operation backed by WDBX blocks:
```zig
var builder = abi.ai.pipeline.chain(allocator, "session-123");
var p = builder
    .withChain(&wdbx_chain)
    .retrieve(.wdbx, .{ .k = 5 })
    .template("Given {context}, respond to: {input}")
    .route(.adaptive)
    .modulate()
    .generate(.{})
    .validate(.constitution)
    .store(.wdbx)
    .build();
const result = try p.run("Hello Abbey!");
```

Key files: `pipeline/mod.zig` (entry), `pipeline/builder.zig` (DSL), `pipeline/executor.zig` (runner), `pipeline/context.zig` (state), `pipeline/persistence.zig` (WDBX adapter), `pipeline/steps/` (10 step implementations). Gated by `feat_reasoning`. Focused test lane: `zig build pipeline-tests`.

The router also exposes `routeAndExecutePipeline()` which builds the standard pipeline and runs it. `AdaptiveModulator.attachWdbx()` enables write-behind persistence of modulation state to WDBX.

### ACP HTTP Server

`abi serve` (or `abi acp serve`) starts an HTTP server on `127.0.0.1:8080` exposing the Agent Communication Protocol. Entry: `src/protocols/acp/server/mod.zig`. Gated by `feat_acp`. The server wires together task management (`src/protocols/acp/server/tasks.zig`) with the ACP protocol layer.

The ACP server also includes:
- **Discord gateway bridge** (`server/discord_routes.zig`) — routes Discord interactions through ACP
- **OpenAPI 3.1.0 spec** (`server/openapi.zig`) — auto-generated API documentation
- **Rich route responses** (`server/routing.zig`) — structured JSON responses with metadata

### Inference Engine

Multi-backend engine (`src/inference/engine.zig`) supports:
- `demo` — synthetic text for testing (default)
- `connector` — resolves provider from `model_id` ("provider/model" format), loads config from env vars, delegates to connector clients (`src/inference/engine/backends.zig`)
- `local` — built-in transformer forward pass (integration point for GGUF loading)

The connector backend dispatches to 12 providers (openai, anthropic, ollama, mistral, cohere, gemini, mlx, huggingface, lm_studio, vllm, llama_cpp, codex). Provider config is loaded from environment variables via `src/connectors/loaders.zig`. Falls back to echo when env vars are missing.

The `abi chat` CLI command wires the profile router to the inference engine: routing decision → `Engine.generate()` → connector dispatch → response.

### Specification

`docs/spec/ABBEY-SPEC.md` — comprehensive mega spec covering architecture, profiles, behavioral model, math foundations, ethics, benchmarks, implementation status, and visual assets.

Additional specifications:
- `docs/spec/abbey-aviva-abi-framework.md` — Profile framework details
- `docs/spec/wdbx-technical-analysis.md` — WDBX storage technical analysis
- `docs/review/` — Code review summaries and improvement plans
- `docs/superpowers/` — Development plans and design specs

## Import Rules

- **Within `src/`**: use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside the module — causes circular "no module named 'abi'" error.
- **From `test/`**: use `@import("abi")` and `@import("build_options")` — these are wired as named module imports by build.zig.
- **Cross-feature imports**: never import another feature's `mod.zig` directly (bypasses the comptime gate). Use conditional: `const obs = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports (Zig 0.16).

## Key Conventions

- The public surface is `abi.<domain>` (e.g., `abi.gpu`, `abi.ai`, `abi.database`). Use `src/root.zig` as the single source of truth for what's exported.
- Struct field renames: grep for `.field_name` (with leading dot) to catch anonymous struct literals that won't match `StructName{` searches.
- `src/core/feature_catalog.zig` is the canonical source of truth for feature metadata.
- `src/core/stub_helpers.zig` provides `StubFeature`, `StubContext`, and `StubContextWithConfig` — reuse these in stubs instead of defining custom lifecycle boilerplate.
- Integration tests in `test/` must use public API accessors (e.g., `manager.getStatus()`) not direct struct field access. This preserves the consumer-API boundary and thread-safety contract.
- Use `linkIfDarwin()` from `build/linking.zig` instead of inline macOS checks — 13 callsites consolidated.
- AI sub-feature stubs under `src/features/ai/*/stub.zig` are domain-specific and intentionally don't use generic `stub_helpers.zig` helpers.
- For non-trivial tasks: read `tasks/lessons.md` and update `tasks/todo.md` before implementation. Keep `tasks/` for workflow notes only — do not confuse with `src/tasks/`.
- **Database engine thread safety**: every public `Engine` method must acquire `db_lock` before reading `vectors_array`, `hnsw_index`, `ai_client`, or `cache`.
- **JSON utilities**: use `foundation/utils/json.zig` for escaping — never reimplement in protocol-specific files (ACP, MCP, etc.).
- **AI pipeline memory**: string literals in `ProfileResponse.content` crash on `deinit` — always `allocator.dupe()` heap copies before storing.
- **Abbey emotion files**: `emotion.zig` and `emotions.zig` both exist — `emotions.zig` is canonical; don't import `emotion.zig`.

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
- `zig fmt .` from root: don't — use `zig build fix` (scopes to `src/`, `build.zig`, `build/`, `test/`)
- IO operations: use `std.Io.Threaded` + `std.Io.Dir.cwd()` pattern (not the removed `std.fs.cwd()`)
- `extern` declarations in platform-gated structs: gate on BOTH `build_options.feat_*` AND `builtin.os.tag`, not just OS. Otherwise symbols leak into feature-disabled builds (ref: `accelerate.zig` fix).
- `foundation.time.timestampSec()` is monotonic from process start — returns 0 in the first second. Use `std.posix.system.clock_gettime(.REALTIME, ...)` for wall-clock timestamps in persisted data.
- `std.process.getEnvVarOwned` removed: use `b.graph.environ_map.get("KEY")` in build.zig
- `std.mem.trimRight` renamed to `std.mem.trimEnd` in Zig 0.16

## Available Agents

The repository includes 8 specialized agents in `.claude/agents/`:

- **abi-stub-fixer** — Automatically updates `stub.zig` files when `mod.zig` public APIs change; runs `zig build check-parity` to verify
- **darwin-build-doctor** — Diagnoses Zig linker failures on macOS 25+ (Darwin Tahoe)
- **abi-expert** — General ABI framework guidance (mod/stub pattern, comptime gating, Zig 0.16 conventions)
- **stub-parity-reviewer** — Reviews mod.zig/stub.zig pairs for API mismatches before full build
- **feature-scaffolder** — Scaffolds new feature modules with mod.zig, stub.zig, types.zig, and build integration
- **build-troubleshooter** — Diagnoses Zig build failures (compile errors, type mismatches, import issues)
- **abi-test-writer** — Writes integration tests following ABI conventions (SkipZigTest guards, public API access, mod.zig wiring)
- **abbey-aviva-abi-architect** — Multi-persona AI pipeline expert (profile routing, persona modulation, constitution validation, WDBX memory)

Invoke these agents via the `Agent` tool with `subagent_type: "<agent-name>"`.

## Available Skills

The repository includes 6 skills in `.claude/skills/`:

- **lessons-review** — Reviews `tasks/lessons.md` for recurring pitfalls before starting work (Zig 0.16 API changes, mod/stub parity, macOS linker, thread safety)
- **stub-audit** — Verifies AI sub-feature stubs match their `mod.zig` public API and use `stub_helpers.zig` appropriately
- **cross-check** — Runs cross-compilation verification for linux, wasi, x86_64 targets; validates comptime feature gating
- **baseline-sync** — Tracks test pass/skip counts from test runs and reports drift from previous baselines
- **full-check** — One-command full validation gate: lint, parity, tests, feature-tests, cross-check
- **pre-commit-check** — Run lint + parity check before committing; catches 80% of CI failures locally

## Code Style

- Functions and variables: `camelCase`
- Types and structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Enum variants: `snake_case`
- Doc comments (`///`) on public API only — not on internal helpers
- GPU backends use a VTable pattern for backend-agnostic dispatch (see `src/features/gpu/`)

## Skill Overrides

- **brainstorming**: For this Zig codebase, skip the full brainstorming workflow for: single-file bug fixes, stub parity fixes, import path updates, and Zig 0.16 migration changes. Use brainstorming only for new features, new modules, or architectural changes.
- **writing-skills / skill-creator**: For this project, keep skills concise. Follow the patterns in `.claude/skills/` as examples of well-scoped project skills.
