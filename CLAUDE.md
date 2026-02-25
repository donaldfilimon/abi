# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Single entry for AI assistants (Claude, Codex, Cursor). Code style, naming, commits, formatting, and workflow requirements are in `AGENTS.md`. Zig 0.16 gotchas are in `.claude/rules/zig.md` (auto-loaded for all `.zig` files).

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | Pinned in `.zigversion` (verify with `zig version` and `cat .zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | `abi.zon` |
| **Test baseline** | Source of truth: `tools/scripts/baseline.zig` |
| **Feature tests** | `zig build feature-tests --summary all` |
| **CLI** | Command surface: `zig build run -- --help` |
| **Features** | Canonical list: `src/core/feature_catalog.zig` |

## Consistency Markers (Script-Validated)

These exact literals are intentionally retained for repository consistency checks and must stay aligned with `tools/scripts/baseline.zig` and `.zigversion`.

- `0.16.0-dev.2637+6a9510c0e`
- `1290 pass, 6 skip (1296 total)`
- `2360 pass (2365 total)`

## Build & Test Commands

```bash
# Toolchain sync
zvm install "$(cat .zigversion)" && zvm use "$(cat .zigversion)"
# Verify: zig version && cat .zigversion && zig build toolchain-doctor
# Fix PATH: export PATH="$HOME/.zvm/bin:$PATH"
```

```bash
zig build                                    # Build with default flags
zig build test --summary all                 # Main test suite
zig build feature-tests --summary all        # Feature inline tests
zig test src/path/to/file.zig                # Test a single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter tests by name
zig fmt .                                    # Format all source
zig build fix                                # Auto-format in place (CI-friendly)
zig build full-check                         # Format + tests + flag validation + CLI smoke
zig build verify-all                         # full-check + consistency + examples + check-wasm (release gate)
zig build validate-flags                     # Compile-check feature flag combinations
zig build cli-tests                          # CLI smoke tests
zig build lint                               # CI formatting check (no writes)
zig build check-consistency                  # Zig version/baseline/0.16 pattern checks
zig build check-imports                      # No circular @import("abi") in feature modules
zig build validate-baseline                  # Verify test baselines match across all files
zig build gendocs                            # Generate API docs
zig build check-perf                         # Build performance verification tool
```

### Running the CLI

```bash
zig build run -- --help                      # CLI help
zig build run -- system-info                 # System and feature status (Feature Matrix)
zig build run -- --list-features             # List features (COMPILED/DISABLED)
zig build run -- status                      # Framework health and feature count
zig build run -- mcp serve                   # Start MCP server (stdio JSON-RPC)
zig build run -- acp card                    # Print agent card JSON
```

## After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` (also runs automatically via hook) |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Feature inline tests | `zig build feature-tests --summary all` |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Test counts | Update `tools/scripts/baseline.zig`, then `/baseline-sync` |
| Anything (full gate) | `zig build full-check` |
| Everything (release gate) | `zig build verify-all` |
| Process requirements | Follow `AGENTS.md` workflow orchestration section |

Hooks auto-enforce: `zig fmt`, stub parity reminders, import rules, baseline drift detection, and test discovery guards. When a hook warns, address it before continuing.

## Critical Gotchas

**Top 7 (cause 80% of failures):**

1. `std.fs.cwd()` → `std.Io.Dir.cwd()` (requires I/O backend init)
2. Editing `mod.zig` without updating `stub.zig` → always keep signatures in sync
3. `defer allocator.free(x)` then `return x` → use `errdefer` (use-after-free)
4. `@tagName(x)` / `@errorName(e)` in format → use `{t}` specifier
5. `std.io.fixedBufferStream()` → removed; use `std.Io.Writer.fixed(&buf)`
6. `@field(build_options, field_name)` requires comptime context — use `inline for` not runtime `for`
7. **API break (v0.4.0):** Facade aliases and flat exports removed — use `abi.ai.agent.Agent` not `abi.ai.Agent`

**Full reference (20 entries):** `.claude/rules/zig.md` auto-loads for all `.zig` files.

**I/O patterns** (code examples in global CLAUDE.md and `.claude/rules/zig.md`):
- **I/O Backend**: `std.Io.Threaded.init(allocator, .{ .environ = ... })` → `.io()` — required for all file/network ops
- **Stdio writer**: `std.Io.File.stdout().writer(io, &buf)` — always call `.flush()` after writing
- **Fixed-buffer writer**: `std.Io.Writer.fixed(&buf)` — replaces removed `std.io.fixedBufferStream()`

**Test runner "failed command":** If `zig build test` fails with `failed command: .../test --cache-dir=... --seed=0x... --listen=-`, a parallel test worker exited non-zero (flaky test, OOM, or timeout). Try: re-run `zig build test --summary all`; or limit build jobs `zig build test -j 1 --summary all`; or run the test binary in single-process mode (build first, then run the cached test binary with only `--cache-dir=./.zig-cache` and `--seed=0x<N>` and no `--listen=-`) to see the failing test name.

## Architecture: Comptime Feature Gating

The central architectural pattern is **comptime feature gating** in `src/abi.zig`. Each
feature module has two implementations selected at compile time via `build_options`:

```zig
// src/abi.zig — this pattern repeats for every feature
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")    // Real implementation
else
    @import("features/gpu/stub.zig");  // Returns error.FeatureDisabled
```

This means:
- Every feature directory has `mod.zig` (real) and `stub.zig` (stub)
- `mod.zig` and `stub.zig` must keep matching public signatures
- Test both paths: `zig build -Denable-<feature>=true` and `=false`
- Disabled features have zero binary overhead

### Module Hierarchy

```
build.zig                → Top-level build script (delegates to build/)
build/                   → Split build system (options, modules, flags, targets, gpu, mobile, wasm)
src/abi.zig              → Public API, comptime feature selection (no flat type aliases)
src/core/                → Framework lifecycle, config builder, registry
src/features/<name>/     → mod.zig + stub.zig per feature
src/services/            → Always-available infrastructure (runtime, platform, shared, ha, tasks)
src/services/connectors/ → LLM provider connectors
src/services/mcp/        → MCP server (JSON-RPC 2.0 over stdio, WDBX + ZLS tools)
src/services/acp/        → ACP server (agent communication protocol)
tools/cli/               → CLI entry point, command registration, TUI dashboards
```

Import convention: public API uses `@import("abi")`, internal modules import
via their parent `mod.zig`. Feature modules CANNOT `@import("abi")` (circular) — use relative imports.

### Framework Lifecycle

The `Framework` struct (`src/core/framework.zig`) manages feature initialization through
a state machine: `uninitialized → initializing → running → stopping → stopped` (or `failed`).

```zig
// 1. Default (all compile-time features enabled)
var fw = try abi.initDefault(allocator);

// 2. Custom config
var fw = try abi.init(allocator, .{ .gpu = .{ .backend = .vulkan } });
```

### Access Patterns

All access uses namespaced submodule paths — no top-level aliases or flat re-exports:

| Module | Access Pattern | Build Flag |
|--------|---------------|------------|
| AI core | `abi.ai.core`, `abi.ai.agent.Agent` | `-Denable-ai` |
| AI LLM | `abi.ai.llm` | `-Denable-llm` |
| AI training | `abi.ai.training.TrainingConfig` | `-Denable-training` |
| AI orchestration | `abi.ai.orchestration` | `-Denable-reasoning` (note: flag ≠ path) |
| GPU | `abi.gpu.unified.MatrixDims`, `abi.gpu.profiling.Profiler` | `-Denable-gpu` |
| SIMD | `abi.simd.vectorAdd` | (always available) |
| Connectors | `abi.connectors.discord`, `abi.connectors.openai` | (always available) |
| Runtime | `abi.runtime.Channel`, `abi.runtime.ThreadPool` | (always available) |
| Shared | `abi.shared.resilience.AtomicCircuitBreaker` | (always available) |

**Removed in v0.4.0:** `abi.ai_core`, `abi.inference`, `abi.training`, `abi.reasoning`
(facade aliases); ~329 flat type aliases. Use submodule paths instead.

## Coding Preferences

- Prefer `std.ArrayListUnmanaged` over `std.ArrayList` — passes allocator per-call, better ownership control:
  ```zig
  var list: std.ArrayListUnmanaged(u8) = .empty;
  defer list.deinit(allocator);
  try list.append(allocator, item);
  ```
- Modern format specifiers: `{t}` for enums/errors, `{B}`/`{Bi}` for byte sizes, `{D}` for durations, `{b64}` for base64
- For null-terminated C strings: `std.fmt.allocPrintSentinel(alloc, fmt, args, 0)` or use string literal `.ptr`

## Testing Patterns

**Two test roots** (each is a separate binary with its own module path):
- `src/services/tests/mod.zig` — main tests; discovers tests via `abi.<feature>` import chain
- `src/feature_test_root.zig` — feature inline tests; can `@import("features/...")` and `@import("services/...")` directly

**Why two roots?** Module path restrictions prevent `src/services/tests/mod.zig` from importing
`src/services/mcp/server.zig` (outside its module path). The feature test root at `src/` level
can reach both `features/` and `services/` subdirectories.

- **Test discovery**: Use `test { _ = @import(...); }` — `comptime {}` does NOT discover tests
- Skip hardware-gated tests with `error.SkipZigTest`
- Parity tests verify `mod.zig` and `stub.zig` export the same interface
- **GPU/database test gap**: Backend source files have Zig 0.16 compatibility issues; they compile through `zig build test` via the named `abi` module but cannot be registered in `feature_test_root.zig`

## Key File Locations

| Need to... | Look at |
|------------|---------|
| Add/modify public API | `src/abi.zig` |
| Change build flags | `build/options.zig` + `build/flags.zig` |
| Feature catalog (canonical list) | `src/core/feature_catalog.zig` |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/main.zig` |
| Add config for a feature | `src/core/config/` |
| Write integration tests | `src/services/tests/` |
| Add a GPU backend | `src/features/gpu/backends/` |
| Security infrastructure | `src/services/shared/security/` |
| Generate API docs | `zig build gendocs` |
| Examples | `examples/` |

### Adding a New Feature Module (9 integration points)

1. `build/options.zig` — add `enable_<name>` field + CLI option
2. `build/flags.zig` — add to `FlagCombo`, `validation_matrix`, `comboToBuildOptions()`
3. `src/core/feature_catalog.zig` — add `Feature` enum variant, `ParitySpec` if needed, and `Metadata` entry
4. `src/features/<name>/mod.zig` + `stub.zig` — implementation + disabled stub
5. `src/abi.zig` — comptime conditional import
6. `src/core/config/mod.zig` — Feature enum, description, Config field, Builder methods, validation
7. `src/core/registry/types.zig` — `isFeatureCompiledIn` switch case
8. `src/core/framework.zig` — import, context field, init/deinit, getter, builder
9. `src/services/tests/stub_parity.zig` — basic parity test

**Stub conventions**: Use `StubContext(ConfigT)` from `src/core/stub_context.zig`. Discard params with `_`, return `error.FeatureDisabled`.

**Shared infrastructure**: `services/shared/resilience/circuit_breaker.zig` (`.atomic`, `.mutex`, `.none` sync). `services/shared/security/rate_limit.zig` for HTTP rate limiting.

## Feature Flags

Canonical list: `src/core/feature_catalog.zig`. All default to `true` except `-Denable-mobile`.

**Usage:** `zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

GPU backends: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`. The `simulated` backend is always enabled as fallback.

**Non-obvious flag mappings:**

| Feature | Build Flag | Gotcha |
|---------|------------|--------|
| `observability` | `-Denable-profiling` | NOT `-Denable-observability` |
| `ai.orchestration` | `-Denable-reasoning` | Flag says "reasoning", access path is `abi.ai.orchestration` |
| `mobile` | `-Denable-mobile` | Defaults to `false` (all others default `true`) |
| AI sub-features | `-Denable-llm`, `-Denable-training`, `-Denable-reasoning` | `embeddings`, `agents`, `personas` share `-Denable-ai` |
| Internal (no catalog) | `-Denable-explore`, `-Denable-vision` | Derived from `-Denable-ai` |

**Runtime overrides:** `abi --enable-<feature>` / `--disable-<feature>` before a command. Only compiled-in features can be enabled at runtime. Use `abi --list-features` or `abi system-info` to check.

## Environment Variables

Connectors follow the pattern `ABI_<PROVIDER>_API_KEY`, `ABI_<PROVIDER>_HOST`, `ABI_<PROVIDER>_MODEL`.

**Non-obvious variables (these break the pattern):**

| Variable | Gotcha |
|----------|--------|
| `ABI_HF_API_TOKEN` | NOT `ABI_HUGGINGFACE_API_KEY` |
| `DISCORD_BOT_TOKEN` | No `ABI_` prefix |
| `ABI_OLLAMA_PASSTHROUGH_URL` | Uses `_URL` not `_HOST` |
| `ABI_MASTER_KEY` | Secrets encryption key (production) |

Fallbacks: `claude` connector checks `ABI_ANTHROPIC_*`; `codex`/`opencode` check `ABI_OPENAI_*`.

## Ralph (Agent Loop)

You are **outside the Ralph loop** unless the user explicitly runs `abi ralph run`. Normal workflow: edit code → `zig build full-check` → done. Ralph config: `ralph.yml`, state: `.ralph/`. Power commands: `abi ralph super --task "goal"`, `abi ralph multi -t "g1" -t "g2"`.

## Custom Skills

| Skill | Invocation | Purpose |
|-------|------------|---------|
| **baseline-sync** | `/baseline-sync` | Sync test baselines from `tools/scripts/baseline.zig` to all doc files |
| **zig-migrate** | `/zig-migrate [path]` | Apply Zig 0.16 migration patterns |
| **zig-build** | `/zig-build` | Build/test/format pipeline (`full` for CLI+flags, `verify` for release) |
| **zig-std** | `/zig-std` | Look up Zig 0.16 std lib API from source at `~/.zvm/master/lib/std/` |
| **new-feature** | `/new-feature` | Scaffold a feature module through all 9 integration points |
| **cli-add-command** | `/cli-add-command` | Scaffold a CLI command with registration and feature gating |
| **connector-add** | `/connector-add` | Scaffold an LLM provider connector |
| **parity-check** | `/parity-check` | Verify mod.zig/stub.zig signature parity |
| **ci-gate** | `/ci-gate [level]` | Run quality gates with failure diagnosis |
| **super-ralph** | `/super-ralph` | Run Ralph agent loop |

## References

- `AGENTS.md` — Code style, naming, imports, error handling, commits, PR checklist
- `.claude/rules/zig.md` — Zig 0.16 gotchas table (auto-loaded for `.zig` files)
- `.claude/skills/` — Skill implementations; index: `.claude/skills/README.md`
- `.codex/skills/workflow-orchestration/` — Canonical workflow orchestration contract
- `tools/scripts/baseline.zig` — Canonical test baseline (source of truth)
- `CONTRIBUTING.md` — Development workflow and PR checklist
