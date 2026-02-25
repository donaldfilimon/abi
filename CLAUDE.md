# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Single entry for AI assistants (Claude, Codex, Cursor). Code style, naming, commits, and formatting rules are in `AGENTS.md`. Full Zig 0.16 gotchas (22 entries) in `.claude/rules/zig.md` (auto-loaded for all `.zig` files).

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2637+6a9510c0e` or newer (pinned in `.zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 1290 pass, 6 skip (1296 total) — source of truth: `tools/scripts/baseline.zig` |
| **Feature tests** | 2360 pass (2365 total), 5 skip — `zig build feature-tests` |
| **CLI** | 30 commands + 8 aliases — `tools/cli/commands/mod.zig` |
| **Features** | 24 comptime-gated modules (see [Feature Flags](#feature-flags)) |

## Build & Test Commands

```bash
# Toolchain sync
zvm install "$(cat .zigversion)" && zvm use "$(cat .zigversion)"
# Or: zvm use master
# Verify: zig version && cat .zigversion && zig build toolchain-doctor
# Fix PATH: export PATH="$HOME/.zvm/bin:$PATH"
```

```bash
zig build                                    # Build with default flags
zig build test --summary all                 # Main test suite (1290 pass, 6 skip)
zig build feature-tests --summary all        # Feature inline tests (2360 pass, 5 skip)
zig test src/path/to/file.zig                # Test a single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter tests by name
zig fmt .                                    # Format all source
zig build full-check                         # Format + tests + flag validation + CLI smoke + imports + consistency + TUI
zig build verify-all                         # full-check + feature tests + examples + check-wasm + check-docs (release gate)
zig build validate-flags                     # Compile-check 34 feature flag combos
zig build cli-tests                          # CLI smoke tests
zig build tui-tests                          # TUI and CLI unit tests
zig build check-consistency                  # Zig version/baseline/0.16 pattern checks
zig build check-imports                      # No circular @import("abi") in feature modules
zig build validate-baseline                  # Verify test baselines match across all files
zig build fix                                # Auto-format in place (CI-friendly, unlike zig fmt)
zig build gendocs                            # Generate API docs
zig build benchmarks                         # Run performance benchmarks
zig build ralph-gate                         # Require live Ralph scoring report and threshold pass
```

## After Making Changes

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` (also runs automatically via hook) |
| Feature `mod.zig` | Also update `stub.zig`, then `zig build -Denable-<feature>=false` |
| Feature inline tests | `zig build feature-tests --summary all` (must stay at 2360+) |
| Build flags / options | `zig build validate-flags` |
| Public API | `zig build test --summary all` + update examples |
| Test counts | Update `tools/scripts/baseline.zig`, then `/baseline-sync` |
| Anything (full gate) | `zig build full-check` |
| Everything (release gate) | `zig build verify-all` |

### Claude Code Hooks (auto-enforcement)

11 hooks run automatically (1 PreToolUse + 10 PostToolUse). When a hook warns, address it before continuing.

| Hook | Trigger | What it enforces |
|------|---------|-----------------|
| **Circular import blocker** | PreToolUse (Write/Edit) | Blocks `@import("abi")` inside `src/features/` |
| **Auto-format** | PostToolUse (Write/Edit `.zig`) | Runs `zig fmt` on saved files |
| **Stub→mod sync** | PostToolUse (Edit `stub.zig`) | Reminds to update matching `mod.zig` |
| **Mod→stub sync** | PostToolUse (Edit `mod.zig`) | Reminds to update matching `stub.zig` |
| **Build options** | PostToolUse (Edit `options.zig`) | Reminds to update `flags.zig` |
| **Build flags** | PostToolUse (Edit `flags.zig`) | Reminds to update `options.zig` |
| **Feature catalog** | PostToolUse (Edit `feature_catalog.zig`) | Reminds about all 9 integration points |
| **Baseline drift** | PostToolUse (Edit test files) | Warns if test counts may have changed |
| **Test discovery guard** | PostToolUse (Write/Edit test files) | Warns if using `comptime {}` instead of `test {}` for imports |
| **Public API reminder** | PostToolUse (Edit `src/abi.zig`) | Reminds to update examples and run tests |
| **Baseline source edit** | PostToolUse (Edit `baseline.zig`) | Reminds to verify numbers and run `/baseline-sync` |

The Bash test output hook also monitors `zig build test`/`feature-tests` output and warns if pass counts diverge from baselines.

## Critical Gotchas

**Top 7 (cause 80% of failures):**

1. `std.fs.cwd()` → `std.Io.Dir.cwd()` (requires I/O backend init)
2. Editing `mod.zig` without updating `stub.zig` → always keep signatures in sync
3. `defer allocator.free(x)` then `return x` → use `errdefer` (use-after-free)
4. `@tagName(x)` / `@errorName(e)` in format → use `{t}` specifier
5. `std.io.fixedBufferStream()` → removed; use `std.Io.Writer.fixed(&buf)`
6. `@field(build_options, field_name)` requires comptime context — use `inline for` not runtime `for`
7. **API break (v0.4.0):** Facade aliases and flat exports removed — use `abi.ai.agent.Agent` not `abi.ai.Agent`

Additional Zig 0.16 patterns, I/O code samples, and stub conventions are in `.claude/rules/zig.md` (auto-loaded for `.zig` files).

### Error Diagnosis

| Symptom | Cause | Fix |
|---------|-------|-----|
| "member not found" on `std.*` | API moved in Zig 0.16 | Read `~/.zvm/master/lib/std/` for current API |
| `error.FeatureDisabled` at runtime | Feature compiled out | Check build flags: `zig build run -- --list-features` |
| Stub parity test failure | `mod.zig` and `stub.zig` diverged | Diff public signatures, update stub |
| `validate-flags` failure | New flag not in matrix | Add to `FlagCombo` + `validation_matrix` in `build/flags.zig` |
| Variable shadowing error | Zig 0.16 is strict about shadowing | Rename inner variable (e.g., `args` → `sub_args`) |
| `catch \|_\| {` compile error | Illegal in 0.16 | Use `catch {` (omit capture entirely) |
| "failed command: .../test --listen=-" | Parallel test worker crashed | Re-run with `-j 1`, or run cached binary without `--listen=-` |

### Project-Specific Traps

- `pages` feature lives at `src/features/observability/pages/`, NOT `src/features/pages/`
- `@embedFile` path is relative to the importing `.zig` file, not the build root
- CLI output: always use `utils.output.printError`/`printInfo`, never `std.debug.print`
- `realpathAlloc` → `realPathFileAlloc(io, sub_path, allocator)` in 0.16 — and the file must exist first
- Main test count 1289-1290 is normal variance (one hardware-gated test flips) — don't adjust baseline for ±1

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
src/abi.zig              → Public API, comptime feature selection
src/core/                → Framework lifecycle, config, registry, errors, stub_context
src/features/<name>/     → mod.zig + stub.zig per feature (24 catalog, 16 dirs, 7 AI sub-features)
src/services/            → Always-available infrastructure:
  connectors/            →   15 LLM providers + scheduler (env: ABI_<PROVIDER>_API_KEY)
  mcp/                   →   MCP server (JSON-RPC 2.0 over stdio)
  shared/security/       →   17 modules (jwt, rbac, secrets, encryption most-used)
  shared/resilience/     →   Circuit breaker (.atomic/.mutex/.none), rate limiter
  tests/                 →   Main test root
tools/cli/               → 30 commands, TUI dashboards, framework (router, completion)
build/                   → options.zig (flags) + flags.zig (combos) are the two critical files
```

Import convention: public API uses `@import("abi")`, internal modules import via parent `mod.zig`. Feature modules CANNOT `@import("abi")` (circular) — use relative imports.

### Framework Lifecycle

`Framework` (`src/core/framework.zig`): state machine `uninitialized → initializing → running → stopping → stopped` (or `failed`). Init via `abi.initDefault(allocator)`, `abi.init(allocator, config)`, or `Framework.builder(allocator).withGpu(.{...}).build()`.

### Access Patterns & Flag Mappings

All access uses namespaced submodule paths — no top-level aliases (removed in v0.4.0):

| Module | Access | Build Flag | Gotcha |
|--------|--------|------------|--------|
| AI orchestration | `abi.ai.orchestration` | `-Denable-reasoning` | Flag ≠ path |
| Observability | `abi.observability` | `-Denable-profiling` | NOT `-Denable-observability` |
| Mobile | `abi.mobile` | `-Denable-mobile` | Defaults to `false` (all others `true`) |
| AI sub-features | `abi.ai.embeddings`, `.agents`, `.personas`, `.constitution` | `-Denable-ai` | Shared flag |
| Internal | `abi.explore`, `abi.vision` | (derived from `-Denable-ai`) | No catalog entry |
| GPU | `abi.gpu.unified.MatrixDims` | `-Denable-gpu` | Use submodule path, not flat |

**Runtime overrides:** `abi --enable-<feature>` / `--disable-<feature>` before a command. Only compiled-in features can be enabled at runtime.

## Coding Preferences

- Prefer `std.ArrayListUnmanaged` over `std.ArrayList` — allocator per-call, better ownership
- `std.log.*` in library code; `std.debug.print` only in CLI/TUI display functions
- End every source file with: `test { std.testing.refAllDecls(@This()); }`

## Testing Patterns

**Two test roots** (each is a separate binary with its own module path):
- `src/services/tests/mod.zig` — main tests; discovers tests via `abi.<feature>` import chain
- `src/feature_test_root.zig` — feature inline tests; can `@import("features/...")` and `@import("services/...")` directly

**Why two roots?** Module path restrictions prevent `src/services/tests/mod.zig` from importing `src/services/mcp/server.zig` (outside its module path). The feature test root at `src/` level can reach both.

- Initialize test structs with `std.mem.zeroes(T)`, never `= undefined` (UB when fields are read)
- **Test discovery**: Use `test { _ = @import(...); }` — `comptime {}` does NOT discover tests
- Skip hardware-gated tests with `error.SkipZigTest`
- **GPU/database test gap**: Backend source files compile through `zig build test` via the named `abi` module but cannot be registered in `feature_test_root.zig`

## Key File Locations

| Need to... | Look at |
|------------|---------|
| Add/modify public API | `src/abi.zig` |
| Change build flags | `build/options.zig` + `build/flags.zig` |
| Feature catalog (canonical list) | `src/core/feature_catalog.zig` |
| Add a CLI command | `tools/cli/commands/`, register in `tools/cli/commands/mod.zig` |
| Write integration tests | `src/services/tests/` |
| Test baselines (source of truth) | `tools/scripts/baseline.zig` |
| TUI dashboards | `tools/cli/tui/` (vtable `Panel` in `panel.zig`: `name`, `render`, `tick`, `handleEvent`, `deinit`) |
| Security infrastructure | `src/services/shared/security/` (17 modules) |

### CLI Command Authoring

Export `pub const meta: command_mod.Meta` and `pub fn run`. Register in `tools/cli/commands/mod.zig`. Commands with `.children` MUST set `.kind = .group` in their `Meta` struct for subcommand dispatch. Commands with subcommands use a sub-directory with their own `mod.zig` (e.g., `commands/ralph/mod.zig`). Tuple ordering in the parent `mod.zig` controls `--help` display order.

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

## Feature Flags

24 feature catalog entries across 17 directories (7 AI sub-features). Canonical list: `src/core/feature_catalog.zig`. All default to `true` except `-Denable-mobile`.

**Usage:** `zig build -Denable-ai=true -Denable-gpu=false -Dgpu-backend=vulkan,cuda`

GPU backends: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `tpu`, `webgl2`, `opengl`, `opengles`, `fpga`, `simulated`. The `simulated` backend is always enabled as fallback.

## Connectors

Located in `src/services/connectors/`. Env var pattern: `ABI_<PROVIDER>_API_KEY` / `_HOST` / `_MODEL`.

**Non-obvious environment variables:**

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

Most-used: `/baseline-sync` (sync test numbers), `/zig-build` (build/test pipeline), `/ci-gate` (quality gates). Full list: `/new-feature`, `/cli-add-command`, `/connector-add`, `/parity-check`, `/zig-migrate`, `/zig-std`, `/super-ralph`.

## References

- `AGENTS.md` — Code style, naming, imports, error handling, commits, PR checklist
- `.claude/rules/zig.md` — Zig 0.16 gotchas (22 entries), I/O patterns, stub conventions (auto-loaded)
- `tools/scripts/baseline.zig` — Canonical test baseline (source of truth)
- `CONTRIBUTING.md` — Development workflow and PR checklist
