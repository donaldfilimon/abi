# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Present plan for approval before implementation on high-stakes changes

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- Aggregate and synthesize subagent results before proceeding

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project
- Patterns to capture: root causes, not just symptoms

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- For UI changes: verify visually; for API changes: test the endpoint

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Simplicity is the ultimate sophistication

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Investigate root cause; fix the disease, not the symptom

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Simplicity First** | Make every change as simple as possible. Minimal code impact. |
| **No Laziness** | Find root causes. No temporary fixes. Senior developer standards. |
| **Minimal Impact** | Changes should only touch what's necessary. Avoid introducing bugs. |
| **Review Lessons** | Review `lessons.md` at session start for the relevant project. |

---

## Quick Reference

| Key | Value |
|-----|-------|
| **Zig** | `0.16.0-dev.2637+6a9510c0e` (pinned in `.zigversion`) |
| **Entry Point** | `src/abi.zig` |
| **Version** | 0.4.0 |
| **Test baseline** | 1290 pass, 6 skip — source of truth: `tools/scripts/baseline.zig` |
| **Feature tests** | 2360 pass, 5 skip — `zig build feature-tests` |
| **CLI** | 30 commands + 8 aliases — `tools/cli/commands/mod.zig` |
| **Features** | 24 comptime-gated modules — `src/core/feature_catalog.zig` |

## Build & Test

```bash
zig build                                    # Default build
zig build test --summary all                 # Main tests (1290 pass, 6 skip)
zig build feature-tests --summary all        # Feature inline tests (2360 pass, 5 skip)
zig build full-check                         # Format + tests + flags + CLI smoke + imports + consistency + TUI
zig build verify-all                         # Release gate (full-check + feature tests + examples + wasm + docs)
zig build validate-flags                     # Compile-check 34 feature flag combos
zig test src/path/to/file.zig                # Single file
zig test src/services/tests/mod.zig --test-filter "pattern"  # Filter by name
zig fmt .                                    # Format all
```

**Toolchain sync:**
```bash
zvm install "$(cat .zigversion)" && zvm use "$(cat .zigversion)"
```

| Changed... | Run |
|------------|-----|
| Any `.zig` file | `zig fmt .` (auto-hook handles this) |
| Feature `mod.zig` | Also update `stub.zig`, test with `-Denable-<feature>=false` |
| Test counts | Update `tools/scripts/baseline.zig` |
| Full gate | `zig build full-check` |

## Architecture

### Comptime Feature Gating

The central pattern — every feature in `src/abi.zig`:

```zig
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")    // Real implementation
else
    @import("features/gpu/stub.zig");  // Returns error.FeatureDisabled
```

Every feature directory has `mod.zig` + `stub.zig` with matching public signatures. Disabled features have zero binary overhead.

### Module Hierarchy

```
src/abi.zig              → Public API, comptime feature selection
src/core/                → Framework lifecycle, config, registry, errors, stub_context
src/features/<name>/     → mod.zig + stub.zig per feature (24 catalog, 16 dirs, 7 AI sub-features)
src/services/            → Always-available infrastructure (connectors, MCP, security, resilience)
tools/cli/               → 30 commands, TUI dashboards, framework (router, completion)
build/                   → options.zig (flags) + flags.zig (combos)
```

**Import rules:**
- Public API: `@import("abi")` — never deep file paths
- Feature modules CANNOT `@import("abi")` (circular) — use relative imports
- No `usingnamespace` — explicit imports only
- Submodule paths only (v0.4.0): `abi.ai.agent.Agent`, NOT `abi.ai.Agent`

### Two Test Roots

| Root | Command | File | Why |
|------|---------|------|-----|
| Main | `zig build test` | `src/services/tests/mod.zig` | Tests via `abi.*` namespace |
| Feature | `zig build feature-tests` | `src/feature_test_root.zig` | Can reach both `features/` and `services/` directly |

Module path restrictions prevent the main root from importing certain files — the feature root at `src/` level can reach both.

### Framework Lifecycle

State machine: `uninitialized → initializing → running → stopping → stopped` (or `failed`).

Init patterns: `Framework.init(allocator, cfg)`, `Framework.initDefault(allocator)`, or `Framework.builder(allocator).withGpu(.{}).build()`.

## Zig 0.16 Gotchas (Top 10)

| Wrong | Correct | Notes |
|-------|---------|-------|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` | Requires I/O backend |
| `std.ArrayList(T).init(alloc)` | `.empty` + per-call allocator | Same for HashMap |
| `pub fn main() !void` | `pub fn main(init: std.process.Init) !void` | New entry |
| `std.json.stringifyAlloc(...)` | `std.json.Stringify.valueAlloc(...)` | Renamed |
| `std.crypto.random` | `std.c.arc4random_buf(&buf, buf.len)` | Removed |
| `std.posix.getenv(...)` | `std.c.getenv(...)` | Moved to libc |
| `std.io.fixedBufferStream()` | `std.Io.Writer.fixed(&buf)` | Removed |
| `@tagName(x)` with `{s}` | Use `{t}` format specifier | Direct formatting |
| `defer alloc.free(x); return x;` | Use `errdefer` | Use-after-free |
| `catch \|_\| {` | `catch {` (omit capture) | Illegal in 0.16 |

Full 22-entry table: `.claude/rules/zig.md` (auto-loaded for `.zig` files).

## Non-Obvious Traps

- `pages` feature lives at `src/features/observability/pages/`, NOT `src/features/pages/`
- Observability uses `-Denable-profiling`, NOT `-Denable-observability`
- AI orchestration uses `-Denable-reasoning` but accessed as `abi.ai.orchestration`
- `@tagName()` returns `[*:0]const u8` — use `std.mem.sliceTo(@tagName(x), 0)` before `dupe()`
- `mobile` is the only feature flag defaulting to `false`
- Main test count 1289-1290 is normal variance — don't adjust baseline for ±1
- `@embedFile` path is relative to the importing `.zig` file, not the build root
- CLI output: always use `utils.output.printError`/`printInfo`, never `std.debug.print`

## Adding a New Feature (9 Integration Points)

1. `build/options.zig` — add `enable_<name>` field + CLI option
2. `build/flags.zig` — add to `FlagCombo`, `validation_matrix`, `comboToBuildOptions()`
3. `src/core/feature_catalog.zig` — add `Feature` enum variant + `Metadata` entry
4. `src/features/<name>/mod.zig` + `stub.zig` — implementation + disabled stub
5. `src/abi.zig` — comptime conditional import
6. `src/core/config/mod.zig` — Feature enum, Config field, Builder methods
7. `src/core/registry/types.zig` — `isFeatureCompiledIn` switch case
8. `src/core/framework.zig` — import, context field, init/deinit, getter, builder
9. `src/services/tests/stub_parity.zig` — basic parity test

## Connectors

15 LLM providers in `src/services/connectors/`. Env var pattern: `ABI_<PROVIDER>_API_KEY`.

**Non-obvious env vars:** `ABI_HF_API_TOKEN` (not HUGGINGFACE), `DISCORD_BOT_TOKEN` (no ABI_ prefix), `ABI_OLLAMA_PASSTHROUGH_URL` (uses _URL), `ABI_MASTER_KEY` (secrets encryption).

## Coding Conventions

- `PascalCase` types, `camelCase` functions, `snake_case.zig` files
- Prefer `std.ArrayListUnmanaged` over `std.ArrayList`
- `std.log.*` in library code; `std.debug.print` only in CLI/TUI
- End every file with: `test { std.testing.refAllDecls(@This()); }`
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- Stubs: use `StubContext(ConfigT)` from `src/core/stub_context.zig`, return `error.FeatureDisabled`
