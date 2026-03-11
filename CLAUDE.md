# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

ABI is a modular Zig 0.16 framework for AI services, vector search, and GPU compute. The core design principle is **comptime feature gating**: every feature in `src/features/<name>/` has a `mod.zig` (real implementation) and `stub.zig` (no-op with matching public signatures). The build system selects which to compile based on `-Dfeat-<name>` flags, giving zero overhead for disabled features.

### Source Layout

- `src/root.zig` — Public package entry point; external code uses `@import("abi")`
- `src/features/` — 19 comptime-gated feature modules (ai, gpu, database, network, web, etc.)
- `src/services/` — Shared runtime services (connectors, LSP, MCP, runtime engine, security)
- `src/core/` — Config, feature catalog, framework lifecycle, registry
- `build/` — Modular build system (options.zig, flags.zig, modules.zig, test_discovery.zig)
- `tools/cli/` — CLI executable with 90+ commands; registry in `tools/cli/registry/`
- `benchmarks/` — Performance benchmark suite

### Key Architecture Concepts

- **Feature gating**: `build_options` (from `build/options.zig`) sets `feat_*` booleans. The root composition layer selects mod vs stub implementations at comptime.
- **mod/stub contract**: When editing any `src/features/<name>/mod.zig`, the corresponding `stub.zig` MUST have identical public function signatures. Sub-module stubs are NOT needed — parent gating covers all children.
- **Import convention**: Within a feature module, use relative imports (`@import("local.zig")`). For the framework API, always use `@import("abi")`. Cross-directory relative imports (e.g., `../../core/`) are fragile and break standalone compilation — avoid them.
- **CLI registry**: After adding/modifying CLI commands, run `zig build refresh-cli-registry` to update the generated snapshot.

## Zig Version

Pinned to `0.16.0-dev.1503+738d2be9d` (see `.zigversion`). Do not upgrade without testing. When repinning, update atomically: `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, and CI config.

## Development Commands

```bash
# Build & format
zig build                             # Build framework and CLI
zig build fix                         # Auto-format codebase
zig build lint                        # Check formatting (may fail on Darwin 25+)
zig fmt --check build.zig build/ src/ tools/  # Safe format check (always works)

# Testing
zig build test --summary all          # Primary test suite (~1290 tests)
zig build feature-tests --summary all # Manifest-driven feature coverage (~2836 tests)
zig test <path> --test-filter "pat"   # Run specific tests by file/pattern

# Validation gates
zig build full-check                  # Local CI gate — mandatory before commits
zig build verify-all                  # Full release validation
zig build validate-flags              # Check feature flag combos
zig build check-cli-registry          # Verify CLI registry is current
zig build check-docs                  # Docs consistency check
zig build check-test-baseline         # Validate test baselines

# CLI
zig build refresh-cli-registry        # Regenerate CLI command metadata
zig build run -- --help               # Run CLI with args
```

### Darwin 25+ Linker Workaround

Zig's pre-built toolchain cannot link on macOS 25+ (undefined symbols in the build runner). Workarounds:

```bash
# Quick: use wrapper that relinks build runner with Apple ld
./tools/scripts/run_build.sh lint
./tools/scripts/run_build.sh test --summary all

# Format check (no linking needed — always works)
zig fmt --check build.zig build/ src/ tools/

# Compile-only validation when linking is still blocked
zig test src/services/tests/mod.zig -fno-emit-bin
```

LLD has **zero** Mach-O support — never set `use_lld = true` for macOS targets.

## Coding Conventions

- **Formatting**: `zig fmt` only. Never manual vertical alignment.
- **Naming**: `lower_snake_case` for files/functions, `PascalCase` for types/structs/error sets.
- **Errors**: Explicit error sets, propagate with `try`, never silently swallow.
- **Commits**: Conventional commits (`fix:`, `feat:`, `docs:`, `chore:`, `style:`). Atomic scope.
- **Format safety**: Never `zig fmt .` from repo root (walks vendored fixtures). Use the targeted paths above.

## Zig 0.16 API Gotchas

These APIs changed from 0.15 and cause recurring mistakes:

- **No `std.time.timestamp()`** — use `std.time.unixSeconds()` (or the Io-based equivalent)
- **No `File.writeAll`** — use `file.writeStreamingAll(io, data)` or `Dir.writeFile(dir, io, .{...})`
- **No `makeDirAbsolute*`** — use `std.Io.Dir.createDirPath(.cwd(), io, path)`
- **No `deleteTreeAbsolute`** — use `std.Io.Dir.deleteTree(.cwd(), io, path)`
- **No `usingnamespace`** — pass parent context as parameters to submodule init functions
- **`LazyPath`**: Use `.cwd_relative` or `.src_path`, not the removed `.path` field
- **`addTest`/`addExecutable`**: Use `root_module` field, not top-level `root_source_file`
- **ZON parsing**: `std.zon.parse.fromSliceAlloc` — use arena-backed parsing and deinit arena at scope end
- **`zig env`** outputs ZON format (`.lib_dir = "..."`) not JSON
- **Enums**: Prefer `@enumFromInt(x)` for int→enum; `std.meta.intToEnum` was removed
- **HashMap iteration**: Use `valueIterator()` / `keyIterator()`, not `.values()` on `AutoHashMapUnmanaged`
- **Allocator vtable**: `alloc`/`resize`/`free` use `alignment: std.mem.Alignment`, not `u8`
- **mem.readInt/writeInt**: Use `std.builtin.Endian.little`/`.big`

## Feature Flags

All features enabled by default. Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false   # Minimal build
zig build -Dgpu-backend=metal                 # GPU backend selection
zig build -Dgpu-backend=cuda,vulkan           # Multiple backends
```

25 `feat_*` flags defined in `build/options.zig` (including 2 internal: `feat_explore`, `feat_vision`). The validation matrix in `build/flags.zig` tests 42 flag combinations (2 baseline + 20 solo + 20 no-X). The feature catalog (`src/core/feature_catalog.zig`) is the source of truth — comptime validation enforces catalog↔BuildOptions consistency.

## CI Pipeline

CI runs on push/PR to `main`/`master` (pinned to `ZIG_VERSION: "0.16.0-dev.1503+738d2be9d"`):

1. **shell-lint** — `bash -n` syntax check on `.cel/*.sh` and `tools/scripts/*.sh`
2. **lint** — `zig fmt --check build.zig build/ src/ tools/`
3. **test** (after lint) — `zig build test` + `zig build feature-tests`
4. **quality-gates** (after test) — `full-check`, `check-test-baseline`, `validate-flags`, `cli-tests`, `check-cli-registry`, `check-docs`
5. **examples** (after lint) — builds example programs

## Environment Variables

| Variable | Description |
|:---------|:------------|
| `ABI_OPENAI_API_KEY` | OpenAI API key |
| `ABI_ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace API token |
| `DISCORD_BOT_TOKEN` | Discord bot token |

## Workflow Rules

1. **Review `tasks/lessons.md`** at session start — it contains corrections for recurring pitfalls.
2. **Plan first** for multi-file changes — write to `tasks/todo.md` before editing.
3. **Validate before completing** — `zig build full-check` must pass (or `zig fmt --check` on Darwin 25+).
4. **Verify stub sync** — any change to `mod.zig` requires matching `stub.zig` update.
5. **Update `tasks/lessons.md`** after fixing any mistake that could recur.
6. **Version pin atomicity** — when changing version strings, grep for all occurrences first, then update all files in one pass.
7. **PR validation trail** — every PR description must include a summary of `zig build full-check` results from the target environment.

## Common Pitfalls

1. **mod.zig ↔ stub.zig sync**: Always update matching `stub.zig` when changing public signatures. Validate with `zig build validate-flags`.
2. **Version pin wave**: Update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, CI config, and `.cel/config.sh` together.
3. **Nightly pin source**: Validate version/commit pairs against `ziglang.org/builds` artifact metadata, not GitHub master HEAD.
4. **Build runner links first**: If `zig build` fails with undefined symbols, the build runner can't link — no `build.zig` workaround helps. Use CEL toolchain.
5. **Test manifest standalone compilation**: Files in `build/test_discovery.zig` must compile with `zig test <file> -fno-emit-bin`. Cross-directory `@import("../../")` breaks this.
6. **Async I/O in TUI**: Use `std.posix.poll` on STDIN instead of `std.time.sleep` in event loops.
7. **Shell script sourcing**: Guard `set -euo pipefail` with `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then ... fi` so strict mode only applies when executed directly.
8. **Bulk find-and-replace can corrupt string literals**: Never do repo-wide text substitution without reviewing each match in context. One bulk operation can cascade into multiple fix waves.
9. **Validation matrix no-X entries**: Each `no-X` flag combo must enable ALL other features, not just disable one. Verify with `zig build validate-flags`.
10. **Vendored fixtures stay out of fmt runs**: Never `zig fmt .` from repo root — vendored Zig test fixtures will be reformatted. Use the repo-safe surface (`zig build fix` or targeted `zig fmt --check`).

## AI Self-Improvement Architecture

The AI subsystem has a feedback→learning loop connecting these components:

```
User Feedback → FeedbackSystem → LearningBridge → SelfLearningSystem
                 (collector.zig)   (learning_bridge.zig)  (self_learning.zig)
                                         ↓
                                   ExperienceBuffer → DPO Optimizer
                                   (experience_buffer.zig)  (dpo_optimizer.zig)
```

- **Agents** (`src/features/ai/agents/`) — Multi-backend conversational agents with optional AdvancedCognition and per-backend performance tracking
- **Multi-agent** (`src/features/ai/multi_agent/`) — Coordinator with parallel/pipeline execution, blackboard shared state, DAG workflows
- **Training** (`src/features/ai/training/`) — Self-learning (text, vision, audio, document), DPO optimization, experience replay
- **Feedback** (`src/features/ai/feedback/`) — Star/thumbs ratings, per-persona analysis, learning bridge for auto-retraining
- **Abbey Advanced** (`src/features/ai/abbey/advanced/`) — Meta-learning, theory of mind, compositional reasoning, self-reflection
- **Ralph** (`tools/cli/commands/ai/ralph/`) — Iterative agent loop with skill storage and quality ranking

## Plugin

`zig-abi-plugin/` provides a Claude Code plugin with build routing, feature scaffolding, and stub-sync validation. Install with `claude --plugin-dir zig-abi-plugin`.

## References

- [AGENTS.md](AGENTS.md) — Canonical workflow contract, acceptance criteria, commit/PR guidelines
- [docs/FAQ-agents.md](docs/FAQ-agents.md) — Detailed style rules and expanded command docs
- [tasks/todo.md](tasks/todo.md) — Active task tracker
- [tasks/lessons.md](tasks/lessons.md) — Correction log for recurring pitfalls
