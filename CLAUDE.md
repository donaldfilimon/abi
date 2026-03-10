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

# Full fix: build CEL toolchain from source
./.zig-bootstrap/build.sh && eval "$(./tools/scripts/use_zig_bootstrap.sh)"
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

## Feature Flags

All features enabled by default. Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false   # Minimal build
zig build -Dgpu-backend=metal                 # GPU backend selection
zig build -Dgpu-backend=cuda,vulkan           # Multiple backends
```

25 `feat_*` flags defined in `build/options.zig` (including 2 internal: `feat_explore`, `feat_vision`). The validation matrix in `build/flags.zig` tests 42 flag combinations (2 baseline + 20 solo + 20 no-X). The feature catalog (`src/core/feature_catalog.zig`) is the source of truth — comptime validation enforces catalog↔BuildOptions consistency.

## Workflow Rules

1. **Review `tasks/lessons.md`** at session start — it contains corrections for recurring pitfalls.
2. **Plan first** for multi-file changes — write to `tasks/todo.md` before editing.
3. **Validate before completing** — `zig build full-check` must pass (or `zig fmt --check` on Darwin 25+).
4. **Verify stub sync** — any change to `mod.zig` requires matching `stub.zig` update.
5. **Update `tasks/lessons.md`** after fixing any mistake that could recur.

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

- [AGENTS.md](AGENTS.md) — Full workflow contract, commit guidelines, acceptance criteria
- [docs/FAQ-agents.md](docs/FAQ-agents.md) — Detailed style rules and expanded command docs
- [tasks/todo.md](tasks/todo.md) — Active task tracker
- [tasks/lessons.md](tasks/lessons.md) — Correction log for recurring pitfalls
