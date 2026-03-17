# zig-abi-plugin

Claude Code plugin for ABI Framework development. Provides smart build routing, Zig 0.16 patterns, feature module scaffolding, and real-time verification.

## Installation

```bash
claude --plugin-dir zig-abi-plugin
```

## Components

### Commands

| Command | Purpose |
|---------|---------|
| `/zig-abi:build [step]` | Smart build with Darwin workaround detection |
| `/zig-abi:check [scope]` | Verification checks (format, imports, stub-sync, deprecated) |
| `/zig-abi:new-feature <name>` | Scaffold new feature module (8-step process) |

### Skills

| Skill | Trigger |
|-------|---------|
| `zig-016-patterns` | Writing Zig code, compilation errors, API questions |
| `abi-architecture` | Feature modules, build system, comptime gating |
| `abi-code-review` | Code review with ABI-specific heuristics |
| `cel-language` | CEL expression syntax, evaluation, policy rules |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/load-context.sh` | SessionStart hook: platform detection, Zig version, task reminders |
| `scripts/check-flag-sync.sh` | Validate feature flag counts across build files |
| `scripts/audit-darwin-targets.sh` | Audit darwinRelink() wiring on executables |
| `skills/abi-code-review/scripts/review_prep.py` | Prepare ABI-specific review context from diffs |

### Agents

| Agent | Purpose |
|--------|---------|
| `stub-sync-validator` | Proactive mod↔stub signature checking after feature edits |

### Hooks

| Event | Action |
|-------|--------|
| `SessionStart` | Loads platform context, checks Zig version and pinned version match |
| `PostToolUse` (Edit/Write) | Warns about stub.zig sync and module import violations |
| `PreToolUse` (Bash) | Warns against `zig fmt .` from root (use specific dirs) |
| `PreToolUse` (Edit) | Warns about `@import("abi")` inside `src/features/` |
| `Stop` | Advisory checklist: stub sync, formatting, CLI registry |

## Quick Reference

```bash
# Build with platform detection
/zig-abi:build full-check

# Verify all aspects
/zig-abi:check all

# Scaffold new feature
/zig-abi:new-feature scheduling
```

## Feature Flags

The ABI Framework uses comptime feature gating with 27 `feat_*` flags and 56 validated combos. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `feat_ai` | true | AI/LLM features |
| `feat_gpu` | true | GPU acceleration |
| `feat_database` | true | Vector database (WDBX) |
| `feat_network` | true | Distributed compute |

All flags default to `true`. Disable with `-Dfeat-<name>=false`.

## Platform Notes

On macOS 25+ (Darwin 25+), the stock Zig linker fails. The plugin auto-detects this and routes to:
1. `run_build.sh` wrapper
2. Fallback validation (`zig fmt --check`, `zig test -fno-emit-bin`)
3. Linux CI when a build step still needs binary emission
