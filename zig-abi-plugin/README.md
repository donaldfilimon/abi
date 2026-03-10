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

### Agents

| Agent | Purpose |
|--------|---------|
| `stub-sync-validator` | Proactive mod↔stub signature checking after feature edits |

### Hooks

| Event | Action |
|-------|--------|
| `PostToolUse` (Edit/Write) | Warns about stub.zig sync and module import violations |

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

The ABI Framework uses comptime feature gating. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `feat_ai` | true | AI/LLM features |
| `feat_gpu` | true | GPU acceleration |
| `feat_database` | true | Vector database (WDBX) |
| `feat_network` | true | Distributed compute |

All flags default to `true`. Disable with `-Dfeat-<name>=false`.

## Platform Notes

On macOS 26+ (Darwin 26+), the stock Zig linker fails. The plugin auto-detects this and routes to:
1. `run_build.sh` wrapper
2. Fallback validation (`zig fmt --check`, `zig test -fno-emit-bin`)
3. Linux CI when a build step still needs binary emission
