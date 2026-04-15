# GEMINI.md — ABI Framework

This file provides guidance to Google Gemini when working with code in this repository. See also CLAUDE.md and QWEN.md for parallel AI-specific guidance.

## Project Overview

ABI is a **Zig 0.16 framework** for AI services, semantic vector storage, GPU acceleration, and distributed runtime. This repository implements a multi-AI orchestration system (Abbey-Aviva-Abi pipeline) with constitutional AI governance.

## Quick Reference

- **Entry point**: `src/root.zig` (exported as `@import("abi")`)
- **Zig version**: Pinned in `.zigversion` (0.16.x)
- **Build wrapper**: `./build.sh` (macOS 26.4+) / `zig build` (Linux)
- **Test gate**: `./build.sh check` or `zig build check`
- **Parity check**: `zig build check-parity` (required after API changes)

## Core Architecture

### Multi-Level AI Pipeline

```
Input → Abi Analyzer → Adaptive Modulator (EMA learning) → Router → Profile Executor → Constitution Check → WDBX Store → Response
         │                    │                                 │              │
         ↓                    ↓                                 ↓              ↓
    Sentiment +         User preferences              Abbey/Aviva/Abi    6 Principles
    Policy + Rules       (adaptive routing)            profile execution   validation
```

### Module Organization

| Directory | Purpose |
|-----------|---------|
| `src/core/` | Framework lifecycle, config, registry |
| `src/features/` | 21 feature modules (mod/stub/types pattern) |
| `src/foundation/` | Utilities: logging, security, time, SIMD, sync |
| `src/runtime/` | Task scheduling, event loops, concurrency |
| `src/inference/` | ML engine: scheduler, sampler, KV cache |
| `src/connectors/` | LLM providers (OpenAI, Anthropic, etc.) |
| `src/protocols/` | MCP, ACP, LSP, HA protocol implementations |

### The Mod/Stub Pattern

Every feature follows this contract:
- `mod.zig` — Full implementation when feature enabled
- `stub.zig` — API-compatible no-ops when feature disabled
- `types.zig` — Shared types for both

**Critical**: Update BOTH `mod.zig` AND `stub.zig` when changing public APIs. Always run `zig build check-parity` after modifications.

### Feature Flags

All features default enabled except `feat-mobile` and `feat-tui`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false    # Disable GPU and AI
zig build -Dgpu-backend=metal                  # Set GPU backend
zig build -Dfeat-tui=true                      # Enable TUI features
```

## Build System

### macOS 26.4+ (Darwin 25.x)
**Critical**: Always use `./build.sh` — it relinks with Apple's native linker (LLD fails on this OS version).

```bash
./build.sh                    # Build static library
./build.sh cli                # Build CLI binary
./build.sh mcp                # Build MCP server
./build.sh test --summary all # Run all tests
./build.sh check              # Lint + test + stub parity
./build.sh full-check         # Full validation gate
```

### Linux/Older macOS
```bash
zig build test --summary all       # Run all tests
zig build check                    # Full gate (lint + parity)
zig build check-parity             # Verify mod/stub API match
zig build cli                      # Build CLI (zig-out/bin/abi)
zig build mcp                      # Build MCP server
```

## Development Conventions

### Naming Standards
- Functions/variables: `camelCase`
- Types/structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Enum variants: `snake_case`

### Import Rules (Critical)
1. **Within `src/`**: Use relative imports ONLY. Never `@import("abi")` from inside — causes circular import.
2. **From `test/`**: Use `@import("abi")` and `@import("build_options")`.
3. **Cross-feature**: Use conditional imports with build_options guards.
4. **Always use `.zig` extension** on path imports.

### Error Handling
| Mechanism | When to Use |
|-----------|-------------|
| `@compileError` | Compile-time contract violations only |
| `@panic` | Unrecoverable invariants; CLI/tests only |
| `unreachable` | Provably impossible branches (compiler-verified) |
| Error unions | All runtime failures in library code |

### Testing Requirements
```zig
test {
    std.testing.refAllDecls(@This());
}
```

**Known pre-existing failures**: 2 inference engine connector tests, 1 auth integration test (not regressions).

### Zig 0.16 Gotchas
- `ArrayListUnmanaged` init: `.empty` not `.{}`
- `std.BoundedArray` removed: use manual `buffer: [N]T` + `len`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- Entry point: `pub fn main(init: std.process.Init) !void`
- `std.mem.trimRight` renamed to `std.mem.trimEnd`

## AI Feature Structure (`src/features/ai/`)

| Sub-directory | Contents |
|---------------|----------|
| `abbey/`, `aviva/`, `abi/` | Three personality profiles |
| `constitution/` | 6-principle AI governance |
| `agents/`, `multi_agent/`, `orchestration/` | Agent systems |
| `llm/`, `embeddings/`, `vision/`, `streaming/` | Core AI capabilities |
| `abbey/`, `reasoning/`, `eval/` | Reasoning systems |
| `training/`, `federated/`, `memory/` | Learning infrastructure |
| `pipeline/` | Composable prompt DSL with WDBX backing |

## Workflows

### Before Starting
1. Read `tasks/lessons.md`
2. Update `tasks/todo.md` for non-trivial changes

### During Development
3. Run `zig build check-parity` after any public API change
4. Use `./build.sh full-check` (macOS) or `zig build full-check` (Linux) as verification gate

### When Done
5. Conventional Commits required
6. Do NOT use `rm` — use safe alternatives only

## Available Resources

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Detailed Claude Code guidance |
| `QWEN.md` | Qwen-specific guidance |
| `AGENTS.md` | Project-wide agent conventions |
| `.zigversion` | Pinned Zig version |
| `build.sh` | macOS 26.4+ build wrapper |
| `tools/zigly` | Zig version manager |
| `docs/spec/ABBEY-SPEC.md` | Comprehensive mega-spec |

## Key Rules Summary

1. **Never `@import("abi")` from `src/`** — cyclic dependency
2. **Always use `./build.sh`** on macOS 26.4+
3. **Always update stub.zig with mod.zig** changes
4. **Always dupe string literals** in structs with `deinit()`
5. **No `rm` command** — use safe alternatives
6. **Parity check required** after any public API modification
