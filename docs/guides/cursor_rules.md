# Cursor Rules

Cursor-specific constraints and task templates for working in the ABI repository.

> **Canonical sources**: [AGENTS.md](../../AGENTS.md) (workflow contract, verification gates, acceptance criteria) and [CLAUDE.md](../../CLAUDE.md) (architecture, build commands, API patterns). This file supplements those with Cursor-specific notes.

## General Rules

1. All generated logic MUST be validated against the Zig 0.16 baseline and repo conventions.
2. **Import rule**: `@import("abi")` for code outside `src/`; relative imports within `src/` (see [CLAUDE.md](../../CLAUDE.md) Import Rules).
3. Every `mod.zig` change requires a matching `stub.zig` update.
4. Run `zig build full-check` before marking any task complete (see [AGENTS.md](../../AGENTS.md) Verification Gates for Darwin fallbacks).
5. Never run `zig fmt .` from repo root — use `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`.
