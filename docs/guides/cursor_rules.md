# Cursor Rules

Cursor-specific constraints and task templates for working in the ABI repository.

> **Status**: Pending formal policy approval. See [AGENTS.md](../../AGENTS.md) Section 4.

## General Rules

1. All generated logic MUST be validated against the Zig 0.16 baseline and repo conventions.
2. Use `@import("abi")` for framework consumption, not cross-directory relative imports.
3. Every `mod.zig` change requires a matching `stub.zig` update.
4. Run `zig build full-check` before marking any task complete.
5. Never run `zig fmt .` from repo root — use `zig fmt --check build.zig build/ src/ tools/`.
