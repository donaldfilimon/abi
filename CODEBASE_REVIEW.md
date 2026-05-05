# ABI Codebase Review

Use `ONBOARDING.md` first, then this map when planning implementation work. See `GLOSSARY.md` for terminology.

## Entry Points

- `src/root.zig` re-exports the public `abi` module surface.
- `src/main.zig` builds the `abi` CLI.
- `src/mcp_main.zig` builds the MCP stdio server.
- `build.zig` wires feature flags, checks, tests, and artifacts.

## Major Subsystems

- `src/foundation/`, `src/runtime/`, and `src/platform/` provide shared utilities, scheduling, memory, OS, and environment abstractions.
- `src/protocols/` contains MCP, ACP, HA, and LSP protocol support.
- `src/features/` contains comptime-gated feature modules with `mod.zig`, `stub.zig`, and shared `types.zig` patterns.
- `src/connectors/`, `src/tasks/`, and `src/inference/` hold external adapters, task orchestration, and inference support.

## Refactor Boundaries

Keep refactors subsystem-scoped. Do not move public exports without compatibility re-exports from `src/root.zig`. Public feature API changes must keep `mod.zig` and `stub.zig` in sync.
