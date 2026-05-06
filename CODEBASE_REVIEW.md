# ABI Codebase Review

Use `ONBOARDING.md` first, then this map when planning implementation work. See `GLOSSARY.md` for terminology.

## Core Information
- **Zig Version**: 0.16.0
- **Framework Entrypoint**: `src/root.zig` (`abi` module)
- **Build System**: `./build.sh` (automatic) or `zig build`

## Entry Points

- `src/root.zig` re-exports the public `abi` module surface.
- `src/public/` groups root wiring for core, services, protocols, features, and metadata.
- `src/main.zig` builds the `abi` CLI.
- `src/mcp_main.zig` builds the MCP server entrypoint.
- `build.zig` wires feature flags, checks, tests, and artifacts.

## Major Subsystems

- `src/foundation/`, `src/runtime/`, and `src/platform/` provide shared utilities, scheduling, memory, OS, and environment abstractions.
- `src/protocols/` contains MCP, ACP, HA, and LSP protocol support.
- `src/features/` contains comptime-gated feature modules with `mod.zig`, `stub.zig`, and shared `types.zig` patterns; core feature metadata lives in `src/features/core/`.
- `src/connectors/`, `src/tasks/`, and `src/inference/` hold external adapters, task orchestration, and inference support.

## Refactor Boundaries

Keep refactors subsystem-scoped. Do not move public exports without compatibility re-exports from `src/root.zig` or its grouped `src/public/` modules. Public feature API changes must keep `mod.zig` and `stub.zig` in sync.
