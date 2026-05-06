# AGENTS.md — ABI Framework

Compact instructions for automation and rapid ramp-up.

## Toolchain & Build
- **Zig Pin**: `.zigversion` (Zig 0.17.0-dev). Use `tools/zigly` to manage.
- **macOS 26.4+**: Always use `./build.sh` (adds `-Dfeat-gpu=false` for stability).
- **Core Commands**:
  - `./build.sh check` — The primary gate (lint + typecheck + parity + tests).
  - `./build.sh cli` — Build CLI (`zig-out/bin/abi`).
  - `./build.sh mcp` — Build MCP server.
  - `zig build check-parity` — Verify `mod.zig` vs `stub.zig` API match.

## Architecture Quirks
- **Mod/Stub Pattern**: Features implement `mod.zig` and `stub.zig`. Changing public APIs requires updates to both. Run parity check after any change.
- **Imports**: Never `@import("abi")` within `src/` (circular dependency). Use relative imports.
- **Error Handling**: Silent error swallowing (`catch {}`) is strictly forbidden in data access, inference, and persistence paths. Errors must be logged or propagated.

## Operational & Debugging
- **Database Engine**: Uses `RwLock` (`db_lock`). Public methods (`index`, `search`, etc.) must acquire shared/exclusive locks.
- **Compute Mesh**: Discovery error handling is fallible; `registerPeer` propagates OOM errors. 
- **GPU Fallback**: `stdgpu/simulated` is the default CPU fallback. Search failures in HNSW GPU acceleration propagate error instead of silently swallowing.
- **Launcher**: Always use `mcp/launcher.sh` for cross-platform MCP server execution.
- **CI**: Commits with `TODO`/`FIXME` are blocked.

## Critical Files
- `.zigversion`: Toolchain pinning.
- `mcp/servers.json`: HA instance configuration.
- `src/features/core/database/engine.zig`: Engine synchronization entrypoint.
- `build.sh`: macOS build wrapper.
