# ABI Summary

ABI is a Zig 0.17-dev framework for AI services, semantic vector storage, GPU acceleration, distributed runtime utilities, and MCP/ACP integration.

The package entrypoint is `src/root.zig`, exposed to consumers as `@import("abi")`. Public root wiring is split under `src/public/` by core, service, protocol, feature, and metadata groups while preserving the `abi.<domain>` surface.

## Current Baseline

- Zig pin: see `.zigversion`.
- Build wrapper: `./build.sh`.
- Parity gate: `./build.sh check-parity`.
- MCP HA config: `mcp/servers.json`.
- MCP launcher: `mcp/launcher.sh`.
- Public API wiring: `src/root.zig` re-exports grouped modules from `src/public/`.
- Doc validation: `.github/workflows/doc-validation.yml` and `scripts/verify-docs.sh`.

## Refactor Rule

Preserve the public `@import("abi")` surface while modularizing internals. When a feature public API changes, update its `mod.zig` and `stub.zig` together and run parity.
