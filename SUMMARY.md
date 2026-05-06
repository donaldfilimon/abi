# ABI Summary

ABI is a **Zig 0.17.0-dev** framework for AI services, semantic vector storage, GPU acceleration, distributed runtime utilities, and MCP/ACP integration.

The package entrypoint is `src/root.zig`, exposed to consumers as `@import("abi")`. Public root wiring is split under `src/public/` by core, service, protocol, feature, and metadata groups while preserving the `abi.<domain>` surface.

## Current Baseline

- Zig pin: see `.zigversion` (0.17.0-dev.251+0db721ec2).
- Build wrapper: `./build.sh` (standard) or `zig build`.
- Parity gate: `./build.sh check-parity`.
- MCP HA config: `mcp/servers.json`.
- MCP launcher: `mcp/launcher.sh`.
- Public API wiring: `src/root.zig` re-exports grouped modules from `src/public/`.
- Doc validation: `.github/workflows/doc-validation.yml` and `scripts/verify-docs.sh`.
- AI Models: `abbeycode` (Gemma4-based, Ollama) for internal chat, trained on Zig 0.17 std lib + ABI source.
- Vector DB: WDBX with 1966 vectors (Zig std lib + ABI src) in `zig_data.db`.
- Internal APIs Only: `abi chat` uses `ollama/abbeycode` connector, no external APIs by default.

## Known Issues

- **Zig 0.17-dev Toolchain Bug**: The `EndOfStream` panic in test runner is a known Zig 0.17-dev toolchain bug (not an ABI codebase issue). This may cause some tests to report errors during finalization. The 3555/3573 test pass rate reflects this upstream issue.

## Refactor Rule

Preserve the public `@import("abi")` surface while modularizing internals. When a feature public API changes, update its `mod.zig` and `stub.zig` together and run parity.
