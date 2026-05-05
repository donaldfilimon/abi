# ABI Glossary

See `ONBOARDING.md` for bootstrap steps and `CODEBASE_REVIEW.md` for the architecture map.

- ABI: the framework package exposed as `@import("abi")`.
- Public root: `src/root.zig`, a compatibility re-export layer for grouped wiring in `src/public/`.
- ACP: Agent Communication Protocol integration points and endpoint checks.
- MCP: Model Context Protocol server support, launched through `mcp/launcher.sh`; supports stdio integrations and SSE/health-check transport when enabled.
- HA: high availability; ABI configures `abi-mcp-1` and `abi-mcp-2` in `mcp/servers.json`.
- Mod/stub parity: each feature's enabled implementation and disabled stub must expose the same public API.
- Zigly: the repository-local Zig/ZLS version manager at `tools/zigly`.
