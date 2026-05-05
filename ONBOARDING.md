# ABI Onboarding

Start here when opening a new ABI session.

## Bootstrap

- Read `AGENTS.md` for agent-specific guidance.
- Use the pinned Zig version from `.zigversion`.
- Run `tools/zigly --bootstrap` to install/link Zig and ZLS.
- On macOS 26.4+ use `./build.sh`; on Linux or older macOS, `zig build` is acceptable.

## First Checks

```bash
zig version
./build.sh check-parity
./build.sh test --summary all
```

## MCP And ACP Readiness

- Build MCP with `./build.sh mcp`.
- Start through `mcp/launcher.sh`; the HA instances are configured in `mcp/servers.json`.
- Check MCP health with `scripts/check-mcp-health.sh`.
- Check MCP-ACP interop with `scripts/check-interop.sh`.
- Inspect ACP endpoints with `scripts/list-acp-endpoints.sh` when `ACP_ENDPOINTS` is set.

## Navigation

- `ONBOARDING_INDEX.md` links the main ramp-up documents.
- `SUMMARY.md` gives a compact repository overview.
- `CODEBASE_REVIEW.md` outlines architecture entry points.
- `GLOSSARY.md` defines repo terms and points back to this onboarding guide.
- `docs/onboarding.md` contains the longer first-day guide.
