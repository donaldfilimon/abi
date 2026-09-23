---
name: mcp
description: Plan abi MCP server work — the 12-tool JSON-RPC 2.0 stdio surface plus its custom loopback HTTP compatibility listener. Use for abi-mcp, tools, transports, or middleware.
---

# mcp

Entry point for the abi MCP server (`crates/abi-mcp/src/`). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Smoke-test abi-mcp + verify the 12-tool contract | `mcp-smoke` |
| Deep-dive the MCP superpower | `abi-superpower-mcp` |
| Transport / middleware / protocol limits detail | `abi-mcp-transport` |

## Frozen contract (do not change without a parity/contract-test update)
- 12 tools, in source order: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`,
  `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`,
  `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- `protocol.MAX_REQUEST_SIZE` = 64 KB; `MAX_JSON_DEPTH` = 32; per-field 16 KB
  cap in `crates/abi-mcp/src/middleware.rs` (declarative validation before dispatch).
- Frozen enums: `connector_test` tool arg `service` ∈ {openai, anthropic, discord,
  twilio, grok}; `ai_train` tool arg `format` ∈ {jsonl, csv, text}.

## Honest boundary
Stdio exits on stdin EOF (not a long-lived daemon). Startup also attempts the
custom loopback listener (`127.0.0.1:8080` by default, configured with
`ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN`); bind failure leaves stdio running.
`GET /sse` opens a persistent MCP 2024-11-05 HTTP+SSE session whose
`POST /message?sessionId=<id>` responses arrive as SSE `message` events (`202`
on the POST; unknown session `404`; at most 16 sessions), while `POST /message`
without a session keeps the one-shot direct-reply mode. It is not Streamable
HTTP (2025-03-26), and non-loopback serving is not supported. Rust handlers return
bounded JSON-RPC errors without exposing internal error chains.

Client registrations (cleaned 2026-09): consistent across grok/claude/codex/opencode/cursor + abi project scopes. Use launcher for abi-mcp everywhere. See `opencode` and `help` skills for the exact current lists.

## Finalized state (2026-09-16)
- Best options: central only ~/.grok/skills + launch.sh sync (idempotent, 0 on re-run); donald-mode; bot-repo authority separate (abbey-bot etc); permission fixes in opencode.jsonc + .grok/config.toml; key comments (GITHUB_TOKEN="", disabled with # requires KEY).
- Global MCPs standardized: .cursor/mcp.json, .claude.json, ~/.grok/config.toml [mcp_servers.*], ~/.config/opencode/opencode.jsonc — core: abi-mcp (via /.../abi/mcp/launcher.sh stdio), skill-loop@0.3.3, context7, fetch(uvx), filesystem(~/.abi/), github(unauth), memory, playwright. (4 more disabled w/ comments).
- Verifs: central sync-clis.py --dry-run = "0 actions/changes"; abi in-repo .agents/skills/sync-clis/launch.sh --dry-run (separate mech); mcp-smoke.sh = "RESULT: PASS — 12/12"; configs parse clean; no drift (timestamps match, dry=0); key files re-read (abs paths).
- Maintain: ALWAYS edit only central ~/.grok/skills/* (or .grok/config.toml for grok MCPs); run sync; never hand-edit copies. Abbey-bot/others: own AGENTS/CLAUDE, separate.
- No loose ends. All per context + final re-checks.
