---
name: mcp-smoke
description: Build the abi MCP server and smoke-test its JSON-RPC tool surface — start `abi-mcp`, send a `tools/list` over the stdio transport, and assert the frozen 12-tool contract (ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run). Use to run/start/smoke-test abi-mcp, verify the MCP tool list, or check the 12-tool contract still holds. Local stdio; the process exits on pipe EOF.
---

# mcp-smoke — assert the MCP server's 12-tool contract

Driver: **`.agents/skills/mcp-smoke/smoke.sh`** (paths relative to repo root).
Builds `abi-mcp` via `./tools/cargo.sh`, sends one JSON-RPC `tools/list` over
**stdio**, and asserts the frozen 12-tool set. Fully local — no network.

## Run (agent path)
```bash
.agents/skills/mcp-smoke/smoke.sh
```
Prints `RESULT: PASS — 12/12 frozen MCP tools present` (exit 0) or `RESULT: FAIL`.

One-liner:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | ./target/debug/abi-mcp 2>/dev/null | jq -r '.result.tools[].name'
```

## Gotchas
- ⚠️ **Do NOT count tools with `grep -c '"name":'` — it returns 13, not 12.**
  Use `jq -r '.result.tools[].name'` or count only tool-list name values.
- **stdio framing is newline-delimited JSON** — not LSP `Content-Length`.
- **No hang, no `timeout` needed.** Pipe EOF exits the stdio loop.
- Optional loopback HTTP/SSE may log on stderr; drop stderr when grepping stdout.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/cargo.sh build -p abi-mcp` then `./tools/check.sh` |
| empty response | trailing `\n` required; drop stderr |
| got 13 tools | over-counting schema `"name"` fields |
| tool set mismatch | reconcile `crates/abi-mcp/src/handlers.rs` with AGENTS.md frozen list |

Historical golden: `tests/golden/mcp-tools-list.json`.
