---
name: mcp-smoke
description: Build the abi MCP server and smoke-test its JSON-RPC tool surface — start `abi-mcp`, send a `tools/list` over the stdio transport, and assert the frozen 12-tool contract (ai_run, ai_complete, ai_learn, ai_train, wdbx_query, scheduler_stats, scheduler_info, connector_test, gpu_status, plugin_list, wdbx_stats, plugin_run). Use to run/start/smoke-test abi-mcp, verify the MCP tool list, or check the 12-tool contract still holds. Local stdio; the process exits on pipe EOF.
---

# mcp-smoke — assert the MCP server's 12-tool contract

Driver: **`.claude/skills/mcp-smoke/smoke.sh`** (paths relative to repo root).
Builds `abi-mcp`, sends one JSON-RPC `tools/list` over the **stdio** transport,
extracts the tool names, and asserts they exactly match the frozen 12-tool set.
Evidence is the `RESULT:` line. Fully local — no network dispatch.

## Run (agent path)
```bash
.claude/skills/mcp-smoke/smoke.sh
```
Prints `RESULT: PASS — 12/12 frozen MCP tools present` (exit 0) or `RESULT: FAIL`
with a `diff` of expected-vs-got tool names (exit 1).

One-liner if you just want the names by hand:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | ./zig-out/bin/abi-mcp 2>/dev/null | jq -r '.result.tools[].name'
```

## Gotchas
- ⚠️ **Do NOT count tools with `grep -c '"name":'` — it returns 13, not 12.**
  `plugin_run`'s `inputSchema` has its own `name` property, so bare `"name":`
  over-counts by one and looks like a broken contract. Count name *values*
  (`grep -oE '"name":"[^"]+"'`) or use `jq -r '.result.tools[].name'`.
- **stdio framing is newline-delimited JSON** — one JSON object per line,
  `\n`-terminated. This is NOT the LSP-style `Content-Length` framing; that
  framing belongs only to the HTTP/SSE transport (`GET /sse`, `POST /message`).
- **No hang, no `timeout` needed.** The stdio loop breaks on EOF, so piping the
  request in (`printf ... | abi-mcp`) closes stdin and the process exits on its
  own. Don't reach for `timeout` — macOS doesn't ship it (`gtimeout` via
  coreutils only). Running `abi-mcp` with an open/interactive stdin is what hangs.
- **`initialize` is not required.** A bare `tools/list` returns the full list;
  you don't need the usual MCP `initialize` handshake first.
- **The build is silent.** `./build.sh mcp` prints ~2 info lines and exits with
  no "success" banner — confirm with `ls zig-out/bin/abi-mcp`, not stdout.
- **Ignore the stderr line** `info: MCP HTTP/SSE server listening on
  http://127.0.0.1:8080` — the loopback HTTP listener always starts; the
  JSON-RPC reply you want is on **stdout**. The driver drops stderr with `2>/dev/null`.
- Request cap is 64 KB; `ABI_MCP_HTTP_TOKEN` gates the HTTP/SSE transport only —
  stdio JSON-RPC stays tokenless local IPC (see CLAUDE.md).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| empty response | You piped to the HTTP framing or forgot the trailing `\n`; use the one-liner above. |
| got 13 tools | You counted bare `"name":`; count `"name":"…"` values or use `jq` (see Gotchas). |
| tool set mismatch | A tool was added/removed/renamed — reconcile `src/mcp/handlers.zig` with the frozen list in CLAUDE.md and `tests/contracts/surface.zig`, then run `zig build test-mcp-contracts`. |

Historical verification: **PASS** on Zig master `0.17.0-dev.1099` — `tools/list` over
stdio returns exactly the 12 frozen tools. For source-level MCP contract review use
the `mcp-contract-auditor` subagent; for transport tests run `zig build test-mcp-server`.
