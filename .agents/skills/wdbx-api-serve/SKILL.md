---
name: wdbx-api-serve
description: Build the abi CLI and drive the WDBX REST server end-to-end — start `abi wdbx api serve` on a loopback port, hit GET /health and GET /stats, and verify ABI_WDBX_REST_TOKEN bearer auth (401 without/with wrong token, 200 with the right one). Use to run/start/smoke-test the WDBX REST API. Loopback only; kills the server on exit.
---

# wdbx-api-serve — drive the WDBX REST server

Driver: **`.agents/skills/wdbx-api-serve/serve.sh`** (paths relative to repo root).
Builds the CLI, launches the REST listener on `127.0.0.1`, exercises it with
`curl`, and tears the server down (EXIT trap). Evidence is the `RESULT:` line.
Fully local, loopback only.

## Run (agent path)
```bash
.agents/skills/wdbx-api-serve/serve.sh        # default ports 8091 (no-auth) + 8092 (auth)
.agents/skills/wdbx-api-serve/serve.sh 9100   # custom base port (uses 9100 + 9101)
```
- Launches `abi wdbx api serve <port>`, polls `GET /health` until up → asserts
  `{"status":"ok"}`.
- `GET /stats` → asserts the store-stats JSON (`"backend"`).
- Restarts with `ABI_WDBX_REST_TOKEN=probe-tok` on `<port>+1` and asserts bearer
  auth: no token → **401**, wrong token → **401**, `Authorization: Bearer probe-tok` → **200**.

Prints `RESULT: PASS` (exit 0) or a FAIL count. Both servers are killed on exit.

Current Rust driver: uses an isolated `ABI_WDBX_PATH`, proves loopback health
and stats, then checks bearer enforcement is exactly 401/401/200.

## Gotchas
- ⚠️ **Loopback + local hardening only.** The listener binds `127.0.0.1`;
  `ABI_WDBX_REST_TOKEN` is a bearer gate, **not** a substitute for a TLS
  fronting layer. Same scheme as the MCP HTTP transport's `ABI_MCP_HTTP_TOKEN`.
- The driver uses two ports (`<port>` and `<port>+1`); pick a base that leaves
  the next port free. It always kills both servers via an EXIT trap — check
  `pgrep -f 'abi wdbx api serve'` returns nothing after a run.
- `auth=off` in the startup log is expected when `ABI_WDBX_REST_TOKEN` is unset;
  it flips to `auth=on` when the env var is present.
- `/stats` reports the backend label separately from `mode=cpu_fallback`; a
  `metal` label alone is not evidence that a kernel executed.
- For a source-level tour of the REST routing core (`rest.rs` `route`) and the
  WAL/checkpoint substrate, use the `wdbx-explorer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check nightly via `./tools/cargo.sh --version`, then `./tools/check.sh`. |
| `server did not come up` | Port in use — pass a free base port; or the build didn't produce the binary. |
| bearer test all 200 | `ABI_WDBX_REST_TOKEN` is not being enforced — check `RestConfig::from_env` and `abi_foundation::http::has_bearer_token`. |
