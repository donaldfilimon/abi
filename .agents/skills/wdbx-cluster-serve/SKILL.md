---
name: wdbx-cluster-serve
description: Build the abi CLI and run a networked WDBX consensus node (`abi wdbx cluster serve <port>`) — background-launch the loopback RequestVote/AppendEntries RPC listener, poll until it is serving, assert readiness, then tear it down. Use when asked to run/start/serve/smoke-test the WDBX cluster node, the consensus RPC listener, or `cluster serve`.
---

# wdbx-cluster-serve — run the networked WDBX consensus node

Driver: **`.agents/skills/wdbx-cluster-serve/cluster-serve.sh`** (paths relative to repo root).
Server-type check — background-launch, poll for readiness, assert, kill. Evidence is the `RESULT:` line.

This is the **networked** consensus node (`cluster_rpc` TCP transport: RequestVote/AppendEntries),
distinct from the **in-process** `abi wdbx cluster demo` that `/cluster-demo-guide` covers and the
single-node `cluster status`. Loopback-only by design.

## Prerequisites
- Pinned/master Zig on PATH (see `/zig-newest-skills`). macOS builds via `./build.sh`.
- `nc` for the port probe (optional — the readiness marker is the primary gate; the probe is skipped if `nc` is absent).

## Run (agent path)
```bash
.agents/skills/wdbx-cluster-serve/cluster-serve.sh          # serve on 127.0.0.1:8092
.agents/skills/wdbx-cluster-serve/cluster-serve.sh 8095     # override the port
```
It builds the CLI, launches `abi wdbx cluster serve <port>` in the background, and asserts the
readiness marker `serving consensus RPC on 127.0.0.1:<port>` (printed to stderr before the accept
loop blocks), an open port, and no bind/panic error — then kills the node via an `EXIT` trap.
Prints `RESULT: PASS — WDBX cluster node served consensus RPC on loopback.` (exit 0) or
`RESULT: FAIL — N check(s).`

Historical verification: **PASS** — marker printed, port `8092` accepting, node torn down with no
lingering process, on Zig master `0.17.0-dev.1099`.

## Gotchas
- **Loopback-only.** The driver binds `127.0.0.1`. The RPC transport is **unauthenticated** — any
  peer that reaches the port can forge votes/log entries. `abi wdbx cluster serve <port> 0.0.0.0`
  (or a routable IP) exists for multi-host, and the CLI warns loudly, but do **not** bind non-loopback
  without a threat review (see `abi-threat-model.md`, "WDBX consensus listener"). This skill never does.
- **The node runs until killed.** It blocks in the accept loop; the driver always kills it on exit
  (`EXIT` trap). Re-run leaves nothing behind — confirm with `pgrep -f 'abi wdbx cluster serve'`.
- **Pick a free port.** A bound port fails the bind and the driver reports FAIL; pass a different `$1`.
- `cluster-serve.sh <non-number>` → usage, exit 2.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Run `/zig-build-doctor` or `./build.sh check` for the real error. |
| no readiness marker / `bind … failed` | Port in use or privileged — pick a higher free port (`8095`). |
| missing marker string | CLI grammar drifted — check `src/cli/handlers/wdbx_runtime.zig` `clusterServe`. |

For source-level questions about the consensus/RPC internals, use the `wdbx-explorer` subagent.
