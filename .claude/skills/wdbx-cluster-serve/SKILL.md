---
name: wdbx-cluster-serve
description: Build the abi CLI and run a networked WDBX consensus node (`abi wdbx cluster serve <port>`) — background-launch the loopback RequestVote/AppendEntries RPC listener, poll until it is serving, assert readiness, then tear it down. Use when asked to run/start/serve/smoke-test the WDBX cluster node, the consensus RPC listener, or `cluster serve`.
---

> **WDBX moved out of this repository on 2026-08-22.** It now lives in the
> sibling repo `~/dev/active/wdbx` together with `abi-compute`,
> `abi-foundation`, `abi-core`, and `abi-telemetry`; `abi` consumes them by
> relative path. Source paths below therefore read `../wdbx/crates/...`. Run
> WDBX-only tests from that repo (`cargo test --workspace`), and `abi`'s gate
> (`./tools/check.sh`) from here.
>
> Under the Abbey System Constitution
> (`docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`) WDBX is the
> **provenance-aware episodic substrate**, not a vector store. Most of the
> evidence half of its specification is unimplemented; the measured gap list is
> in `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`. Do not
> describe an episodic capability as Current on the strength of the vector-store
> features that do exist.

# wdbx-cluster-serve — run the networked WDBX consensus node

Driver: **`.agents/skills/wdbx-cluster-serve/cluster-serve.sh`** (paths relative to repo root).
Server-type check — background-launch, poll for readiness, assert, kill. Evidence is the `RESULT:` line.

This is the **networked** consensus node (`cluster_rpc` TCP transport: RequestVote/AppendEntries),
distinct from the **in-process** `abi wdbx cluster demo` that `/cluster-demo-guide` covers and the
single-node `cluster status`. Loopback-only by design.

## Prerequisites
- Nightly Rust via `./tools/cargo.sh` (never bare Homebrew `cargo`).
- `nc` for the port probe (optional — the readiness marker is the primary gate; the probe is skipped if `nc` is absent).

## Run (agent path)
```bash
.agents/skills/wdbx-cluster-serve/cluster-serve.sh          # serve on 127.0.0.1:8092
.agents/skills/wdbx-cluster-serve/cluster-serve.sh 8095     # override the port
```
It builds the CLI, launches `abi wdbx cluster serve <port>` in the background, and asserts the
readiness marker `wdbx cluster RPC serving on 127.0.0.1:<port>` (printed to stderr before the accept
loop blocks), an open port, and no bind/panic error — then kills the node via an `EXIT` trap.
Prints `RESULT: PASS — WDBX cluster node served consensus RPC on loopback.` (exit 0) or
`RESULT: FAIL — N check(s).`

Current Rust driver: requires the readiness marker and accepting loopback port,
then tears the node down and checks for bind/panic failures.

## Gotchas
- **Loopback-only.** The driver binds `127.0.0.1` in compatibility mode. A
  non-loopback bind fails closed unless `ABI_WDBX_CLUSTER_TOKEN` is configured,
  but the reference transport still has no TLS and is not production multi-host.
- **The node runs until killed.** It blocks in the accept loop; the driver always kills it on exit
  (`EXIT` trap). Re-run leaves nothing behind — confirm with `pgrep -f 'abi wdbx cluster serve'`.
- **Pick a free port.** A bound port fails the bind and the driver reports FAIL; pass a different `$1`.
- `cluster-serve.sh <non-number>` → usage, exit 2.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Run `./tools/check.sh` for the real error. |
| no readiness marker / `bind … failed` | Port in use or privileged — pick a higher free port (`8095`). |
| missing marker string | CLI grammar drifted — check `crates/abi-cli/src/wdbx/cluster.rs`. |

For source-level questions about the consensus/RPC internals, use the `wdbx-explorer` subagent.
