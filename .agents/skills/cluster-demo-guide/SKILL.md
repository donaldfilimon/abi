---
name: cluster-demo-guide
description: Build the abi CLI and run the in-process WDBX Raft-style consensus demo — leader election, log replication, leader failover, and re-election — plus the single-node cluster status. Use when asked about WDBX clustering, consensus, Raft, or failover behavior.
---

# cluster-demo-guide — drive the WDBX consensus demo

Driver: **`.agents/skills/cluster-demo-guide/cluster.sh`** (paths relative to repo root).
Read-only CLI capture — evidence is the `RESULT:` line + the election/replication trace.

## Run (agent path)
```bash
.agents/skills/cluster-demo-guide/cluster.sh        # 3 nodes (default)
.agents/skills/cluster-demo-guide/cluster.sh 5      # N nodes
```
Builds the CLI, runs `abi wdbx cluster status` and `abi wdbx cluster demo <n>`,
and asserts `leader_elected=true`, `replicate(`, and `re-election`. Prints
`RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — elect → replicate
(acks=3 quorum=2) → down leader → re-elect (term=2) on a 3-node in-process cluster.

## Gotchas
- This is **in-process** consensus. The CLI itself notes networked RPC transport
  is a Phase-2 item; `cluster serve <port>` exposes the RequestVote/AppendEntries
  RPC but the demo is single-process.
- `cluster status` (no demo) reports the single-node default (`nodes=1`).
- Malformed node count → usage (exit 2).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `/zig-build-doctor` or `./build.sh check`. |
| missing `leader_elected=true` | check `src/features/wdbx/cluster.zig`; use the `wdbx-explorer` subagent. |
