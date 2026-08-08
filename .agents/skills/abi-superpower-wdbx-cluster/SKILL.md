---
name: abi-superpower-wdbx-cluster
description: WDBX cluster superpower. Raft/RPC plus signed membership, exact replication/read repair, deterministic placement/rebalance, and authenticated single-host multi-process proof.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["status", "demo", "local-demo", "serve", "rpc-test"]
      description: "Cluster action"
    - name: "port"
      type: "integer"
      description: "Port for serve (default 8090)"
    - name: "nodes"
      type: "integer"
      description: "Number of nodes for demo (default 3)"
    - name: "host"
      type: "string"
      description: "Host for serve (default 127.0.0.1)"
    - name: "node-id"
      type: "string"
      description: "Node ID for serve"
---

# ABI Superpower: WDBX Cluster

Exposes WDBX cluster capabilities. **Honest scope**: exact committed-transaction
replication/read repair, signed membership, rendezvous placement, verified
resumable rebalance, and authenticated local processes are tested on one host.
This is not a production multi-host distributed database.

## Actions

### status
Show in-process Raft state-machine status (single node that elects itself leader):
```
abi wdbx cluster status
```

### demo
Run in-process consensus demo (election, quorum replication, failover):
```
abi wdbx cluster demo 3
```

### serve
Serve a networked consensus RPC node:
```
abi wdbx cluster serve 8090 node1 127.0.0.1
```
- Loopback (`127.0.0.1`) allowed by default
- Non-loopback bind **refuses to start** without `ABI_WDBX_CLUSTER_TOKEN`
- Optional peer allowlist via `ABI_WDBX_CLUSTER_PEERS` (comma-separated node IDs)

### local-demo
Run the authenticated exact-data proof with 3–9 local processes:
```bash
abi wdbx cluster local-demo 3 --json
```
This proves single-host process/RPC behavior only, not real separate-host,
hosted/Windows, or production operation.

### RPC tests

Authenticated loopback vote/append coverage lives in the `abi-wdbx` test
suite. There is no public `rpc-test` CLI subcommand.

## Architecture

| Layer | Source | Status |
|-------|--------|--------|
| Raft Core | `crates/abi-wdbx/src/cluster.rs` | Current — leader election, majority-quorum replication, failover, quorum-loss detection |
| RPC Transport | `crates/abi-wdbx/src/cluster_rpc.rs` | Partial — real TCP RequestVote/AppendEntries, shared-secret frames, optional peer allowlist, loopback-tested |
| Exact data plane | `crates/abi-wdbx/src/{v2/replication,cluster/{replication,repair,rebalance,placement}}.rs` | Current — exact identity, conflict retention, stable fan-out, resumable plans |
| Signed membership | `crates/abi-wdbx/src/cluster/membership.rs` | Current — signed membership records and verification |
| CLI Surface | `crates/abi-cli/src/wdbx/cluster.rs` | Current — `status/demo/local-demo/serve` |

## Auth & Network

- **Shared-secret**: `ABI_WDBX_CLUSTER_TOKEN` — required for non-loopback binds, included in RequestVote/AppendEntries frames
- **Peer allowlist**: `ABI_WDBX_CLUSTER_PEERS` — optional comma-separated node IDs to restrict accepted peers
- **Transport**: Raw TCP with line-delimited JSON frames (`crates/abi-wdbx/src/net_line.rs`)
- **TLS/mTLS**: NOT implemented — deploy behind network policy/proxy for non-loopback

## Gap to Production (§3.5 wdbx-north-star.mdx)

| Missing | Required for Production |
|---------|------------------------|
| Multi-host deployment | TLS/mTLS or equivalent network policy |
| Production membership operations | Signed local records do not prove production control-plane operations |
| Production sharding | Local placement/replication APIs do not prove a production distributed database |
| Cross-host/hosted tests | Current exact multi-process proof is single-host |

## Build and runtime boundary

`abi-wdbx` is a normal Rust workspace crate; there is no `feat-wdbx` switch or
`FeatureDisabled` stub. The real CLI surface is `abi wdbx cluster
status|demo|local-demo|serve`; `rpc-test` is not a public CLI subcommand.

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx` and `docs/spec/wdbx-north-star.mdx`:
- ✅ In-process Raft consensus (election, replication, failover)
- ✅ Real TCP RPC transport with auth + peer allowlist
- ✅ Exact committed-object quorum replication and conflict-preserving read repair
- ✅ Signed membership, rendezvous placement, stable fan-out, verified resumable rebalance
- ✅ Authenticated 3–9 process local demo
- ❌ NOT production multi-host deployment
- ❌ NOT production sharding despite local placement/replica APIs
- ❌ The legacy cluster RPC itself has no TLS/mTLS; the separate gateway's local TLS/mTLS tests do not change that transport boundary
- ❌ NOT production dynamic-membership operations; signed generation-checked membership records are locally tested data/control primitives

Present honestly — current evidence is exact and useful, but single-host scoped.
