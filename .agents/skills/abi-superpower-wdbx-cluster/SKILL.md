---
name: abi-superpower-wdbx-cluster
description: WDBX cluster consensus superpower. In-process Raft demo, networked RPC transport, node serving, loopback-tested.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["status", "demo", "serve", "rpc-test"]
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

Exposes WDBX consensus/cluster capabilities as a superpower. **Honest scope**: In-process Raft core + real TCP RequestVote/AppendEntries transport over loopback. Non-loopback binds require `ABI_WDBX_CLUSTER_TOKEN`. NOT production multi-host distributed database.

## Actions

### status
Show in-process Raft state-machine status (single node that elects itself leader):
```
/abi-superpower-wdbx-cluster status
```

### demo
Run in-process consensus demo (election, quorum replication, failover):
```
/abi-superpower-wdbx-cluster demo --nodes 3
```

### serve
Serve a networked consensus RPC node:
```
/abi-superpower-wdbx-cluster serve --port 8090 --node-id node1 --host 127.0.0.1
```
- Loopback (`127.0.0.1`) allowed by default
- Non-loopback bind **refuses to start** without `ABI_WDBX_CLUSTER_TOKEN`
- Optional peer allowlist via `ABI_WDBX_CLUSTER_PEERS` (comma-separated node IDs)

### rpc-test
Run authenticated loopback multi-node vote+append round (verifies quorum and peer logs):
```
/abi-superpower-wdbx-cluster rpc-test
```

## Architecture

| Layer | Source | Status |
|-------|--------|--------|
| Raft Core | `src/features/wdbx/cluster.zig` | Current — leader election, majority-quorum replication, failover, quorum-loss detection |
| RPC Transport | `src/features/wdbx/cluster_rpc.zig` | Partial — real TCP RequestVote/AppendEntries, shared-secret frames, optional peer allowlist, loopback-tested |
| CLI Surface | `src/cli/handlers/wdbx_runtime.zig` | Current — `abi wdbx cluster status/demo/serve` |

## Auth & Network

- **Shared-secret**: `ABI_WDBX_CLUSTER_TOKEN` — required for non-loopback binds, included in RequestVote/AppendEntries frames
- **Peer allowlist**: `ABI_WDBX_CLUSTER_PEERS` — optional comma-separated node IDs to restrict accepted peers
- **Transport**: Raw TCP with line-delimited JSON frames (`src/features/wdbx/net_line.zig`)
- **TLS/mTLS**: NOT implemented — deploy behind network policy/proxy for non-loopback

## Gap to Production (§3.5 wdbx-north-star.mdx)

| Missing | Required for Production |
|---------|------------------------|
| Multi-host deployment | TLS/mTLS or equivalent network policy |
| Dynamic membership | Add/remove nodes at runtime |
| Data sharding | Partition vectors/keys across nodes |
| Cross-host transport tests | Currently loopback-only |

## Feature Gates

Requires `feat-wdbx=true` (default). When disabled, returns `FeatureDisabled`.

## Claim Boundary

Per `docs/contracts/external-claims-audit.mdx` and `docs/spec/wdbx-north-star.mdx`:
- ✅ In-process Raft consensus (election, replication, failover)
- ✅ Real TCP RPC transport with auth + peer allowlist
- ❌ NOT production multi-host deployment
- ❌ NOT data sharding
- ❌ NOT TLS/mTLS
- ❌ NOT dynamic membership

Present honestly — this is a reference-scoped demonstration of consensus primitives.