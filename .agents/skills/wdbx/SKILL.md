---
name: wdbx
description: Plan abi WDBX vector-store work — in-process KV+vector store, HNSW, WAL/segment persistence, loopback cluster RPC, REST serve, bench, and reference-grade secure demos. Use when asked about wdbx, vector search, persistence, clustering, or the secure demo. Routes to wdbx-roundtrip/api-serve/cluster-serve/bench, secure-demo, abi-superpower-wdbx*, and abi-wdbx-persistence. Demos are reference-grade, NOT production FHE/AES/sharding.
---

# wdbx

Entry point for abi's WDBX vector store (`src/features/wdbx/`). Routes:

| You want to… | Use |
| --- | --- |
| Insert→query round-trip smoke | `wdbx-roundtrip` |
| Serve loopback REST (`abi wdbx api serve`) | `wdbx-api-serve` |
| In-process Raft consensus demo / cluster serve | `wdbx-cluster-serve`, `cluster-demo-guide` |
| Benchmark | `wdbx-bench` |
| `abi wdbx secure demo` (quant / Huffman / autoencoder / HE / FHE) | `secure-demo` |
| Persistence (WAL / segments / recovery) deep-dive | `abi-wdbx-persistence` |
| Superpower deep-dives | `abi-superpower-wdbx`, `-wdbx-cluster`, `-wdbx-compute`, `-wdbx-secure` |

## Honest boundaries (from `docs/contracts/external-claims-audit.mdx` §WDBX)
- Real: in-process KV + vector store, HNSW, SIMD cosine, MVCC snapshots,
  WAL/segment checkpoints, loopback Raft RPC (RequestVote/AppendEntries),
  int8 quantization, order-0 Huffman, in-process autoencoder, additive HE,
  DGHV somewhat-HE (reference parameters).
- **NOT** production multi-host distributed deployment or data sharding.
- **NOT** production-secure or bootstrapped full FHE; no AES/RBAC WDBX storage.
- **NOT** a production / learned-SOTA compression codec.
- Non-loopback bind refuses without `ABI_WDBX_CLUSTER_TOKEN`; REST is
  127.0.0.1-only (`ABI_WDBX_REST_TOKEN` for bearer auth). TLS env vars are
  validated but native TLS is not linked — deploy behind a TLS-terminating proxy.
