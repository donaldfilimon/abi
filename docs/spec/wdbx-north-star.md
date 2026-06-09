# WDBX North-Star: ABI as a Cognitive Runtime

**Date:** 2026-06-01
**Status:** **VISION / ROADMAP — not a statement of current capabilities.**
**Scope:** Long-term direction for evolving ABI from a local multi-persona assistant into a WDBX-centered cognitive runtime spanning storage, indexing, compute, security, clustering, and transport.

> **Claims discipline (read first).** This document is aspirational. Every item below is tagged **Current**, **Partial**, or **Proposed**. Only **Current** items are backed by repo source, tests, or benchmark artifacts today. **Proposed** items — distributed clustering, Apple Neural Engine / TPU execution, neural compression, homomorphic encryption, and any throughput/latency figure — are **not** repo-proven and MUST NOT be presented as existing ABI capabilities in external collateral. This file is the proposal record the external-claims policy permits; see `docs/contracts/external-claims-audit.md` and the `tests/contracts/public_docs.zig` claim-boundary test. When this vision and the source disagree, the source wins (`build.zig`, `src/`, contract tests).

---

## 1. Final Mission

Make **WDBX** the shared memory and reasoning substrate for all ABI agents and personas (Abbey, Aviva, Abi): persistent, verifiable, hardware-accelerated, and — over the long horizon — scalable across CPU, GPU, NPU, and TPU environments and multiple nodes. ABI evolves from a single-process assistant into a cognitive runtime where retrieval, persona routing, and integrity are first-class.

```
ABI
 ↓
Persona Router          (Current)
 ↓
WDBX Runtime
 ├─ Storage Layer       (Current — in-process + segment/WAL persistence)
 ├─ Index Layer         (Current — HNSW/SIMD + snapshot-persisted temporal/causal ranker)
 ├─ Compute Layer       (Partial — CPU SIMD + GPU fallback; NPU/TPU Proposed)
 ├─ Security Layer      (Partial — SHA-256 chain + snapshot checksum)
 ├─ Cluster Layer       (Proposed)
 └─ Transport Layer     (Partial — stdio/JSON-RPC + loopback HTTP/SSE)
```

---

## 2. Success Criteria → Current Status

Each north-star capability mapped to honest repo state. Evidence is a pointer to source/tests for **Current**/**Partial** rows.

| Capability | Status | Evidence / Gap |
| --- | --- | --- |
| Persistent long-term memory | **Current (single-node)** | JSONL snapshot codec (`persistence.zig`), CRC32-framed write-ahead log with replay + corruption detection (`wal.zig`), automatic segment/WAL recovery for runtime CLI reads/writes (`recovery.zig`, `wdbx_db.zig`), and a multi-segment checkpoint store with epoch reclamation (`segments.zig`). `abi wdbx db init/verify`, `block insert/get`, and `query` exercise the segment/WAL lifecycle; the legacy snapshot file remains a compatibility mirror. Gap: full MVCC visibility and compaction policy for larger stores. |
| Temporal-causal retrieval | **Current (single-node/MCP)** | `temporal.zig` implements temporal decay + causal BFS graph scoring, `retrieval.zig` composes HNSW semantic search with the `semantic × temporal × causal × persona` hybrid ranker, JSONL snapshots persist `temporal_node` / `temporal_edge` records, and MCP `wdbx_query` uses hybrid ranking by default. Gap: broader CLI semantic query UX if that becomes a product surface. |
| Multi-persona memory routing | **Current** | `src/features/ai/router.zig` keyword-weighted Abbey/Aviva/Abi routing; conversation blocks labeled by profile in `src/features/wdbx/chain.zig`. |
| SIMD-accelerated search | **Current** | HNSW `@Vector` cosine distance: `src/features/wdbx/hnsw.zig`. |
| GPU-accelerated retrieval | **Partial** | HNSW distance routes through `gpu.vectorOps().cosineSimilarity()` with deterministic CPU fallback; `compute.zig` adds a CPU/GPU/NPU/TPU backend selector. Gap: native Metal/CUDA/Vulkan compute kernels are not linked. |
| Apple Neural Engine / TPU integration | **Partial** | `compute.zig` enumerates `npu_ane`/`tpu_remote` as first-class selectable backends with dynamic selection + deterministic CPU fallback (unit-tested parity). Gap: native ANE/TPU dispatch is not linked. |
| Distributed clustering | **Partial** | `cluster.zig` implements an in-process Raft-style core: leader election, majority-quorum log replication, and leader failover (4 unit tests; `abi wdbx cluster demo`). Gap: a networked RPC transport across hosts. |
| Cryptographic verification | **Current** | SHA-256-linked blocks + `verifyBlocks()`; snapshot integrity line rejects tampering (`error.ChecksumMismatch`); WAL CRC32 frames: `chain.zig`, `persistence.zig`, `wal.zig`. |
| Zero-copy memory access | **Partial** | Pool/arena allocators in `src/foundation/pool_allocator.zig`; padded contiguous vector storage. Gap: a proven zero-copy read path. |
| Neural / embedding compression | **Partial** | `compression.zig` scalar int8 embedding quantization, ~4× with bounded reconstruction error (3 tests; `abi wdbx secure demo`). Gap: a learned/entropy codec. |
| Homomorphic encryption | **Partial** | `crypto_he.zig` additive single-key homomorphism over GF(p): ciphertext sums decrypt to plaintext sums (5 tests; `secure demo`). Gap: multiplication / full FHE (research horizon). |

---

## 3. Layered Architecture

### 3.1 Storage Layer
- **Current:** in-process key/value, fixed-capacity padded vectors, SHA-256 block chain, 3D spatial records; JSONL snapshot save/load with integrity checksum; **write-ahead log** (`wal.zig`) with CRC32-framed append-only records, deterministic replay (reuses `persistence.deserialize`), and corruption detection. `abi wdbx db verify` replays the WAL and cross-checks it against the current checkpoint; runtime `block`/`query` commands recover WAL-ahead state before reading or writing; `segments.zig` provides immutable epoch checkpoints, reset, active epoch listing, and watermark reclamation. CLI runtime commands now write/open segment checkpoints by default and keep a monolithic snapshot as a compatibility mirror.
- **Proposed:** full MVCC visibility, larger-store compaction policy, and cross-process/concurrent checkpoint coordination.
- **Invariants:** append-only, deterministic, verifiable — enforced today by the snapshot checksum, WAL CRC frames, and segment manifest.

### 3.2 Index Layer
- **Current:** HNSW graph with SIMD cosine distance and ordered result contracts; **temporal/causal graph + hybrid ranker** (`temporal.zig`): exponential recency decay, causal BFS proximity, and the combined ranking key below (unit-tested and used by MCP `wdbx_query`).
- **Scoring model (implemented in `temporal.zig`, composed with HNSW in `retrieval.zig`, and default for MCP-local `wdbx_query`):**
  ```
  score = semantic × temporal × causal × persona
  ```
  `semantic` comes from HNSW cosine, `temporal` from recency half-life decay, `causal` from causal-edge hop distance, `persona` from the router profile weight. JSONL snapshots persist temporal nodes and causal edges.

### 3.3 Compute Layer
- **Current:** CPU SIMD via Zig `@Vector`; GPU *capability reporting* with deterministic CPU fallback; `compute.zig` dynamic backend selector across CPU (`scalar`/`avx2`/`avx512`/`neon`, host-detected), GPU (`cuda`/`metal`/`vulkan`), NPU (`ane`), and TPU (`remote`), each degrading to the CPU SIMD path with verified CPU/GPU parity.
- **Proposed:** native compute kernels for the accelerator backends (CUDA / Metal / Vulkan Compute / ANE / remote TPU). Selection is already **dynamic** and always degrades to the deterministic CPU path.

### 3.4 Security Layer
- **Current:** SHA-256 block chaining with integrity verification; snapshot checksum + WAL CRC32 frames with clean rejection of tampered/out-of-range records; `crypto_he.zig` additive homomorphic encryption (encrypted aggregation under a single key); connector credential validation; constitution governance on responses.
- **Proposed:** at-rest snapshot encryption, signed snapshots, and a fully homomorphic (multiplicative) query path (Phase 4). No encryption/RBAC claims beyond what is implemented and tested.

### 3.5 Cluster Layer
- **Current (in-process):** `cluster.zig` Raft-style core — leader election, majority-quorum log replication, leader failover, quorum-loss detection — exercised over an in-process node array with a deterministic step model (4 unit tests; `abi wdbx cluster demo`).
- **Proposed:** a networked RPC transport so the consensus core spans separate hosts. Until that lands, ABI is **not** a distributed multi-host deployment.

### 3.6 Transport Layer
- **Current:** MCP JSON-RPC 2.0 over stdio (64 KB request cap) + optional loopback HTTP/SSE (`GET /sse`, `POST /message`) on `127.0.0.1:8080`; **WDBX REST listener** (`rest.zig`) serving `POST /insert /query /verify` and `GET /health /stats` on a loopback port (`abi wdbx api serve [port]`, default 8081), with a fully unit-tested pure routing core.
- **Proposed:** gRPC, WebSocket streaming, cluster RPC; hardening the REST listener for non-loopback use.

---

## 4. API Surface

### Current
The contract-tested public surfaces are the CLI (`src/abi_cli/usage.zig`) and the MCP tool list (`src/mcp/handlers.zig`): `ai_run`, `ai_complete`, `ai_train`, `wdbx_query`, `wdbx_stats`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `plugin_run`.

### REST (implemented, loopback — `rest.zig`)
```
POST /insert     POST /query     POST /verify
GET  /health     GET  /stats
```
Served by `abi wdbx api serve [port]` (default 8081). The routing core is a pure, unit-tested `route(method, path, body)` function; `serve` wraps it on a 127.0.0.1 listener. Future: gRPC, WebSocket streaming, cluster RPC, and hardening for non-loopback exposure.

---

## 5. CLI Surface

**Current frozen top-level commands** (contract-tested, do not break): `help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx` (+ the `abi --tui` shortcut).

**`wdbx` command namespace — implemented** (`src/abi_cli/handlers/wdbx.zig`, contract row added to `tests/contracts/surface.zig`):
```
wdbx db init <path>        wdbx db verify <path>        # segment checkpoint + WAL integrity
wdbx block insert <path> <profile> <metadata>           # writes segment checkpoint + WAL
wdbx block get <path>
wdbx query <path>                                       # store stats manifest
wdbx benchmark [count]                                  # local in-memory timing
wdbx cluster status                                     # honest single-node report
wdbx cluster demo [nodes]                               # in-process election/quorum/failover
wdbx compute info                                       # CPU/GPU/NPU/TPU backends + selection
wdbx secure demo                                        # int8 compression + additive HE aggregation
wdbx gpu info                                           # GPU backend capabilities
wdbx api serve [port]                                   # loopback REST listener (default 8081)
```
`cluster status` reports `nodes=1 role=standalone` (no distributed claim); `cluster demo` runs the in-process Raft-style core (§3.5). `api serve` starts the loopback REST listener implemented in `rest.zig` (`POST /insert /query /verify`, `GET /health /stats`; §3.6/§4). Native distributed transport, NPU/TPU dispatch, learned compression, and full (multiplicative) FHE remain **Proposed** (§2).

---

## 6. Benchmarks

**Policy:** publish numbers only from a reproducible benchmark artifact checked into the repo and cited (per `docs/spec/multi-persona-technical.md` §8). The list below is *what to capture*, not measured claims.

- Dimensions to track: insert latency, query latency, throughput, memory bandwidth, GPU utilization, ANE utilization.
- Distribution metrics: P50 / P95 / P99.
- Current state: `src/benchmarks.zig` covers functional vector-op / HNSW / chain / routing / constitution timing; it does not yet emit a checked-in latency/throughput artifact. Until it does, treat all of the above as targets.

---

## 7. Testing Strategy

| Suite | Status | Notes |
| --- | --- | --- |
| Unit — storage / indexing / checksums | **Current** | Inline tests across `wdbx/*` incl. persistence round-trip with vector/block/spatial/temporal records, tamper, and integrity tests. |
| Integration — insert / query / verify | **Current** | `src/integration_tests.zig`, contract tests in `tests/contracts/`. |
| GPU — CPU/GPU parity | **Current** | HNSW distance parity test + `compute.zig` dot-product parity across cpu/gpu/npu backends. Broaden as native kernels land. |
| Recovery — WAL replay / corruption | **Current** | `wal.zig` tests: append→replay reconstruction, flipped-byte CRC rejection, bad-header rejection; `recovery.zig` prefers segment checkpoints over legacy snapshots and selects WAL-ahead state over stale checkpoints; CLI runtime tests cover `block get`, `query`, the next `block insert` recovering/checkpointing WAL-ahead state, and reopening from segment checkpoints without the snapshot mirror. Retrieval tests cover hybrid semantic/temporal/causal/persona re-ranking and the MCP contract checks `ranking=hybrid`. |
| Cluster — node failure / failover / replication | **Current (in-process)** | `cluster.zig` tests: single-leader election, quorum replication + commit, leader failover at higher term, quorum-loss unavailability. Gap: cross-host transport tests. |

---

## 8. Phased Roadmap

| Phase | Theme | Gate |
| --- | --- | --- |
| **1** | Single-node cognitive runtime: durable storage (WAL + segments + recovery), persona-weighted scoring, always-on persistence, REST `/insert /query /verify`. | **Landed for single-node/MCP** — WAL + replay/corruption recovery, segment checkpoints as the default runtime checkpoint source, epoch reclamation, runtime CLI auto-recovery, snapshot-persisted temporal/causal records, MCP default hybrid ranking, the `wdbx` CLI, and the loopback REST listener are landed and tested. |
| **2** | Multi-node cluster: membership, replication, leader failover, cluster RPC. | First **Proposed** layer; gate on real tests before any "distributed" claim. |
| **3** | Neural compression of stored embeddings/state. | Requires measured compression artifact before claiming ratios. |
| **4** | Homomorphic encryption for query-over-encrypted-memory. | Research horizon; claim only with a working, tested path. |
| **5** | Self-optimizing planner (adaptive backend + index selection). | Builds on dynamic backend selection + scheduler/metrics. |
| **6** | Autonomous cognitive fabric: agents + personas operate over WDBX as shared substrate across hardware tiers. | Culmination; depends on all prior phases. |

**Near-term execution order after Phase 1:** keep larger-store compaction, broader CLI semantic retrieval UX, and future transport/compute/security layers behind the existing contract discipline. Each step keeps mod/stub parity and passes `./build.sh check` before it is marked done in `tasks/todo.md` / `tasks/roadmap-next.md`.

---

## 9. Maintenance

This is a north-star, so it drifts faster than the source. Before reusing any line externally, reconcile against `build.zig`, `src/`, and `tests/contracts/`, and downgrade any row that the source does not yet prove. Promote a row from **Proposed** → **Partial** → **Current** only when source + a passing test back it.
