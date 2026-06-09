# External Claims Audit

This audit reconciles public collateral claims from the Drive files opened on 2026-05-25 with the current ABI checkout. Treat repo source, `build.zig`, `src/abi_cli/usage.zig`, `src/mcp/handlers.zig`, this `docs/contracts/` directory, and `tests/contracts/` as the source of truth.

## Audited Sources

- `MLAI Infrastructure Components and Specifications` spreadsheet.
- `High-Performance Neural State Management: Technical Analysis of WDBX and Multi-Persona Architectures` Google Doc.

## Repo-Backed Claims

| Claim area | Current repo evidence | Safe external wording |
| --- | --- | --- |
| Toolchain | `.zigversion`, `build.zig`, README quick start | ABI targets Zig `0.17.0-dev.813+2153f8143`; older Zig 0.16 wording is stale. |
| WDBX vector store | `src/features/wdbx/mod.zig`, `hnsw.zig`, `tests/contracts/surface.zig` | WDBX provides an in-process vector store with fixed-capacity padded vectors, HNSW-style indexing, SIMD cosine distance, ordered search-result contract coverage, and disabled-feature stubs. |
| Block history | `src/features/wdbx/chain.zig`, `tests/contracts/surface.zig` | WDBX provides SHA-256-linked conversation blocks with snapshot iteration and integrity verification coverage. |
| Spatial records | `src/features/wdbx/spatial_3d.zig`, `Store.stats()` | WDBX includes an in-memory 3D spatial index with Euclidean, Manhattan, and cosine-distance searches. |
| AI profiles | `src/features/ai/router.zig`, `src/features/ai/mod.zig` | ABI has local Abbey, Aviva, and Abi profile routes selected by deterministic keyword-weighted heuristics, with optional EMA state persistence. |
| Completion persistence | `CompletionRequest.store_result`, `completeWithStore()`, contract tests | Completion persistence is opt-in; when WDBX is enabled it records query/response vectors, metadata, and append-linked blocks in a caller-provided store. |
| MCP/CLI | `src/abi_cli/usage.zig`, `src/mcp/handlers.zig`, contract tests | Public CLI commands and MCP tools are small, frozen surfaces guarded by contract tests. |
| GPU | `src/features/gpu/`, `build.zig`, contract tests | GPU support is capability/status reporting plus vector operations that deterministically fall back to CPU when native kernels are unavailable. |
| Connectors | `src/connectors/`, `docs/contracts/public-api.md` | OpenAI, Anthropic, Discord, Grok, and Twilio connectors validate local/live boundaries; live HTTP dispatch is explicit. |
| WDBX roadmap demos | `src/features/wdbx/{wal,recovery,segments,temporal,retrieval,cluster,compute,compression,crypto_he,rest}.zig`, `src/abi_cli/handlers/wdbx*.zig`, `docs/spec/wdbx-north-star.md` | The `abi wdbx` namespace adds a durable write-ahead log (replay + corruption detection), runtime recovery from WAL-ahead checkpoints, segment checkpoints as the default runtime checkpoint source, a snapshot-persisted temporal/causal graph, and MCP `wdbx_query` hybrid ranking, plus **in-process demonstrations**: `cluster demo` (single-host Raft-style consensus), `compute info` (CPU/GPU/NPU/TPU selection with deterministic CPU fallback), `secure demo` (int8 quantization + additive single-key homomorphic aggregation), and `api serve` (loopback REST). Present these as single-node/in-process/roadmap surfaces only — **not** distributed clustering, native NPU/TPU execution, learned compression, or full FHE. |

## Claims To Remove Or Downgrade

| External claim | Repo status | Replacement wording |
| --- | --- | --- |
| WDBX is a distributed database with intelligent sharding. | Not currently proven by repo source or tests. | WDBX is currently an in-process vector/key-value/block store. |
| WDBX proves `12,000 QPS`, `8.2 ms` latency, or `20-30%` lower latency. | Not currently proven by repo source, tests, or benchmark artifacts. | The repo validates functional search/block contracts; publish performance only with fresh benchmark artifacts. |
| WDBX includes AES-256 encryption or RBAC. | Not currently implemented as WDBX storage features. | Do not claim WDBX encryption/RBAC. Mention connector credential validation only where relevant. |
| ABI/WDBX is implemented in Swift 6 or Swift 6.2 Span. | Not present in the repo. | ABI/WDBX are Zig modules; macOS builds may link Metal/Foundation/Objective-C for GPU status paths. |
| ABI is implemented as a Python/JavaScript framework using TensorFlow, PyTorch, PostgreSQL, MongoDB, RabbitMQ, Kafka, Docker, Kubernetes, or Jenkins. | Not currently reflected by this repo's implementation or build. | ABI is a Zig codebase with local connectors, feature modules, build-time plugin registry generation, and Zig build/test gates. |
| ABI proves `295x` GPU matmul speedup or `13x` neural-network speedup. | Not currently proven by repo benchmark outputs. | GPU paths report backend capabilities and CPU fallback behavior. |
| WDBX proves `10,000 req/s`, `50 ms` latency, or `95%` accuracy. | Not currently proven by repo tests or benchmark artifacts. | Keep these as future benchmark targets unless a reproducible ABI benchmark artifact exists. |
| Abbey/Aviva/Abi have measured empathy, technical-accuracy, SQuAD, CodeSearchNet, or model-comparison scores. | Not currently proven by repo tests. | The repo has deterministic local routing/profile behavior and validation helpers, not external model-quality benchmarks. |
| The multi-persona system proves `15 kWh/1k` energy use or `25%` efficiency gains. | Not currently measured by repo tests or tooling. | Do not publish energy-efficiency claims without a reproducible measurement artifact. |
| ABI performs dynamic neural persona blending using model-output interpolation formulas. | Not implemented as neural output blending. | ABI uses keyword-weighted profile selection and optional EMA smoothing of route weights. |
| ABI is deployed on H100/A100 clusters, Kubernetes auto-scaling, InfiniBand/NVLink, or decentralized blockchain nodes. | Not currently proven by this repo. | Present those only as external deployment proposals, not current ABI repo capabilities. |
| ABI is certified for GDPR, CCPA, HIPAA, ISO 27001, or compliance-by-design guarantees. | Not currently proven by repo source or tests. | The repo has local constitution checks, connector validation, and credential-file permission handling, but no regulatory certification evidence. |

## Reusable Delta For External Artifacts

Use this concise replacement paragraph when updating Drive docs:

> Current ABI repo evidence supports a Zig 0.17 local AI orchestration framework with deterministic Abbey/Aviva/Abi profile routing, an in-process WDBX vector/key-value/block store, segment checkpoint plus WAL persistence with runtime recovery, a compatibility JSONL snapshot mirror, epoch reclamation helpers, snapshot-persisted temporal/causal graph records, MCP hybrid WDBX query ranking, HNSW-style cosine search, SHA-256-linked conversation blocks, 3D spatial search, feature-off stubs, CLI/MCP contract coverage, explicit connector live-mode boundaries, and GPU capability reporting with CPU fallback. The repo does not currently prove distributed sharding, AES/RBAC WDBX storage, Swift/Python/TensorFlow implementation claims, Kubernetes/H100 deployment claims, regulatory certifications, QPS/latency/accuracy targets, GPU speedup figures, energy-efficiency metrics, or SQuAD/CodeSearchNet/GPT comparative scores.
