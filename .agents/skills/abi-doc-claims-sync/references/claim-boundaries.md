# ABI Documentation Claim Boundaries

Authoritative policy files:

- `/Users/donaldfilimon/abi/docs/contracts/external-claims-audit.mdx`
- `/Users/donaldfilimon/abi/docs/contracts/public-api.mdx`
- `/Users/donaldfilimon/abi/docs/spec/wdbx-north-star.mdx`
- `/Users/donaldfilimon/abi/abi-threat-model.md`

Safe repo-backed wording:

- ABI is a nightly-Rust local AI orchestration framework with deterministic Abbey/Aviva/Abi profile routing.
- WDBX is an in-process vector/key-value/block store with segment checkpoints, WAL recovery, temporal/causal records, HNSW-style cosine search, and hybrid ranking where wired.
- CLI and MCP surfaces are frozen and contract-tested.
- MCP stdio is local IPC. Startup also attempts a custom loopback HTTP listener
  that can require `ABI_MCP_HTTP_TOKEN`; its one-event `/sse` discovery response
  is not persistent MCP HTTP+SSE.
- WDBX REST is loopback and can require `ABI_WDBX_REST_TOKEN`.
- WDBX cluster RPC uses real TCP RequestVote/AppendEntries, supports `ABI_WDBX_CLUSTER_TOKEN`, optional `ABI_WDBX_CLUSTER_PEERS`, and refuses non-loopback binds without a token.
- GPU support includes capability/status reporting and vector operations with
  linked Metal fused cosine/dot/L2 plus multi-pass `reduce_sum_kernel` on macOS
  when native kernels initialize, else deterministic CPU fallback.
- Compression/FHE surfaces are implemented demos or reference schemes unless stronger evidence exists.

Claims to remove, downgrade, or frame as proposed:

- Distributed sharding or production multi-host database deployment.
- AES/RBAC WDBX storage unless implemented and tested.
- Regulatory certifications.
- Kubernetes/H100/A100/InfiniBand/NVLink deployment claims.
- QPS, latency, accuracy, speedup, energy, empathy, SQuAD, or CodeSearchNet numbers without fresh artifacts.
- Native local CUDA/Vulkan/ANE execution when the source only proves detection, reporting, or fallback (macOS Metal map + multi-pass reduce is real when initialized — still not a blanket GPU speedup claim).
- Production/SOTA learned compression.
- Production-secure or bootstrapped full FHE.
- Production-ready non-loopback MCP/WDBX HTTP without TLS, authz, rate limits, and deployment controls.

Instruction-file sync checklist:

- Command list and forbidden legacy names.
- WDBX grammar including `db compact`, `cluster serve <port> [node] [host]`, and `api serve [port]`.
- MCP 12-tool names and transport details.
- Feature flags and FoundationModels gating.
- Nightly-Rust idioms and `./tools/cargo.sh` vs bare `cargo` caveat.
- `./build.sh check`, `full-check`, docs validation, parity, and cross-smoke guidance.
