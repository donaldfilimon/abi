# ABI Claim Boundaries

Use this as the quick reference before editing ABI docs. The authoritative
sources are still `build.zig`, `src/`, `tests/contracts/`,
`docs/contracts/external-claims-audit.mdx`, and
`docs/spec/wdbx-north-star.mdx`.

## Safe Current Wording

- ABI is a Zig 0.17 local AI orchestration framework.
- ABI has deterministic Abbey/Aviva/Abi profile routing with optional adaptive
  state persistence.
- WDBX is an in-process key/value, vector, block, spatial, temporal/causal
  memory substrate.
- Completion persistence is opt-in through `CompletionRequest.store_result`.
- WDBX supports segment checkpoints, a compatibility JSONL snapshot mirror, WAL
  replay/corruption detection, and runtime recovery.
- MCP stdio and loopback HTTP/SSE surfaces are contract-tested; loopback HTTP
  can require a bearer token.
- WDBX REST is loopback-scoped and can require a bearer token.
- Cluster RPC is a real RequestVote/AppendEntries TCP transport with shared
  secret and optional peer allowlist controls, but remains reference-scoped.
- GPU/backend support is capability reporting plus vector operations that
  deterministically fall back to CPU unless native kernels are actually linked
  and reported by the runtime.
- Compression and FHE surfaces are reference/demo scoped unless audited
  production artifacts prove otherwise.

## Do Not Claim Without New Evidence

- Production multi-host distributed deployment or data sharding.
- Production-ready non-loopback MCP/WDBX HTTP exposure without TLS, authz,
  rate limiting, and deployment review.
- Native local accelerator execution for CUDA, Vulkan, Metal compute kernels,
  ANE, or TPU.
- Production/SOTA learned compression.
- Production-secure or bootstrapped full FHE.
- AES/RBAC WDBX storage.
- Swift, Python, TensorFlow, PyTorch, Kubernetes, H100/A100, InfiniBand, or
  blockchain implementation/deployment claims.
- Regulatory certification claims such as GDPR, CCPA, HIPAA, or ISO 27001.
- QPS, latency, accuracy, model-quality, energy, or speedup numbers without a
  reproducible checked-in benchmark artifact.

## Required Sync

When changing durable conventions, keep `AGENTS.md`, `CLAUDE.md`, and
`GEMINI.md` in sync. This includes command lists, MCP tools, feature flags,
build gates, Zig idioms, generated-file rules, listener/auth behavior, and
OpenCode setup notes.
