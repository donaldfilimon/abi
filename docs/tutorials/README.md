# Tutorials (consolidated outlines)

Outline-stage how-to material, consolidated from three Zig-era stub tutorials
(`advanced-llm-usage.md`, `gpu-backend-selection.md`,
`vector-db-optimization.md`). Not in Mintlify navigation. These are topic
outlines, not finished guides — expand a section into its own page only when it
has runnable, claim-honest content. Claim boundaries:
[`contracts/external-claims-audit.mdx`](../contracts/external-claims-audit.mdx).

## GPU backend selection

ABI's GPU support is a **claim-honest capability report** (`abi backends`, MCP
`gpu_status`), not a backend picker. Metal is preferred on macOS; when native
kernels are not linked the report says `accelerated=false` and vector ops use
the deterministic CPU SIMD fallback. CUDA/Vulkan/ANE execution is a non-claim.
General considerations if evaluating backends elsewhere: target OS and hardware
availability, feature parity and performance tradeoffs, driver support and
installation complexity.

## Vector DB (WDBX) optimization

Strategies for optimizing vector-store usage for retrieval-augmented
generation:

- Choosing embeddings dimensionality
- Index type selection (HNSW, IVF, PQ)
- Batch insertion and compaction
- Metrics to monitor (recall, latency)

Checklist:

- [ ] Use normalized vectors
- [ ] Tune HNSW parameters (M, efConstruction)
- [ ] Run periodic index rebuilds for large insert volumes

## Advanced LLM usage

Planned topics (no runnable ABI example yet — the original page's example was
Zig pseudo-code for an API that no longer exists):

- Fine-tuning overview
- LoRA quickstart
- Prompt engineering patterns

Acceptance bar for promoting this to a real tutorial: at least one runnable
example (or pseudo-code clearly labeled), plus common pitfalls and performance
tips.
