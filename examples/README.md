# ABI Framework Examples

37 runnable examples demonstrating all major features of the ABI framework.

## Quick Start

```bash
zig build examples              # Build all examples
zig build run-hello             # Run a specific example
```

## All Examples

### Getting Started

| Example | Run Command | Description |
|---------|-------------|-------------|
| `hello.zig` | `run-hello` | Basic framework initialization and version check |
| `config.zig` | `run-config` | Configuration system with Builder pattern |
| `registry.zig` | `run-registry` | Feature registry for runtime feature management |

### AI & ML

| Example | Run Command | Description |
|---------|-------------|-------------|
| `agent.zig` | `run-agent` | AI agent with conversational chat interface |
| `ai_core.zig` | `run-ai-core` | AI core: agents, tools, prompts, personas, memory |
| `ai_inference.zig` | `run-ai-inference` | LLM inference, embeddings, vision, streaming |
| `ai_training.zig` | `run-ai-training` | Training pipelines and federated learning |
| `ai_reasoning.zig` | `run-ai-reasoning` | Abbey reasoning, RAG, eval, templates |
| `embeddings.zig` | `run-embeddings` | Vector embeddings and similarity operations |
| `llm.zig` | `run-llm` | Local LLM inference with GGUF models |
| `llm_real.zig` | `run-llm-real` | Real inference via Ollama/LM Studio/vLLM |
| `orchestration.zig` | `run-orchestration` | Multi-model routing with ensemble and fallback |
| `streaming.zig` | `run-streaming` | Streaming response handling for AI models |
| `training.zig` | `run-training` | Model training with optimizers and checkpointing |
| `training/train_demo.zig` | `run-train-demo` | Focused LLM training demo |
| `train_ava.zig` | `run-train-ava` | Train Ava assistant from gpt-oss models |

### Compute & GPU

| Example | Run Command | Description |
|---------|-------------|-------------|
| `compute.zig` | `run-compute` | SIMD compute: dot product, cosine similarity, vector add |
| `gpu.zig` | `run-gpu` | GPU acceleration and SIMD operations |
| `tensor_ops.zig` | `run-tensor-ops` | Tensor/matrix ops with SIMD-accelerated kernels |
| `concurrency.zig` | `run-concurrency` | Lock-free concurrency primitives (MPMC, Chase-Lev) |
| `concurrent_pipeline.zig` | `run-concurrent-pipeline` | Multi-stage pipeline with Channel, ThreadPool, DagPipeline |

### Data & Storage

| Example | Run Command | Description |
|---------|-------------|-------------|
| `database.zig` | `run-database` | Vector database: insert, search, statistics |
| `cache.zig` | `run-cache` | In-memory LRU/LFU cache with TTL |
| `search.zig` | `run-search` | Full-text search with BM25 scoring |
| `storage.zig` | `run-storage` | Unified file/object storage |

### Infrastructure

| Example | Run Command | Description |
|---------|-------------|-------------|
| `network.zig` | `run-network` | Network cluster setup and node management |
| `gateway.zig` | `run-gateway` | API gateway: routing, rate limiting, circuit breaker |
| `messaging.zig` | `run-messaging` | Event bus, pub/sub, message queues |
| `web.zig` | `run-web` | Web/HTTP framework and middleware |
| `cloud.zig` | `run-cloud` | Cloud adapters (AWS, GCP, Azure) |
| `mobile.zig` | `run-mobile` | Mobile platform support |
| `pages.zig` | `run-pages` | Dashboard/UI pages with URL routing |

### Operations

| Example | Run Command | Description |
|---------|-------------|-------------|
| `auth.zig` | `run-auth` | Authentication and security |
| `analytics.zig` | `run-analytics` | Event tracking and experiments |
| `observability.zig` | `run-observability` | Metrics and tracing (counters, gauges, histograms) |
| `ha.zig` | `run-ha` | High availability: replication, failover, PITR |
| `discord.zig` | `run-discord` | Discord bot integration (requires `DISCORD_BOT_TOKEN`) |

### C Interop

| File | Description |
|------|-------------|
| `c_test.c` | C API bindings test |

## Learning Path

1. **Start**: `hello.zig` → `config.zig` → `registry.zig`
2. **Data**: `database.zig` → `cache.zig` → `search.zig`
3. **Compute**: `compute.zig` → `concurrency.zig` → `gpu.zig` → `tensor_ops.zig`
4. **AI**: `agent.zig` → `embeddings.zig` → `llm.zig` → `training.zig`
5. **Infrastructure**: `network.zig` → `gateway.zig` → `observability.zig` → `ha.zig`
6. **Advanced**: `orchestration.zig` → `concurrent_pipeline.zig` → `train_ava.zig`

## See Also

- [Training Guide](training/README.md) — Detailed training pipeline documentation
- [API Reference](../docs/api/) — Auto-generated API docs
- [CLAUDE.md](../CLAUDE.md) — Build commands and architecture guide
