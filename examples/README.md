# ABI Examples

Runnable examples demonstrating the ABI framework's features. Each example is a standalone program that initializes the framework and exercises one or more modules.

## Running Examples

Build and run all examples:

```bash
zig build examples
```

Run an individual example:

```bash
zig build run-<name>
```

For example: `zig build run-hello`, `zig build run-gpu`, `zig build run-database`.

> **Darwin 25+**: Use `./tools/scripts/run_build.sh run-<name>` instead of `zig build` directly.

## Examples by Category

### AI & ML

| File | Description | Feature Flag |
|------|-------------|--------------|
| `ai_suite.zig` | Consolidated AI suite: inference, reasoning, training, and multimodal agents | `-Dfeat-ai` |
| `embeddings.zig` | SIMD-accelerated vector operations for embeddings (dot product, cosine similarity) | `-Dfeat-ai` |
| `llm_real.zig` | Live LLM inference via Ollama, LM Studio, or vLLM backends | `-Dfeat-llm` |
| `streaming.zig` | Streaming API with SSE encoding and circuit breaker resilience | `-Dfeat-ai` |
| `train_ava.zig` | Fine-tune the Ava assistant model from a GGUF base | `-Dfeat-training` |
| `training/train_demo.zig` | LLM training pipeline with synthetic data and loss tracking | `-Dfeat-training` |

### GPU & Compute

| File | Description | Feature Flag |
|------|-------------|--------------|
| `gpu.zig` | Unified GPU API: device discovery, buffer management, profiling, multi-GPU | `-Dfeat-gpu` |
| `gpu_training.zig` | GPU + AI training integration with distributed gradient sync | `-Dfeat-gpu`, `-Dfeat-training` |
| `compute.zig` | SIMD vector operations: dot product, cosine similarity, L2 distance | `-Dfeat-compute` |
| `tensor_ops.zig` | Matrix multiply, tensor transforms, and SIMD vector ops end-to-end | `-Dfeat-compute` |
| `concurrency.zig` | Lock-free concurrency primitives: MPMC queue, Chase-Lev work-stealing deque | (always available) |
| `concurrent_pipeline.zig` | Multi-stage pipeline with channels, thread pool, and DAG orchestration | (always available) |

### Database & Storage

| File | Description | Feature Flag |
|------|-------------|--------------|
| `database.zig` | WDBX vector database: creation, vector insertion, similarity search, backup | `-Dfeat-database` |
| `storage.zig` | Unified object storage abstraction with metadata and memory backend | `-Dfeat-storage` |
| `cache.zig` | In-memory LRU/LFU cache with TTL, eviction policies, and statistics | `-Dfeat-cache` |
| `search.zig` | Full-text search with inverted index and BM25 ranking | `-Dfeat-search` |

### Networking

| File | Description | Feature Flag |
|------|-------------|--------------|
| `network.zig` | Distributed compute network with node registration and Raft consensus | `-Dfeat-network` |
| `distributed_db.zig` | Cross-module integration: vector DB + Raft consensus + circuit breakers | `-Dfeat-database`, `-Dfeat-network` |
| `messaging.zig` | Pub/sub messaging with MQTT-style topic patterns and wildcards | `-Dfeat-messaging` |

### Web & API

| File | Description | Feature Flag |
|------|-------------|--------------|
| `web.zig` | Web module: HTTP client, persona routing, chat handling, JSON utilities | `-Dfeat-web` |
| `gateway.zig` | API gateway with radix-tree routing, rate limiting, and circuit breakers | `-Dfeat-gateway` |
| `pages.zig` | Dashboard UI pages with URL routing, templates, and path parameters | `-Dfeat-pages` |
| `web_observability.zig` | Web server + observability integration with metrics middleware | `-Dfeat-web`, `-Dfeat-profiling` |

### Infrastructure

| File | Description | Feature Flag |
|------|-------------|--------------|
| `observability.zig` | Observability subsystem: counters, gauges, histograms, distributed tracing | `-Dfeat-profiling` |
| `analytics.zig` | Analytics module: event tracking, sessions, funnels, statistics | `-Dfeat-analytics` |
| `auth.zig` | Security infrastructure: JWT, API key management, RBAC, rate limiting | `-Dfeat-auth` |
| `cloud.zig` | Cloud providers: AWS Lambda, GCP Functions, Azure serverless wrappers | `-Dfeat-cloud` |
| `ha.zig` | High availability: multi-region replication, backup, PITR, failover | `-Dfeat-database` |
| `mobile.zig` | Mobile platform: detection, lifecycle, sensors, notifications | `-Dfeat-mobile` |

### Framework

| File | Description | Feature Flag |
|------|-------------|--------------|
| `hello.zig` | Minimal hello-world showing framework initialization | (always available) |
| `config.zig` | Configuration system with Builder pattern for GPU, AI, and DB settings | (always available) |
| `registry.zig` | Feature registry: comptime and runtime toggle modes, lifecycle management | (always available) |
| `discord.zig` | Discord bot connector: API auth, guild listing, message sending | `-Dfeat-web` |

## Environment Variables

Some examples require API keys or service endpoints:

| Variable | Used By |
|----------|---------|
| `DISCORD_BOT_TOKEN` | `discord.zig` |
| `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL` | `llm_real.zig` |
| `ABI_OPENAI_API_KEY` | `ai_suite.zig` |
| `ABI_ANTHROPIC_API_KEY` | `ai_suite.zig` |

## Feature Flags

All features are enabled by default. Disable a feature with `-Dfeat-<name>=false`. See the main project [README](../README.md) for the full flag reference.
