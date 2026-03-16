# ABI Framework Examples

Example programs demonstrating ABI framework features. Each example is a
standalone Zig source file that imports `@import("abi")`.

## Building Examples

```bash
# Build all examples
zig build examples

# On Darwin 25+ / macOS 26+, prefer a host-built or otherwise known-good Zig.
# If stock prebuilt Zig is linker-blocked before build.zig runs, use the
# fallback wrapper:
./tools/scripts/run_build.sh examples
```

All examples are compiled as part of `zig build verify-all`.

## Catalog

### AI & Machine Learning

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `ai_suite.zig` | End-to-end AI pipeline: agents, profiles, training | `feat-ai` |
| `embeddings.zig` | Text embedding generation and similarity | `feat-ai` |
| `llm_real.zig` | Real LLM backend integration (requires API key) | `feat-ai`, `feat-llm` |
| `train_ava.zig` | Training workflow for the Ava model | `feat-ai`, `feat-training` |
| `training/train_demo.zig` | Minimal training loop demo | `feat-ai`, `feat-training` |

### GPU & Compute

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `gpu.zig` | GPU device enumeration and kernel dispatch | `feat-gpu` |
| `gpu_training.zig` | GPU-accelerated training pipeline | `feat-gpu`, `feat-training` |
| `compute.zig` | General compute operations | `feat-compute` |
| `tensor_ops.zig` | Tensor creation, manipulation, and math | `feat-compute` |
| `concurrency.zig` | Work-stealing, thread pools, parallel pipelines | `feat-concurrency` |
| `concurrent_pipeline.zig` | Multi-stage concurrent data pipeline | `feat-concurrency` |

### Database & Search

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `database.zig` | WDBX vector store: insert, query, backup | `feat-database` |
| `distributed_db.zig` | Distributed database with replication | `feat-database` |
| `search.zig` | Semantic and full-text search | `feat-database` |

### Networking & Web

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `network.zig` | TCP/UDP sockets, HTTP client | `feat-network` |
| `web.zig` | HTTP server and route handling | `feat-web` |
| `web_observability.zig` | Web server with metrics and tracing | `feat-web`, `feat-observability` |
| `gateway.zig` | API gateway with rate limiting and routing | `feat-network` |
| `streaming.zig` | Streaming data processing pipelines | `feat-network` |
| `messaging.zig` | Message queues and pub/sub patterns | `feat-messaging` |

### Infrastructure & Platform

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `auth.zig` | Authentication and RBAC | `feat-auth` |
| `cache.zig` | In-memory and tiered caching | `feat-cache` |
| `cloud.zig` | Cloud provider integration | `feat-cloud` |
| `storage.zig` | File and object storage | `feat-storage` |
| `mobile.zig` | Mobile platform bindings | `feat-mobile` |
| `ha.zig` | High availability: failover, health checks | (always on) |
| `pages.zig` | Static page serving and templating | `feat-web` |

### Observability & Operations

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `observability.zig` | Metrics, tracing, and structured logging | `feat-observability` |
| `analytics.zig` | Event tracking and analytics pipelines | `feat-analytics` |

### Integration & Configuration

| Example | Description | Feature Flags |
|---------|-------------|---------------|
| `config.zig` | Configuration loading and validation | (always on) |
| `registry.zig` | Service registry and discovery | (always on) |
| `discord.zig` | Discord bot integration | `feat-network` |
| `hello.zig` | Minimal "hello world" app | (always on) |

## Feature Flags

All features default to enabled. Disable with `-Dfeat-<name>=false`:

```bash
zig build examples -Dfeat-gpu=false -Dfeat-mobile=false
```

Feature definitions live in `build/options.zig`. See [CLAUDE.md](../CLAUDE.md) for the
full list of flags and conventions.
