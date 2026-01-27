# ABI Framework API Reference

> Comprehensive API documentation auto-generated from source code

---

## Quick Links

| Module | Description |
|--------|-------------|
| [abi](./abi.md) | Main framework entry point and public API |
| [config](./config.md) | Unified configuration system with builder pattern |
| [framework](./framework.md) | Framework orchestration and lifecycle management |
| [runtime](./runtime.md) | Runtime infrastructure (engine, scheduling, memory) |
| [runtime-engine](./runtime-engine.md) | Work-stealing task execution engine |
| [runtime-scheduling](./runtime-scheduling.md) | Futures, cancellation, and task groups |
| [runtime-memory](./runtime-memory.md) | Memory pools and custom allocators |
| [runtime-concurrency](./runtime-concurrency.md) | Lock-free concurrent primitives |
| [gpu](./gpu.md) | GPU acceleration framework (Vulkan, CUDA, Metal, WebGPU) |
| [ai](./ai.md) | AI module with agents, LLM, embeddings, and training |
| [ai-agents](./ai-agents.md) | Agent runtime and orchestration |
| [ai-embeddings](./ai-embeddings.md) | Vector embeddings generation |
| [ai-llm](./ai-llm.md) | Local LLM inference |
| [ai-training](./ai-training.md) | Training pipelines and fine-tuning |
| [connectors](./connectors.md) | API connectors (OpenAI, Ollama, Anthropic, HuggingFace) |
| [database](./database.md) | Vector database (WDBX with HNSW/IVF-PQ) |
| [network](./network.md) | Distributed compute and Raft consensus |
| [ha](./ha.md) | High availability (backup, PITR, replication) |
| [observability](./observability.md) | Metrics, tracing, and monitoring |
| [registry](./registry.md) | Plugin registry (comptime, runtime, dynamic) |
| [web](./web.md) | Web utilities and HTTP support |
| [security](./security.md) | TLS, mTLS, API keys, and RBAC |

---

## Core Framework

### [abi](./abi.md)

Main framework entry point and public API

**Source:** [`src/abi.zig`](../../src/abi.zig)

### [config](./config.md)

Unified configuration system with builder pattern

**Source:** [`src/config/mod.zig`](../../src/config/mod.zig)

### [framework](./framework.md)

Framework orchestration and lifecycle management

**Source:** [`src/framework.zig`](../../src/framework.zig)

## Compute & Runtime

### [runtime](./runtime.md)

Runtime infrastructure (engine, scheduling, memory)

**Source:** [`src/runtime/mod.zig`](../../src/runtime/mod.zig)

### [runtime-engine](./runtime-engine.md)

Work-stealing task execution engine

**Source:** [`src/runtime/engine/mod.zig`](../../src/runtime/engine/mod.zig)

### [runtime-scheduling](./runtime-scheduling.md)

Futures, cancellation, and task groups

**Source:** [`src/runtime/scheduling/mod.zig`](../../src/runtime/scheduling/mod.zig)

### [runtime-memory](./runtime-memory.md)

Memory pools and custom allocators

**Source:** [`src/runtime/memory/mod.zig`](../../src/runtime/memory/mod.zig)

### [runtime-concurrency](./runtime-concurrency.md)

Lock-free concurrent primitives

**Source:** [`src/runtime/concurrency/mod.zig`](../../src/runtime/concurrency/mod.zig)

### [gpu](./gpu.md)

GPU acceleration framework (Vulkan, CUDA, Metal, WebGPU)

**Source:** [`src/gpu/mod.zig`](../../src/gpu/mod.zig)

## AI & Machine Learning

### [ai](./ai.md)

AI module with agents, LLM, embeddings, and training

**Source:** [`src/ai/mod.zig`](../../src/ai/mod.zig)

### [ai-agents](./ai-agents.md)

Agent runtime and orchestration

**Source:** [`src/ai/agents/mod.zig`](../../src/ai/agents/mod.zig)

### [ai-embeddings](./ai-embeddings.md)

Vector embeddings generation

**Source:** [`src/ai/embeddings/mod.zig`](../../src/ai/embeddings/mod.zig)

### [ai-llm](./ai-llm.md)

Local LLM inference

**Source:** [`src/ai/llm/mod.zig`](../../src/ai/llm/mod.zig)

### [ai-training](./ai-training.md)

Training pipelines and fine-tuning

**Source:** [`src/ai/training/mod.zig`](../../src/ai/training/mod.zig)

### [connectors](./connectors.md)

API connectors (OpenAI, Ollama, Anthropic, HuggingFace)

**Source:** [`src/connectors/mod.zig`](../../src/connectors/mod.zig)

## Data & Storage

### [database](./database.md)

Vector database (WDBX with HNSW/IVF-PQ)

**Source:** [`src/database/mod.zig`](../../src/database/mod.zig)

## Infrastructure

### [network](./network.md)

Distributed compute and Raft consensus

**Source:** [`src/network/mod.zig`](../../src/network/mod.zig)

### [ha](./ha.md)

High availability (backup, PITR, replication)

**Source:** [`src/ha/mod.zig`](../../src/ha/mod.zig)

### [observability](./observability.md)

Metrics, tracing, and monitoring

**Source:** [`src/observability/mod.zig`](../../src/observability/mod.zig)

### [registry](./registry.md)

Plugin registry (comptime, runtime, dynamic)

**Source:** [`src/registry/mod.zig`](../../src/registry/mod.zig)

### [web](./web.md)

Web utilities and HTTP support

**Source:** [`src/web/mod.zig`](../../src/web/mod.zig)

## Utilities

### [security](./security.md)

TLS, mTLS, API keys, and RBAC

**Source:** [`src/shared/security/mod.zig`](../../src/shared/security/mod.zig)

---

## Additional Resources

- [Getting Started Guide](../tutorials/getting-started.md)
- [Architecture Overview](../architecture/overview.md)
- [Feature Flags](../feature-flags.md)
- [Troubleshooting](../troubleshooting.md)

---

*Generated automatically by `zig build gendocs`*
