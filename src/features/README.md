# Features Module (Legacy)

Implementation layer for optional features. Most features have been migrated to
top-level modules as part of the 2026-01-17 refactoring.

## Current Status (2026-01-17)

| Directory | Status | Description |
|-----------|--------|-------------|
| `ai/` | Active | Full AI implementation (re-exported via `src/ai/`) |
| `connectors/` | Active | API connectors (OpenAI, Ollama, Anthropic, etc.) |
| `ha/` | Active | High availability (backup, PITR, replication) - exported via `abi.ha` |

**Deleted:**
- `monitoring/` - Consolidated into `src/observability/` (2026-01-17)

## Refactoring Summary

The following modules have been fully migrated away from features/:

| Module | New Location | Status |
|--------|--------------|--------|
| GPU | `src/gpu/` | Fully migrated |
| Database | `src/database/` | Fully migrated |
| Network | `src/network/` | Fully migrated |
| Web | `src/web/` | Fully migrated |
| Runtime | `src/runtime/` | Fully migrated (was in compute/) |
| Registry | `src/registry/` | New system (comptime, runtime, dynamic) |

## Architecture

Features are accessible via two paths:

1. **Top-level modules** (preferred): `src/gpu/`, `src/ai/`, `src/database/`, etc.
   These provide Context structs for Framework integration.

2. **This module** (implementation): Direct access to feature code.

```
Fully Migrated (primary location is top-level):
  src/gpu/           ->  GPU acceleration (primary implementation)
  src/database/      ->  Vector database (primary implementation)
  src/network/       ->  Distributed compute (primary implementation)
  src/web/           ->  Web utilities (primary implementation)
  src/runtime/       ->  Runtime infrastructure (primary implementation)
  src/observability/ ->  Metrics/tracing (consolidated from features/monitoring/)

Still Here (intentionally):
  src/features/ai/         ->  AI implementation (re-exported via src/ai/)
  src/features/connectors/ ->  API connectors (OpenAI, Ollama, Anthropic, etc.)
  src/features/ha/         ->  High availability (exported via abi.ha)
```

## Feature Flags

| Feature | Flag | Default | Description |
|---------|------|---------|-------------|
| AI | `-Denable-ai` | true | LLM inference, embeddings, RAG, connectors |
| Monitoring | `-Denable-profiling` | true | Metrics, tracing, OpenTelemetry |

## Sub-modules

| Directory | Public Export | Status | Description |
|-----------|---------------|--------|-------------|
| `ai/` | `abi.ai` | Active | AI features (LLM, embeddings, RAG, agents) |
| `connectors/` | `abi.connectors` | Active | API connectors (OpenAI, Ollama, etc.) |
| `ha/` | `abi.ha` | Active | High availability (backup, PITR, replication) |

## Usage

**Preferred: Using top-level modules with Framework**

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .withAi(.{ .llm = .{} })
    .withDatabase(.{ .path = "./data" })
    .build();
defer fw.deinit();

const ai_ctx = try fw.getAi();
```

**Direct access (advanced)**

```zig
const abi = @import("abi");

// AI (if enabled)
const response = try abi.ai.inferText(allocator, "Hello!");
```

## See Also

- [src/README.md](../README.md) - Source overview
- [API Reference](../../API_REFERENCE.md)
- [ROADMAP.md](../../ROADMAP.md) - Migration plans
