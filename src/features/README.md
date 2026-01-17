# Features Module

Implementation layer for optional features. This module is being incrementally
migrated to top-level modules.

## Migration Status (2026-01-17)

| Directory | Status | New Location |
|-----------|--------|--------------|
| `ai/` | Partial | `src/ai/` (wrapper) + here (implementation) |
| `connectors/` | Active | Keep here (API integrations) |
| `ha/` | Active | Keep here (HA components) |
| `monitoring/` | Deprecated | Use `src/observability/` |

## Architecture

Features are accessible via two paths:

1. **Top-level modules** (preferred): `src/gpu/`, `src/ai/`, `src/database/`, etc.
   These provide Context structs for Framework integration.

2. **This module** (implementation): Direct access to feature code.

```
Fully Migrated (no longer here):
  src/gpu/mod.zig        ->  primary implementation
  src/database/mod.zig   ->  primary implementation
  src/network/mod.zig    ->  primary implementation
  src/web/mod.zig        ->  primary implementation
  src/runtime/mod.zig    ->  primary implementation

Partially Migrated:
  src/ai/mod.zig         ->  re-exports from  ->  src/features/ai/mod.zig

Still Here (intentionally):
  src/features/connectors/  ->  API connectors (OpenAI, Ollama, HuggingFace, etc.)
  src/features/ha/          ->  High availability (backup, PITR, replication)
  src/features/monitoring/  ->  @deprecated - use src/observability/
```

## Feature Flags

| Feature | Flag | Default | Description |
|---------|------|---------|-------------|
| AI | `-Denable-ai` | true | LLM inference, embeddings, RAG, connectors |
| Monitoring | `-Denable-profiling` | true | Metrics, tracing, OpenTelemetry |

## Sub-modules

| Directory | Top-Level Module | Status | Description |
|-----------|------------------|--------|-------------|
| `ai/` | `src/ai/` | Partial migration | AI features (LLM, embeddings, RAG) |
| `connectors/` | (via ai) | Active | API connectors |
| `ha/` | (internal) | Active | High availability components |
| `monitoring/` | `src/observability/` | Deprecated | Use observability instead |

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
