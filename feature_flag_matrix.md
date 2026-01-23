---
title: "feature_flag_matrix"
tags: []
---
# Feature‑Flag Matrix

This table maps each `build_options.enable_*` flag to the files that depend on it. It helps developers understand the impact of toggling a feature.

| Feature Flag | Primary Modules (uses `if (build_options.enable_…)`) |
|--------------|----------------------------------------------------|
| `enable_gpu` | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/gpu/...`, `src/ai/abbey/neural/...`, `src/ai/llm/ops/gpu.zig` |
| `enable_ai`  | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/ai/...` (all AI sub‑modules), `src/ai/multi_agent/mod.zig` |
| `enable_llm` | `src/ai/mod.zig`, `src/ai/llm/mod.zig`, `src/ai/llm/ops/...` |
| `enable_database` | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/database/...` |
| `enable_network` | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/network/...` |
| `enable_profiling` | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/observability/...` |
| `enable_web` | `src/abi.zig`, `src/config.zig`, `src/framework.zig`, `src/web/...`, `src/ai/abbey/client.zig` |

*The list is not exhaustive but captures the main entry points where a feature flag gates compilation.*


