# ABI Master Specification

This document serves as the master reference for the ABI Framework, encompassing both the architectural design of the core system and the plugin ecosystem.

## 1. Architectural Vision (Refactor 2026-05-14)
The ABI framework is built on a data-oriented composition model optimized for Zig 0.17.0, moving from nested structures to a flat, registry-based lifecycle.

### Core Principles
- **Explicit Memory Management**: Custom allocators (Arena, Pool).
- **Data-Oriented Design**: Cache-friendly layout, SIMD optimized.
- **Mod/Stub Contract**: Feature-gated via `build_options` to allow tree-shaking.
- **Registry-Based Lifecycle**: Managed by `src/core/registry.zig`.

## 2. Directory & Module Structure
```
src/
├── root.zig           # Public API and feature exports
├── core/              # Foundational systems (registry, memory, scheduler)
├── foundation/        # OS and primitive abstractions (io, time, sync)
├── features/          # Domain-specific modules
│   ├── ai/            # Abbey-Aviva-Abi Pipeline (Router, profiles, governance)
│   └── wdbx/          # Vector Storage & Block Chain (HNSW index, MVCC chain)
└── tests/             # Integration and stress tests
```

## 3. Core Features (WDBX Substrate)
- **HNSW Index**: Hierarchical Navigable Small World index using SIMD-accelerated distance calculations.
- **Block Chain Memory**: Cryptographically chained conversation blocks (SHA-256) with MVCC-based persistent storage for immutable state management.

## 4. AI Pipeline: Abbey-Aviva-Abi
- **Routing**: Sentiment-based, multi-weight routing across Abbey, Aviva, and Abi profiles.
- **Governance**: Constitution-driven validation of response integrity against 6 core principles (Safety, Honesty, Privacy, Fairness, Autonomy, Transparency).

### Plugin System Implementation
The plugin system is implemented via build-time registry generation:
1. `tools/generate_plugin_registry.zig`: Scans `src/plugins/`, validates structure (Mod/Stub pattern), and generates `src/plugin_registry.zig`.
2. `build.zig`: Automatically triggers generation during `abi` build.
3. `src/registry.zig`: Imports `plugin_registry.zig` and invokes `registerPlugins()`.
4. CLI: `abi plugin list` provides discovery interface.


### Plugin Discovery (abi-plugin.json)
Each plugin must provide a manifest:
```json
{
  "name": "plugin-name",
  "version": "0.1.0",
  "targetFeature": "ai",
  "entryPoint": "src/plugin.zig",
  "dependencies": { "abi": "0.17.0" }
}
```

### CLI & Build Integration
- **CLI**: `abi plugin` commands (list, install, link, validate) manage plugins globally (`~/.abi/plugins/`) or locally.
- **Build System**: The CLI automates plugin registration, generating `src/plugins.zig` and adding plugin paths to `build.zig`.
- **Validation**: Plugins are verified against framework invariants and Mod/Stub parity requirements at build-time.
- **Security**: No dynamic loading (shared libraries) is allowed; static compilation integrity is maintained.
