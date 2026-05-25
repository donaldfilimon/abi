# ABI Master Specification

This document serves as the master reference for the ABI Framework, encompassing both the architectural design of the core system and the plugin ecosystem.

## 1. Architectural Vision (Refactor 2026-05-14)
The ABI framework is built on a data-oriented composition model optimized for Zig 0.17.0, with feature-gated modules, explicit ownership, and a generated static plugin registry that preserves plugin manifest metadata.

### Core Principles
- **Explicit Memory Management**: Custom allocators (Arena, Pool).
- **Data-Oriented Design**: Cache-friendly layout, SIMD optimized.
- **Mod/Stub Contract**: Feature-gated via `build_options` to allow tree-shaking.
- **Registry-Based Lifecycle**: Managed by `src/core/registry.zig`, with memory and scheduler helpers exposed from `src/core/memory.zig` and `src/core/scheduler.zig`.

## 2. Directory & Module Structure
```
src/
├── root.zig           # Public API and feature exports
├── core/              # Registry, config, memory, scheduler
│   ├── registry.zig
│   ├── memory.zig
│   └── scheduler.zig
├── interfaces.zig     # Cross-module contract types
├── foundation/        # OS and primitive abstractions (io, time, sync)
├── features/          # Domain-specific modules
│   ├── ai/            # Abbey-Aviva-Abi Pipeline (Router, profiles, governance)
│   └── wdbx/          # Vector Storage & Block Chain (HNSW index, MVCC chain)
├── plugins/           # Static plugin manifests and local plugin manager
├── integration_tests.zig
└── benchmarks.zig
```

## 3. Core Features (WDBX Substrate)
- **HNSW Index**: Hierarchical Navigable Small World index using SIMD-accelerated distance calculations.
- **Block Chain Memory**: Cryptographically chained conversation blocks (SHA-256) with MVCC-based persistent storage for immutable state management.

## 4. AI Pipeline: Abbey-Aviva-Abi
- **Routing**: Sentiment-based, multi-weight routing across Abbey, Aviva, and Abi profiles.
- **Governance**: Constitution-driven validation of response integrity against 6 core principles (Safety, Honesty, Privacy, Fairness, Autonomy, Transparency).

### Plugin System Implementation
The plugin system is implemented via build-time registry generation:
1. `tools/generate_plugin_registry.zig`: Scans plugin manifests under `src/plugins/*/abi-plugin.json`, validates required manifest fields, and generates `src/plugin_registry.zig`.
2. `build.zig`: Automatically triggers generation during `abi` build.
3. `src/core/registry.zig`: Imports `plugin_registry.zig` and invokes `registerPlugins()`.
4. `src/plugins/plugin_manager.zig`: Provides required-field manifest validation and local load/list/unload APIs for plugin directories.
5. CLI: `abi plugin list` provides the current discovery interface, including plugin version and target feature metadata.


### Plugin Discovery (abi-plugin.json)
Each plugin must provide a manifest. `entry_point` must be a safe relative `.zig` path (no absolute paths or `..` traversal):
```json
{
  "name": "plugin-name",
  "version": "0.1.0",
  "description": "What this plugin provides",
  "target_feature": "ai",
  "entry_point": "mod.zig"
}
```

### CLI & Build Integration
- **CLI**: `abi plugin list` lists the statically generated registry contents with version and target feature metadata.
- **Build System**: `build.zig` runs `tools/generate_plugin_registry.zig`, generating `src/plugin_registry.zig` before CLI/check builds.
- **Validation**: Plugin feature surfaces with `mod.zig`/`stub.zig` pairs are checked by `zig build check-parity`; generated registry metadata is covered by `tests/contracts/plugin_registry.zig`.
- **Security**: No dynamic loading (shared libraries) is allowed; static compilation integrity is maintained.

### Connector Boundary
Discord connector calls validate credentials, numeric snowflake-like IDs, and Discord's 2000-byte message size limit before local acknowledgements or live HTTP dispatch.
