# ABI Master Specification

This document serves as the master reference for the ABI Framework, encompassing both the architectural design of the core system and the plugin ecosystem.

## 1. Architectural Vision (Refactor 2026-05-14)
The ABI framework is built on a data-oriented composition model targeting Zig 0.17.0, with feature-gated modules, explicit ownership, and a generated static plugin registry that preserves plugin manifest metadata. Public claims should stay inside what the current source and validation gates prove.

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
- **Block Chain Memory**: Cryptographically chained conversation blocks (SHA-256) with MVCC-based snapshot lookup for immutable state management; contract tests cover metadata round-tripping and snapshot access.
- **Claim Boundary**: The current repo does not prove distributed sharding, AES/RBAC, Swift/Python/TensorFlow runtime support, Kubernetes/H100 deployments, regulatory certifications, production QPS/latency/accuracy, energy efficiency, or comparative model benchmark scores.

## 4. AI Pipeline: Abbey-Aviva-Abi
- **Routing**: Sentiment-based, multi-weight routing across Abbey, Aviva, and Abi profiles.
- **Governance**: Constitution-driven validation of response integrity against 6 core principles (Safety, Honesty, Privacy, Fairness, Autonomy, Transparency).

### Plugin System Implementation
The plugin system is implemented via build-time registry generation:
1. `tools/generate_plugin_registry.zig`: Scans plugin manifests under `src/plugins/*/abi-plugin.json`, validates required manifest fields, and generates `src/plugin_registry.zig`; bundled fixtures include baseline and WDBX-targeted example plugins.
2. `build.zig`: Automatically triggers generation during `abi` build.
3. `src/core/registry.zig`: Imports `plugin_registry.zig` and invokes `registerPlugins()`.
4. `src/plugins/plugin_manager.zig`: Provides required-field manifest validation and local load/list/unload APIs for plugin directories.
5. CLI: `abi plugin list` provides the current discovery interface, including plugin count, version, target feature, safe entry point, and description metadata.


### Plugin Discovery (abi-plugin.json)
Each plugin must provide a manifest. `entry_point` must be a safe relative `.zig` path whose file exists under the plugin directory (no absolute paths, `..` traversal, empty path segments, backslashes, or drive separators). The generator and plugin manager also accept `targetFeature` / `entryPoint` aliases:
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
- **CLI**: `abi plugin list` lists the statically generated registry contents with plugin count, version, target feature, entry point, and description metadata.
- **Build System**: `build.zig` runs `tools/generate_plugin_registry.zig`, generating `src/plugin_registry.zig` before CLI/check builds.
- **Validation**: Plugin feature surfaces with `mod.zig`/`stub.zig` pairs are checked by `zig build check-parity`; this checks top-level public declaration names, not complete signatures. Generated multi-plugin registry metadata is covered by `tests/contracts/plugin_registry.zig`.
- **Security**: No dynamic loading (shared libraries) is allowed; static compilation integrity is maintained.

### Connector Boundary
Discord connector calls validate printable non-whitespace credentials, numeric snowflake-like client/channel/author IDs, and Discord's 2000-byte message size limit before local acknowledgements or live HTTP dispatch. Twilio connector calls validate account SIDs as `AC` plus 32 hex characters, auth tokens as 32 hex characters, base URL, timeout, explicit `.live` transport selection, XML/form escaping, and ConversationRelay payload aliases/wrong-typed payloads before local responses or live TwiML/form dispatch. OpenAI and Anthropic local streaming helpers remain deterministic unless explicit live methods are used.

### Feature/GPU Completion Contract
Every feature surface under `src/features/` has a real implementation and disabled stub selected by `src/features/mod.zig`. `tools/check_feature_stubs.sh` compiles every `-Dfeat-*` disabled path, runs focused `test-feature-contracts` coverage, runs feature-aware public `test-contracts` coverage for every disabled feature, and covers `-Dfeat-mobile=true` because mobile defaults off. GPU backend selection is runtime behavior: Metal may initialize on macOS, but vector operations and status reporting fall back deterministically to CPU when native kernels are unavailable.
