# ABI Master Specification

This document serves as the master reference for the ABI Framework, encompassing both the architectural design of the core system and the plugin ecosystem.

## 1. Architectural Vision (Refactor 2026-05-14)
The ABI framework is built on a data-oriented composition model targeting Zig 0.17, with feature-gated modules, explicit ownership, and a generated static plugin registry that preserves plugin manifest metadata. Public claims should stay inside what the current source and validation gates prove.

### Core Principles
- **Explicit Memory Management**: Custom allocators (Arena, Pool).
- **Data-Oriented Design**: Cache-friendly layout, SIMD optimized.
- **Mod/Stub Contract**: Feature-gated via `build_options` to allow tree-shaking.
- **Registry-Based Lifecycle & Observability (Significantly Advanced)**: 
  - `core/scheduler.zig` now drives real user work (`abi agent train` submits prioritized TrainTasks; dashboard emits live ticks; MCP owns a long-lived scheduler instance exposed via `scheduler_stats` tool). `Scheduler.stats()` + `MemoryTracker` attachment provide deep visibility.
  - Metrics counters (`submitted`/`completed`/`failed`) are live on task lifecycle when `-Dfeat-metrics`.
  - The default-on `telemetry` feature exposes lightweight `record` / `increment` hooks.
  - MemoryTracker + TrackingAllocator are wired into production CLI paths (agent train arenas + dashboard scheduler), WDBX store hot paths, SEA adaptive-weight persistence, and AI training internals (dataset inspection plus metadata persistence) with integration tests.
  - WDBX `Store` + HNSW acceleration status is updated on vector operations and surfaced in stats/manifest/MCP/CLI without claiming native kernels when the backend falls back.
  - The `abi wdbx` namespace operates local snapshots, WAL verification, blocks, queries, benchmarks, GPU info, loopback REST, and in-process roadmap demos.
  - All of this is exercised by contract tests and the new accelerated HNSW GPU path coverage.

## 2. Directory & Module Structure
```
src/
├── root.zig           # Public API and feature exports
├── core/              # Registry, config, memory, scheduler
│   ├── registry.zig
│   ├── config.zig
│   ├── memory.zig
│   └── scheduler.zig
├── interfaces.zig     # Cross-module contract types
├── foundation/        # OS and primitive abstractions (io, time, sync)
├── features/          # Comptime-gated domain modules (each a mod.zig + stub.zig):
│   │                  #   ai, wdbx, sea, gpu, accelerator, shaders, mlir,
│   │                  #   os_control, mobile, metrics, tui, hash, telemetry
│   ├── mod.zig        # Feature dispatcher: selects real mod vs stub per build flag
│   ├── ai/            # Abbey-Aviva-Abi pipeline (router, profiles, governance, training)
│   └── wdbx/          # Vector storage & runtime (HNSW, chain, snapshots, WAL, demos)
├── connectors/        # openai, anthropic, discord, twilio, grok, http (+ live transport)
├── mcp/               # MCP server (JSON-RPC stdio + HTTP/SSE), tool dispatch, durable state
├── cli/               # CLI command dispatch + handlers
├── plugins/           # Static plugin manifests and local plugin manager
├── integration_tests.zig
└── benchmarks.zig
```

## 3. Core Features (WDBX Substrate)
- **HNSW Index**: Hierarchical Navigable Small World index. Cosine distance uses a portable SIMD implementation (`cosineDistanceSIMD` with `@Vector` + scalar fallback) and routes query/candidate scoring through `gpu.vectorOps().cosineSimilarity` / `batchCosineSimilarity`. Those vector ops use native Metal kernels only when the backend reports initialized native kernels; otherwise they deterministically fall back to vectorized CPU. The vector and HNSW paths are exercised by contract tests in `tests/contracts/feature_modules.zig` and focused distance tests in `src/features/wdbx/hnsw_distance.zig`.
- **Block Chain Memory**: Cryptographically chained conversation blocks (SHA-256) with MVCC-based snapshot lookup for immutable state management; contract tests cover metadata round-tripping and snapshot access.
- **Durability**: JSONL snapshots include a SHA-256 integrity line; the WAL uses CRC32-framed append records with replay and corruption detection. `abi wdbx db verify` cross-checks local snapshot and WAL state.
- **Roadmap Modules**: `cluster`, `compute`, `compression`, `crypto_he`, and `rest` provide in-process demonstrations and loopback APIs. They are intentionally scoped as demos unless source/tests later prove distributed transport, native accelerator dispatch, learned compression, or full FHE.
- **Observability & Lifecycle**: `Store` exposes `stats()` + `accelerationStatus()` (backend/mode/message) updated on vector/spatial ops. Acceleration reporting is also surfaced via `exportManifest()`.
- **Claim Boundary**: The current repo does not prove distributed sharding, AES/RBAC, Swift/Python/TensorFlow runtime support, Kubernetes/H100 deployments, regulatory certifications, production QPS/latency/accuracy, energy efficiency, comparative model benchmark scores, or general native GPU acceleration. The repo proves a GPU/vector abstraction, batched cosine API, HNSW integration, backend status reporting, and deterministic CPU fallback; native Metal execution is claimed only when the runtime backend reports initialized native kernels.

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
- **WDBX CLI**: `abi wdbx <db|block|query|benchmark|cluster|compute|secure|gpu|api>` is a contract-tested local runtime surface. The `cluster`, `compute`, and `secure` subcommands are in-process demonstrations, not distributed or native-accelerator claims.
- **Build System**: `build.zig` runs `tools/generate_plugin_registry.zig`, generating `src/plugin_registry.zig` before CLI/check builds.
- **Validation**: Plugin feature surfaces with `mod.zig`/`stub.zig` pairs are checked by `zig build check-parity`; this checks top-level public declaration names, not complete signatures. Generated multi-plugin registry metadata is covered by `tests/contracts/plugin_registry.zig`.
- **Security**: No dynamic loading (shared libraries) is allowed; static compilation integrity is maintained.

### Connector Boundary
Discord connector calls validate printable non-whitespace credentials, numeric snowflake-like client/channel/author IDs, and Discord's 2000-byte message size limit before local acknowledgements or live HTTP dispatch. Twilio connector calls validate account SIDs as `AC` plus 32 hex characters, auth tokens as 32 hex characters, base URL, timeout, explicit `.live` transport selection, XML/form escaping, and ConversationRelay payload aliases/wrong-typed payloads before local responses or live TwiML/form dispatch. OpenAI and Anthropic local streaming helpers remain deterministic unless explicit live methods are used.

### Feature/GPU Completion Contract
Every feature surface under `src/features/` has a real implementation and disabled stub selected by `src/features/mod.zig`. `tools/check_feature_stubs.sh` compiles every `-Dfeat-*` disabled path, runs focused `test-feature-contracts` coverage, runs feature-aware public `test-contracts` coverage for every disabled feature, and covers `-Dfeat-mobile=true` because mobile defaults off.

**GPU/Vector Acceleration (Advanced in Phase 2)**: Vector operations (`gpu.vectorOps()`) provide `dot` / `squaredL2` / `cosineSimilarity` / `batchCosineSimilarity`. They use native Metal kernels on macOS only when `feat-gpu`, backend acceleration, and initialized native kernels are all present; otherwise the same API falls back to vectorized CPU. HNSW cosine distance in `wdbx/hnsw.zig` uses that vector abstraction for query/candidate scoring while preserving the pure-SIMD fallback. Contract tests exercise the vector surface, batched cosine API, HNSW search path, and disabled-feature behavior. WDBX `Store.accelerationStatus()` and `runAccelerationKernel` report the selected mode without fabricating native acceleration.
