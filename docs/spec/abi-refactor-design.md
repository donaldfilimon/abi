# ABI Framework Refactor Design (Zig 0.17.0)

**Date:** 2026-05-14
**Status:** Implemented baseline; refreshed against current source layout
**Scope:** Full codebase refactor based on current ABI source, executable contracts, and WDBX/Abbey design inputs. Performance and deployment-scale claims require separate benchmark evidence.

---

## 1. Architectural Vision: Data-Oriented Composition

The current tree uses a flat, registry-based composition model optimized for Zig 0.17.0. Treat `build.zig`, `src/features/mod.zig`, `src/abi_cli/usage.zig`, and contract tests as source truth when this design drifts.

### 1.1 Core Principles
- **Explicit Memory Management**: Use dedicated allocators (Arena, Pool) for different component lifecycles.
- **Data-Oriented Design**: Organize data (embeddings, blocks) for cache locality and SIMD friendliness.
- **Comptime-Gated Features**: Use the Mod/Stub pattern enforced by `build_options` to allow aggressive tree-shaking.
- **Registry-Based Lifecycle**: A central `Registry` in `src/core/registry.zig` manages generated plugin loading, with memory and scheduler helpers exposed from `src/core/memory.zig` and `src/core/scheduler.zig`.

---

## 2. Directory Structure

```
src/
├── root.zig           # Public API and feature exports
├── main.zig           # CLI entry point (delegates to abi_cli/)
├── interfaces.zig     # Cross-module contract types (greenfield rewrite)
├── core/              # Registry, config, memory, scheduler
│   ├── registry.zig
│   ├── memory.zig
│   ├── scheduler.zig
│   └── config.zig
├── foundation/        # OS and primitive abstractions
│   ├── mod.zig        # Re-exports all foundation modules
│   ├── io/            # Split IO module
│   │   ├── mod.zig    # Entry point, re-exports, utility functions
│   │   ├── stats.zig  # IOStats, IOStatsSnapshot
│   │   ├── reader.zig # BufferedReader
│   │   ├── writer.zig # BufferedWriter
│   │   └── filestream.zig  # FileStream
│   ├── time.zig       # Unified time (unixMs)
│   ├── sync.zig       # RwLock, SpinLock
│   ├── os.zig         # OSController, command policies
│   ├── logger.zig     # Structured logger with levels
│   ├── utils.zig      # String/path helpers, containsIgnoreCase
│   ├── credentials.zig # API key/credential storage
│   ├── errors.zig     # Centralized error types
│   ├── pool_allocator.zig # Arena/pool allocator utilities
│   ├── plugin_validator.zig # Plugin manifest schema validation
│   └── validation.zig # General validation helpers
├── features/          # Domain-specific modules (mod/stub pattern)
│   ├── mod.zig        # Feature selection via build_options
│   ├── backend_utils.zig # Shared backend utilities
│   ├── ai/            # Abbey-Aviva-Abi Pipeline (flat layout)
│   │   ├── mod.zig    # Public API: run, complete, train, profiles, router
│   │   ├── stub.zig   # No-op stubs when feat-ai disabled
│   │   ├── helpers.zig # countNonEmptyLines, textEmbedding, responseEmbedding
│   │   ├── router.zig # AdaptiveModulator, analyzeSentiment, profile routing
│   │   ├── constitution.zig # 6-principle governance validation
│   │   ├── pipeline.zig # Training pipeline
│   │   └── streaming.zig # Local OpenAI-compatible SSE server
│   ├── gpu/           # GPU status & vector operations (mod/stub)
│   │   ├── mod.zig         # Entry point, re-exports, Metal init
│   │   ├── stub.zig        # No-op stubs when feat-gpu disabled
│   │   ├── backends.zig    # Backend enum, detection, capabilities
│   │   ├── vector_ops.zig  # VectorOps: dot, squaredL2, cosineSimilarity
│   │   ├── reporting.zig   # backendStatusReport, isAvailable
│   │   └── metal_shared.zig # Metal ObjC runtime context
│   ├── wdbx/          # Vector Store & Block Chain (mod/stub, flat layout)
│   │   ├── mod.zig    # Store, putVector, search, appendBlock
│   │   ├── stub.zig   # No-op stubs when feat-wdbx disabled
│   │   ├── hnsw.zig   # HNSW index with SIMD cosine distance
│   │   ├── chain.zig  # Block chain with MVCC snapshots
│   │   └── spatial_3d.zig # In-memory 3D spatial index
│   ├── accelerator/   # Backend selection metadata (mod/stub)
│   ├── shaders/       # Local shader validation (mod/stub)
│   ├── mlir/          # Textual MLIR lowering (mod/stub)
│   ├── tui/           # Diagnostics dashboard (mod/stub, enabled by default)
│   ├── mobile/        # Mobile platform surface (mod/stub, disabled by default)
│   └── os_control/    # Safe OS command policy controls (mod/stub)
├── connectors/        # External service connectors
│   ├── mod.zig        # Re-exports and connector registration
│   ├── connector.zig  # ConnectorError, TransportMode, Response
│   ├── http.zig       # HTTP helpers, basic auth, form encoding
│   ├── json.zig       # JSON string escaping
│   ├── openai.zig     # OpenAI connector
│   ├── anthropic.zig  # Anthropic connector
│   ├── discord.zig    # Discord connector with credential/snowflake-like/message validation
│   └── twilio.zig     # Twilio ConversationRelay simulator with SID/token/base-url/timeout validation
├── abi_cli/           # CLI dispatch, handlers, usage
│   ├── dispatch.zig   # Top-level command routing
│   ├── usage.zig      # Source of truth for CLI help text
│   └── handlers/      # Per-command handler implementations
│       ├── mod.zig
│       ├── agent.zig
│       ├── auth.zig
│       ├── backends.zig
│       ├── dashboard.zig
│       ├── plugin.zig
│       ├── train.zig
│       └── twilio.zig
├── mcp/               # MCP JSON-RPC 2.0 server
│   ├── main.zig       # stdio loop + loopback HTTP/SSE
│   ├── handlers.zig   # Tool call implementations
│   ├── json_helpers.zig # JSON parsing utilities
│   ├── protocol.zig   # JSON-RPC protocol types
│   └── server.zig     # Server transport layer
├── plugins/           # Plugin manifests and local plugin manager
│   ├── plugin_manager.zig      # Load/unload/list from required JSON manifests
│   ├── example-plugin/         # Baseline example plugin fixture
│   └── example-wdbx-plugin/    # WDBX-targeted registry fixture
├── plugin_registry.zig # Auto-generated metadata registry (do not edit)
├── testing/           # Test infrastructure
│   └── test_helpers.zig # TestAllocator, TempDir, mocks, assertions
├── integration_tests.zig
└── benchmarks.zig

tests/                 # External contract tests
├── contracts/
│   ├── surface.zig    # CLI/MCP surface contract tests
│   ├── mcp_tools.zig  # MCP tool contract tests
│   └── plugin_registry.zig # Generated plugin metadata contract tests

tools/                 # Build helpers and verification scripts
├── build.sh           # macOS/Darwin build wrapper
├── check_parity.zig   # Feature/plugin mod/stub top-level declaration-name parity checker
├── check_feature_stubs.sh # Feature stub compilation plus feature/public contract smoke tests
├── generate_plugin_registry.zig # Plugin manifest scanner
└── run_contract_cli.sh # CLI contract test runner
```

---

## 3. WDBX Substrate: HNSW Index (Priority 1)

The Hierarchical Navigable Small World (HNSW) index is the foundation for semantic retrieval.

### 3.1 Data Structures
- **VectorStorage**: A contiguous array of `f32` vectors, optimized for SIMD `@Vector`.
- **Node**: 
  - `id: u32`
  - `edges: [MAX_LAYERS]ArrayList(u32)`

### 3.2 Algorithms
- **SIMD Distance**: Cosine similarity using `std.simd` primitives for Zig 0.17.
- **Concurrent Insert**: Serialized graph mutation guarded by the WDBX HNSW synchronization primitives; concurrent edge-update semantics are internal implementation details.
- **Heuristic Search**: Deterministic neighborhood traversal returning scores in non-increasing order. The current source does not publish QPS, latency, sharding, or distributed-storage guarantees.

---

## 4. WDBX Substrate: Block Chain Memory (Priority 2)

Cryptographically chained conversation blocks for append-only state history and integrity checks.

### 4.1 ConversationBlock
```zig
const ConversationBlock = struct {
    id: [32]u8,           // SHA-256 block hash
    prev_id: [32]u8,      // Previous block hash or genesis hash
    timestamp_ms: i64,    // foundation.time.unixMs()
    profile: []const u8,  // Abbey, Aviva, or Abi label
    query_id: u32,        // Stored query vector id
    response_id: u32,     // Stored response vector id
    metadata: []const u8, // JSON or key/value metadata snapshot
};
```

### 4.2 MVCC (Multiversion Concurrency Control)
- Use a SHA-256 chained block list with MVCC-style snapshot iteration.
- Readers acquire a snapshot/iterator view while writers append through the chain API; concurrency details stay behind the chain API.
- Contract tests verify appended metadata can be retrieved through both the store's last-block view and a block-chain snapshot lookup.

---

## 5. AI Pipeline: Abbey-Aviva-Abi

The three-way weighted routing and blending pipeline.

### 5.1 Routing (Abi)
- **Sentiment Analysis**: Fast-path intent detection.
- **Weighting**: Calculate `w_abbey`, `w_aviva`, `w_abi` from keyword-weighted local routing heuristics plus optional adaptive EMA persistence.

### 5.2 Constitution (Governance)
- Six principles: truthfulness, safety, helpfulness, fairness, privacy, transparency.
- Post-generation validation via `evaluateResponse()`.

---

## 6. Maintenance Strategy

1. **Source truth first**: Reconcile architecture prose against `build.zig`, `src/features/mod.zig`, `src/abi_cli/usage.zig`, `docs/contracts/public-api.md`, and contract tests before changing public surfaces.
2. **Mod/stub parity**: Any public feature API change must update both `mod.zig` and `stub.zig`, including disabled-feature semantics, then pass `zig build check-parity`. The parity checker covers top-level public declaration names for feature/plugin pairs, not full signature equivalence. The check gate also runs focused feature-off behavior smoke tests and feature-aware public contracts for every `-Dfeat-*` flag.
3. **Generated registry**: Do not edit `src/plugin_registry.zig`; update plugin manifests and regenerate through the build step. Required manifest fields are `name`, `version`, `description`, `target_feature`, and `entry_point`; `targetFeature` / `entryPoint` aliases are accepted. `entry_point` must be a safe relative `.zig` path that exists under the plugin directory. Contract coverage expects multiple bundled plugin fixtures to remain discoverable.
4. **Import boundaries**: Library modules under `src/` use relative imports with `.zig` extensions. MCP executable/handler files (`src/mcp/main.zig`, `src/mcp/handlers.zig`) may import the public `abi` package because `build.zig` wires that package explicitly; never do so from modules re-exported by `src/root.zig`.
5. **Verification gates**: For source changes run `./build.sh check`; for release/readiness changes run `./build.sh full-check` (`check` plus integration tests, benchmarks, and TUI smoke).
6. **External claims**: Do not reuse spreadsheet or analysis-document claims for AES/RBAC, Swift/Python/TensorFlow stacks, distributed sharding, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy use, or SQuAD/CodeSearchNet/GPT comparisons unless the claim is tied to a current repo test, source implementation, or benchmark artifact.
