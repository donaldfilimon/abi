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
├── interfaces.zig     # Cross-module contract types (small shared request/response structs for MCP/CLI surfaces; originally scoped as larger greenfield rewrite)
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
│   │   ├── types.zig  # Public request/result/profile types re-exported by mod.zig
│   │   ├── stub_types.zig / stub_profile.zig / stub_constitution.zig
│   │   ├── completion.zig # Completion routing, scheduler task, and optional WDBX persistence
│   │   ├── training.zig # Training/evaluation workflow, scheduler task, and WDBX training records
│   │   ├── training_support.zig # Dataset/profile validation and profile embeddings used by training paths
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
│   │   ├── types.zig  # Shared constants and record/stat types re-exported by mod.zig
│   │   ├── stub_types.zig / stub_index.zig / stub_storage.zig / stub_spatial_3d.zig
│   │   ├── hnsw.zig   # HNSW index with SIMD cosine distance
│   │   ├── chain.zig  # Block chain with MVCC snapshots
│   │   ├── persistence.zig # JSONL snapshot serialize/restore with SHA-256 integrity line
│   │   ├── wal.zig    # Write-ahead log: CRC32-framed records, replay, corruption detection
│   │   ├── temporal.zig # Temporal/causal graph + semantic×temporal×causal×persona hybrid ranking
│   │   ├── cluster.zig # In-process Raft-style consensus demo: election/quorum/failover (no networked transport)
│   │   ├── compression.zig # int8 embedding quantization round-trip demo (not a learned codec)
│   │   ├── crypto_he.zig # Additive single-key homomorphic aggregation demo (not full FHE)
│   │   ├── compute.zig # CPU/GPU/NPU/TPU backend selector w/ deterministic CPU fallback (native dispatch not linked)
│   │   ├── rest.zig   # Loopback REST listener: POST /insert /query /verify, GET /health /stats
│   │   └── spatial_3d.zig # In-memory 3D spatial index
│   ├── accelerator/   # Backend selection metadata (mod/stub)
│   ├── shaders/       # Local shader validation (mod/stub)
│   ├── mlir/          # Textual MLIR lowering (mod/stub)
│   ├── hash/          # Hash utility surface (mod/stub, enabled by default)
│   ├── metrics/       # Optional observability counters (mod/stub, disabled by default)
│   ├── telemetry/     # Fixed-capacity process-wide event/counter hooks (mod/stub, enabled by default)
│   ├── tui/           # Diagnostics dashboard + live terminal redraw helpers (mod/stub, enabled by default)
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
│   ├── twilio.zig     # Twilio ConversationRelay simulator with SID/token/base-url/timeout validation
│   └── grok.zig       # Grok/xAI connector (local deterministic + .live opt-in)
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
│       ├── twilio.zig
│       └── wdbx.zig   # WDBX runtime control surface (db/block/query/benchmark/cluster/compute/secure/gpu/api)
├── mcp/               # MCP JSON-RPC 2.0 server
│   ├── main.zig       # stdio loop + loopback HTTP/SSE
│   ├── handlers.zig   # Tool call implementations
│   ├── json_helpers.zig # JSON parsing utilities
│   ├── protocol.zig   # JSON-RPC protocol types
│   ├── rpc.zig        # Shared JSON-RPC request processing
│   ├── shutdown.zig   # Signal/shutdown coordination
│   ├── state.zig      # Long-lived MCP feature state
│   └── server.zig     # Stdio and HTTP/SSE transport layer
├── plugins/           # Plugin manifests and local plugin manager
│   ├── abi-plugin.json         # Top-level plugin manifest (core ABI surface)
│   ├── plugin_manager.zig      # Load/unload/list from required JSON manifests
│   ├── example-plugin/         # Baseline example plugin fixture (abi-plugin.json + mod.zig + stub.zig)
│   └── example-wdbx-plugin/    # WDBX-targeted registry fixture (abi-plugin.json + mod.zig + stub.zig)
├── plugin_registry.zig # Auto-generated metadata registry (do not edit)
├── testing/           # Test infrastructure
│   └── test_helpers.zig # TestAllocator, TempDir, mocks, assertions
├── integration_tests.zig
└── benchmarks.zig

tests/                 # External contract tests
├── contracts/
│   ├── surface.zig         # CLI/MCP surface contract tests
│   ├── mcp_tools.zig       # MCP tool contract tests
│   ├── plugin_registry.zig # Generated plugin metadata contract tests
│   ├── feature_modules.zig # Per-feature mod/stub + public contract smoke (feature-on/off)
│   └── public_docs.zig     # Public documentation + claim boundary contracts

tools/                 # Build helpers and verification scripts
├── build.sh           # macOS/Darwin build wrapper (delegates to tools/build.sh)
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
  - `edges: [MAX_LAYERS]std.ArrayListUnmanaged(u32)`

### 3.2 Algorithms
- **Distance Path**: Cosine similarity routes through `gpu.vectorOps().cosineSimilarity()` when available and falls back to the local SIMD implementation for deterministic behavior.
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

### 4.3 Snapshot Persistence

`src/features/wdbx/persistence.zig` serializes a `Store` to a line-delimited JSON (JSONL) snapshot and restores it deterministically.

- **Format**: a `# ABI-WDBX v1` header line, one minified JSON object per record (`kv`, `vector`, `block`, `spatial`), and a trailing `# checksum:<sha256-hex>` integrity line covering the record body.
- **Integrity**: the SHA-256 checksum is verified on load when present (checksum-less snapshots remain loadable for backward compatibility); a truncated or tampered body is rejected with `error.ChecksumMismatch` rather than restoring partial state.
- **Faithful restore**: vectors restore to their original monotonically-assigned ids (a mismatch is `error.CorruptVectorId`); blocks restore with their original timestamps so the SHA-256 chain hashes reproduce exactly and `verifyBlocks()` still holds.
- **Untrusted input**: integer fields out of `u32` range fail with `error.FieldOutOfRange` instead of panicking, so a corrupt snapshot fails cleanly.
- **IO**: `saveToPath`/`loadFromPath` wrap serialize/deserialize over `std.Io` with a 64 MB read cap. WDBX-disabled builds expose no persistence surface.

### 4.4 WDBX Runtime Control Surface (`abi wdbx`) and Roadmap Modules

`src/abi_cli/handlers/wdbx.zig` adds an `abi wdbx <db|block|query|benchmark|cluster|compute|secure|gpu|api>` namespace — the 11th frozen CLI command, with its contract row in `tests/contracts/surface.zig`. It is comptime-gated on `build_options.feat_wdbx` and backed by the in-process store, JSONL snapshots, and the write-ahead log:

- `db init|verify`, `block insert|get`, `query` — snapshot + WAL lifecycle; `db verify` cross-checks WAL replay against the snapshot block count.
- `benchmark [count]` — local in-memory insert/search timing (explicitly *not* a published throughput claim).
- `gpu info` — GPU backend capability report.

The namespace also exercises forward-looking WDBX modules that are deliberately scoped as honest, **in-process demonstrations — not production or distributed capabilities**. See `docs/spec/wdbx-north-star.md` for the Current/Partial/Proposed mapping and `docs/contracts/external-claims-audit.md` for the claim boundary:

- `cluster status|demo` (`cluster.zig`) — an in-process Raft-style core (leader election, majority-quorum replication, failover). There is **no** networked RPC transport; `cluster status` reports `nodes=1 role=standalone`.
- `compute info` (`compute.zig`) — a CPU/GPU/NPU/TPU backend selector that always degrades to the deterministic CPU SIMD path; native accelerator dispatch is not linked.
- `secure demo` (`compression.zig` + `crypto_he.zig`) — int8 embedding quantization round-trip plus additive single-key homomorphic aggregation. This is not a learned codec and not full (multiplicative) FHE.
- `api serve [port]` (`rest.zig`) — a loopback-only REST listener (`POST /insert /query /verify`, `GET /health /stats`, default port 8081) built on a pure, unit-tested routing core.

All of these preserve mod/stub parity: `mod.zig` exports `wal`, `temporal`, `cluster`, `compression`, `crypto_he`, `compute`, and `rest`, while `stub.zig` carries matching empty parity markers so `zig build check-parity` holds with `-Dfeat-wdbx=false`.

---

## 5. AI Pipeline: Abbey-Aviva-Abi

The three-way weighted routing and blending pipeline.

### 5.1 Routing (Abi)
- **Sentiment Analysis**: Fast-path intent detection.
- **Weighting**: Calculate `w_abbey`, `w_aviva`, `w_abi` from keyword-weighted local routing heuristics plus optional adaptive EMA persistence.

### 5.2 Constitution (Governance)
- Six principles: truthfulness, safety, helpfulness, fairness, privacy, transparency.
- Post-generation validation via `evaluateResponse()`.

### 5.5 Core Lifecycle Integration (Registry + Scheduler + Memory)
The `src/core/` modules (`registry.zig`, `scheduler.zig`, `memory.zig`) and `src/foundation/pool_allocator.zig` realize the "Registry-Based Lifecycle" and "Explicit Memory Management" principles. `abi.scheduler`, `abi.memory`, and `abi.registry` are unconditionally exported from `src/root.zig`.

**Current state (as of 2026-06):** Real usage exists and is exercised on key surfaces:
- Scheduler drives actual high-priority training work in `abi agent train` (TrainTask submission + `runAll`, with Arena-wrapped contexts).
- Scheduler-backed completion is exposed through `completeWithScheduler()` and used by CLI/MCP completion paths while preserving direct completion APIs.
- Live stats and cooperative refresh tasks in the CLI/TUI dashboard, with terminal input polling and flicker-free redraw helpers mirrored in the TUI stub.
- Long-lived Scheduler instance owned by the MCP server with dedicated `scheduler_stats` / `scheduler_info` tools in the static, contract-tested MCP descriptor list.
- `MemoryTracker` + `TrackingAllocator` attached via `setMemoryTracker` in the training path and dashboard; allocations performed under scheduler tasks are recorded.
- Cross-feature observability wiring: Scheduler conditionally records task lifecycle metrics when `-Dfeat-metrics` is enabled. The default-on `feat-telemetry` feature (`src/features/telemetry/`) provides allocation-free `record(name)` / `increment(name, delta)` hooks plus process-wide readback (`counterValue`, `totalEvents`, `distinctCounters`, `droppedEvents`, `reset`) through a fixed-capacity counter table. It complements the opt-in `metrics` registry, and mod/stub parity is preserved for disabled telemetry builds.
- Dedicated integration test ("scheduler drives training tasks") validates end-to-end submission, execution, stats, and memory tracking.

- Registry remains focused on plugin descriptors (RwLock-protected) today; it is the coordinator for the plugin execution seam (`plugin_manager` + `abi plugin run` / MCP `plugin_run`) but is not yet the broader cross-feature lifecycle registry envisioned originally.
- Deeper adoption opportunities remain: MemoryTracker in more stages of the AI pipeline and HNSW storage internals; making the Registry a more general component lifecycle coordinator; potential unification or clearer boundary between `core/memory` and `foundation/pool_allocator`. WDBX `Store.putVector` and `Store.search` now expose hot-path allocation activity through optional `MemoryTracker` instrumentation.

See `tasks/roadmap-next.md` (Streams 1-2), `tasks/todo.md` (Core scheduler + memory row marked Done), and `tasks/scheduler-memory-wireup.md` for the detailed surfaces and the integration sketches that were executed. All integration changes preserved mod/stub parity, used relative `.zig` imports inside `src/`, and passed the full `./build.sh check` + `check-parity` + feature-off contract matrix. The original "primary remaining gap" language has been retired because the observability + real-work scheduling vision has been substantially delivered.

---

## 6. Maintenance Strategy

1. **Source truth first**: Reconcile architecture prose against `build.zig`, `src/features/mod.zig`, `src/abi_cli/usage.zig`, `docs/contracts/public-api.md`, and contract tests before changing public surfaces.
2. **Mod/stub parity**: Any public feature API change must update both `mod.zig` and `stub.zig`, including disabled-feature semantics, then pass `zig build check-parity`. The parity checker covers top-level public declaration names for feature/plugin pairs, not full signature equivalence. The check gate also runs focused feature-off behavior smoke tests and feature-aware public contracts for every `-Dfeat-*` flag.
3. **Generated registry**: Do not edit `src/plugin_registry.zig`; update plugin manifests and regenerate through the build step. Required manifest fields are `name`, `version`, `description`, `target_feature`, and `entry_point`; `targetFeature` / `entryPoint` aliases are accepted. `entry_point` must be a safe relative `.zig` path that exists under the plugin directory. Contract coverage expects multiple bundled plugin fixtures to remain discoverable.
4. **Import boundaries**: Library modules under `src/` use relative imports with `.zig` extensions. MCP executable/handler files (`src/mcp/main.zig`, `src/mcp/handlers.zig`) may import the public `abi` package because `build.zig` wires that package explicitly; never do so from modules re-exported by `src/root.zig`.
5. **Verification gates**: For source changes run `./build.sh check`; for release/readiness changes run `./build.sh full-check` (`check` plus integration tests, benchmarks, and TUI smoke).
6. **External claims**: Do not reuse spreadsheet or analysis-document claims for AES/RBAC, Swift/Python/TensorFlow stacks, distributed sharding, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy use, or SQuAD/CodeSearchNet/GPT comparisons unless the claim is tied to a current repo test, source implementation, or benchmark artifact.
7. **Design doc hygiene**: When refreshing this file (or any architecture prose), treat the following as executable source of truth and diff against them before publishing updates:
   - `build.zig` (feature flag defaults and wiring)
   - `src/features/mod.zig` (mod/stub selection + re-exports)
   - `src/abi_cli/usage.zig` (frozen CLI command surface)
   - `src/root.zig` (public namespace exports)
   - `docs/contracts/public-api.md`
   - `tests/contracts/*.zig` (surface, mcp_tools, feature_modules, plugin_registry, public_docs)
   - Actual `git ls-files` layout under `src/`, `tests/`, `tools/`, and `plugins/`
   Always call out (or correct) any intentional exceptions. The design is descriptive history + guardrails, not the contract.
