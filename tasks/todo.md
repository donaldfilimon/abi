# TODO — ABI Framework

Status tracking for incomplete work items. Reference: `docs/spec/abi-refactor-design.md`

North-star vision (long-horizon direction, current-vs-proposed mapping): `docs/spec/wdbx-north-star.md`. Near-term Phase 1 work (durable WAL/segment storage, recovery tests, persona-weighted scoring, loopback REST transport) is tracked there; only **Current**/**Partial** rows are repo-backed.

### North-Star Phase 1 — landed this pass (all `./build.sh check` green, parity OK)

| Item | Status | Notes |
| ---- | ------ | ----- |
| WDBX write-ahead log + recovery | ✅ Done | `src/features/wdbx/wal.zig`: CRC32-framed append-only records for kv, blocks, temporal nodes, and causal edges; deterministic replay (reuses `persistence.deserialize`); flipped-byte + bad-header rejection. `src/features/wdbx/recovery.zig` selects WAL-ahead state over stale checkpoints. CLI runtime commands now recover before `block get`, `query`, and the next `block insert`, while `db verify` still surfaces divergence. |
| Multi-segment storage + epoch reclamation | ✅ Done (default runtime checkpoint) | `src/features/wdbx/segments.zig`: manifest-backed immutable segment checkpoints, monotonic epochs, `loadLatest`, active epoch listing, reset, and watermark reclamation. `src/features/wdbx/recovery.zig` prefers segment checkpoints over legacy snapshots, and `abi wdbx db/block/query` now writes/opens through the segment checkpoint path while keeping the monolithic snapshot as a compatibility mirror. |
| Temporal/causal graph + hybrid ranker | ✅ Done (persisted + default MCP path) | `src/features/wdbx/temporal.zig`: recency half-life decay, causal BFS hop weight, `semantic × temporal × causal × persona` ranker. `src/features/wdbx/retrieval.zig` composes HNSW semantic search with the hybrid ranker, JSONL snapshots persist temporal nodes/edges, and MCP `wdbx_query` now returns hybrid-ranked local matches with component scores. |
| `wdbx` CLI namespace | ✅ Done | `src/abi_cli/handlers/wdbx.zig`: `db init/verify`, `block insert/get`, `query`, `benchmark`, `cluster status/demo`, `compute info`, `secure demo`, `gpu info`, `api serve`. Frozen-CLI contract = 11 commands (`tests/contracts/surface.zig`). `db verify` cross-checks WAL replay vs the current checkpoint; runtime commands recover WAL-ahead state. Comptime-gated on `feat_wdbx`. |
| Mod/stub parity for new wdbx modules | ✅ Done | `wal`, `temporal`, `recovery`, `retrieval`, `segments`, `cluster`, `cluster_rpc`, `compression`, `neural_compress`, `crypto_he`, `fhe`, `compute`, `rest` exported from `mod.zig`; matching empty parity markers in `stub.zig`; `zig build check-parity` green with `-Dfeat-wdbx=false`. |
| In-process cluster consensus (demo) | ✅ Done (in-process) | `src/features/wdbx/cluster.zig`: Raft-style leader election, majority-quorum replication, leader failover, quorum-loss detection over an in-process node array; `abi wdbx cluster demo`. 4 named tests. **Not** networked/multi-host. |
| Compute backend selector | ✅ Done | `src/features/wdbx/compute.zig`: CPU (`scalar`/`avx2`/`avx512`/`neon`, host-detected) / GPU / NPU / TPU enumeration + dynamic selection, always degrading to the deterministic CPU SIMD path; `abi wdbx compute info`. 3 named tests. Native ANE/TPU/CUDA/Metal dispatch **not linked**. |
| Embedding compression (demo) | ✅ Done | `src/features/wdbx/compression.zig`: int8 scalar quantization round-trip (~4×, bounded error); `abi wdbx secure demo`. 3 named tests. **Not** a learned/entropy codec. |
| Additive homomorphic aggregation (demo) | ✅ Done | `src/features/wdbx/crypto_he.zig`: additive single-key homomorphism over GF(p) — ciphertext sums decrypt to plaintext sums; `abi wdbx secure demo`. 5 named tests. **Not** full (multiplicative) FHE. |
| Loopback REST listener | ✅ Done | `src/features/wdbx/rest.zig`: pure unit-tested `route(method, path, body)` core + 127.0.0.1 listener serving `POST /insert /query /verify`, `GET /health /stats`; `abi wdbx api serve [port]` (default 8081). 4 named tests. **Loopback only** — not hardened for external exposure. |

**Phase 1 single-node status:** the retrieval/persistence path now includes runtime WAL recovery, segment-backed checkpoint loading/writing, snapshot-persisted temporal/causal graph records, and hybrid-ranked MCP `wdbx_query` results. See `docs/spec/wdbx-north-star.md` §2/§8 for the Current/Partial/Proposed mapping.

### WDBX V18 cognitive-runtime pass — landed this session (all `./build.sh full-check` green; merged to `main`)

Advances the V18 success criteria beyond the Phase-1 demos above. **10 of 11 criteria are now implemented and tested in pure Zig 0.17;** the 11th (ANE execution) is intentionally out of scope under the "100% Zig" constraint (last row).

| Item | Status | Notes |
| ---- | ------ | ----- |
| Persistent long-term memory (crash-safe) | ✅ Done | `wal.zig` epoch-tagged delta WAL + `recovery.zig` epoch-gated MERGE (checkpoint + WAL delta on top, manifest commit as the barrier, no double-apply). `Store.putVector` is now WAL-logged; crash-recovery + corruption tests added. |
| Multi-persona memory routing/isolation | ✅ Done | `retrieval.hybridSearchScoped` returns ONLY a scoped persona's vectors (vs blended); `wdbx query <path> [text] [persona]`. |
| Benchmark P50/P95/P99 | ✅ Done | `benchmarks.zig` artifact schema bumped `abi-bench/v2`; `wdbx benchmark` CLI reports percentiles. |
| Distributed clustering (networked RPC) | ✅ Done (loopback-tested) | `cluster_rpc.zig`: real TCP RequestVote/AppendEntries over `std.Io.net`; election / replication / downed-node tests over 127.0.0.1. Multi-host = a routable bind address. |
| Neural compression | ✅ Done | `neural_compress.zig`: in-process-trained tanh autoencoder codec (hand-written backprop + SGD); reconstruction-error and compression-ratio tests. A real learned codec, reference-scoped (not SOTA). |
| Homomorphic encryption (add + multiply) | ✅ Done | `fhe.zig`: DGHV somewhat-homomorphic scheme over native `i1024`; homomorphic XOR + AND on encrypted bits, depth-2 circuit tests. Reference parameters / bounded depth — **not** security-audited. Supersedes the additive-only demo. |
| SIMD-accelerated search (host-matched width) | ✅ Done | `compute.dot` uses `std.simd.suggestVectorLength` (AVX2/AVX512/NEON) instead of a fixed vector. |
| Zero-copy memory access | ✅ Done | `Store.getVector` returns a borrowed slice aliasing the backing buffer; test proves pointer aliasing (no copy/alloc). |
| GPU-accelerated retrieval (CPU/GPU parity) | ✅ Done | Metal path + `gpu/vector_ops.zig` parity test of the active backend vs a scalar reference. |
| Apple Neural Engine | ◑ Detection only | `compute.aneHardwarePresent()` truthfully reports ANE presence on Apple-Silicon macOS; native dispatch is **not** linked. ANE *execution* requires CoreML/ObjC (not pure Zig) plus on-device profiling to verify residency — intentionally out of scope under "100% Zig" (user-accepted). |

**V18 status:** 10 of 11 success criteria implemented and tested in pure Zig; ANE execution is the sole remaining item and is out of scope by the 100% Zig constraint (a CoreML/ObjC bridge + on-device profiling would be required). **Still Proposed** beyond this: **native** CUDA/Vulkan/Metal-kernel/ANE/TPU compute dispatch, and hardening the REST listener for non-loopback exposure.

## Priority: HIGH

| Item | Status | Notes |
| ---- | ------ | ----- |
| WDBX HNSW index implementation | ✅ Done | SIMD cosine distance, concurrent insert with SpinLock |
| WDBX block chain with MVCC | ✅ Done | SHA-256 chained blocks, snapshot isolation |
| AI pipeline router (Abbey-Aviva-Abi) | ✅ Done | Sentiment analysis, adaptive weighting, profile routing |
| Constitution governance module | ✅ Done | 6-principle validation with scoring |
| LLM connectors (OpenAI, Anthropic, Discord, Grok/xAI) | ✅ Done | Deterministic local responses plus opt-in live HTTP methods requiring credentials/network; Grok/xAI is exported and covered by connector/MCP contracts |
| MCP JSON-RPC 2.0 server | ✅ Done | stdio transport, initialize/tools/ping/shutdown |
| AI streaming server (OpenAI SSE) | ✅ Done | SSE streaming + non-streaming JSON |
| Aviva & Abi profile implementations | ✅ Done | Creative/exploratory + concise/action-oriented |
| Zig 0.17 MCP/stdin hardening pass | ✅ Done | MCP stdio loop, routing case-folding, silent catch cleanup, feat_mobile build option |
| Zig 0.17 remaining scaffold pass | ✅ Done | MCP JSON safety, local AI semantics, streaming, stub parity, feature gates, local connector/shader/MLIR behavior |
| Zig 0.17 external-boundary pass | ✅ Done | Connector live-mode errors, native GPU/toolchain status APIs, safe OS execute allow-list, plugin manifest validation |
| Zig 0.17 live-surface/build-gate pass | ✅ Done | Opt-in live HTTP connector methods, escaped request body builders, CLI/MCP builds and connector tests included in `check` |
| Zig 0.17 dirty-checkout gate recovery | ✅ Done | Scheduler/registry/WDBX/OS ownership fixes; `check` and `full-check` green |
| Twilio voice AI support connector | ✅ Done | Local ConversationRelay simulator, escalation payload contracts, Twilio credentials, and CLI simulation surface |
| Zig 0.17 ABI modernization and expansion | ✅ Done | MCP std.Io.net migration, TUI dashboard wiring/stub parity, HNSW locking, GPU fallback safety, walkthrough and AI guidance docs refreshed, checks green |
| GPU/WDBX/model completion expansion | ✅ Done | Backend capability reporting, WDBX stats/manifest APIs, local completion APIs, CLI/MCP completion surfaces verified |
| Codebase readiness/build/docs pass | ✅ Done | Manifest-driven plugin registry, plugin manager module coverage, full-check integration+benchmark gate, TUI scheduler snapshot, docs refreshed |
| Full module/GPU contract completion pass | ✅ Done | Root/feature namespace contracts, GPU real/stub fallback tests, all feature-off `test-feature-contracts -Dfeat-*=false` smoke checks, connector hardening, MCP/plugin contract expansion |
| Core scheduler + memory lifecycle integration | ✅ Done | `scheduler.stats()` + live data in CLI dashboard/TUI + agent train. MCP owns long-lived Scheduler + new tools (scheduler_stats etc). `abi scheduler status` reports a one-shot CLI scheduler probe with task/memory counters. Real training tasks now submitted via scheduler (see integration test + agent handler). Phase 1+2 complete. |

## Priority: MEDIUM

| Item | Status | Notes |
| ---- | ------ | ----- |
| Core scheduler implementation | ✅ Done | Task queue, priority handling, execution tracking, O(log N) heap dispatch (see HIGH "lifecycle integration" row for dashboard observability progress) |
| Core memory management | ✅ Done | Allocation tracking, pool basics |
| Framework config expansion | ✅ Done | Feature toggles, paths, limits |
| Foundation logger | ✅ Done | Structured logger with levels |
| Foundation utils | ✅ Done | String helpers, path manipulation |
| Foundation errors | ✅ Done | Centralized error types |
| OS controller commands | ✅ Done | File ops, process info, platform detection |
| Foundation IO module | ✅ Done | Async read/write, buffered reader/writer, file stream, path utils |
| Plugin manager | ✅ Done | Load/unload/list + real `run` dispatch for bundled example plugins (actual mod.zig run() invoked via AOT import; CLI `abi plugin run` + MCP `plugin_run` now execute real plugin code) |
| Integration test suite | ✅ Done | 9 tests covering WDBX, AI routing, constitution, connectors, MCP |
| Benchmark suite | ✅ Done | 7 benchmarks (vector ops, HNSW, chain, routing, constitution) |
| Test helpers module | ✅ Done | TestAllocator, TempDir/TempFile, mocks, assertions |

## Priority: LOW

| Item | Status | Notes |
| ---- | ------ | ----- |
| GPU backend expansion | ✅ Done | Added webgpu, opengl, webgl2 variants to mod + stub |
| Accelerator backend expansion | ✅ Done | Expanded to match GPU backend variants |
| Foundation IO optimization | ✅ Done | Async IO layer with buffered reader/writer |
| Plugin registry enhancements | ✅ Done | PluginManager with manifest validation, load/unload/list |
| Cross-compilation CI | ✅ Done | GitHub Actions native checks plus Linux/macOS cross-compile smoke builds |
| GPU backend stubs completion | ✅ Done | Metal framework linked on macOS with Objective-C runtime initialization path; vector operations fall back safely when native kernels are unavailable |
| Mobile mod/stub pair | ✅ Done | feat-mobile mod.zig + stub.zig created; runtime profile/mode artifacts, layout validation, disabled-stub parity, and feature-on/off contract coverage are verified |
| Twilio live transport | ✅ Done | httpPostForm helper, ConversationRelayEventLive with Basic auth, TwiML builder, configurable escalation |
| Modernization follow-up contract and release-note pass | ✅ Done | Added plugin validator/WDBX edge coverage, disabled WDBX manifest shape parity, and CHANGELOG release-note highlights |

## Known Test Failures (Pre-existing)

- None currently reproduced; `zig build test-integration` passes locally.

## Current Goals & Roadmap Focus (abi-goals-roadmap, post-audit)

See `tasks/roadmap-next.md` for the full refreshed view. High-level priorities:

**Stabilization**
- Tests for real scheduler usage in training: ✅ added ("scheduler drives training tasks" integration test + agent handler now submits real TrainTask via sched).
- Live transport integration tests: ✅ Done (LoopbackHttpServer + httpPostJson/Form round-trip tests).
- MemoryTracker in WDBX hot paths (putVector, search): ✅ Done (non-fallible trackAllocNoTag/trackFreeNoTag + integration test). Tagged allocations now use live-record semantics: `trackFree` removes the newest matching pointer record, `trackResize` updates pointer/size accounting, and `TrackingAllocator` keeps current usage/record count live across free/remap/resize paths.
- Pool allocator adoption in WDBX: ✅ Done for padded vector buffers and small spatial payload copies through `Store.initWithConfig`.
- TUI interactivity: ✅ Done (InteractiveTerminal in tui/mod.zig owns raw mode + key reading; dashboard.zig simplified).

**Integration & Architecture**
- AI scheduler/memory integration: ✅ Advanced (`submitCompletionTask`/`submitTrainingTask`/`submitAgentTask`, CLI/MCP completion wiring, agent plan/train helper wiring, scheduled CLI task result ownership; broader AI internals remain future work).
- GPU vector-op integration for WDBX search: ✅ Advanced (HNSW pairwise distance and neighbor-expansion batch scoring route through `gpu.vectorOps()` with SIMD fallback); native/batched backend expansion beyond local vector ops remains future work.
- Accelerator selection report: ✅ Improved (`selectionReport` exposes workload, selected/fallback backend, native availability, and GPU availability/acceleration flags; CLI `abi backends` reports it).
- Shader validation artifact: ✅ Improved (`validateDetailed` reports language, detected entry point, source bytes, checksum; delimiter-balance validation now shared by real/stub shader modules).
- MLIR textual lowering artifact: ✅ Improved (`analyze` validates module symbols, reports operation count/checksum; `lower` escapes operation attributes; real/stub surfaces match).
- Mobile runtime profile artifact: ✅ Improved (`profile` reports runtime mode, screen profile, hardware label, status message, and explicit native-dispatch status; disabled stub mirrors the profile contract).
- Plugin execution: **Real dispatch implemented** for bundled plugins.

**MCP / Observability / UX**
- MCP tool expansion: ✅ Advanced (contract-tested `connector_test`, `plugin_list`, `scheduler_info` alias plus existing scheduler/GPU/WDBX/plugin execution tools).
- Interactive TUI event loop: ✅ Done (InteractiveTerminal + dashboard loop integration).

**Docs**
- Continuous reconciliation of roadmap/todo/design docs against source after changes.

Zig 0.17 classic syntax work is largely complete. Focus is now architectural integration and test coverage for new functionality.

## Session Summary (redesign/refactor + organization pass — June 2026)

Multi-agent pass against `main` (work coordinated across disjoint slices; `./build.sh full-check` green throughout). Toolchain forward-ported to **Zig `0.17.0-dev.813`** (`.zigversion` bumped; `std.meta.fields`→`std.enums.values`, `@typeInfo(enum).fields`→`@hasField`, `error_set.?` removed — see `~/.claude` memory `zig-017-dev-gotchas`).

**Completions (functionality that did nothing now works):**
- **telemetry** — was an inert on-by-default feature (`record`/`increment` no-ops, 0 callsites). Now a real process-wide, lock-guarded, allocation-free counter sink with `counterValue`/`totalEvents`/`distinctCounters`/`droppedEvents`/`reset`; stub mirrors for parity. (`src/features/telemetry/`)
- **metrics.snapshotGauges** — was a placeholder returning empty while `setGauge` populated the map (gauge data unreadable). Implemented to mirror `snapshotCounters`; added `getGauge`; fixed a key-string leak in `deinit`. (`src/features/metrics/`)
- **live TUI** — dashboard loop blocked on `readKey` so it never auto-refreshed. Added `InteractiveTerminal.pollInput` (poll(2)); loop now redraws on a ~1s timer, responds to `q`/`r` instantly, flicker-free redraw (`homeScreen`+`clearToEnd`). (`src/features/tui/`, `src/abi_cli/handlers/dashboard.zig`)

**Organization / dead-code:**
- **MCP** `handlers.zig` decomposed: `connector_tools.zig`, `plugin_tools.zig`, `state.zig`, `ai_tools.zig`, rpc/shutdown split (584L → slim dispatch facade).
- **WDBX** shared type extraction + disabled-stub-module organization; **CLI** `wdbx` handler responsibilities split.
- **AI** disabled stub modules organized; public type definitions extracted.
- **connectors** inline tests extracted to `src/connectors/tests.zig` (`mod.zig` 634L → 23L, re-export surface intact).
- **dead code removed**: `OSController.execute` + its `Command` enum + `Registry.getOSController` + the `SystemInfo.total_memory_mb` placeholder (zero callers). (`src/foundation/os.zig`, `src/core/registry.zig`)

**Verification:** `./build.sh full-check` (check + integration + benchmarks + TUI smoke) green; `check-parity` green; legacy-Zig-pattern sweep of `connectors/foundation/core/abi_cli/plugins` came back clean (already idiomatic).

## Things To Do Next

**Honest stubs — keep disclosed, do NOT fake-complete** (would violate `docs/contracts/external-claims-audit.md`):
- `accelerator`, `shaders`, `mlir`, and `mobile` now provide stronger local selection/validation/lowering/profile artifacts but still disclose that native accelerator/platform dispatch and external compiler/toolchains are not linked. Leave as-is unless wiring real native dispatch/toolchains.

**Still Proposed (in-process demos exist; production forms do not):**
- Networked RPC transport for the cluster core (multi-host); native ANE/TPU/CUDA/Metal compute dispatch; learned/entropy compression codec; full multiplicative FHE; non-loopback REST hardening.

**Remaining real work (disjoint, candidate next slices):**
- Deeper scheduler/memory wiring into lower-level AI internals; broader native backend acceleration beyond the current local batched WDBX vector-op path.

## Status Format

- `✅ Done` — Implemented and passing tests
- `🟡 In progress` — Work started, not yet complete
- `⚪ Not started` — Not yet begun
- `🔴 Blocked` — Waiting on dependency

## Session Summary (this continuation)
- tools/build/ orphan fully removed.
- feat-hash + feat-metrics fully wired (build, features/mod, contracts, stub checks, parity).
- Scheduler lifecycle: real training dispatch + stats + MCP long-lived + dashboard live ticks + dedicated integration test.
- Plugin system: real run() dispatch implemented (calls actual plugin code for examples); CLI/MCP surfaces execute it.
- Grok/xAI connector: exported through `src/connectors/mod.zig`, deterministic local chat/stream paths, explicit live-transport methods, config validation, and MCP `connector_test` coverage.
- All example plugins made valid self-contained .zig modules.
- **MemoryTracker in WDBX hot paths**: Added non-fallible trackAllocNoTag/trackFreeNoTag to MemoryTracker; wired into Store.putVector and Store.search; integration test added.
- **MemoryTracker live records**: `trackFree` removes matching live records, `trackResize` updates pointer/size accounting, and TrackingAllocator free/resize/remap hooks keep current usage live instead of cumulative-only.
- **GPU vector-op integration for HNSW**: HNSW cosine distance now routes through `gpu.vectorOps().cosineSimilarity()` and falls back to the existing SIMD path if the GPU/vector abstraction errors; focused test coverage added.
- **Live transport tests (Stream 5)**: Added LoopbackHttpServer to test_helpers.zig; httpPostJson and httpPostForm round-trip integration tests against loopback.
- **Loopback live transport hardening**: Removed nanosleep retry loops from httpPostJson/Form integration tests; listener bind is now treated as readiness, failures unblock/join the server thread and report the original client error.
- **Loopback live transport deterministic cleanup**: `LoopbackHttpExchange` now captures server-thread request/error state, and JSON/form live transport tests use unconditional wake/join cleanup with no retry masking after listener readiness.
- **TUI InteractiveTerminal**: Moved raw mode + key reading into tui/mod.zig; dashboard.zig simplified to use it; stub parity maintained.
- **AI scheduler helper surface**: Added `CompletionTaskContext`, `TrainingTaskContext`, `AgentTaskContext`, `submitCompletionTask`, `submitTrainingTask`, and `submitAgentTask`; wired CLI/MCP completion plus `abi agent plan/train` through the helpers with feature-disabled fallback. `abi agent train` now reports the scheduled task results directly instead of re-running training, and deinitializes the owned results.
- **Scheduler CLI status**: Added `abi scheduler status` as a contract-smoked one-shot scheduler probe that reports task counters plus attached `MemoryTracker` state without claiming a long-running CLI daemon.
- **Accelerator selection artifact**: Added `SelectionReport`/`selectionReport`/`workloadName`; `abi backends` now prints selected/fallback backend plus native/GPU availability flags. Stubs mirror disabled CPU fallback reporting.
- **Shader validation artifact**: Added `ValidationReport`/`validateDetailed`; compile artifacts now use detected entry points and checksum/source-byte metadata, while stubs mirror validation before returning disabled artifacts.
- **MLIR textual lowering artifact**: Added `ModuleAnalysis`/`analyze`; textual lowering now validates module symbols, emits operation count/checksum metadata, and escapes operation names in attributes; stubs mirror validation before disabled output.
- **Mobile runtime profile artifact**: Added `RuntimeMode`/`MobileProfile`/`profile`/`runtimeModeName`; real and stub modules report mode/screen/hardware/status plus explicit `native_dispatch=false` instead of implying platform dispatch.
- **WDBX pool allocator adoption**: `StoreConfig.pool_alloc` now feeds both padded vector/search buffers and small `SpatialIndex3D` payload copies, with heap fallback for oversized payloads; stub and contract surface updated.
- **ArrayList cleanup**: Removed the final live `std.ArrayList(...)` use from `src/testing/test_helpers.zig`; `rg "std\\.ArrayList\\("` now returns no source hits.
- **WDBX runtime recovery**: `abi wdbx block get`, `query`, and subsequent `block insert` now open through `wdbx.recovery.open`, so a WAL-ahead checkpoint is recovered for normal runtime reads/writes; `db verify` still reports the divergence until a write checkpoints it.
- **WDBX temporal WAL records**: `wal.appendTemporalNode` and `wal.appendTemporalEdge` persist temporal/causal graph records through the same framed WAL replay path used by snapshots.
- **WDBX batched vector scoring**: HNSW neighbor expansion now uses `gpu.VectorOps.batchCosineSimilarity()` through a distance helper, retaining deterministic SIMD fallback and avoiding native-dispatch claims.
- **Docs reconciliation**: roadmap-next.md and todo.md reflect all new closures.
- **MemoryTracker roadmap reconciliation**: `tasks/scheduler-memory-wireup.md` now matches current source: `trackFree` is pointer-aware, `trackResize` updates live records, WDBX hot paths use no-tag aggregate tracking, and dashboard/MCP/AI scheduler consumers are no longer listed as missing.
- `./build.sh check` + parity green after WDBX pool/spatial payload expansion.
