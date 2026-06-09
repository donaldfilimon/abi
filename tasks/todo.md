# TODO — ABI Framework

Status tracking for incomplete work items. Reference: `docs/spec/abi-refactor-design.md`

North-star vision (long-horizon direction, current-vs-proposed mapping): `docs/spec/wdbx-north-star.md`. Near-term Phase 1 work (durable WAL/segment storage, recovery tests, persona-weighted scoring, loopback REST transport) is tracked there; only **Current**/**Partial** rows are repo-backed.

### North-Star Phase 1 — landed this pass (all `./build.sh check` green, parity OK)

| Item | Status | Notes |
| ---- | ------ | ----- |
| WDBX write-ahead log + recovery | ✅ Done | `src/features/wdbx/wal.zig`: CRC32-framed append-only records, deterministic replay (reuses `persistence.deserialize`), flipped-byte + bad-header rejection. 3 unit tests. |
| Temporal/causal graph + hybrid ranker | ✅ Done | `src/features/wdbx/temporal.zig`: recency half-life decay, causal BFS hop weight, `semantic × temporal × causal × persona` ranker. 4 unit tests. |
| `wdbx` CLI namespace | ✅ Done | `src/abi_cli/handlers/wdbx.zig`: `db init/verify`, `block insert/get`, `query`, `benchmark`, `cluster status/demo`, `compute info`, `secure demo`, `gpu info`, `api serve`. Frozen-CLI contract = 11 commands (`tests/contracts/surface.zig`). `db verify` cross-checks WAL replay vs snapshot. Comptime-gated on `feat_wdbx`. |
| Mod/stub parity for new wdbx modules | ✅ Done | `wal`, `temporal`, `cluster`, `compression`, `crypto_he`, `compute`, `rest` exported from `mod.zig`; matching empty parity markers in `stub.zig`; `zig build check-parity` green with `-Dfeat-wdbx=false`. |
| In-process cluster consensus (demo) | ✅ Done (in-process) | `src/features/wdbx/cluster.zig`: Raft-style leader election, majority-quorum replication, leader failover, quorum-loss detection over an in-process node array; `abi wdbx cluster demo`. 4 named tests. **Not** networked/multi-host. |
| Compute backend selector | ✅ Done | `src/features/wdbx/compute.zig`: CPU (`scalar`/`avx2`/`avx512`/`neon`, host-detected) / GPU / NPU / TPU enumeration + dynamic selection, always degrading to the deterministic CPU SIMD path; `abi wdbx compute info`. 3 named tests. Native ANE/TPU/CUDA/Metal dispatch **not linked**. |
| Embedding compression (demo) | ✅ Done | `src/features/wdbx/compression.zig`: int8 scalar quantization round-trip (~4×, bounded error); `abi wdbx secure demo`. 3 named tests. **Not** a learned/entropy codec. |
| Additive homomorphic aggregation (demo) | ✅ Done | `src/features/wdbx/crypto_he.zig`: additive single-key homomorphism over GF(p) — ciphertext sums decrypt to plaintext sums; `abi wdbx secure demo`. 5 named tests. **Not** full (multiplicative) FHE. |
| Loopback REST listener | ✅ Done | `src/features/wdbx/rest.zig`: pure unit-tested `route(method, path, body)` core + 127.0.0.1 listener serving `POST /insert /query /verify`, `GET /health /stats`; `abi wdbx api serve [port]` (default 8081). 4 named tests. **Loopback only** — not hardened for external exposure. |

**Remaining Phase 1 (not yet done):** multi-segment storage + epoch reclamation, automatic startup recovery, and wiring the temporal/causal hybrid ranker into the default `wdbx_query` path + persisting the causal graph. **Still Proposed** — in-process demos exist (rows above), but the production/scaled forms do not, so do not present them as shipped: a **networked** RPC transport so the cluster core spans separate hosts, **native** ANE/TPU/CUDA/Metal compute dispatch, a **learned** compression codec, full **multiplicative** FHE, and hardening the REST listener for non-loopback exposure. See `docs/spec/wdbx-north-star.md` §2/§8 for the Current/Partial/Proposed mapping.

## Priority: HIGH

| Item | Status | Notes |
| ---- | ------ | ----- |
| WDBX HNSW index implementation | ✅ Done | SIMD cosine distance, concurrent insert with SpinLock |
| WDBX block chain with MVCC | ✅ Done | SHA-256 chained blocks, snapshot isolation |
| AI pipeline router (Abbey-Aviva-Abi) | ✅ Done | Sentiment analysis, adaptive weighting, profile routing |
| Constitution governance module | ✅ Done | 6-principle validation with scoring |
| LLM connectors (OpenAI, Anthropic, Discord) | ✅ Done | Deterministic local responses plus opt-in live HTTP methods requiring credentials/network |
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
| Core scheduler + memory lifecycle integration | ✅ Done | `scheduler.stats()` + live data in CLI dashboard/TUI + agent train. MCP owns long-lived Scheduler + new tools (scheduler_stats etc). Real training tasks now submitted via scheduler (see integration test + agent handler). Phase 1+2 complete. |

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
| Mobile mod/stub pair | ✅ Done | feat-mobile mod.zig + stub.zig created, parity verified, 4 tests |
| Twilio live transport | ✅ Done | httpPostForm helper, ConversationRelayEventLive with Basic auth, TwiML builder, configurable escalation |
| Modernization follow-up contract and release-note pass | ✅ Done | Added plugin validator/WDBX edge coverage, disabled WDBX manifest shape parity, and CHANGELOG release-note highlights |

## Known Test Failures (Pre-existing)

- None currently reproduced; `zig build test-integration` passes locally.

## Current Goals & Roadmap Focus (abi-goals-roadmap, post-audit)

See `tasks/roadmap-next.md` for the full refreshed view. High-level priorities:

**Stabilization**
- Tests for real scheduler usage in training: ✅ added ("scheduler drives training tasks" integration test + agent handler now submits real TrainTask via sched).
- Live transport integration tests: ✅ Done (LoopbackHttpServer + httpPostJson/Form round-trip tests).
- MemoryTracker in WDBX hot paths (putVector, search): ✅ Done (non-fallible trackAllocNoTag/trackFreeNoTag + integration test).
- Pool allocator adoption in WDBX: ✅ Done for padded vector buffers and small spatial payload copies through `Store.initWithConfig`.
- TUI interactivity: ✅ Done (InteractiveTerminal in tui/mod.zig owns raw mode + key reading; dashboard.zig simplified).

**Integration & Architecture**
- AI scheduler/memory integration: ✅ Advanced (`submitCompletionTask`/`submitTrainingTask`, CLI/MCP completion wiring, agent training helper wiring; broader AI internals remain future work).
- GPU vector-op integration for HNSW distance calculations: ✅ Done (routes through `gpu.vectorOps()` with SIMD fallback); broader native/batched WDBX acceleration remains future work.
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
- `accelerator`, `shaders`, `mlir`, `mobile` real `mod.zig` are validation/metadata/simulation only; each discloses its limitation. Leave as-is unless wiring real native dispatch.

**Still Proposed (in-process demos exist; production forms do not):**
- Networked RPC transport for the cluster core (multi-host); native ANE/TPU/CUDA/Metal compute dispatch; learned/entropy compression codec; full multiplicative FHE; non-loopback REST hardening.

**Remaining real work (disjoint, candidate next slices):**
- WDBX Phase 1 finish: multi-segment storage + epoch reclamation, automatic startup recovery, wire the temporal/causal hybrid ranker into the default `wdbx_query` path + persist the causal graph.
- `MemoryTracker.trackFree` ignores `ptr` (records are append-only; `getRecordCount` is cumulative, not live) — `src/core/memory.zig`. Decide: make it live, or rename to reflect cumulative semantics.
- Deeper scheduler/memory wiring into more AI-pipeline stages; broader native/batched GPU acceleration beyond HNSW distance.
- Loopback integration tests skip-and-leak on connection flake (`src/integration_tests.zig`) — fail hard or use a deterministic ready signal instead of nanosleep-retry + `detach`.

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
- All example plugins made valid self-contained .zig modules.
- **MemoryTracker in WDBX hot paths**: Added non-fallible trackAllocNoTag/trackFreeNoTag to MemoryTracker; wired into Store.putVector and Store.search; integration test added.
- **GPU vector-op integration for HNSW**: HNSW cosine distance now routes through `gpu.vectorOps().cosineSimilarity()` and falls back to the existing SIMD path if the GPU/vector abstraction errors; focused test coverage added.
- **Live transport tests (Stream 5)**: Added LoopbackHttpServer to test_helpers.zig; httpPostJson and httpPostForm round-trip integration tests against loopback.
- **TUI InteractiveTerminal**: Moved raw mode + key reading into tui/mod.zig; dashboard.zig simplified to use it; stub parity maintained.
- **AI scheduler helper surface**: Added `CompletionTaskContext`, `TrainingTaskContext`, `submitCompletionTask`, and `submitTrainingTask`; wired CLI/MCP completion and agent training through the helpers with feature-disabled fallback.
- **WDBX pool allocator adoption**: `StoreConfig.pool_alloc` now feeds both padded vector/search buffers and small `SpatialIndex3D` payload copies, with heap fallback for oversized payloads; stub and contract surface updated.
- **ArrayList cleanup**: Removed the final live `std.ArrayList(...)` use from `src/testing/test_helpers.zig`; `rg "std\\.ArrayList\\("` now returns no source hits.
- **Docs reconciliation**: roadmap-next.md and todo.md reflect all new closures.
- `./build.sh check` + parity green after WDBX pool/spatial payload expansion.
