# TODO — ABI Framework

Status tracking for incomplete work items. Reference: `docs/spec/abi-refactor-design.md`

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
- Live transport integration tests (still open).
- **New (approved plan)**: Deeper observability pass — Phase 1 + Phase 3 hygiene complete. Phase 2 HNSW in progress (multi-agents). **Current focus @src/abi_cli**: Wire MemoryTracker into production CLI handlers (agent train + dashboard) to complete real-user observability. Gate restored green.

**Integration & Architecture**
- Extend scheduler/memory into more AI paths + MemoryTracker in hot paths.
- GPU acceleration of HNSW/WDBX (beyond reporting).
- Plugin execution: **Real dispatch implemented** for bundled plugins (example-*/mod.zig run() actually invoked from PluginManager.run + CLI/MCP surfaces). The metadata-only gap is closed. (Stubs also made compilable.)

**MCP / Observability / UX**
- Further MCP tool expansion now that dynamic infrastructure exists.
- TUI interactive event loop.

**Docs**
- Continuous reconciliation of roadmap/todo/design docs against source after changes.

Zig 0.17 classic syntax work is largely complete. Focus is now architectural integration and test coverage for new functionality.

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
- All docs (todo, roadmap) reconciled.
- `./build.sh check` + parity green post-changes (356 tests).
