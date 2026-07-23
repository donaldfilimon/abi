# Changelog

All notable ABI Framework changes are recorded here. The executable gates remain the source of truth for readiness: `./build.sh check`, `zig build check-parity`, and `./build.sh full-check`.

## Unreleased

### Added

- feat(gpu): Metal elementwise `add_kernel` wired through `compute_api.map(.add)` / `reduce(.add)` with CPU fallback and parity tests — still not a general GPU speedup / CUDA / ANE claim.
- feat(gpu): bounded Metal softmax (`softmax_kernel` + `softmax_norm_kernel`) with host-side max/partition-sum, CPU fallback, and CPU/GPU parity tests — demo-grade; CUDA/Vulkan/ANE remain non-goals.
- docs(ops): claim-honest non-loopback REST/MCP HTTP threat review (`docs/spec/non-loopback-rest-threat-review.mdx`) — proxy TLS preferred, native TLS deferred, not a hardened-expose claim.
- docs(ops): cluster mTLS/membership ops guidance (`docs/spec/cluster-mtls-ops.mdx`) — proxy mTLS preferred; dynamic membership and sharding stay Proposed.
- docs(plan): Phase D cutover HITL checklist (`docs/spec/phase-d-cutover-plan.mdx`) — scaffold is not cutover.

### Changed

- fix(auth): `abi auth status` discloses non-macOS `ABI_CREDENTIALS_BACKEND=keychain` as unsupported (Windows/Linux Proposed) instead of implying an active macOS keychain; Backend line prints before credential load so `KeychainUnsupported` no longer hides the disclosure.
- docs(claims): sync north-star / external-claims / README / public-api / claim-boundaries with demo-grade Metal softmax alongside fused cosine/dot/L2 + multi-pass reduce.
- perf(gpu): Metal dispatch CQ — shared `MetalDispatch` helper; map+reduce and fused cosine chain multi-pass `reduce_sum` in one command buffer (no host re-upload thrash); Metal failures log then explicit CPU fallback. Broader kernels / CUDA / ANE remain Proposed.
- refactor(wdbx): dedupe REST missing-token vs wrong-bearer 401 insert fixture (`expectUnauthorizedInsert`).
- perf(gpu): Metal multi-pass `reduce_sum_kernel` — loop 256-wide threadgroup partials until one scalar (`runReduceSum`); host `sumF32` remains the CPU / failure fallback. Broader kernels / CUDA / ANE remain Proposed.
- docs(claims): sync north-star / external-claims / README / public-api with Metal fused cosine/dot/L2 + multi-pass reduce; list demo rANS/order-1 (`ans.zig`) beside Huffman — still not CUDA/Vulkan/ANE, SOTA compression, or GPU speedup claims.

### Fixed

- fix(gpu): remove redundant Metal buffer releases in `runSoftmax` error paths that double-freed buffers already covered by `defer`.
- test(wdbx): REST wrong-bearer path returns 401 + `WWW-Authenticate` (parity with MCP HTTP); assert challenge header on missing-token too.
- fix(tui): `/diff --stat` never worked from the live REPL line (`runDiff` re-parsed a hardcoded `"/diff "` literal); `/open` leaked the transient read buffer on every successful load; malformed `--soul-alpha` was silently coerced to 0.5 instead of usage + exit 2 per the CLI numeric-arg contract.

### Changed

- refactor: leaf-extraction wave across connectors/core/CLI/MCP — SSE streaming types + parsing from `connectors/http.zig` into `sse_stream.zig` (314); os_control policy types/validation from `os_control/mod.zig` into `policy.zig` (181); `Task` types from `core/scheduler.zig` into `core/task.zig`; completion handlers from `cli/handlers/train.zig` into `complete_handlers.zig` (287); connector test monolith `connectors/tests.zig` (651) split into per-connector `*_tests.zig` leaves (anthropic/discord/grok/http/json/openai/twilio); MCP HTTP request parsing into `mcp/http_parse.zig` (118). Public APIs unchanged; parity green.
- refactor(tui): final REPL organization split — `repl.zig` 1050→538 as a thin dispatch hub; raw/line input loops, key decode, redraw, tab completion, and Ctrl-R reverse search extracted to `repl_io.zig` (248); `completePrompt`, stream callback contexts, and `resolveFileMentions` extracted to `repl_complete.zig` (292); duplicated `runSyncClis` consolidated in `repl_git_commands.zig`. Public TUI API unchanged (parity green). `docs/spec/abi-refactor-design.mdx` §2 synced with the current module tree.

### Added

- feat(tui): `/pane` toggle for a post-turn split view (chat left, `git diff --stat` or open-file summary right) in `repl_pane.zig`; reuses `dashboard_render` visible-column fit helpers; stays unsplit below 80 cols or when terminal width is unknown; layout math unit-tested.
- test(tui,cli): `runOpen` packed-context round-trip/budget/leak coverage, `--soul`/`--soul-alpha` usage-contract tests (+ `wiring.parseSoulAlpha`), and an `agent os dry-run` handler success-path test.
- docs(skills): complete the 8 referenced-but-empty skills (opencode, ai-plan, gpu, mcp, sea, tui, wdbx, agent-status-reporter) with honest-stub framing; add tools/check_skills.sh frontmatter validator; remove 6 dead repo-root abi-superpower-* orphans.
- **TUI feature-parity improvements** — Ctrl-R reverse history search, Alt-Enter multi-line input, Ctrl-K/U/W/L Emacs keys, `/sessions` list command, `/clear` screen command, colorized `/diff` (green +/red -/cyan hunks), `/diff --stat`, unified file context budgets (32 KiB `/open`, 16 KiB `@file`), `estimateTokens()` helper.
+ **SoulLayout neural routing wired into CLI** — `src/features/ai/router.zig` adds `routeInputWithSoul()` (keyword-sentiment + 3-output neural softmax blend via `blend_alpha`), `blendWeights()`, and `src/cli/handlers/train.zig` adds `handleSoulComplete` path behind `abi complete --soul <file.json> [--soul-alpha <0.0-1.0>]`. `SoulLayout.fromJson` + `bootstrap` now exercised end-to-end; point_neural_net parity maintained in stub.
- **9 superpower skills from docs/specs** — `abi-superpower-agent-orchestration` (multi/spawn/browser local orchestration), `abi-superpower-constitution` (6-principle constitutional audit), `abi-superpower-wdbx-cluster` (Raft + RPC), `abi-superpower-wdbx-compute` (CPU/GPU/NPU/TPU selector), `abi-superpower-wdbx-secure` (compression + HE demos), `abi-claims-validator` (external-claims audit), `abi-wdbx-persistence` (WAL + segments + recovery), `abi-mcp-transport` (JSON-RPC stdio + HTTP/SSE), `abi-plugin-system` (manifest + registry). All in `.agents/skills/` (symlinked to `.opencode/skills/`). All include honest claim boundaries per `docs/contracts/external-claims-audit.mdx`. AGENTS.md/CLAUDE.md/GEMINI.md updated with skill references.
- **Priority A security + REPL (G1–G5)** — pure REPL line editor (`src/features/tui/line_editor.zig`) with CSI decode, cursor, history, and tab slash-completion wired into raw-mode `agent tui`; MCP/stdio shared JSON nesting depth bound (`protocol.MAX_JSON_DEPTH=32`); live connectors require `https://` base URLs; POSIX no-echo `auth signin`; `ai_train` dataset/artifact confinement under cwd or `ABI_TRAIN_DATA_ROOT` (symlink escape rejected); ambient durable-store parent dirs created/repaired as owner-only `0700` on POSIX. No new CLI commands or MCP tools.
- Local multi-agent orchestration: `src/features/ai/orchestration.zig` plus CLI `abi agent multi|spawn|browser` (scheduler workers, claim-honest browser plan with `embedded_browser=false`, no new top-level CLI or MCP tools). Contract smoke via `tools/contract_cli/agent_orchestration.sh`; `agent-plan-train` / `run-abi` skills exercise the paths.
- Docs contributing guide (`docs/contributing.mdx`) linked from the Mintlify hub; `abi-refactor-design.mdx` and multi-persona overview refreshed for orchestration.
- Codex coordinator agent (`.codex/agents/abi.toml`) and modern-refactor plan archival under `modern-refactor/examples/`.

- `src/foundation/temp_path.zig` — cross-platform `getTempDir()`/`tempFilePath()` helper; 30 hardcoded `/tmp/` references replaced across 13 foundation/feature/plugin/MCP files.

- SEA typed memory taxonomy: `src/features/sea/types.zig` with `MemoryKind` (9 variants: note, user_preference, project_decision, code_fact, tool_output, benchmark, constraint, contradiction, summary) + `Authority` (5 rungs: inferred, user_stated, tool_verified, file_verified, system_pinned) enums with parse/text/score round-trip tests.
- SEA multi-signal scorer: `src/features/sea/scorer.zig` with `SeaSignals` (8 orthogonal signals), `DEFAULT_SEA_WEIGHTS` (sums to 1.0), `seaScore()` weighted combiner (clamped to [0,1]), `adjustWeightsForTask()` for code_repair/project_recall/benchmark_review tuning, `selectSeaCandidates()` budgeted greedy selector with token/record-count/per-cluster budgets plus ≥0.92 high-score escape hatch, and `contextPack()` evidence renderer. 11 tests total.
- AI IoT stream monitoring: `src/features/ai/iot_monitor.zig` with z-score anomaly detector using Welford's online mean/variance algorithm, configurable threshold, history clear/reset, and 4 tests.
- AI multimodal fusion: `src/features/ai/multimodal_fusion.zig` with deterministic `VisionProcessor` (64-d), `AudioProcessor` (32-d), `IotProcessor` (16-d) text/telemetry embedders (character-bucket hashing + L2 normalization) and concatenative `fuse()` combinator. 5 tests.
- All four modules wired through `mod.zig` + `stub.zig` for both SEA (`src/features/sea/`) and AI (`src/features/ai/`) with full declaration-name parity; `zig build check-parity` passes.

- **Trainable `PointNeuralNetwork` + soul-prompt seeding** — `src/features/ai/point_neural_net.zig` (real trainable MLP: `ActivationFunction` relu/tanh/sigmoid/linear + derivatives, `Point.fromText`, `DenseLayer`, `ForwardTrace`, `PointNeuralNetwork` with `forwardWithTrace`/`backtraceTeach`/`train`/`optimizeTopology` (weight-pruning + neuron-pruning dead-neuron removal)/`save` JSON/`telemetry`, `NeuralTelemetry`, `SoulRecord`, `SoulLayout.fromJson`/`bootstrap`) and `src/features/ai/soul_layout.zig` (`SoulLayout.fromJson` parsing + `bootstrap` that persists each record into WDBX, trains the net ~100 epochs, and runs topology optimization with `NeuralTelemetry` reporting). Wired through `mod.zig`/`stub.zig`/`stub_types.zig` with full parity; `./build.sh check` 39/39 green. Honest status: real trainable net (not a disclosed stub); not yet invoked by `training.zig` (still metadata-only).

- **Per-turn `PipelineTelemetry` snapshot** — `src/features/ai/pipeline_telemetry.zig` with `PipelineTelemetry` (owned per-turn snapshot: `ethical_scores`, `escore`, `vetoed`, `NeuralTelemetryView`, provider/retrieval/cluster/governance/guardrail/`prompt_version` summaries, `p99_latency_ms`, `total_errors`, `responses_total`) and `ObservabilityHub` (`recordResponse`/`recordError`/`recordConstitutionBlock` fire-and-forget counters + `snapshot`/`finishTurn` mandatory tail folding the `AuditResult` ethics scores, `NeuralTelemetry`, and string summaries into one owned struct freed via `PipelineTelemetry.deinit`). Wired into `mod.zig`/`stub.zig` with full parity (inner structs renamed `PipelineTelemetryImpl`/`ObservabilityHubImpl` in the stub to avoid ambiguous references); 3 tests. `optimizeTopology` weight-pruning threshold corrected from an absolute (~0.002, which almost never triggered) to a relative magnitude (a fraction of the max absolute weight), so pruning deterministically fires on near-zero weights.

- **Discord gateway loop** — `src/connectors/discord_gateway.zig` with `GATEWAY_URL`/`API_BASE`/`DEFAULT_INTENTS` (`(1<<9)|(1<<12)|(1<<15)`)/`MAX_MESSAGE_CONTENT_BYTES` (1900) constants, `DiscordCommand` enum, `GatewayConfig`/`GatewayStats`/`Gateway.run` (Hello→Identify→heartbeat→MESSAGE_CREATE op 10/2/1/0), `parseDiscordCommand`/`routeDiscordMessage`/`truncate` (prefix-scoped, bot-safe, token-safe), JSON helpers (`parseOp`/`parseSeq`/`parseHeartbeatInterval`/`extractMessageCreate`/`buildIdentify`/`buildHeartbeat`), `FakeTransport` (offline, owns its queued messages), `WebSocketClient` (real WS handshake + masked client frames; TLS not linked — deploy behind a TLS-terminating proxy), `WebSocketTransport`, and `connectAndRun`. Wired into `src/connectors/mod.zig`; 5 tests pass. No autonomous token-bearing POST fires without a live transport.

- **`PointNeuralNetwork` now invoked by the `train` path** — `src/features/ai/training.zig` `train` trains a real `[3,8,3]` `PointNeuralNetwork` autoencoder on an available dataset (jsonl/text/csv → `Point.fromText` records via new `training_support.datasetToPoints`) and persists weights to `artifact_dir/neural_net_<profile>.json`, reporting `model trained on N records`. Falls back to the prior metadata-only disclosure when no dataset is present. 2 new tests (`datasetToPoints` format coverage; `train` trains + persists). Closes the remaining "not yet wired into training.zig" slice noted in the WDBX extract §8.

### Changed

- `docs/spec/sea-design-extract.mdx` status from DESIGN REFERENCE / PROPOSED → PARTIAL — MemoryKind, Authority, 8-signal scorer with task-aware weight deltas, cluster-diversity selection, and context-pack now promoted to Current ABI Zig source. Mapping table updated.
- `docs/spec/wdbx-rust-capability-extract.mdx` status from DESIGN REFERENCE / PROPOSED → PARTIAL — IoT stream monitoring and multimodal fusion now promoted to Current ABI Zig source. Mapping table rows and "What ABI would gain" section updated.
- `docs/spec/wdbx-rust-capability-extract.mdx` status from PARTIAL → CURRENT — `PipelineTelemetry` per-turn snapshot (`src/features/ai/pipeline_telemetry.zig`) and the Discord gateway loop (`src/connectors/discord_gateway.zig`) now promoted to Current ABI Zig source; all five WDBX designs ported. §8 mapping rows (Governance E-score, Telemetry emission discipline, Discord gateway) and the "What ABI would gain" section updated to reflect zero remaining Proposed items.

- OpenCode project config: `opencode.json` at repo root, `.opencode/` directory with `config.json` pinned instructions + MCP servers, and `.opencode/skills/` symlinked to `.agents/skills/` for canonical skill discovery.
- Module-level `//!` doc comments to 7 source files: `src/root.zig`, `src/features/mod.zig`, `src/features/ai/mod.zig`, `src/features/ai/stub.zig`, `src/features/ai/models.zig`, `src/cli/usage.zig`, `src/mcp/handlers.zig`.
- Executable `pin.sh` drivers for `zig-pin` skill in both `.agents/skills/zig-pin/` and `.claude/skills/zig-pin/` — both were missing standalone shell drivers.
- `sync-clis/launch.sh` — canonical `.agents/skills/` to target CLI directories syncing.
- GPU honesty throughout: `vector_ops.zig` `executeKernel()` now returns `.cpu_fallback` always (removed `native_gpu`/`simulated_gpu` conditional); `runtime.zig` `defaultAcceleration()` also returns `.cpu_fallback`; `tests/contracts/feature_modules.zig` accepts `.cpu_fallback` alongside `.native_gpu`/`.simulated_gpu`. No false claims of native GPU dispatch.
- SEA adaptive completion: `completion.zig` gains `completeAdaptive()` and `completeWithStoreAdaptive()` routing through persisted `AdaptiveModulator` weights; `sea/learn_loop.zig` switches to the adaptive path; `ai/mod.zig`+`ai/stub.zig` export `completeWithStoreAdaptive` in parity.
- WDBX cluster RPC: `cluster_rpc.zig` adds `runAuthenticatedLoopbackRound()` — deterministic multi-node vote + append helper that verifies quorum and peer logs; test validates 4-node round.
- Connector validation sections to AGENTS.md (Discord/Twilio credential and payload validation rules, matching CLAUDE.md/GEMINI.md).
- GEMINI.md now has Zig 0.17 patterns (inline test/refAllDecls, naming conventions, @panic/unreachable), a Project Skills section listing all 24 skills, and an OpenCode Setup section.

### Changed

- XDG compliance: `credentials.zig` `getCredentialsPath()` now resolves via `ABI_CREDENTIALS_PATH` → `XDG_CONFIG_HOME/abi/` → `~/.abi/`; `durable_store.zig` `resolveConfig()` resolves via `ABI_WDBX_PATH` → `XDG_DATA_HOME/abi/wdbx` → `~/.abi/wdbx`.
- `AGENTS.md` compacted 88→75 lines; stale doc references updated in `abi-threat-model.md`, `docs/contracts/public-api.mdx`, `docs/spec/abi-refactor-design.mdx`.
- `CLAUDE.md` compacted 138→78 lines, `GEMINI.md` compacted 148→76 lines — removed per-command flag docs, AI subsystem internals, and full skill lists; matching AGENTS.md compact format. All three instruction files now share identical conventions sections.
- `walkthrough.md` fixed 3 stale `/tmp/abi-demo.wdbx.jsonl` paths → `./abi-demo.wdbx.jsonl`.
- Skill scripts: 6 stale `/tmp/<...>-build.log` hardcoded paths replaced with `mktemp` + `trap` cleanup in `scheduler-status/status.sh`, `dashboard-smoke/dashboard.sh`, `mcp-smoke/smoke.sh` (both `.agents/skills/` and `.claude/skills/`).
- `src/core/scheduler.zig` fixed `catch null` → `catch "unknown"` on error-message OOM fallback.
- `sync-clis/launch.sh` `REPO_ROOT` path corrected from `$SCRIPT_DIR/../..` → `$SCRIPT_DIR/../../..`.

- Fixed 18 agent shell scripts in `.agents/skills/` that referenced stale `.claude/skills/` paths — all paths point to `.agents/skills/` now.
- Fixed `complete-base/SKILL.md` model alias: `Codex-fable-5` → `claude-fable-5` (verified against `src/features/ai/models.zig`).
- Bumped `skill-loop-cli` from `0.2.3` → `0.3.3` in `.mcp.json`.
- Dashboard GPU status: `dashboard.zig` extracts `GpuSnapshot` struct, caches GPU detection once per render instead of re-calling inside the render loop; passes snapshot through call chain.
- `docs/superpowers/archive/plans/2026-07-01-mkdocs-to-mintlify.md` status updated from "PLAN ONLY" to "COMPLETED".
- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` synced — AGENTS.md now has Connector Validation, GEMINI.md now has Zig patterns + Skills + OpenCode sections.

### Removed

- Dead `PathConfig` struct removed from `src/core/config.zig` — 5 unused `/tmp/abi/*` default paths (`data_dir`/`cache_dir`/`log_dir`/`config_dir`/`plugin_dir`) and `paths` field on `Config` deleted; nothing read them.
- `applyVote`/`applyAppend`/`VoteReply`/`AppendReply` moved from `cluster_rpc.zig` → `cluster.zig` (Raft state application rejoined with state machine; `cluster_rpc.zig` re-exports through `cluster.`).

- Deleted stale `mcp/servers.json` (redundant duplicate of `.mcp.json`; no code references it).
- Deleted `.agents/skills/.plugins-synced-from-central` stale marker.
- Deleted dead `src/features/ai/plan.zig` (16-line noop: `parsePlan` returned `&.{}`, `formatPlanResponse` returned `"plan"`, neither called anywhere). Parity-synced `src/features/ai/mod.zig` (removed `pub const plan`/`PlanStep`/`parsePlan` re-exports) and `src/features/ai/stub.zig` (removed parity-matched `PlanStep`/`parsePlan`/`plan` block). `zig build check-parity` passes; no contract test referenced the removed names.
- Deleted stray `mutex_check.o` (3 MB object file in repo root, not in `build.zig.zon` `paths`, not referenced by `build.zig`); added `/mutex_check.o` to `.gitignore`.

- Deleted orphaned `tests/contracts/completion_persistence.zig` (referenced in earlier guard text but no longer imported by any test; removing it eliminates the `completion_persistence` symbol-set the prior `memory_record` guard asserted on, which the `kv=1, completion:<id> only` contract now makes unreachable).

### Changed (file splits + param bundling)

- Wave-2 file extractions (5 splits + 1 relocation):
  - `src/cli/dispatch.zig` → `suggest.zig`: edit-distance suggestion engine extracted (dispatch 473→341).
  - `src/cli/registry.zig` → `completion.zig` + `help_json.zig`: shell completion scripts and JSON help metadata extracted (registry 1,033→646).
  - `src/features/tui/mod.zig` → `dashboard.zig`: dashboard rendering and all pane-layout helpers extracted (mod 636→153); `mod.zig` is now a pure re-export hub for 5 submodules (repl/types/sanitize/terminal/dashboard).
  - `src/cli/handlers/dashboard.zig` → `dashboard_json.zig`: JSON serialization and pane-list writer extracted (dashboard 824→485).

- Split four large source files into focused siblings, preserving all public APIs:
  - `src/mcp/server.zig` → `stdio_transport.zig` + `http_transport.zig` (server.zig is now a thin re-export shim).
  - `src/features/wdbx/rest.zig` → `rest_parse.zig` (response / vector-field / header / bearer-token / read helpers) + `rest_handlers.zig` (`route` core) — `rest.zig` re-exports both for the existing public surface.
  - `src/features/nn/mod.zig` → `model.zig` (types, `Model`/`Scratch`/`Grads`, `forwardLoss`/`backward`/`evalLoss`/`forwardLossNoTarget`) + `train.zig` (`initModel`/`trainModel`/`trainOnText`/`trainOnJsonl`/`extractCorpusFromJsonl`) — `nn/mod.zig` now imports both and keeps the existing public API.
  - `src/features/tui/` adds `types.zig` (shared status / state / dashboard types), `sanitize.zig` (control-byte sanitizer), `terminal.zig` (raw-mode terminal); `mod.zig` and `stub.zig` re-export through them. `writeDashboard`/`writeDiagnostics` writers added to `mod.zig`.
- Param bundling: `src/features/wdbx/wal.zig` `appendBlock` now takes a `BlockRecord` struct (5-field bundle) instead of 5 trailing args; `src/cli/handlers/train.zig` `handleComplete` now takes a `CompleteOptions` struct (5-field bundle) instead of 5 trailing args. `cli/registry.zig` adapts.
- `src/cli/handlers/agent.zig` `handleAgent` split into `handleAgentPlan` / `handleAgentTrain` / `handleAgentTui` / `handleAgentOs` per-subcommand functions; dispatch reduced to a short `sub_cmd` switch.
- `src/cli/registry.zig` `scheduler` migrated from raw `(io, alloc, argv)` shim to the typed argument parser (`typedCmd("scheduler", &scheduler_args, schedulerHandler)`); `scheduler status` is the only valid subcommand, enforced both at parse time and in the handler.
- `src/features/tui/repl.zig` `syncclis` slash command no longer hardcodes an in-repo path; it resolves `~/.grok/skills/sync-clis/launch.sh` via `$HOME` and refuses cleanly if the launcher is missing.
- WDBX `cluster status`, `cluster demo`, and `secure demo` now print an explicit north-star status line (Partial / Phase 1-2 vs Proposed / Phase 2 per `docs/spec/wdbx-north-star.mdx` §2/§3.4-3.5).
- Instruction files: `AGENTS.md` rewritten (110 lines, every section kept in sync with the refactor), `CLAUDE.md` adds `test-plugins` and CI cross-smoke note, `GEMINI.md` adds the CI cross-smoke note. `docs/spec/abi-refactor-design.mdx` and archived `docs/superpowers/archive/specs/ABI-MASTER-SPEC.md` reconciled to the current tree.

### Re-pointed (skill telemetry)

- Re-anchored 8 abi-reviewer agent files (`.claude/agents/{ai-constitution-reviewer,compression-security-reviewer,instruction-sync,plugin-system-reviewer,sea-evidence-analyst,tui-navigation-guide,wdbx-explorer,zig-build-doctor}.md`) so bare `mod.zig`/`stub.zig`/`router.zig`/`constitution.zig`/`entropy.zig`/`neural_compress.zig`/`fhe.zig`/`evidence.zig`/`dashboard.zig`/`plugin.zig`/`plugin_manager.zig`/`std.mem.trimEnd` references are now qualified to their real paths. `npx skill-loop init` + `inspect` now report zero broken refs for these agents; previously 18 broken refs across `local` skills (16 of them ours).
- Model catalog local-offline provider alias routing: `models.zig` `providerOf()` now classifies `ollama-`/`ollama/`, `lmstudio/`, `llama-cpp/`/`llama/`, `vllm/`, `mlx-`/`mlx/` prefix ids as `.local` explicitly (instead of silent fallthrough), with inline regression tests asserting all 8 prefix forms route to `.local`, stay `!isKnown`, and pass through `canonical` unchanged. Deterministic offline, no cloud credentials. No `Provider` enum / `catalog` / `resolve` / `canonical` surface change; parity unaffected (`models.zig` is std-only, shared by `mod.zig` + `stub.zig`). Maps issue #647's stale `mcp/src/inference/engine/backends.zig` alias-consistency item to the real current path.
- `std.testing.refAllDecls(@This())` coverage added to all 32 plugin mod/stub files under `src/plugins/*/` and the 9 non-exempt build-covered modules previously lacking it (connector façades `connector`/`http`/`anthropic`/`discord`/`twilio`/`tests`, `src/foundation/pool_allocator.zig` + `validation.zig`, `src/mcp/server.zig`).

### Fixed

- MCP shutdown use-after-free: the `shutdown` JSON-RPC method tore down shared state in-band from whichever transport thread handled it, so `deinitScheduler()` could free the `Scheduler` while the peer transport was inside a tool call holding a `*Scheduler`. The stdio (`server.zig`) and HTTP (`rpc.zig`) handlers now only *signal* shutdown; `main` runs `deinitScheduler()`/`deinitWdbxStore()` via LIFO `defer`s after the HTTP thread is joined, so teardown can't race an in-flight call (deinit is idempotent; the WAL covers crash recovery).
- MCP shared-state lazy-init TOCTOU race: `ensureScheduler`/`ensureWdbxStore` flipped the `initialized` atomic to `true` *before* constructing the backing optional, so a concurrent reader (HTTP/SSE thread + stdio loop both call `getScheduler`/`getWdbxStore`) could observe `initialized==true` while the optional was still `null` and panic on `.?`. Switched to double-checked locking under dedicated init spinlocks that publish the flag with release ordering only after construction. No contract/API/parity change.
- WDBX write-ahead-log double-frees: `Store.putVector` and `Store.store` could double-free / dangle a buffer when a WAL append IO error followed the in-memory commit. The `errdefer` now stays the sole owner across the fallible append — `putVector` moves the append above the padded-buffer free; `store` disarms the owned-key/value `errdefer`s with commit flags — preserving the deliberate memory-first / WAL-after ordering. (Latent on the persistent path; no default-build test reproduced them.)
- WDBX remote-compute reference listener (`serveOnce`) now guards the untrusted `dim * 2` against `usize` overflow via `std.math.mul` before allocating, looping, and slicing.
- `zig build check-parity` now fails when a feature/plugin leaf ships `mod.zig` with no sibling `stub.zig` (previously a silent pass); only the intentional `src/features/mod.zig` dispatcher is exempt.
- SEA learn loop logs (rather than swallows) a router weight-save failure on the durable persistence path; the CLI dashboard handler `defer`s `MemoryTracker.deinit()` to match the train/agent handlers.
- Reconciled stale docs against source: threat-model `src/abi_cli/` → `src/cli/` paths after the CLI rename; corrected the apple-fm `@c`-shim wording in CLAUDE/AGENTS/GEMINI (the shim exists and is linked; it is not a nonexistent `@_cdecl` shim) and softened the unbacked "runtime-verified on Apple-Intelligence hardware" claim per the external-claims policy.

### Added

- Hardened credential-file persistence: `abi auth` now creates/repairs `~/.abi` as owner-only (`0700` on POSIX-capable targets) and opens/truncates `credentials.json` with owner-only file permissions (`0600`) before writing secret bytes, with regression coverage for existing permissive files.
- Redacted Discord/Twilio connector logs that previously emitted local Discord message content and Twilio live response bodies; logs now report IDs/status plus byte counts, with helper regression tests.
- Added optional bearer-token enforcement for the MCP loopback HTTP/SSE transport via `ABI_MCP_HTTP_TOKEN`, with HTTP-boundary tests for unauthorized and authorized requests. Stdio transport remains unchanged.
- Added optional bearer-token enforcement for the WDBX loopback REST transport via `ABI_WDBX_REST_TOKEN`, with HTTP-boundary tests for unauthorized and authorized requests.
- Added `wdbx.entropy`, an exact order-0 canonical Huffman codec for arbitrary byte payloads with no-expansion fallback, deterministic round-trip tests, and explicit claim boundaries below production/SOTA learned compression.
- SEA `runLearnLoop` gains an optional `LearnLoopConfig.tracker` that makes adaptive persona-router weight persistence observable through a `MemoryTracker` (balanced, non-escaping; default off → no call-site change).
- `runCli` behavioral tests covering help/no-args (exit 0) and unknown-command / missing-required-positional (exit 2) dispatch paths.

- Hardened the modernization contract suite around root/feature namespaces, CLI/MCP tools, generated plugin registry metadata, and feature-off stub behavior.
- Added `ABI_MCP_HTTP_PORT` support for moving the MCP loopback HTTP/SSE transport off the default `127.0.0.1:8080` port.
- Expanded generated plugin metadata to include `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point`; added a second bundled plugin fixture for multi-plugin registry contract coverage.
- Added manifest validation coverage for unsafe plugin entry points, missing entry files, camelCase manifest aliases, and nested safe entry files.
- Tightened AI/WDBX completion semantics so `CompletionRequest.store_result=false` leaves WDBX unchanged, while persisted completions record query/response vectors, metadata, and chain blocks.
- Expanded WDBX store manifest output with spatial record counts and disabled-stub manifest fields that preserve the real manifest shape; added contract coverage for ordered vector search, block metadata round-tripping, and snapshot lookup.
- Hardened connector boundaries: Discord validates token shape, numeric IDs, inbound author IDs, and message size; Twilio validates credential shape, base URL, timeout, explicit `.live` transport selection, ConversationRelay aliases/wrong-typed fields, TwiML XML escaping, and URL-encoded form payloads before local/live dispatch.
- Tightened disabled-feature stubs: AI mirrors empty-input/training/agent validation while preserving requested completion models, MLIR/shader stubs validate inputs before disabled artifacts, and WDBX nested writes report disabled behavior without recording phantom vectors or blocks.
- Added AI/WDBX edge coverage for empty completion input, disabled-WDBX training degradation, append-linked completion blocks, and WDBX block-chain ownership/tamper detection.
- Added an external-claims audit doc and public-doc contract test so Drive/investor collateral can reuse only repo-backed ABI/WDBX claims.
- Added WDBX JSONL snapshot integrity, CRC32-framed write-ahead log replay/corruption detection, temporal/causal ranking primitives, and a frozen `abi wdbx` CLI namespace.
- Added honest in-process WDBX roadmap demonstrations for local consensus, backend selection with CPU fallback, int8 embedding quantization, additive aggregation, and loopback REST.
- Added the default-on `telemetry` feature surface for lightweight event/counter hooks with mod/stub parity.

### Performance

- Removed redundant work from WDBX HNSW search, WAL append/replay, and block-chain hot paths (no behavioral change; covered by existing gates).

### Tests

- Added second-pass audit coverage (additive only, no production change): `segments.readManifest` rejects a corrupt manifest (bad header / non-numeric `next_epoch` / non-numeric active token) with `InvalidManifest` instead of silently dropping live segments; SEA `gatherEvidence` populated-recall + `staticProfileLabel` attribution paths; and the SEA persist→recall round-trip (a persisted turn recalled as evidence on a later related turn).

### Validation

- Use `./build.sh check` as the baseline gate for source changes.
- Use `zig build check-parity` after public feature API changes.
- Use `./build.sh full-check` for release/readiness checks (`check` + integration tests + benchmarks + dashboard smoke + `agent tui` line-mode smoke).
- Public docs intentionally avoid unproven external claims for distributed sharding, AES/RBAC, Swift/Python/TensorFlow implementations, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy use, and comparative model benchmarks.
