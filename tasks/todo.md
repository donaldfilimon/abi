# TODO — ABI Framework

Forward-looking tracker for **incomplete and in-flight** work. Completed history is **not** kept here — it lives in `git log` and `CHANGELOG.md`. This file is the lightweight active board; long-horizon direction is `docs/spec/wdbx-north-star.mdx` (§2/§8 Current/Partial/Proposed mapping).

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: do **not** add "Session Summary" logs here — that is what git history and the CHANGELOG are for. When an item closes, delete its row (or move a one-line note to "Recently landed"), don't append a narrative.

---

## Open work

### Honest stubs — keep disclosed, do NOT fake-complete

These ship real local artifacts but truthfully disclose that native/external dispatch is not linked. "Completing" them with simulated capability would violate `docs/contracts/external-claims-audit.mdx`. Leave as-is unless wiring genuine native dispatch/toolchains.

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| `accelerator` backend dispatch | ◑ Selection report only | Native CUDA/Metal/NPU/TPU dispatch not linked; CPU SIMD fallback is the real path. |
| `shaders` validation | ◑ Validate + checksum only | No real shader compiler/toolchain linked. |
| `mlir` lowering | ◑ Textual analyze/lower only | No external MLIR/LLVM toolchain linked. |
| `mobile` runtime profile | ◑ Profile reporting only | `native_dispatch=false` reported explicitly; no platform runtime. |

### Still Proposed (in-process demos exist; production forms do not)

| Item | Status | Gap to production |
| ---- | ------ | ----------------- |
| Native compute dispatch | ⚪ Not started | ANE/TPU/CUDA/Metal-kernel execution (vs the current local SIMD/Metal-fallback path). ANE execution is **out of scope** under the 100% Zig constraint — see Non-goals. |
| Production/SOTA learned compression codec | ◑ Partial / disclosed | Exact order-0 Huffman entropy (`entropy.zig`), int8 embedding quantization, and the reference `neural_compress.zig` autoencoder exist; no ANS/arithmetic/context-model or production-scale learned codec is proven. |
| Security-audited FHE | ⚪ Not started | `fhe.zig` (DGHV; chained multiplicative depth 3 tested) is reference-parameter, bounded-depth, **not** audited. |
| Non-loopback REST hardening | ◑ Partial / disclosed | `rest.zig` remains 127.0.0.1-only and can require `Authorization: Bearer` via `ABI_WDBX_REST_TOKEN`; external exposure still needs TLS, rate limiting, authz, and threat review. |
| Multi-host cluster | ◑ Authenticated routable bind + local multi-node RPC loop / ops story missing | `cluster_rpc.zig` runs real TCP RequestVote/AppendEntries, includes an authenticated loopback multi-node vote+append round that verifies quorum and peer logs, and `cluster serve <port> [node] [host]` can bind a routable host only when `ABI_WDBX_CLUSTER_TOKEN` is set. `ABI_WDBX_CLUSTER_PEERS` can limit accepted node ids. Multi-host production still needs TLS/mTLS or a fronting network policy, deployment controls, dynamic membership, and sharding. |

### Feature-parity north-star (honest status vs source)

Target: feature parity with local inference runtimes (llama-cpp, MLX) and CLI/TUI tools (codex, claude-code) while staying 100% Zig on the 0.17 master branch. Status below is pinned to source — not marketing.

| Item | Status | Gap / Notes |
| ---- | ------ | ------------ |
| Local llama-cpp / OpenAI-compat inference bridge | ◑ Landed (HTTP client) | `src/connectors/local_bridge.zig` + `complete` path: prefix models (`llama-cpp/`, `ollama/`, `vllm/`, …) hit loopback HTTP with health-check fallback to in-process persona router. Env: `ABI_LLAMA_CPP_ENDPOINT` (default `127.0.0.1:8080`). Not an embedded ggml runtime. Remaining: broader SSE/stream parity when the local server streams tokens. |
| MLX bridge / on-device FM | ◑ Partial | MLX **HTTP bridge** via `mlx/`/`mlx-` prefixes + `ABI_MLX_ENDPOINT` (default `127.0.0.1:8081`). Apple **FoundationModels** is separate (`apple-fm` + `feat-foundationmodels`, arm64 macOS). ANE dispatch remains a disclosed non-goal. |
| Codex/claude-code TUI feature parity | ◑ Partial | Landed: raw-mode line editor, slash-commands (`/open`, `/diff`, `/commit`, `/context`, `/features`, `/learn`, `/live`, `/save`/`/load`, `/sessions`, `/clear`, `/pane`), plugin commands + context providers, multi-turn ring history (`MAX_TURN_HISTORY`) injected into completions, session save/load in `repl_session.zig`, Ctrl-R reverse search, Alt-Enter multi-line, Ctrl-K/U/W Emacs keys, colorized `/diff`, `/diff --stat`, unified context budgets, post-turn `/pane` split (chat left, `git diff --stat` right via `repl_pane.zig`), live Anthropic SSE via `/live` + `streamMessageLiveIncremental`. Remaining: true token-by-token in-process generation (template model). No new top-level CLI commands (frozen surface). |
| Streaming token-by-token completion | ◑ Partial (improved) | Local persona path: post-hoc ~4-byte chunks via `stream_callback` with tty-aware flush. Local-bridge models: OpenAI-compatible SSE via `httpPostJsonStreamingIncremental`. Live Anthropic: Messages SSE via `streamMessageLiveIncremental` on `complete --live --stream` and TUI `/live`. Residual: true incremental in-process generation (template model returns full text). |
| File-aware agent context | ◑ Landed (v2) | `file_context.zig`: `@file` + fair-share budget (open > @file > tree > git), bounded workspace tree listing, budgeted `git diff`/`--stat` for `agent plan`/`multi`. REPL `/open` + `@file` remain. No new MCP tools (frozen 12-tool surface). |
| Trainable neural net (PointNeuralNetwork + soul-prompt seeding) | ✅ Done | `src/features/ai/point_neural_net.zig` (real trainable MLP: backprop/SGD, topology optimization, weight pruning, JSON save, telemetry, `SoulLayout`) + `src/features/ai/soul_layout.zig` (`SoulLayout.fromJson` + `bootstrap` with WDBX persistence) landed and wired through `mod.zig`/`stub.zig`/`stub_types.zig` with parity; `./build.sh check` 39/39 green. Now invoked by `training.zig`: `train` trains a `[3,8,3]` autoencoder on an available dataset and persists weights to `artifact_dir/neural_net_<profile>.json` (metadata-only fallback when no dataset). |
| Per-turn `PipelineTelemetry` snapshot | ✅ Done | `src/features/ai/pipeline_telemetry.zig` (`PipelineTelemetry` owned snapshot + `ObservabilityHub` counter aggregation + `snapshot`/`finishTurn` mandatory tail folding `AuditResult` ethics scores, `NeuralTelemetry`, and string summaries) landed and wired through `mod.zig`/`stub.zig` with parity; 3 tests. Closes the WDBX §3 mandatory per-turn telemetry design gap. |
| Discord gateway loop | ✅ Done | `src/connectors/discord_gateway.zig` (WS gateway Hello→Identify→heartbeat→MESSAGE_CREATE loop, prefix-scoped/bot-safe/token-safe command routing, `WebSocketClient` masked frames; TLS not linked) landed and wired into `src/connectors/mod.zig`; 5 tests. No autonomous token POST without a live transport. Closes the WDBX §7 gateway-loop design gap. |

### Candidate next slices (real remaining work)

| Item | Status | Notes |
| ---- | ------ | ----- |
| Broader native/batched GPU acceleration | 🟡 In progress | HNSW pairwise + neighbor-expansion batch scoring route through `gpu.vectorOps()` with SIMD fallback. AI completion/SEA paths delegate similarity to `store.search` (already GPU-routed), so the remaining expansion is native kernel dispatch — the deferred 100%-Zig-constraint item, not a completable gap. |
| Windows runtime verification for cross builds | ⚪ Not started | `.github/workflows/ci.yml` runs `zig build check` + `zig build cross-smoke` (linux-gnu/windows-gnu/aarch64-macos). Remaining (out of scope from a macOS host): actual Windows runtime verification. `/tmp`/`std.c.getpid()` test-helper cleanup complete. |
| modern-refactor Phase 2–4 (docs hub / tools split / polish) | ✅ Done | Docs hub + contributing page; contract CLI factoring + agent orchestration smoke; plugin registry Zig-string generator; plan archived under `modern-refactor/examples/`; design specs refreshed for multi-agent orchestration. |

### Priority A security + REPL (G1–G5)

| Item | Status | Notes |
| ---- | ------ | ----- |
| G1 REPL line editor | ✅ Done | Pure `line_editor.zig` (CSI decode, cursor, history) wired into raw-mode REPL; unit tests + TUI/dashboard smoke. |
| G2 MCP/REST JSON depth | ✅ Done | `MAX_JSON_DEPTH=32` in `protocol.validateRequest`; shared by stdio + HTTP `processJsonRpc`; oversize/bearer tests retained. |
| G3 credential/HTTPS hygiene | ✅ Done | Live `joinUrl` requires `https://`; POSIX no-echo signin; Windows ACL/keychain remain disclosed gaps. |
| G4 `ai_train` path sandbox | ✅ Done | Dataset/artifact confined to cwd or `ABI_TRAIN_DATA_ROOT`; rejects `..`, abs outside root, symlink escape. |
| G5 store dir `0700` | ✅ Done | Durable store parent dirs created/repaired owner-only on POSIX; not a multi-host production claim. |

---

## Constraints & intentional non-goals

These are decisions, not unfinished work — do not "fix" them.

- **ANE execution** requires CoreML/ObjC + on-device profiling; excluded by the 100% Zig constraint (user-accepted). Detection (`compute.aneHardwarePresent()`) is truthful; dispatch is not linked.
- **`rest.zig` ↔ `src/mcp/server.zig` HTTP-framing duplication is intentional.** `src/mcp/` is its own module root and cannot import a shared `src/foundation/` leaf (confirmed by compile error). See memory `mcp-module-root-isolation`.
- **Never force-push `main`.** Local and remote histories were reconciled at `848ec2c8` (the old no-common-ancestor state is resolved); the memory `origin-main-unrelated-history` is stale.
- **External-claims policy** (`docs/contracts/external-claims-audit.mdx`): no unbacked sharding/AES/RBAC/cert/QPS/latency/accuracy claims; frame unproven metrics as targets.

---

## Known test failures

- None currently reproduced. Latest review gates: all `*.zig` files pass standalone `zig ast-check`; `zig build lint` passes (errors=0); `zig build check-parity` passes (exit 0); pin gate green on `0.17.0-dev.1398+cb5635714`; `./build.sh check` passes (39/39 steps, 1750 tests); `./build.sh full-check` passes (47/47 steps).

---

## Recently landed (digest — full detail in git + CHANGELOG)

One-line pointers only; the authoritative record is `git log` and `CHANGELOG.md`.
- **REPL `/pane` split** — `repl_pane.zig` post-turn composed split (chat left, `git diff --stat` / open-file summary right); reuses `dashboard_render` fit helpers; refuses <80 cols / non-tty; unit-tested layout math.
- **REPL finalization** — `repl.zig` 1050→538 (thin dispatch hub); input loops/keys/redraw/tab/Ctrl-R → `repl_io.zig` 248; `completePrompt` + stream contexts + `resolveFileMentions` → `repl_complete.zig` 292. Fixed `/diff --stat` (hardcoded `"/diff "` re-parse), deduped `runSyncClis` into `repl_git_commands.zig`, fixed `/open` transient-buffer leak, malformed `--soul-alpha` now usage exit 2. New tests: runOpen packing round-trip/budget/leak, `--soul` arg contract, `agent os dry-run` success. Design doc §2 synced.
- **OS-control policy hardening** — trusted absolute POSIX command specs, empty child environments, retained opened cwd handles, options-only `ls`, and escaped policy-validated dry-runs; feature-off hard-denies and the primary gate is green.
- Skill-set completion: authored opencode/ai-plan/gpu/mcp/sea/tui/wdbx/agent-status-reporter; added tools/check_skills.sh validator; removed 6 repo-root abi-superpower-* orphans. ./tools/check_skills.sh green; ./build.sh check green.
- **TUI dashboard_render extract** — `dashboard.zig` 858→399; composition/split helpers → `dashboard_render.zig` 399 (wired to existing `dashboard_widgets`/`dashboard_panes`; storage pane scope row synced); 5 pure-helper tests. Public TUI API unchanged.
- **Docs hub + Approach-1 plan closeout** — Mintlify index dedupe/dead `wdbx-rust` card drop; plan/design residual checkboxes closed honestly; `modernized/` stays pointer-only (Phase D reimagine HITL-blocked, no scaffold).
- **Dead code + duplication cleanup (wave 5)** — removed duplicated `SoulLayout`/`SoulRecord` from `point_neural_net.zig` (676→543, dead copy of `soul_layout.zig`); deduplicated `"completion:{d}"` format string via shared `COMPLETION_KEY_FMT` in `types.zig`/`stub_types.zig` (parity); deduplicated profile labels via shared `PROFILE_LABELS` constant; kept `streaming.zig` at 16-byte chunks (SSE JSON framing requires larger chunks than the in-process 4-byte path).
- **File splits (wave 5)** — `twilio.zig` 581→254 + `twilio_relay.zig` 346 (ConversationRelay types/parsing); `repl.zig` 1182→1005 + `repl_git_commands.zig` 217 (git/session slash-command handlers as free functions); `plugin_manager.zig` 661→624 (data-driven comptime dispatch table replacing 16-branch if-chain). All splits green.
- **Zig 0.17 pin bump (0.17.0-dev.1398)** — `.zigversion` + CI + all instruction files bumped to `0.17.0-dev.1398+cb5635714`; build + 1752 tests + parity + lint + full-check all green.
- **TUI test-gap extractions (wave 4)** — 6 pure functions + 27 tests extracted into `repl_commands.zig`: `colorizeDiff`, `diffArgv`/`diffWantsStat`/`commitAddArgv`/`commitArgvFor`, `accumulateCommitMessage`, `homeEnvVarName`/`syncClisLauncherPath`, `formatPluginCommandAck`, `buildCompletionContext`/`formatTurnHistoryPreview`. `repl.zig` wired to use all extractions; deduplicated ring-buffer walk between `completePrompt` and `showContext`. `./build.sh check` 39/39 green.
- **Dead code cleanup** — removed unreferenced `routeInputAdaptive` (router.zig + stub_profile.zig); removed 6 orphaned imports (wdbx/mod.zig, gpu/mod.zig, ai/completion.zig, ai/stub_types.zig, ai/training.zig, cli/completion.zig).
- **File splits (wave 4)** — `discord_gateway.zig` 757→436 + `discord_ws_client.zig` 228 + `discord_routing.zig` 126; `dashboard.zig` 703→572 + `dashboard_widgets.zig` 87 + `dashboard_panes.zig` 76. All splits green.
- **Streaming improvement** — `STREAM_CHUNK_SIZE` reduced from 16→4 bytes for more granular post-hoc chunk emission in the in-process persona path; instruction files updated.
- **TUI REPL extract (wave 3)** — `repl.zig` 1399→1110; session/state leaf modules `repl_types.zig` + `repl_session.zig` (serialize/apply/save/load + leak-free tests); `stub.zig`/`mod.zig` parity for `SessionFile`. `./build.sh check` green.
- **Local-bridge SSE streaming** — fix SSE callback context + token accumulation; `agent tui` + `complete --stream` for bridge models; WDBX `rate_limiter`/`tls_config` landed loopback-honest.
- **Module restructure wave** — `registry.zig` 657->398 (handler closures + arg specs -> `wiring.zig` 274); `repl.zig` 656->520 (slash-command parsing + formatting -> `repl_commands.zig` 153); `agent.zig` 528->392 (help text + arg parsers -> `agent_help.zig` 206); `wdbx/mod.zig` 616->61 (`Store` struct -> `store.zig` 582). All slices green: `./build.sh check` 39/39 steps, all tests pass, parity holds.
- **Zig 0.17 pin bump** — `.zigversion` + CI `ZIG_VERSION` bumped to `0.17.0-dev.1275+59a628c6d` (forward-compat verified: build + 910 tests + parity + lint + binary launch all green on master).
- **Design doc sync** — `docs/spec/abi-refactor-design.mdx` Section 2 refreshed: wdbx mod/store split, TUI repl/repl_commands split, CLI registry/wiring + agent/agent_help split, 16 plugin fixtures listed, foundation env/temp_path + gpu compute_api + ai models/iot_monitor/multimodal_fusion + sea scorer/types + connectors fm.zig added.
- **Feature-parity roadmap** — todo.md now includes Proposed north-star items: llama-cpp local inference bridge, MLX on-device inference, codex/claude-code TUI parity, streaming token-by-token completion, file-aware agent context.
- **Per-turn `PipelineTelemetry` snapshot (WDBX §3 closed)** — `src/features/ai/pipeline_telemetry.zig` (`PipelineTelemetry` + `ObservabilityHub` with mandatory `snapshot`/`finishTurn` tail folding `AuditResult` ethics scores + `NeuralTelemetry` + summaries); wired through `mod.zig`/`stub.zig` with parity; 3 tests. `./build.sh check` 39/39 green.
- **Discord gateway loop (WDBX §7 closed)** — `src/connectors/discord_gateway.zig` (Hello→Identify→heartbeat→MESSAGE_CREATE loop, prefix-scoped/bot-safe/token-safe routing, `WebSocketClient` masked frames; TLS not linked) wired into `src/connectors/mod.zig`; 5 tests. `./build.sh check` 39/39 green. Also fixed `optimizeTopology` pruning threshold (absolute → relative magnitude) so weight pruning deterministically fires.
- **`PointNeuralNetwork` wired into the `train` path** — `training.zig` `train` now trains a real `[3,8,3]` autoencoder on an available dataset (jsonl/text/csv → `Point.fromText` via `training_support.datasetToPoints`) and persists weights to `artifact_dir/neural_net_<profile>.json`; metadata-only fallback when no dataset. Closes the last WDBX-extract §8 "not yet wired" slice. `./build.sh check` 39/39 green.

- **Priority A G1–G5** — REPL line editor; MCP JSON depth bound; HTTPS-only live connectors + POSIX no-echo signin; `ai_train` path sandbox (`ABI_TRAIN_DATA_ROOT`); durable store parent dirs `0700` on POSIX.
- **Local agent orchestration + MCP depth** — `agent multi|spawn|browser` now expose scheduler-backed local workers and claim-honest browser planning; background submission is failure-transactional, feature-off stubs preserve type ownership, CLI runtime smoke covers the new surface, and MCP HTTP has transport-level wrong-bearer + oversized-body regression tests.
- **modern-refactor Phase 1** — filled advertised skill `references/` (`analysis-checklist.md`, `implementation-playbook.md` + example); layout verifier; `.gitignore` allowlist + honest README; docs archive isolation + standard extract disclaimers; `tools/goal_capture.sh` SCRATCH via env/`TMPDIR`. (Phases 2–4 later completed — see open-work row ✅ Done; product reimagine stays separate/HITL-blocked.)
- **File extractions (wave 2)** — `dispatch.zig`→`suggest.zig` (473→341), `registry.zig`→`completion.zig`+`help_json.zig` (1033→646), `tui/mod.zig`→`dashboard.zig` (636→153), `handlers/dashboard.zig`→`dashboard_json.zig` (824→485), `cluster_rpc.zig`→`cluster.zig` (cluster_rpc 645→615, cluster 252→292).
- **`src/foundation/temp_path.zig`** — `getTempDir()`/`tempFilePath()` created; 30 hardcoded `/tmp/` refs replaced across 13 files.
- **XDG compliance** — `credentials.zig` now checks `ABI_CREDENTIALS_PATH`→`XDG_CONFIG_HOME`→`~/.abi/`; `durable_store.zig` checks `XDG_DATA_HOME`→`~/.abi/wdbx`.
- **Dead PathConfig removed** — 5 misleading `/tmp/abi/*` defaults stripped from `config.zig`.
- **`sync-clis/launch.sh` REPO_ROOT fix** — path corrected in launcher script.
- **`scheduler.zig` null→unknown fix** — `catch null` → `catch "unknown"`.
- **Instruction files compacted** — AGENTS.md 88→75, CLAUDE.md 138→78, GEMINI.md 148→76 lines; all three now share identical conventions sections.
- **`walkthrough.md` stale paths fixed** — 3 `/tmp/abi-demo.*` → `./abi-demo.*`.
- **MCP concurrency hardening** — shutdown use-after-free closed (teardown deferred to `main` after the HTTP thread joins); TOCTOU lazy-init race in shared scheduler/store closed (double-checked locking, release/acquire ordering).
- **Credential-file hardening** — `abi auth` now creates/repairs `~/.abi` as owner-only (`0700`) and opens/truncates `credentials.json` as owner-only (`0600`) before writing secrets on POSIX-capable targets; still plaintext, with keychain/Windows ACL/zeroing left as disclosed future work.
- **Connector log redaction** — Discord local send/receive logs and Twilio live response logs now emit metadata/byte counts instead of message or provider-response bodies.
- **MCP/REST loopback auth hardening** — optional bearer-token enforcement added for MCP HTTP/SSE (`ABI_MCP_HTTP_TOKEN`) and WDBX REST (`ABI_WDBX_REST_TOKEN`); still not a production non-loopback exposure claim without TLS/authz/rate-limit review.
- **WDBX/SEA correctness** — WAL double-free guards on `putVector`/`store`; `remote_compute` overflow guard; corrupt-manifest rejection; SEA persist→recall round-trip + evidence-recall coverage.
- **SEA adaptive learning loop + WDBX RPC loop** — learned completions now route through persisted `AdaptiveModulator` weights, and `cluster_rpc.zig` has a deterministic authenticated loopback multi-node vote+append round that verifies quorum and peer logs; this is still not production multi-host orchestration or sharding.
- **WDBX perf** — redundant work removed from HNSW/WAL/block-chain hot paths.
- **WDBX segment compaction** — `abi wdbx db compact <path> [keep]` now retains the newest segment checkpoints and reclaims older manifest-listed checkpoints while preserving recovery.
- **WDBX compression** — exact order-0 Huffman entropy codec added beside int8 embedding quantization and the reference autoencoder; still no SOTA/production learned-compression claim.
- **Build/parity** — `check-parity` now fails on a `mod.zig` leaf missing its `stub.zig`.
- **AI training observability** — `training_support.inspectDatasetTracked` routes dataset path/read/JSONL parse allocations through `MemoryTracker`, and `trainWithStore` now falls back to the attached store tracker for the initial training phase.
- **WDBX north-star Phase 1 + V18 cognitive runtime** — WAL+recovery, multi-segment checkpoints, temporal/causal hybrid ranker, persona-scoped retrieval, P50/P95/P99 benchmarks, loopback REST, in-process consensus/compression/FHE demos. (10/11 V18 criteria; ANE execution is the disclosed non-goal.)
- **Whole-tree Zig hygiene review** — all 196 `.zig` files pass standalone `zig ast-check`; fixed the standalone `example-plugin` stub unused-parameter failure and corrected the linked `.agents` `zig-newest-skills` driver path.
- **Codebase analysis + cleanup pass (modern-refactor skills)** — loaded/installed codebase-analysis, refactor-strategy, modern-patterns, code-review; systematic scan per checklist (boundaries, legacy patterns, god files >400LOC, duplication, residue); confirmed excellent post-extraction state (Zig 0.17 clean, no critical silent catches in hot paths, surfaces centralized, parity clean). No high-risk slices; small hygiene only. Gates (lint, check-parity) green.
- **Cross-compilation CI** — `.github/workflows/ci.yml` runs `zig build check` + `zig build cross-smoke` across linux-gnu/windows-gnu/aarch64-macos compile/link targets; Windows runtime execution remains an open verification item above.
- **CLI/TUI command-surface redesign** — typed CLI specs now drive help/validation for migrated commands, typo hints, `help --json` command/subcommand/shortcut/completion-shell metadata, metadata-driven `help --completion <bash|zsh|fish>` scripts, dashboard/TUI pane selection, pane metadata listing, compact selected-pane rendering, plain/no-color, forced one-shot, refresh-interval rendering, JSON snapshots with layout metadata, `abi --tui` shortcut flags, and `agent tui` slash-command status/model validation are contract-smoked; OpenCode MCP config connects both local servers.
- Dead-code cleanup (plan.zig deletion + parity sync, mutex_check.o removal)
- Local-provider model alias routing in models.zig (ollama/lmstudio/llama-cpp/vllm/mlx prefixes → .local, deterministic offline)
- Module declaration coverage cleanup (9 modules + 32 plugin files)
- Whole-tree refactoring wave — param bundling (CompleteOptions, BlockRecord), 4 large file splits (tui, wdbx rest, mcp server, nn), refAllDecls coverage, instruction-file sync.
- **TUI feature-parity improvements** — Ctrl-R reverse history search, Alt-Enter multi-line input, Ctrl-K/U/W/L Emacs keys, `/sessions` list command, `/clear` screen command, colorized `/diff` (green +/red -/cyan hunks), `/diff --stat`, unified file context budgets (32 KiB `/open`, 16 KiB `@file`), `estimateTokens()` helper. `./build.sh check` green (39/39, 1624 tests).
- **9 new superpower skills from docs/specs** — `abi-superpower-agent-orchestration` (multi/spawn/browser), `abi-superpower-constitution` (6-principle audit), `abi-superpower-wdbx-cluster` (Raft + RPC), `abi-superpower-wdbx-compute` (CPU/GPU/NPU/TPU selector), `abi-superpower-wdbx-secure` (compression + HE demos), `abi-claims-validator` (external-claims audit), `abi-wdbx-persistence` (WAL + segments + recovery), `abi-mcp-transport` (JSON-RPC stdio + HTTP/SSE), `abi-plugin-system` (manifest + registry). All in `.agents/skills/` (symlinked to `.opencode/skills/`). `./build.sh check` green.
- **SoulLayout neural routing wired into CLI** — `src/features/ai/router.zig` adds `routeInputWithSoul()` (keyword-sentiment + 3-output neural softmax blend via `blend_alpha`), `blendWeights()`, and `src/cli/handlers/train.zig` adds `handleSoulComplete` path behind `abi complete --soul <file.json> [--soul-alpha <0.0-1.0>]`. `SoulLayout.fromJson` + `bootstrap` now exercised end-to-end; point_neural_net parity maintained in stub. `./build.sh check` 39/39 green.

---

## References

- `docs/spec/wdbx-north-star.mdx` — Current/Partial/Proposed capability mapping
- `docs/contracts/external-claims-audit.mdx` — what public docs may and may not claim
- `CHANGELOG.md` — release-note record of landed changes
