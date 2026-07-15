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
| Codex/claude-code TUI feature parity | ◑ Partial | Landed: raw-mode line editor, slash-commands (`/open`, `/diff`, `/commit`, `/context`, `/features`, `/learn`, `/save`/`/load`), plugin commands + context providers, multi-turn ring history (`MAX_TURN_HISTORY`) injected into completions, session save/load in `repl_session.zig`. Remaining: pane split (chat + diffs), true token-by-token terminal redraw, richer context-window policy. No new top-level CLI commands (frozen surface). |
| Streaming token-by-token completion | ◑ Partial (bridge SSE landed) | Local persona path: post-hoc ~16-byte chunks via `stream_callback`. Local-bridge models: OpenAI-compatible SSE via `httpPostJsonStreaming` + `completeLiveStreaming`, wired into `agent tui` and `complete --stream`. Residual: true incremental HTTP read (current SSE parse is full-body then emit) + live remote provider streaming in TUI. |
| File-aware agent context | ◑ Landed (budgeted `@file`) | `file_context.zig` (8 KiB budget, cwd sandbox) wired into `agent plan`, `agent multi`, and `agent tui`. Remaining: workspace tree awareness, multi-file budget policy, deeper diff-in-context for plan/multi (REPL has `/diff`). No new MCP tools (frozen 12-tool surface). |

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
- **`origin/main` shares no common ancestor** with local `main` (different roots). Never force-push to reconcile. See memory `origin-main-unrelated-history`.
- **External-claims policy** (`docs/contracts/external-claims-audit.mdx`): no unbacked sharding/AES/RBAC/cert/QPS/latency/accuracy claims; frame unproven metrics as targets.

---

## Known test failures

- None currently reproduced. Latest review gates: all 196 `*.zig` files pass standalone `zig ast-check`; `zig build lint --summary all` passes (2/2 steps, errors=0); `zig build check-parity` passes (exit 0); pin gate green on `0.17.0-dev.1275+59a628c6d` (`.agents/skills/zig-pin/pin.sh` exit 0); `zig-newest-skills` PASS on Zig master `0.17.0-dev.1275+59a628c6d`; `./build.sh check` passes (39/39 steps, unchanged); `./build.sh full-check` passes (47/47 steps).

---

## Recently landed (digest — full detail in git + CHANGELOG)

One-line pointers only; the authoritative record is `git log` and `CHANGELOG.md`.
- **TUI REPL extract (wave 3)** — `repl.zig` 1399→1110; session/state leaf modules `repl_types.zig` + `repl_session.zig` (serialize/apply/save/load + leak-free tests); `stub.zig`/`mod.zig` parity for `SessionFile`. `./build.sh check` green.
- **Local-bridge SSE streaming** — fix SSE callback context + token accumulation; `agent tui` + `complete --stream` for bridge models; WDBX `rate_limiter`/`tls_config` landed loopback-honest.
- **Module restructure wave** — `registry.zig` 657->398 (handler closures + arg specs -> `wiring.zig` 274); `repl.zig` 656->520 (slash-command parsing + formatting -> `repl_commands.zig` 153); `agent.zig` 528->392 (help text + arg parsers -> `agent_help.zig` 206); `wdbx/mod.zig` 616->61 (`Store` struct -> `store.zig` 582). All slices green: `./build.sh check` 39/39 steps, all tests pass, parity holds.
- **Zig 0.17 pin bump** — `.zigversion` + CI `ZIG_VERSION` bumped to `0.17.0-dev.1275+59a628c6d` (forward-compat verified: build + 910 tests + parity + lint + binary launch all green on master).
- **Design doc sync** — `docs/spec/abi-refactor-design.mdx` Section 2 refreshed: wdbx mod/store split, TUI repl/repl_commands split, CLI registry/wiring + agent/agent_help split, 16 plugin fixtures listed, foundation env/temp_path + gpu compute_api + ai models/iot_monitor/multimodal_fusion + sea scorer/types + connectors fm.zig added.
- **Feature-parity roadmap** — todo.md now includes Proposed north-star items: llama-cpp local inference bridge, MLX on-device inference, codex/claude-code TUI parity, streaming token-by-token completion, file-aware agent context.

- **Priority A G1–G5** — REPL line editor; MCP JSON depth bound; HTTPS-only live connectors + POSIX no-echo signin; `ai_train` path sandbox (`ABI_TRAIN_DATA_ROOT`); durable store parent dirs `0700` on POSIX.
- **Local agent orchestration + MCP depth** — `agent multi|spawn|browser` now expose scheduler-backed local workers and claim-honest browser planning; background submission is failure-transactional, feature-off stubs preserve type ownership, CLI runtime smoke covers the new surface, and MCP HTTP has transport-level wrong-bearer + oversized-body regression tests.
- **modern-refactor Phase 1** — filled advertised skill `references/` (`analysis-checklist.md`, `implementation-playbook.md` + example); layout verifier; `.gitignore` allowlist + honest README; docs archive isolation + standard extract disclaimers; `tools/goal_capture.sh` SCRATCH via env/`TMPDIR`. Phases 2–4 deferred.
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
- **9 new superpower skills from docs/specs** — `abi-superpower-agent-orchestration` (multi/spawn/browser), `abi-superpower-constitution` (6-principle audit), `abi-superpower-wdbx-cluster` (Raft + RPC), `abi-superpower-wdbx-compute` (CPU/GPU/NPU/TPU selector), `abi-superpower-wdbx-secure` (compression + HE demos), `abi-claims-validator` (external-claims audit), `abi-wdbx-persistence` (WAL + segments + recovery), `abi-mcp-transport` (JSON-RPC stdio + HTTP/SSE), `abi-plugin-system` (manifest + registry). All in `.agents/skills/` (symlinked to `.opencode/skills/`). `./build.sh check` green.

---

## References

- `docs/spec/wdbx-north-star.mdx` — Current/Partial/Proposed capability mapping
- `docs/contracts/external-claims-audit.mdx` — what public docs may and may not claim
- `CHANGELOG.md` — release-note record of landed changes
