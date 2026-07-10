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

### Candidate next slices (real remaining work)

| Item | Status | Notes |
| ---- | ------ | ----- |
| Broader native/batched GPU acceleration | 🟡 In progress | HNSW pairwise + neighbor-expansion batch scoring route through `gpu.vectorOps()` with SIMD fallback. AI completion/SEA paths delegate similarity to `store.search` (already GPU-routed), so the remaining expansion is native kernel dispatch — the deferred 100%-Zig-constraint item, not a completable gap. |
| Windows runtime verification for cross builds | ⚪ Not started | `.github/workflows/ci.yml` runs `zig build check` + `zig build cross-smoke` (linux-gnu/windows-gnu/aarch64-macos). Remaining (out of scope from a macOS host): actual Windows runtime verification. `/tmp`/`std.c.getpid()` test-helper cleanup complete. |
| modern-refactor Phase 2–4 (docs hub / tools split / polish) | ◑ Partial | Phase 1, Mintlify Card hub/nav, extracted CLI contract scripts, build-derived feature matrix, MCP wrong-bearer/413 transport tests, and coordinator-agent packaging landed. Remaining: docs contributor guidance/extract organization, generator/tool dedup, final polish + plan archival. |

---

## Constraints & intentional non-goals

These are decisions, not unfinished work — do not "fix" them.

- **ANE execution** requires CoreML/ObjC + on-device profiling; excluded by the 100% Zig constraint (user-accepted). Detection (`compute.aneHardwarePresent()`) is truthful; dispatch is not linked.
- **`rest.zig` ↔ `src/mcp/server.zig` HTTP-framing duplication is intentional.** `src/mcp/` is its own module root and cannot import a shared `src/foundation/` leaf (confirmed by compile error). See memory `mcp-module-root-isolation`.
- **`origin/main` shares no common ancestor** with local `main` (different roots). Never force-push to reconcile. See memory `origin-main-unrelated-history`.
- **External-claims policy** (`docs/contracts/external-claims-audit.mdx`): no unbacked sharding/AES/RBAC/cert/QPS/latency/accuracy claims; frame unproven metrics as targets.

---

## Known test failures

- None currently reproduced. Latest review gates: all 196 `*.zig` files pass standalone `zig ast-check`; `zig build lint --summary all` passes (2/2 steps, errors=0); `zig build check-parity` passes (exit 0); pin gate green on `0.17.0-dev.1252+e4b325c19` (`.agents/skills/zig-pin/pin.sh` exit 0); `zig-newest-skills` PASS on Zig master `0.17.0-dev.1275+59a628c6d`; `./build.sh check` passes (39/39 steps, unchanged); `./build.sh full-check` passes (47/47 steps).

---

## Recently landed (digest — full detail in git + CHANGELOG)

One-line pointers only; the authoritative record is `git log` and `CHANGELOG.md`.

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
- **Cross-compilation CI** — `.github/workflows/ci.yml` runs `zig build check` + `zig build cross-smoke` across linux-gnu/windows-gnu/aarch64-macos compile/link targets; Windows runtime execution remains an open verification item above.
- **CLI/TUI command-surface redesign** — typed CLI specs now drive help/validation for migrated commands, typo hints, `help --json` command/subcommand/shortcut/completion-shell metadata, metadata-driven `help --completion <bash|zsh|fish>` scripts, dashboard/TUI pane selection, pane metadata listing, compact selected-pane rendering, plain/no-color, forced one-shot, refresh-interval rendering, JSON snapshots with layout metadata, `abi --tui` shortcut flags, and `agent tui` slash-command status/model validation are contract-smoked; OpenCode MCP config connects both local servers.
- Dead-code cleanup (plan.zig deletion + parity sync, mutex_check.o removal)
- Local-provider model alias routing in models.zig (ollama/lmstudio/llama-cpp/vllm/mlx prefixes → .local, deterministic offline)
- Module declaration coverage cleanup (9 modules + 32 plugin files)
- Whole-tree refactoring wave — param bundling (CompleteOptions, BlockRecord), 4 large file splits (tui, wdbx rest, mcp server, nn), refAllDecls coverage, instruction-file sync.

---

## References

- `docs/spec/wdbx-north-star.mdx` — Current/Partial/Proposed capability mapping
- `docs/contracts/external-claims-audit.mdx` — what public docs may and may not claim
- `CHANGELOG.md` — release-note record of landed changes
