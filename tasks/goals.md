# Goals

## Finish all completable tasks/todo.md hardening
status: done
- Close the two optional Rust #647 hardening rows, finish the in-flight OS-control safety work, preserve the five disclosed product residuals as Partial, and prove the result with focused tests plus the repository health gates.
- **Already on `main` as PR [#770](https://github.com/donaldfilimon/abi/pull/770) (`35b3a99`)** — the `cursor/goal-ledger-open-slices` tree is byte-identical to `origin/main`, so the "merge into main" ask was satisfied before this turn. No merge was performed and nothing was pushed to `main`.
- **The acceptance criterion was not actually met when #770 landed: `main` was red.** The self-hosted CI run for `35b3a99` failed `./tools/check.sh` on two `abi-cli` os-control tests, and the local gate could not reach its final step at all. Both repaired 2026-08-01 in PR [#772](https://github.com/donaldfilimon/abi/pull/772), so this goal closes against a gate that genuinely completes rather than one assumed green.
- **Fix 1 — spurious `WriterBusy` (`crates/abi-wdbx/src/durable.rs`).** The advisory writer lock lives on the open file description, so any `fork` in the process duplicates it into the child until `exec` closes it via O_CLOEXEC. A store dropped just before an unrelated `Command::spawn` stays locked for the width of that window. Not test-only: `abi agent os execute` holds the audit store open *while spawning the command it audits*, so a second `abi` invocation could be told the store was busy when no writer owned it. `acquire_writer_lock` now waits out `WouldBlock` for 50 ms in 1 ms steps; a genuinely held lock still reports `WriterBusy`, and a real filesystem error is never retried. Diagnosed empirically: an `lsof` probe at the moment of failure found **no holder**, and a retry probe recovered on the **first** 1 ms attempt in 6/6 caught failures.
- **Fix 2 — `./tools/check.sh` could not finish on a fresh checkout.** `crates/abi-connectors/build.rs` and `crates/abi-gpu/build.rs` located `target/<profile>` via `OUT_DIR.ancestors().nth(3)`. cargo 1.99.0-nightly nests `build/<pkg>/<hash>/out` where older releases used `build/<pkg>-<hash>/out`, so that resolved to `target/<profile>/build` and both shim dylibs landed one directory from where `@loader_path` looks. `./target/debug/abi` then refused to launch, aborting the benchmark step. Both scripts now find the `build` component and take its parent. Masked in the primary checkout only by a stale dylib from an older cargo layout; CI never saw it because the test step failed first.
- **Claim correction:** `tasks/todo.md`'s `Windows credential ACL CI ✅` → `◑`. The `cfg(windows)` tests are written and the job is configured, but the GitHub API reports *"The job was not started because your account is locked due to a billing issue."* for every hosted job — `windows-acl` has **never executed**, so the ACL behavior is unproven at runtime on any host. `AGENTS.md`, `CLAUDE.md`, and `tools/check.sh` now state both halves: the self-hosted macOS ARM64 runner does execute the gate; every GitHub-hosted job is refused at dispatch.
- Verification: `./tools/check.sh` exit 0, `check: all green`, including the benchmark gate that previously aborted. The stress that reproduced the flake (`os::` at `--test-threads 8`) went from 18/40 and 11/20 failures to **0/50** and **0/40**. A new `durable::` regression test fails deliberately with the retry budget set to 0. Documented smoke walkthrough re-run on the real binary.
- Working-copy repair: `AGENTS.md` had been clobbered by a generic `abbey init` stub; the canonical file is restored (stub saved to the session scratchpad). The now-obsolete "Known gap" note about it was dropped from `CLAUDE.md`, and the verified CI wording landed in #772 instead.
- **2026-08-01 slice:** all #647 and OS-control checklist rows were already ✅ in `tasks/todo.md`, but CI (`.github/workflows/ci.yml`, self-hosted) had gone red on `main` with `WriterBusy`/"writer already open" panics in `os::audit::tests::a_recorded_block_is_reachable_after_reopening_the_store` and `os::tests::execute_and_timeout_audits_reach_the_scratch_store_but_dry_run_does_not` — a green ledger row does not mean the gate stayed green, so this needed a real look, not a rubber stamp.
- Root-caused as a scratch-path hygiene gap, not a locking bug in `DurableStore` itself (no background thread holds the lock; `_writer_lock: File` releases via normal Drop): both tests built their scratch dir from `{pid}-{thread_id:?}`, and the libtest thread pool reuses `ThreadId`s across test functions in the same binary. A panicked prior run skips its own cleanup, so a leftover locked-looking dir from a crashed run — confirmed on this machine at `/var/folders/.../abi-os-audit-reopen-9685-ThreadId(58)` from the exact failing CI run, owned by a since-dead PID — can be reused by a later run on the same (pid, thread) pair. Local repro attempts (10+ runs, filtered/full/single-threaded) never reproduced the panic, consistent with a rare collision rather than a deterministic bug.
- **Correction to the bullet above (2026-08-01, measured).** That root cause is not the one. Two independent findings contradict it:
  1. **A dead process cannot hold an `flock`.** The kernel releases it at process exit, so a leftover scratch dir "owned by a since-dead PID" cannot produce `WriterBusy` at all. And `scratch_store` already did `remove_dir_all` before creating, so a stale directory was never the input.
  2. **It does reproduce locally, easily.** `os::` at `--test-threads 8` fails **18/40** and **11/20**; the narrower set `os::audit:: + os::tests::` fails 2/25. The earlier "never reproduced" runs were at the default thread count, which is too low to open the window.
- **Measured cause:** the advisory writer lock lives on the *open file description*, so any `fork` duplicates it into the child until `exec` closes it via O_CLOEXEC. An `lsof` probe at the instant of failure found **no holder**, and a retry probe recovered on the **first** 1 ms attempt in 6/6 caught failures — a live transient holder, not a stale path. This is why the scratch-path rename cannot fix it: the reopen races a `fork` on the *same* path it just created.
- **Head-to-head, same harness, 40 runs each:** scratch-path hardening (`1fad195`) alone → **3/40 still fail**; writer-lock retry ([#772](https://github.com/donaldfilimon/abi/pull/772)) alone → **0/40**; both together → **0/40**. The path hardening is good hygiene and worth keeping, but it is not the fix. Land both; do not land `1fad195` alone and call the flake closed.
- **Landed 2026-08-01 on "merge all into main":** PR [#772](https://github.com/donaldfilimon/abi/pull/772) squash-merged to `main` as `09ece5c` — the writer-lock retry, both build-script `profile_dir` fixes, the Windows-ACL claim downgrade, and the corrected CI wording. `main`'s gate completes end-to-end again.
- The parallel PR [#771](https://github.com/donaldfilimon/abi/pull/771) was closed as superseded once the head-to-head numbers were posted. Its scratch-path hardening is independently worth keeping, so it was rebased onto post-#772 `main` (merge, no force-push), its false `./tools/check.sh green` row corrected, its superseded `CLAUDE.md` hunk dropped, re-verified (gate exit 0, `os::` 0/40), and re-opened as PR [#773](https://github.com/donaldfilimon/abi/pull/773) framed as hygiene rather than a flake fix.
- Nothing else is unmerged: `cursor/goal-ledger-open-slices` and `worktree-fix-os-audit-writerbusy` are stale pre-#772 branches whose only "diff" against `main` would *revert* #772 — they carry no unique content and must not be merged.
- **Both landed (2026-08-01).** `main` is now `35bb4f1`: [#772](https://github.com/donaldfilimon/abi/pull/772) → `09ece5c` (writer-lock retry, both `profile_dir` build-script fixes, Windows-ACL claim downgrade, corrected CI wording) and [#773](https://github.com/donaldfilimon/abi/pull/773) → `35bb4f1` (scratch-path hygiene + corrected flake attribution).
- **Episode worth remembering:** a parallel agent force-pushed the #773 branch, which silently reverted the ledger correction and restored the false "the path rename fixed the flake / `./tools/check.sh` green" row. Caught by re-diffing the remote head before merging, and re-applied as a normal commit on top (`5cb3f79`) rather than a counter-force-push. Verify the *remote* head content, not your local branch, before merging anything another agent has touched.
- Branch cleanup: `cursor/goal-ledger-open-slices` and `fix/wdbx-writer-lock-fork-window` were deleted only after proving they carried nothing unique — the former's entire "contribution" was the pre-#772 regressed code (`ancestors().nth(3)`, the non-retrying `try_lock`), the latter's was the uncorrected todo row.
- **CI confirmed green on `main` (2026-08-01):** run 30687475587, `check (self-hosted)` job `completed/success` on `35bb4f1`. The `09ece5c` run's gate step also read `completed/success`; its job showed `cancelled` only because `ci.yml`'s `cancel-in-progress` killed the post-checkout cleanup. **Read the step, not the job conclusion** — the single runner serializes everything, so back-to-back merges routinely cancel a job after the gate has already finished. Two stale runs on the merged-and-deleted `cursor/fix-audit-scratch-path-flake` branch (one live since 05:57 on the vanished commit `d9918bb`) were holding the runner and had to be cancelled before `main` could be verified.
- Fix: both tests now build their scratch path with `abi_foundation::temp_path::temp_file_path()` (PID + monotonic per-process counter), the same collision-resistant helper `abi-wdbx`'s own `durable.rs` test fixture already uses — no product code touched. `./tools/check.sh` green after (fmt, clippy `-D warnings`, full workspace test + bench-regression + docs).
- Also corrected a stale CLAUDE.md claim ("GitHub Actions is billing-locked, so `check.sh` is the only real gate") — `ci.yml` runs `check.sh` for real on a self-hosted macOS ARM64 runner.
- **Landed:** rebased the fix (`1fb35ae` → cherry-picked as `1fad195`) onto current `origin/main` — which by then already had `35b3a99`/PR #770, the exact commit whose CI run first hit this flake. Applied cleanly, `./tools/check.sh` green against that base. Pushed as `cursor/fix-audit-scratch-path-flake`, opened draft PR [#771](https://github.com/donaldfilimon/abi/pull/771).
- Not yet `done`: the fix is verified locally (`./tools/check.sh` green, repeatedly) but unproven against a real CI run, since the original failure was rare (10+ local repro attempts, 0 failures) and this is a self-hosted, long-lived runner — only a future CI run on PR #771 (or after merge) can confirm the flake is actually gone. Close this goal once that run is green.

## Make the Abbey Core Identity and Operating Specification canonical
status: done
- Align Abbey, Aviva, ABI routing, and WDBX context/memory behavior with the supplied identity, mission, personality, operating, privacy, accessibility, and epistemic requirements.
- Preserve the Primary Declaration as the product direction while auditing every capability and architecture statement against source, tests, and documented Current/Partial/Proposed boundaries.
- Completion requires an integrated behavioral contract, focused tests, documentation/claims validation, and runtime evidence; aspirational distributed, multimodal, security, memory, or benchmark properties must remain explicitly labeled until verified.
- Landed through PRs #720-#722: canonical identity/specification and claim matrix; Abbey-primary/Aviva-direct/ABI-orchestration routing; exact explicit persona selection; WDBX/SEA provenance minimization, structured trust parsing, authority-weighted deterministic ranking; fixed-trio role alignment; agent/skill mirrors and contract coverage.
- Verification: independent check-work PASS after correcting four findings; merged-main `./build.sh full-check` 48/48; pinned primary gate, feature-off matrix, parity, integration, benchmarks, TUI/CLI smoke, cross-smoke CI, Mintlify validation, and skill checks all green.
- Historical integration evidence: local `main` matched `origin/main` at `2987eec` when this goal closed. Later verified local-only capability waves intentionally moved local `main` ahead; aspirational distributed/multimodal/security/benchmark capabilities remain explicitly Current/Partial/Proposed rather than claimed complete.

## Complete and perfect all ~/abi
status: done
- Mega-goal (claim-honest): green `main` gate, Rust-only tree, frozen surfaces, honest residual catalog — never fake-complete stubs/ANE/sharding/FHE.
- **Rust rewrite on `main` (2026-07-30):** PR #756 → `34c35d5`; gate is `./tools/check.sh` (nightly). Zig tree removed.
- **Hygiene closeout:** critical agent skills/smoke drivers (`goals`, `run-abi`, `mcp-smoke`, `complete-base`, `nn-demo`) teach Rust gates; `nn` demo JSON checkpoint (`--out` / `--checkpoint`).
- **Definition of done met:** default branch Rust-only; `./tools/check.sh` green; 13 CLI + 12 MCP live; store safety + claim discipline documented; residuals permanently labeled Partial on `tasks/todo.md`.
- Permanent product Partial (new goals if pursued): full ggml/llama.cpp, broader native CUDA/Vulkan execution, Windows runtime CI, production separate-host cluster deployment, and native cluster TLS termination. The incremental sampler and Discord/Twilio rustls paths subsequently shipped and are no longer residuals.

## Fix run-abi smoke test failure
status: done
- Fix the mismatch between expected string "GPU backend report" and actual output "Compute Backends:" in smoke.sh
- Ensure `./.agents/skills/run-abi/smoke.sh` passes successfully

## Implement TUI/CLI North-Star Features
status: done
- Ported to Rust on `main` via rewrite: agent line-mode TUI, dashboard one-shot + raw-mode, file_context (`@file`/tree/git), complete local/live/bridge/neural/soul, Anthropic SSE + local-bridge SSE.
- Residual product (not this goal): full ggml/llama.cpp; pane-split chat+diffs REPL; live remote TUI beyond current SSE paths. The bounded incremental sampler subsequently shipped.

## Extract TUI dashboard_render helpers
status: done
- Historical Zig implementation: composition/split/ANSI fit helpers moved into `dashboard_render.zig` with five characterizing tests. The current Rust dashboard is decomposed under `crates/abi-cli/src/dashboard/` and is gated by `./tools/check.sh`.

## Integrate and maintain modern-refactor skills
status: done
- Historical pre-Rust skill wave: updated the then-current modern-refactor skills and mirrors to ABI conventions, frontmatter, and base-dir notes. The canonical repository skills are now Rust-oriented under `.agents/skills/` and synchronized by the repository tooling.
- Synced via sync-clis to .claude and .grok.
- Updated lessons.md, goals.md, instruction files consistency.
- Ran full skill scan (59 skills), build check.

## External claims audit (Drive collateral vs. repo)
status: done
- Full content lives in `docs/contracts/external-claims-audit.mdx` (Toolchain, WDBX, block/spatial, AI profiles, MCP/CLI, GPU, shaders/MLIR, connectors, roadmap demos, "Claims To Remove Or Downgrade", reusable external-delta paragraph).
- Historical 2026-07-18 Zig snapshot retained for provenance. Current evidence lives in the Rust `abi-cli`, `abi-mcp`, `abi-wdbx`, `abi-compute`, `abi-gpu`, and `abi-ai` crates; the downgrade table remains authoritative for production sharding/QPS/AES/RBAC/H100/cert claims.
- Re-open only if Drive collateral or major capability docs drift again.

## Merge all branches into main
status: done
- Audited every local/remote branch and integrated all unique, current work through PR #703 (`6c47a16c`): router-state validation, Zig-pin public-doc contracts, compact canonical agent instructions, and Windows cross-build fixes.
- Superseded branch trees were excluded where their patches were already upstream or would regress current source; branch labels were retained because deletion was not explicitly requested.
- Verified with `cross-smoke`, `./build.sh check`, `./build.sh full-check`, run-abi smoke, 75/75 skill checks, and Mintlify validation; substantive CI checks passed.

## Audit → production-hardening wave (6 fixes + deferred claim-honest)
status: done
- Landed in PR #738: MCP fail-closed, incomplete HTTP→400, putVector id-burn, WAL torn-tail+fsync, fixed-work bearer, scheduler OOM-safe error_msg.
- Do-all claim-honest follow-on: failed-auth rate limit, HNSW rollbackLastInsert on WAL fail, parent-dir fsync (POSIX best-effort).
- Leave labeled: Windows runtime ACL, ggml, Phase D cutover, native TLS/sharding.

## Rewrite ABI fully in Rust nightly and remove Zig
status: done
### Intent
Replace the entire Zig implementation of `~/abi` with a **nightly Rust**
workspace that is strictly better operationally (tooling, tests, docs, claims
honesty) while preserving user-visible contracts and on-disk safety. Delete
every tracked Zig source, build entrypoint, and toolchain pin only after the
Rust gate covers the same surfaces. Never invent production capability
(native GPU kernels, multi-host sharding, audited FHE, live Discord/Twilio TLS
without a proxy).

### Definition of done
1. **Default branch is Rust-only** — `main` has 0 tracked `*.zig` / `build.zig*`;
   primary gate is `./tools/check.sh` (nightly via `./tools/cargo.sh`, never bare
   Homebrew `cargo`).
2. **Frozen surfaces ship** — 13 CLI commands (`help`, `complete`, `train`,
   `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`,
   `scheduler`, `nn`) and 12 MCP tools (`ai_run`, `ai_complete`, `ai_learn`,
   `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`,
   `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`)
   work on real binaries with contract/golden coverage.
3. **Store compatibility** — real WDBX segments under `~/.abi/` remain readable;
   tests never open the live path (scratch / env discipline).
4. **Claim-honest** — GPU reports `accelerated=false` without native kernels;
   residuals are labeled product non-goals, not fake-completed ports.
5. **Docs teach Rust** — README / AGENTS / CLAUDE / GEMINI / walkthrough point at
   `./tools/check.sh`; `tasks/todo.md` is a Rust board; `RUST-REWRITE-PLAN.md`
   records rewrite COMPLETE.
6. **Landed** — rewrite branch merged to `main` (PR #756 or successor); post-merge
   smoke green.

### Out of scope (disclosed residuals — not blockers)
- Broader Metal coverage, verified CUDA/Vulkan runtime execution, and verified ANE residency (the current CoreML surface proves only a compute-unit request)
- Live Discord `wss://` / Twilio media WebSocket without TLS-terminating proxy
- External shader compiler / MLIR-LLVM toolchains; mobile `native_dispatch`
- Production-secure/audited FHE, production multi-host deployment/sharding, and SOTA learned-compression claims (current TFHE/PQ/autoencoder paths are locally tested, reference-scoped artifacts)
- Windows runtime CI for ACLs (needs Windows host)
- ggml/GGUF in-process LLM sampler (optional product expansion)

### Phases
| Phase | Outcome | Status |
| ----- | ------- | ------ |
| **0 — Workspace + gate** | Cargo nightly workspace, `tools/cargo.sh` / `tools/check.sh`, golden fixtures | ✅ |
| **1 — Foundation / core / connectors / WDBX** | Pure crates + durable store + demos, claim-honest | ✅ |
| **2 — AI / SEA / NN / GPU detect / plugins / MCP** | Personas, learn loop, char-LM demo, 12 tools stdio | ✅ |
| **3 — Full CLI + TUI/dashboard** | All 13 commands; one-shot + raw-mode dashboard; agent line-mode | ✅ |
| **4 — Deferred ports** | Soul, Twilio local, Discord offline, local_bridge, MCP HTTP/SSE, FM Swift shim | ✅ |
| **5 — Zig teardown** | One-shot delete; 0 tracked Zig; `./build.sh` → check.sh | ✅ |
| **6 — Docs + board** | Plan COMPLETE banner; README/todo Rust-first | ✅ (board + plan); long-form history scrub optional |
| **7 — Land on `main`** | Push, PR #756, merge, post-merge smoke | ✅ landed (`34c35d5` + #757 hygiene) |

### Landed (2026-07-30)
- Squash-merged PR [#756](https://github.com/donaldfilimon/abi/pull/756) onto
  `main` as `34c35d5` (`Rust nightly rewrite of abi (complete) (#756)`).
- Remote `rust-rewrite` deleted with the merge.
- CI self-hosted `./tools/check.sh` green after
  `fix(mcp): disclose wdbx_stats when durable store cannot open`.
- Post-merge on `main`: Zig inventory 0; `./tools/check.sh` green; CLI/MCP smoke.
- Product residuals remain on `tasks/todo.md` (not rewrite reopeners).

### Verification commands
```bash
./tools/check.sh
git ls-files '*.zig' 'build.zig*'   # must be empty
./tools/cargo.sh build -p abi-cli -p abi-mcp
./target/debug/abi backends
./target/debug/abi complete "hi"
# MCP: tools/list must enumerate exactly 12 tools
# Store: never point tests at real ~/.abi/
```

## Cleanup and refactor Rust 2024 nightly abi codebase
status: done
- Workspace is already `edition = "2024"` on nightly (`Cargo.toml`, `rust-toolchain.toml`); this goal is decomposition/dedup of existing code, not an edition migration.
- Target the six files originally flagged as hotspots (single-responsibility violations). **All six are closed**, and `tools/check_rust_sizes.sh` now enforces the repository contract: every Rust module is at most 1,000 lines and `crates/abi-cli/src/main.rs` is at most 200. Current largest modules are `hnsw.rs` 958, `store.rs` 933, `dashboard.rs` 932, `multiway.rs` 930, and `v2/lifecycle.rs` 921; `wal.rs` is 755 and scheduler production code is 464. `crates/abi-cli/src/wdbx.rs` remains decomposed into the `wdbx/` module directory.
- Checklist tracked in `tasks/todo.md`. Every slice must land through `./tools/check.sh` green with no behavior change (frozen CLI/MCP surfaces + golden fixtures must still pass byte-for-byte).
- Not a rewrite: no new features, no claims changes, no touching the store-safety or claims-discipline rules in `AGENTS.md`.
- **Slice 1 landed (branch hygiene, 2026-07-30):** pruned 11 fully-merged local branches (5 stale `cursor/*` + 6 `worktree-agent-*`, all 0 commits ahead of `main`); 49 → 38 local branches. No code touched, no gate needed.
- **Audit finding — do NOT bulk-merge:** `chore/zig-residual-agent-tooling`, `cursor/agent-spawn-workers`, `cursor/cloud-agent-*`, `cursor/ce-setup` (and likely `cursor/metal-vectorops-clamp`, `cursor/self-improve-review-loop-8e79`, `cursor/abi-skills-health-062909`) branch from *before* the Rust-rewrite product work. Merging them would delete shipped code: `abi-connectors/src/tls_ws.rs` (653 lines), Metal kernel files, `abi-ai/src/orchestration.rs` (676 lines), `abi-nn/src/{gguf_demo,sample_inc}.rs` — all `✅` in `tasks/todo.md`. Treat as dead ends; cherry-pick only if something specific is still wanted.
- **Resolved 2026-08-08:** the stale `.claude/worktrees/` and `~/abi-fix2` entries are gone. `git worktree list --porcelain` now contains only the primary `/Users/donaldfilimon/abi` worktree; no old Zig edits were applied to the Rust tree.
- **Slice 2 landed (2026-07-30):** `crates/abi-wdbx/src/multiway.rs` `evolve()` — extracted the frontier-draining inner loop into `process_frontier() -> Option<Termination>`; elapsed-time bookkeeping now happens once instead of at 4 duplicated early-return sites. 12/12 multiway unit tests (byte-deterministic export-hash + resume oracles) + full `./tools/check.sh` green before and after. Landed via PR [#767](https://github.com/donaldfilimon/abi/pull/767), squash-merged to `main`.
- **Slices 3–6 landed (2026-07-31):** `wdbx.rs` split, `format.rs` split, `wal.rs` `Mutation` dedup, and `complete.rs` arg-parse/dispatch separation all closed with goldens byte-identical. The stale "next slice: wdbx.rs split" note that stood here was contradicted by `tasks/todo.md` — corrected.
- **Slice 7 landed (2026-07-31) — last acceptance gap closed:** the line-mode REPL moved out of `crates/abi-cli/src/agent.rs` into `crates/abi-cli/src/repl.rs` (help text, `valid_model_id`, `LineModeState`, ten slash handlers, the stdin loop, and the eight tests that only exercise them). `agent_tui_line_mode` → `repl::line_mode()`; no logic changed. agent.rs 942 → 520 lines, off the hotspot watch list; repl.rs is 442. Verified pure before moving: no item was referenced outside `agent.rs` and no golden pins REPL text. `./tools/check.sh` green, goldens byte-identical, and `abi agent tui` fed `/status /model fable-5 /profile /help /quit` emits the same bytes.
- **All six flagged hotspots are now closed.** Every row in the cleanup table is ✅.
- **Later closure evidence:** the three follow-on goals below shipped through the Rust source and are reflected in `tasks/todo.md`. Their supported interfaces are complete; PowerShell completion and a new top-level `--completion` flag remain separate public-v2 choices rather than hidden acceptance gaps.

## Generate shell completions dynamically from live command definitions
status: done
- `crates/abi-cli/src/completion.rs` generates Bash, Zsh, and Fish from the live `usage::COMMANDS` / shortcut metadata and command grammar; byte-exact tests compare every generated script with the checked-in golden fixtures.
- Deep grammar covers the supported `--model` flag and the complete `--pane` value set. The frozen public entry remains `abi help --completion <bash|zsh|fish>`.
- PowerShell and a new top-level `abi --completion` flag are not claimed: both change the frozen interface, and Windows runtime behavior remains unavailable on this Mac. They require an explicit public-v2 goal rather than reopening the completed generator.

## Wire SEA 8-signal scorer into learn loop
status: done
- `abi-sea/src/evidence.rs` constructs `SeaSignals`, applies task-adjusted weights, calls `select_sea_candidates`, and enforces stable-ID deduplication plus bounded candidate/record/token/cluster budgets.
- Recency, metadata fit, contradiction, semantic, keyword, importance, authority, and causality signals are covered; prompt snippets are UTF-8-safe and byte bounded.
- The TTY agent REPL exposes session-local `/sea on|off|status|toggle`; focused evidence/scorer/learn-loop tests and the full gate prove the supported path.

## Add tab-completion and line-editing to TUI and REPL
status: done
- `repl_editor.rs` provides the bounded hand-rolled Unicode-safe editor, history/draft restoration, and Tab completion for slash commands, model IDs, and SEA routes without adding a runtime dependency.
- Dashboard Tab/Shift-Tab/h/l navigation and bounded SGR mouse pane selection are implemented with guard-scoped capture/restoration.
- Unit coverage plus dashboard/TUI PTY drivers prove selection, exit, stream discipline, and terminal cleanup. The implementation choice is the in-tree editor rather than `rustyline`/`reedline`.

## Improve OS control safety and flexibility
status: done
- Ledger correction (2026-07-31): two checklist items were already shipped in `crates/abi-cli/src/os.rs` but still listed as open here.
- ✅ Command timeout — 30s in `exec_command_with_timeout`, with stdout/stderr drained on dedicated threads so a chatty command cannot deadlock on a full pipe buffer before the timeout applies.
- ✅ Environment filtering — `filter_env` does `env_clear()` then re-adds only vars not matching `ABI_*` / `*SECRET*` / `*TOKEN*` / `*KEY*` / `*PASSWORD*` / `*CREDENTIAL*`.
- ✅ **Broaden dry-run to accept any command (read-only by design)** — landed 2026-07-31. Dry-run no longer gates on the allowlist; it renders the plan and appends `policy=allowed|denied`, plus a `note:` naming the allowlist when execute would refuse. Unknown commands are marked `(unresolved)` rather than implying the bare name resolves on PATH. Execute keeps the gate — `execute --confirm rm` still exits 1. Verified on the real binary (`agent os dry-run rm -rf /tmp/...` exits 0, spawns nothing, target path never created) and via 5 unit tests; full `./tools/check.sh` green. Also removed a duplicated copy of the old dry-run test from `agent.rs`.
- ✅ **WDBX audit block for executed commands** — landed 2026-07-31. `crates/abi-cli/src/os/audit.rs` appends one vector (the command line embedded), an `os-cmd:<vector_id>` KV entry with canonical `serde_json` metadata (argv, cwd, exit_code, elapsed_ms, timed_out, env_filtered), and one `os-control` audit-chain block per executed command. **Only `execute` is recorded** — writing a block for `dry-run` would put commands in the chain that never ran. The store is injected, so tests use a scratch `DurableStore` and never open `~/.abi/`; a missing or failing store is disclosed as `audit=skipped(no-store)` / `audit=failed(...)` on the `[os-cmd]` line rather than silently dropping the trail.
- ✅ **`~/.abi/os-policy.toml`** — landed 2026-07-31. `crates/abi-cli/src/os/policy.rs`. **Design decision to revisit if you disagree: the file can only narrow, never widen.** `allow` is intersected with the compiled `CEILING`; a name outside it is ignored and reported. Reason: the file sits in the user's home directory, so a widening policy would turn "can write `~/.abi/`" into "can run arbitrary commands through `abi`". Widening the ceiling stays a code change. Unknown keys and malformed files are errors, never a silent fallback to the permissive default (failing open would hand a user who typo'd a *restricting* policy the full allowlist). Strict TOML subset, no new dependency. Path overridable via `ABI_OS_POLICY`.
- ✅ **Configurable timeout** — `timeout_secs` in the same policy file, bounded 1..=3600 (both ends are errors, not silent clamps); default stays 30s. Tested by proving a 1s timeout kills a 30s sleep.
- Verification: 25 os-control unit tests, full `./tools/check.sh` green, and a real-binary smoke against a scratch store + scratch policy — narrowed `allow = ["pwd","ls"]` denies `whoami`, `allow=[...,"rm"]` still denies `rm`, and the audit block landed in the scratch store with `~/.abi/` provably untouched.

## Goal-ledger scope note (2026-08-08 reconciliation)
status: done
- The former blocked note is closed by current Rust evidence: byte-exact Bash/Zsh/Fish generation, the eight-signal SEA evidence path, the bounded hand-rolled REPL editor, and dashboard keyboard/mouse navigation all ship and are gate-tested.
- The old note incorrectly treated proposed interface expansion as part of the Current acceptance boundary. PowerShell, a new top-level completion flag, hosted Windows runtime proof, production multi-host behavior, and CUDA/Vulkan runtime execution remain explicit evidence gaps; none is inferred from the locally completed work.
- The v2 expansion now proves byte-exact committed-transaction replication, conflict-preserving read repair, deterministic placement and resumable rebalance, and a 3–9 node single-host multi-process demo. These close the local proof slices only; real separate-host operation, production deployment, and hosted validation remain open.
- Versioned PQ and persisted-autoencoder artifacts, optional TFHE-rs execution, DGHV educational refresh, the cycle-free `abi-compute` crate, and the Rust size gate are current locally tested surfaces. They do not establish SOTA compression, production cryptography, accelerator speedups, or external audit. HawkScan and Semgrep evidence is unavailable in this checkout.
- The authenticated bounded gRPC/WebSocket gateway is integrated locally with all eight RPCs, metadata-only events, two explicit listeners, and plaintext/TLS/mTLS runtime tests. This establishes a Current local gateway only; hosted production deployment, separate-host operations, DAST, certificate lifecycle proof, and external review remain open evidence boundaries.

## Complete ABI-owned agent, model, and worker foundations for Abbey
status: in_progress
- Adopt and extend the already-merged `abi-agent-runtime` and `abi-models` crates rather than recreating their contracts. The staged acceptance ledger is in `tasks/todo.md` under "Abbey runtime foundation train."
- Current foundation boundary: `abi-agent-runtime` defines provider-neutral model, event, tool-description, policy, audit, usage, budget, cancellation, and deterministic-fixture contracts. It does **not** execute tools or perform model inference.
- Registry-delivery slice implemented: `abi-models` now validates pinned HTTPS artifact URLs, performs bounded exact-range Rustls downloads without redirects, resumes into fsynced partials, verifies mandatory hashes, and publishes without clobbering an existing destination. Ed25519 manifest envelopes are accepted only from configured publisher keys, and consent is bound to the accepting principal plus license/model/revision/artifact digests. Unsigned registry loaders remain explicit local/test-provenance APIs; the crate still performs no model loading or inference and stores no weights, datasets, or adapters in the repository.
- Agent-host slice implemented: the additive `abi-agent-host` crate performs startup-owned schema compilation, validation-before-policy, audited authorization, bounded object-safe execution, correlated tool-result continuation, duplicate/post-terminal rejection, and cooperative cancellation/deadline/event/output/tool/round/run enforcement. It adds no CLI or MCP command.
- No authenticated worker protocol exists yet. `abi-model-runtime` and `abi-worker` remain open and must preserve the frozen 13-command CLI and 12-tool MCP catalogs.
- Every ABI slice starts from clean `origin/main`, runs `./tools/check.sh` through the pinned nightly wrapper, and records source/test/rustdoc evidence separately from hosted or hardware runtime proof.
