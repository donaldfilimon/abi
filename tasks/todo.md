# TODO — ABI Framework (Rust 2024, stable 1.99.1)

## Approved complete-ABI execution (2026-10-23)

- Planning supplement (2026-10-25):
  [complete ABI, then qualify its LYLX migration](../docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md).
  Retains C01–C44 and separates local delivery, operational release and replacement
  qualification; no implementation milestone is closed by writing the plan.
- Correction (2026-10-25): the supplement no longer says the lockfile selects Tokio
  1.53.1 or targets 1.53.3. Baseline `main` `ff40bdd` pins 1.54.1, the floor; the
  next A1 item is binding the consumed WDBX/Abbey source set, then one C01–C44 ledger.
- [x] (2026-10-25, A1 partial) Merge origin/main `42f1446` into local `main`
  without discarding either history (`f65d5e5`) and repair the README so the
  doc policy holds (`2113824`); full gate exit 0 at `2113824` with the receipt
  recorded in `docs/verification/2026-10-23-completion-matrix.md`.
- [x] (2026-10-26, A1) Bind the consumed source set: the sibling WDBX owner's
  staged changes landed upstream, so WDBX sits clean at `96c4b13`; a fresh
  complete `./tools/check.sh < /dev/null` at ABI `main` `aaf2929` exited 0
  against it (1,241 Rust executions, 186 Python policy tests OK with 8 skips,
  `cargo-audit` scan run, benchmark regression PASS) and the receipt is in the
  C01–C44 matrix execution record. Remaining A1 items stay external: WDBX's
  own gate and the Abbey root/desktop gates are separate owners' actions, and
  pin promotion/publication is the owner's call (CI pin stays `2969d71`).
- [x] (2026-10-27, A1/A2 evidence) Bound gate on ABI `1dd7f0b` + WDBX `9d386dd`
  (clean shared clones): exit 0, 1,248 Rust / 186 Python OK. Receipt
  `docs/verification/2026-10-27-a1-gate-receipt.md`; 44-row ledger
  `docs/verification/2026-10-27-c01-c44-execution-ledger.md`. Training stack
  (C23–C28) handed to LYLX (Donald). Finish plan for C38/C39/C11/C10/C32:
  `docs/plans/2026-10-27-finish-abi.md`.
- [x] (2026-10-27, finish-ABI) C38 metrics/health, C39 runbooks with drills,
  C11 governed Discord/Twilio executor, C10 governed macOS app activation and
  C32 unsigned Mac package/installer, all moved `M` → `U`. C05/C06 moved to
  `Q local Mac` (A2 exit). Receipts: `docs/verification/2026-10-27-{a1,a2,finish}-gate-receipt.md`.
- [ ] Complete milestones 1–11 of
  `docs/superpowers/plans/2026-10-23-complete-abi.md` against the authoritative
  C01–C44 matrix in `docs/verification/2026-10-23-completion-matrix.md`.
- [x] Collect planning baseline process exit 0 and capture initial ABI/WDBX/Abbey
  hashes, file modes, revisions and locked package/default-feature metadata under
  `/Users/donald/.codex/verification/abi-completion-20261023/`.
  The unfrozen planning baseline is not final cohort qualification.
- [x] Local worker persistence fault acceptance and fresh post-authority
  dispatch-time guard; interrupted admissions retain no fresh authority.
- [x] Implement local worker authority-time repair, real-file fault acceptance
  and closed governed Unix pwd adapter with OS receipts. Focused suites/review
  pass; full ABI gate is exit 0 (1,234 Rust executions) and copied Mac CLI/MCP
  candidate acceptance passes. Real Abbey durable-owner integration and the
  remaining ecosystem milestones stay open. Mac is the immediate runtime target.

Current status ledger for the Rust rewrite closeout, completed hardening slices,
and explicitly disclosed residuals. Detailed history also lives in
`RUST-REWRITE-PLAN.md`, `git log`, and `CHANGELOG.md`. Claims gate:
`docs/contracts/external-claims-audit.mdx`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: keep completion evidence compact and keep residual constraints
> explicit. Source and tests override prose. Gate: `./tools/check.sh`.

---

## Continuation qualification (2026-10-20)

- [ ] Governed process adapter (2026-10-23): design/buildability review complete;
  implementation begins after the current Abbey qualification source handoff.
  Closed startup-pinned cwd observation, bounded Unix supervisor and real OS
  postcondition evidence are planned. General platform/messaging and other OS
  runtime acceptance remain open. Plan:
  `docs/superpowers/plans/2026-10-23-governed-process.md`.

- [x] 2026-10-23 governed completed-history reopen acceptance: the real private
  worker log and encrypted outbox reopen, then return recorded completion with
  unavailable local approval state and refusing fresh safety facts. Packet and
  manifest bytes, file contents/mtime and exactly-one adapter count stay unchanged;
  no approval file is created. Control-plane library: 83 passed, zero ignored
  (`/tmp/abi-governed-reopen-suite-20261023.log`). Final whole gate is tracked
  separately; the twelve-point local interruption continuation is recorded below.
  General platform, separate-host and power-loss acceptance remain open.

Production ecosystem implementation (approved 2026-10-22) is tracked in
`docs/verification/2026-10-22-production-ecosystem.md`, with the full release boundary in
`docs/superpowers/plans/2026-10-22-abi-production-ecosystem.md`.

- [ ] Durable Abbey approvals and governed tenant tools (2026-10-22): local
  approval/lifecycle ownership, explicit live host boundary and encrypted tool
  relay admission are implemented. Native TCP/PTY approval/denial, restart replay,
  owner isolation and prepare/post-effect crash acceptance pass; final ABI, WDBX
  and prior Abbey root/desktop gates pass. The twelve-point local scratch
  interruption/duplicate-delivery acceptance is implemented and independently
  reviewed, including parent SIGKILL before/after consumer import commit and
  confirmed TCP publication/ACK response loss. Final ABI gate: 1,212 Rust
  executions, zero failures/ignored; source identities stayed unchanged. Receipt:
  `/Users/donald/.codex/verification/abi-continuation-20261023/abi-consumer-final-receipt.json`.
  Updated Abbey cohort qualification is coordinated separately. Concrete adapters
  beyond scratch files, separate-host partitions, power loss and cross-platform
  acceptance remain open. See `docs/spec/durable-governed-tools.md` and
  `docs/superpowers/plans/2026-10-22-durable-governed-tools.md`. This is a
  continuation of the full ecosystem objective, not its completion.

- [x] Durable encrypted output redelivery and worker reconnect recovery local
  source slice: private synced outbox, read-only recorded-outcome recovery,
  replay-only current-authority grants, transient buffer restoration and atomic
  verified client imports. Actual process-kill/TCP-response-loss fixtures retain
  exact ciphertext and provider counts; package suites and Clippy pass. Final
  whole-workspace gate/source receipt is recorded separately in
  `docs/verification/2026-10-22-production-ecosystem.md`. Contract and implementation
  plan: `docs/spec/encrypted-relay-recovery.md` and
  `docs/superpowers/plans/2026-10-22-encrypted-relay-recovery.md`. Public/separate-host,
  power-loss and Windows/Linux runtime acceptance remain open.

- [x] Encrypted enrolled-tenant inference relay: signed X25519 enrollment,
  metadata-only durable queue, bounded transient HPKE ciphertext, 2/10 quota
  defaults, fresh action-bound outbound node proofs/dispatch permits, durable
  user-node execution and two-tenant TCP acceptance. Browser accepts sealed
  job files; native client owns prompt/result decryption. Local probe only;
  governed tools, separate hosts and public operation remain open.
- [x] Encrypted incremental results and cooperative worker cancellation: ordered
  node-signed HPKE frames, bounded backpressure, metadata-only browser watching,
  local prefix verification, durable cancel-request/worker-stop distinction,
  racing-frame reconciliation and two-tenant TCP acceptance. Durable redelivery
  is implemented in the continuation above; noncooperative process termination
  and separate-host qualification remain open.

- [x] Local public-account/enrollment foundation: separate Rust service and
  browser client, verified OIDC/PKCE, durable tenants/sessions, exact locally
  signed pairing, application credentials, revocation/deletion and replay tests.
  See `crates/abi-control-plane/README.md`; Task 2 now has local encrypted two-tenant job acceptance; public-provider
  and separate-host qualification remain open.

- [x] WDBX ledger-hardening consumer regressions: storage accounting,
  compensation admission and writer-lock recovery remain in sibling WDBX.
  Both gateway transport regressions pass against that source; the signed
  Abbey SEA composition regression also passes. Baseline full ABI gate
  stopped at its required dependency scanner (neither scanner was on PATH).
  The installed `~/.cargo/bin/cargo-audit` was then located; the full gate
  passed with that directory on PATH, without installation or policy bypass.
  Immutable CI pins remain unchanged.
- [x] Complete local ledger-hardening gate runs: retain the
  initial scanner-PATH failure and tool timeouts. WDBX's pinned-rustup full gate
  and ABI's full gate passed after the REST overload correction. Abbey's earlier
  gate stopped on concurrent TUI formatting; the later run passed that check
  but failed the daemon memory-capability process test. A fresh isolated Cargo
  target subsequently passed the complete four-mode Abbey gate. Feature-skew
  scratch reproduction is not proof of the historical failure's cause.
- [ ] Freeze and qualify the final immutable ledger-hardening cohort before CI
  pin changes: unrelated Abbey legacy-migration test work changed during local
  qualification. The successful local commands do not establish a published
  exact-revision pairing; CI and publication remain separate.

- [x] Repair buffered tool admission after provider budget exhaustion: two
  actual failing token regressions now pass; final usage, explicit token/event
  stops, cumulative usage, retained text and governed no-intent controls cover
  the shared driver. Independent scoped review approved the repair.
- [x] Repair runtime terminal cancellation/deadline observation: three failing
  regressions now pass without changing sticky stops or exact event/token
  admission semantics. Final host/runtime suites: 128 passed, none ignored;
  package format and warning-denied Clippy pass.
- [ ] Complete broad current-source qualification after the sibling owner
  handoff. The later complete ABI gate exited 0 with 1,082 Rust executions,
  none failed/ignored, and 173 Python cases (8 skipped). Preserve earlier
  scanner-PATH, sibling formatting and repaired host-lint failures as separate
  receipts. CUDA compilation is unavailable without nvcc; WDBX emitted an
  unused-import warning. This green run does not qualify an immutable changing
  three-repository cohort, publication or later owner source changes.
- [x] Keep UTF-8 bounded capture a true stream prefix: two unit regressions and
  one public runtime regression failed before the fix and now pass. Once a
  character is dropped, later chunks count as dropped instead of filling the
  gap. Chunk-boundary equivalence and unchanged streamed events are covered;
  runtime tests and warning-denied Clippy pass; independent review approved.
- [x] Mitigate governed authority expiry across prepare/stage-12 callbacks.
  Guards check the exact admitted grant, source, latest facts and supplied
  approval at one trusted timestamp; intentionally cancel undispatched work
  with the existing `NotDispatched / CancelledBeforeDispatch` terminal. Nine
  actual RED regressions plus a passing control became ten GREEN tests;
  independent review approved. Consumed approval and caller cancellation remain
  unchanged. Existing real Abbey governed integration tests passed (9), and a
  separate scratch expiry fixture passed (5 tests, 8 scenarios) with the actual
  RuntimeStore, reopen, replay refusal and compensation rejection. No shared
  lifecycle or Abbey source edits were required. Broader fresh revocation/
  permission revalidation still needs same-intent consumed-approval semantics
  from the durable owner; the expiry guard is not atomic authority/effect proof.

- [x] Current-source continuation: reviewed durable replay refusal is local ABI
  commit `7b9519b`; real host/store tests and full declared-pair gate passed.
  Abbey atomic PID fixture repair passed ten runs and review; the existing stable
  migration is on canonical main with combined root (3,831 Rust executions) and
  separate desktop gates passing on unchanged captured sources.
- [x] Qualify refreshed canonical ABI dependencies with current WDBX: full stable
  gate passed (1,055 Rust executions), code/lock bytes stable; copied release
  artifacts and hook/docs checks passed.
- [x] Record Abbey owner exact-pin root/desktop qualification: 3,831/49 Rust
  executions, 98/18 Python cases, unchanged candidate; installer rollback passed.
- [ ] Qualify later owner decoder/route-log edits after their source handoff;
  these postdate the accepted ABI and Abbey cohorts.
  CI remains paused; no commits/pushes; publication remains separate.
  Evidence: docs/verification/2026-10-20-current-canonical-qualification.md.

- [x] Qualify the selected stable source checkpoint: capability/recording host,
  same-store Abbey lifecycle, scoped evidence and artifact join. Repaired fixed
  ABI cohort: full gate exit 0 on 1.99.1 (1,055 Rust test executions); older fixed
  Abbey consumer: full candidate exit 0 (3,827 executions, four modes), no stripping
  warnings. Both manifests/modes unchanged. Later source and Abbey's newer
  candidate remain separately owned. CI/publication/live effects remain paused.
  Report: docs/superpowers/reports/2026-10-20-stable-source-qualification.md.
- [x] Qualify the local immutable pairing with WDBX 2969d71: exact-source own
  gate 647 passed plus repaired paired ABI gate. Original 08a69b3 rustfmt failure
  and the earlier formatting-repaired dirty-source success remain separate.
- [x] Close final producer/consumer mismatches: actual host + RuntimeStore
  cancellation/refusal tests reproduced 5 RED, then 8 passed; host/core 96 and
  runtime 138 passed, strict clippy and independent scoped review approved.
  NotDispatched and genuine post-effect uncertainty persist correctly; stage 11
  denial records preserve atomic successful admission. Replay diagnostic
  Infrastructure(InvalidState) remains disclosed, never a replayed effect.
- [ ] External owner packet: WDBX member-erasure/retention/key-custody contracts
  are absent; do not fabricate a competing deletion path. Abbey owns its newer
  stable candidate, canonical pin promotion and approval UI. CI/publication,
  real models/training, live integration and deployment require separate scope.
- [x] Restore the mandatory dependency scanner on this host: installed
  `cargo-audit 0.22.2` through `./tools/cargo.sh`; the required scan passed
  with the existing policy exceptions unchanged.
- [x] Resolve the sibling compute formatting mismatch without changing behavior;
  ABI nightly workspace formatting and WDBX stable 1.99.1 formatting both pass.
- [x] Earlier nightly baseline `./tools/check.sh` exited 0: required audit, size/format checks,
  warning-denied Clippy, workspace build/tests, sibling contract tests, Metal
  tests/docs, benchmark guard, rustdoc and release CLI/MCP builds passed.
  Eight central-sync policy tests skipped because their personal driver is
  absent; CUDA compilation was unavailable without `nvcc`. Local WDBX is at
  `08a69b3`, ahead of ABI's `cf2e326` CI pin; this does not qualify a pin update
  or an exact-commit CI receipt. Pending hook and architecture edits preserved.
- [x] Qualify the CLI/MCP smoke harness: override inherited live-store settings,
  use the MCP launcher with a freshly built release binary, and stop on build
  failure before stale binaries can run. Build into the directory the harness
  actually executes even with an inherited `CARGO_TARGET_DIR`. Three isolated
  regressions pass after reproducing failures; real smoke passed all 18 checks
  and the final full gate exited 0. The run-abi skill documents these boundaries.
- [x] Qualify the dedicated MCP smoke driver: use safe store settings, locked
  release builds and the launcher; reject failed processes and mismatched
  JSON-RPC response envelopes before checking tool names. Five isolated tests
  pass after reproducing six failing assertions; real smoke verified the exact
  12-tool set and the final full gate exited 0. Build diagnostics print before
  temporary-log cleanup, and the skill uses the required launcher.

## ABI source-release hardening (2026-09-29)

- [x] Regress and fix gateway credential diagnostics and error teardown, malformed provider-turn admission, and release-gate build identity.
- [x] Require locked builds, WDBX contract conformance, the existing dependency-security policy, and release binaries in the gate.
- [x] Complete local source acceptance: 143 Python policy tests, 924 Rust tests (zero ignored), required audit, benchmark, docs, and release binaries passed. Mintlify and actionlint passed separately.
- [x] Final review fixes: failed CI target allocation exits without export; retained binary handoff includes both required macOS libraries, loader/signature/hash checks, and copied CLI golden help. Focused release tests, actionlint, and docs validation passed. The preceding 143/924 result belongs to the prior source SHA; a fresh exact-commit gate and CI receipt are pending.
- Publication acceptance requires a successful self-hosted CI receipt whose `headSha` equals published `main`; the delivered release record carries this post-publication proof and artifact checksums without embedding a self-referential source SHA here.
- Scope is `donaldfilimon/abi` only. The completed Rust rewrite and historical Discord PR reconciliation remain closed. Bot deployment, music/voice acceptance, real providers, sibling changes, and new public commands are outside this adaptation.

## Historical Discord PR reconciliation (2026-09-29)

- Donald selected this canonical checkout (`~/dev/active/abi`, remote
  `donaldfilimon/abi`) when reviewing the similarly named
  `donald-filimon/abi` PR #1. The two GitHub repositories are distinct.
- The original `wos8hs-codex/design-architecture-for-discord-ai-bot-abbey`
  branch landed here as [PR #14](https://github.com/donaldfilimon/abi/pull/14),
  **Add Discord gateway skeleton**, on 2025-06-07, merge
  `eed2ed34b1ea98fcb13ffc5a3bfce66ce6bcb1e8`. Canonical PR #1 is the unrelated
  **Add Zig agent client**, also already merged.
- The other repository's open PR #1 points to XFOSS/abi head
  `f04bff2dbd63981ff77e168ecaa223e02bc8a22c`: 1,227 commits and 1,896 changed
  files against its own base. That historical Zig snapshot has no common Git
  ancestor with current canonical `main`; it is not a pending Rust change to
  import wholesale. No historical branch was merged or force-pushed.
- Current implementations are `crates/abi-connectors/src/discord_gateway.rs`,
  `discord_routing.rs`, `discord_ws.rs`, and `tls_ws.rs`. The complete
  `./tools/check.sh` gate passed with exit 0 on canonical
  `37d8b6b51719a9afd96cbd0a2583905ddafc97ad`, including gateway routing,
  malformed WebSocket handling, local REST transport, and process-local TLS
  tests. This is local implementation evidence, not a live Discord deployment
  or proof that every historical Abbey AI proposal is implemented.
- No open PR remained in `donaldfilimon/abi` at inspection. The external PR
  remains open; the signed-in account has no push access to its destination.
- Follow-up CI run `36524711811` exposed a stale WDBX pin: `abi-sea` calls
  `VersionedStore::record_vector_pair`, which the pinned `3ac03f0` did not
  provide. Update `WDBX_REVISION` to the clean, published local-gate substrate
  `9fee98ff5ccb92fa86a2ed44f93abd65e7e181ae`; keep the sibling path layout and
  exact revision pin. GitHub's separate hosted CodeQL job did not execute
  because of the account billing lock, and supplies no validation evidence.
- The complete local gate passed again with the updated pin and exact WDBX
  revision (`./tools/check.sh`, exit 0). CUDA feature compilation remains
  unavailable on this host because `nvcc` is absent.

---

## Rewrite closeout

| Item | Status | Notes |
| ---- | ------ | ----- |
| Zig teardown | ✅ | 0 tracked `*.zig` / `build.zig*` |
| Frozen CLI (14) + MCP (12) | ✅ | Golden + unit coverage; `improve` is the 14th CLI command |
| FoundationModels shim | ✅ | `libabi_fm_shim.dylib` on arm64 macOS; honest offline |
| Local OpenAI bridge + MCP HTTP transport | ✅ | Loopback persistent MCP 2024-11-05 HTTP+SSE sessions are contract-tested against stdio responses; one-shot `POST /message` remains for compatibility. Not Streamable HTTP (2025-03-26). |
| Land `rust-rewrite` on `main` | ✅ | Squash-merged [#756](https://github.com/donaldfilimon/abi/pull/756) as `34c35d5` |

---

## Cleanup / refactor backlog (goal: "Cleanup and refactor Rust 2024 nightly abi codebase")

| Item | Status | Notes |
| ---- | ------ | ----- |
| `../wdbx/crates/abi-wdbx/src/multiway.rs` decomposition | ✅ | `evolve()`'s frontier-draining loop extracted to `process_frontier()`; elapsed-time bookkeeping now happens once instead of at 4 duplicated early-return sites. 12/12 multiway tests + full gate green. |
| `crates/abi-cli/src/wdbx.rs` split | ✅ | Thin `wdbx/mod.rs` + `db`/`block`/`query`/`cluster`/`api`/`benchmark`/`compute`/`secure`/`gpu`; goldens byte-identical. |
| `../wdbx/crates/abi-wdbx/src/format.rs` split | ✅ | Facade + `hash`/`record`/`segment`/`manifest`; 34 format tests green. |
| `../wdbx/crates/abi-wdbx/src/wal.rs` dedup | ✅ | Shared `Mutation` serde enum for all `append_*` frames. |
| `crates/abi-cli/src/agent.rs` decomposition | ✅ | `util::open_store` + `os` module extracted, then the line-mode REPL moved to `repl.rs`. agent.rs 942 → 520 lines. |
| `crates/abi-cli/src/complete.rs` split | ✅ | `parse_complete_args` + `CompleteRequest` separated from dispatch. |
| `complete --neural` handler extraction | ✅ | The unchanged character-LM demo lives in `complete/neural.rs`; the focused CLI test, source-size gate, goldens, and full `./tools/check.sh` pass. |

> Each row: pure refactor, no behavior change. `./tools/check.sh` green + golden fixtures byte-identical before marking ✅.

---

## Active finish-all wave

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Make the gate portable to a hostile `PATH` | ✅ | The gate's Python tooling needs 3.11+ (`tomllib`); macOS `/usr/bin/python3` is 3.9 and could resolve first, killing the gate with a `ModuleNotFoundError` traceback. `tools/check.sh` and `tools/cargo.sh` now resolve a capable interpreter themselves (versioned `python3.X` first, then `python3`/`python`; `ABI_PYTHON` overrides) and fail closed with an actionable message. The required dependency scanner lives in `~/.cargo/bin`, so `tools/security/run-dep-scan.sh` and `tools/cargo.sh` add it to `PATH` — the required scan now actually RUNS instead of failing closed on an unexported PATH. The full gate passes under a deliberately stripped `PATH="/usr/bin:/bin:/opt/homebrew/bin"`. A new regression pins the fail-closed diagnostic. Note: Python 3.12+ `unittest` exits 5 when discovery finds no tests (3.9 exited 0), so the gate fixture now carries a real test. |
| Keep the runner-hardening checklist from aborting early | ✅ | `tools/github-runner/harden-macos.sh` read the pinned channel out of `rust-toolchain.toml` with a bare `sed` pipe: under `set -euo pipefail` a missing/unreadable manifest aborted the whole checklist before its diagnostic, and the Disk step's `df` on a missing install dir aborted it again — so the script never printed its `RESULT:` summary. Both now degrade to a diagnostic/warn instead of aborting, and the script completes its checklist. Two regressions cover it: the pinned channel is read from a fixture manifest (and a floating `nightly` is never selected), and an unreadable manifest still reports and summarises. |
| Close the Python-probe fail-open in the gate | ✅ | The interpreter probe tested only "exit 0", which any program ignoring its `-c` argument satisfies (`/bin/echo`, `/bin/true`, a broken shim). `ABI_PYTHON=/bin/echo` therefore made the gate's Python policy tests and the Abbey Python oracle no-ops while the gate still reported "all green", and it disabled the compiler-substitution policy in `tools/cargo.sh` — the check that exists precisely to reject a substituted `RUSTC`. The probe now uses a distinguishing status (`sys.exit(3 if capable else 4)`, accept only 3) and the versioned-interpreter filter is pure bash, so a stripped `PATH` cannot silently drop every candidate. Verified by mutation: with the weak probe the gate wrongly exits 0 and the new regression fails; with the fix it exits non-zero. A second regression pins that a non-interpreter is rejected. The `HOME`-guarded scanner PATH and a self-contained test PATH (the old one was coupled to the host's `/usr/bin` Python layout) are also fixed. |
| Report the self-learning cycle that actually happened | ✅ | `abi complete --learn` printed `self_learn_cycle=persistence+adaptation+recall+evidence` unconditionally, including when the learning write was audit-rejected, disabled, or of unknown outcome — contradicting the `persisted=` and `wdbx_status=` fields on the same output. The field now derives from the persistence status: `persistence+adaptation+recall` only when committed, `adaptation-only` when skipped or disabled, `adaptation-only+write-unknown` when the outcome is unknown. A regression covers both the committed and skipped shapes. The metadata block moved to `crates/abi-cli/src/complete/report.rs` to stay inside the 1,000-line size gate (`complete.rs` 1011 → 945). Every metadata field stays on the first output line: the checked-in SEA smoke reads only that line, so it is the anti-spoofing boundary against markers repeated in generated response text. |
| Reconcile the agent-mirror trees with the real source layout | ✅ | The `.codex/agents/*.toml` tree was missed by the earlier pass and disagreed with its `.claude/agents/*.md` counterparts: six auditors still pointed at pre-move ABI-local substrate paths (`crates/abi-wdbx/`, `crates/abi-core/`, `crates/abi-foundation/`), one at `crates/sea/` plus a removed `feat-sea` gate, one at the deleted `crates/abi-cli/src/dashboard.rs`, and one claimed `CLI (13)`. Every affected agent is a read-only auditor that would have reported "source not found" on exactly the surface it exists to check. Corrected against the `.claude` counterparts. Three claims in the `.claude` tree were also false and were corrected there: a `log::warn` in `abi-sea` that does not exist (the crate has no `log` dependency and silently breaks to zero evidence — now flagged as a real observability gap), a `feat-os-control` build gate that does not exist, and the deleted `dashboard.rs` path. |
| Realign shipped output and docs with the pinned toolchain | ✅ | The CLI's own `abi backends` banner printed "Rust nightly" and a test pinned it, contradicting `rust-toolchain.toml` (`channel = "1.99.1"`) and the external-claims audit. It now prints "Rust stable 1.99.1"; the banner's unit and process tests were updated together. The same stale nightly claim was removed from the marketing site, `README.md`, `walkthrough.md`, `CLAUDE.md`, `CHANGELOG.md`, `abi-threat-model.md`, `ABI-MEGA-PLUGIN.md`, the runner docs, the agent/skill mirrors, and `build.sh -l`. `tools/github-runner/harden-macos.sh` no longer asserts a floating `nightly` toolchain — it reads the pinned channel from `rust-toolchain.toml`, so a correct host no longer fails the hardening check. |
| Reconcile frozen-surface and transport claims with source | ✅ | The frozen CLI is 14 commands (`improve` included), not 13 — corrected in the ledgers, site, analysis notes, golden README, and agent files. `docs/contracts/public-api.mdx` was missing the `abi wdbx episode` subcommand entirely. The site and `docs/README.md` had drifted: the site claimed the loopback listener was "not a persistent MCP SSE channel" while `crates/abi-mcp/src/{http,sse}.rs` ships persistent 2024-11-05 HTTP+SSE sessions contract-tested against stdio bytes, and `docs/README.md` claimed a `docs-validate` CI job that no workflow defines. `abi-model-runtime`'s `runtime_verified` now states in its message and in the claims audit that it carries the `abi-compute` meaning ("verified on the runtime that ran"), not the `abi-gpu` CPU-oracle-parity bar. `docs/spec/wdbx-north-star.mdx` no longer claims a `std::simd` path that the sibling removed in favour of stable lane-unrolling, and the Windows ACL row no longer says "compile-only" now that PR #794 runtime-verified it. |
| Central sync-clis regression coverage | ✅ | Eight isolated integration tests cover an unchanged second run, dry-run/apply agreement, support-entry repair and preservation, and native Codex/Claude/OpenCode agent destinations. Discovered by `./tools/check.sh`; missing personal driver/manifest installations explicitly skip. Current central source and runtime configuration are unchanged. |
| Keep the required dependency scan green against a yanked release | ✅ | The in-flight bulk dependency bump landed `tokio` on 1.54.0, which crates.io has yanked, so `cargo-audit` failed the required scan with `Warning: yanked` — the gate step that must RUN now failed on a release nobody can install. `tokio` is pinned back to the non-yanked 1.53.1 (a semver-compatible patch release; every requirement still resolves) and the required scan passes. The remaining 50 bumps are untouched. |
| Generate Bash/Zsh/Fish completions from live metadata | ✅ | Production generator uses `usage::COMMANDS`/`SHORTCUTS`; captured scripts remain independent byte-exact oracles and live byte comparisons pass. PowerShell and a new top-level flag remain frozen-surface changes, not implicit cleanup. |
| Wire the SEA eight-signal scorer into evidence recall | ✅ | Stable-ID deduplication, all current signals, task weights, deterministic budgets, 100-candidate public/defense-in-depth cap, indexed timestamps, provenance, raw routing input, prompt-byte cap, and scratch-store persona regression are covered. |
| Keep audit-rejected completions out of SEA learning | ✅ | The shared learn loop saves weights and completion evidence only after a passing, non-vetoed audit. SEA recall excludes explicit audit rejections while retaining their durable records; scratch-store write and recall/recency regressions, SEA unit tests, and accepted/vetoed CLI smokes pass. |
| Recover valid SEA evidence behind rejected hits | ✅ | A bounded retry widens sparse recall to the existing 100-candidate ceiling when rejected or missing-metadata hits exhaust the initial pool. A red-to-green scratch-store regression proves an eligible 21st hit survives 20 nearer audit rejections; all 31 SEA tests, the CLI learn-loop smoke, and `./tools/check.sh` pass. |
| Commit completion evidence as one WDBX transaction | ✅ | Sibling WDBX now commits the vector pair, metadata, and audit block in one causal transaction; an invalid second vector leaves no partial record. CLI local completion, MCP `ai_complete`, and SEA learning use it. Scratch tests verify one transaction per completion, vetoed learning is reported as skipped, and store errors are conservatively reported as outcome-unknown. WDBX and ABI full gates plus the SEA CLI smoke pass. |
| Commit training evidence as one WDBX transaction | ✅ | MCP `ai_train` and CLI `agent train` now use the existing atomic vector-pair/metadata/audit operation. Scratch tests verify one transaction on success and no partial vector or metadata after vector-width rejection. Both report a storage error as outcome-unknown; focused tests and `./tools/check.sh` pass. |
| Add TTY line editing and dashboard navigation | ✅ | Bounded Unicode-column-aware editor, history, Tab completion, single-stream output, Ctrl-C/D restoration, session-local SEA state, keyboard cycling, and bounded SGR mouse pane selection are covered. Capture enable/disable is guard-scoped; unit tests plus the dashboard and `tui` PTY drivers prove selection, exit, and cleanup. |
| Harden MCP transports found in code review | ✅ | 64 KiB physical-frame discard/recovery, absent-ID notification semantics, explicit-null validation, port-zero shutdown, hostile-Origin rejection, and exact HTTP 202/no-body behavior are covered without changing the frozen 12-tool catalog. |
| Bound MCP SSE pending responses | ✅ | Each session admits at most 32 queued/reserved responses; a full session returns `503` before dispatch. Notifications and failed sends release their slots. The MCP transport tests, warning-denied clippy, Mintlify validation, and `./tools/check.sh` pass. |
| Bound MCP HTTP request reads | ✅ | A 30-second absolute deadline ends trickling requests; the reader checks shutdown at most every 250 ms. Focused MCP tests, warning-denied clippy, Mintlify validation, and `./tools/check.sh` pass. |
| Keep MCP HTTP accept responsive under slow peers | ✅ | At most 16 request workers keep slow reads off the accept loop; overload gets `503` before dispatch, with a separate 25 ms rejection-read cap. Fair-service, saturation, and shutdown tests passed 20 consecutive concurrent runs; 80 MCP unit tests, warning-denied clippy, and `./tools/check.sh` passed. |
| Honor declared HTTP request body length | ✅ | The shared foundation reader now returns exactly the declared request, dropping coalesced surplus bytes. A red-to-green framing regression, MCP no-dispatch socket test, and REST no-mutation route test cover the boundary; WDBX and ABI full gates passed. |
| Reject ambiguous HTTP body framing | ✅ | The shared reader rejects duplicate/non-decimal `Content-Length`, unsupported `Transfer-Encoding`, and invalid UTF-8 headers as malformed. MCP, browser studio, and WDBX REST return 400 before dispatch/routing. Focused regressions plus WDBX and ABI full gates passed. |
| Bound REST and browser-studio request reads | ✅ | MCP's absolute-deadline reader now lives in the shared HTTP foundation. REST and studio cap each request read at 5 seconds; a partial open request gets 400 and the next request succeeds. MCP keeps its 30-second deadline and shutdown polling. Focused regressions, both full gates, and both loopback smoke drivers passed. The later bounded-worker change removes the sequential accept-loop wait. |
| Keep REST and studio accepts responsive under slow peers | ✅ | Both listeners use at most 16 connection workers and return 503 when saturated; slow partial requests no longer hold the accept loop. WDBX keeps store routing serialized on the server thread and one-shot APIs remain synchronous. Concurrent insert, overload, and stop regressions passed 20 repeats each; WDBX and ABI full gates and both loopback smokes passed. |
| Disclose agent-training persistence outcomes | ✅ | CLI `agent train` and MCP `ai_train` count a WDBX record only after its complete transaction succeeds. MCP reports observed store deltas with `records=0` on an unknown write outcome; no-store inspection does not count dataset lines as persisted, and CLI distinguishes a configured store-open failure. Scratch success/failure tests and the full ABI gate pass. |
| Distinguish completion store-open failure from no-store | ✅ | CLI local completion now reports `persisted=false` with a store-open-failed status when a configured WDBX path cannot open; `--learn` fails explicitly instead of silently falling back. Deliberately disabled persistence still uses the existing no-store completion. An obstructed scratch-path regression covers both states; focused lint and the full ABI gate passed. |
| Validate loopback HTTP Host and Origin before routing | ✅ | Shared exact loopback Host validation rejects hostile, duplicate, and malformed supplied authorities across REST, studio, and MCP; REST also rejects hostile/duplicate Origin before rate limiting or mutation. Studio retains required Host, while REST/MCP preserve Host-less native-client compatibility. Focused non-mutation/dispatch regressions, WDBX and ABI full gates, MCP transport tests, and both loopback smoke drivers passed. This is loopback hardening, not production web security. |
| Bound WDBX v3 episode ledger appends | ✅ | The append path checks the same 64 MiB whole-ledger ceiling enforced on reopen, independent of per-guild budgets. A scratch-store regression covers rejection without mutation and successful append/reopen after space is restored; WDBX and ABI full gates passed. |
| Keep WDBX forgetting tombstones out of live-memory edges | ✅ | A zero-payload `forgets` record remains auditable but cannot itself be forgotten, superseded, quarantined, or contradicted as a live memory. A red-to-green scratch regression and edge tests cover append and reopen; WDBX and ABI full gates pass. |
| Parse episode CLI endpoints before bearer-token dispatch | ✅ | A URL with loopback-looking userinfo and a remote host passed the old text-split check. The CLI now rejects userinfo and classifies the host from Tonic's parsed URI before loading the token or connecting. Red-to-green parser/ordering tests, live loopback/TLS/mTLS episode tests, and `./tools/check.sh` pass. |
| Parse connector HTTP destinations before loopback exemption | ✅ | `http://127.0.0.1:443@evil.com` passed the prior prefix check, but ureq would address the remote host with the configured Authorization header. Validation now uses the same `http::Uri` authority semantics as ureq and rejects userinfo for cleartext loopback. Red-to-green URL/request regressions, default and no-default connector tests, local connector smoke, and `./tools/check.sh` pass. |
| Keep Twilio WebSocket URL secrets out of parse errors | ✅ | Malformed stream URLs previously echoed the entire URL, including query tokens; an `@` in the path also falsely counted as authentication. URI parsing now emits secret-free errors, rejects userinfo, and recognizes only non-empty `token`/`AuthToken` query values. Red-to-green parser/auth tests, 107 connector tests, and `./tools/check.sh` pass; no remote Twilio call was made. |
| Encode supplied Twilio WebSocket auth as one query value | ✅ | The explicit `auth_token` previously entered the HTTP upgrade request target raw, so `&` changed query structure and CR/LF could inject a header line. The appended value now percent-encodes reserved, control, and UTF-8 bytes while preserving an existing URL token. Red-to-green request-framing regression, 108 connector tests, and `./tools/check.sh` pass; no remote Twilio call was made. |
| Bound live WebSocket server frames before buffering | ✅ | The shared decoder previously waited on arbitrary declared payload lengths and could overflow header-plus-length arithmetic. It now rejects payloads above 1 MiB from the frame header and uses checked length addition. Red-to-green decoder and `WsClient::read_text` regressions, 110 connector tests, and `./tools/check.sh` pass; no remote peer was used. |
| Require a verified WebSocket upgrade response | ✅ | The TLS client previously accepted a missing `Sec-WebSocket-Accept` when a peer supplied `X-Abi-Test-Peer`, and loose substring checks could mistake unrelated headers for upgrade fields. It now requires exact `Upgrade` and `Connection` tokens plus the computed accept; the process-local TLS peer sends a real accept. Red-to-green handshake regression, 111 default-feature and 100 no-default-feature connector tests, and `./tools/check.sh` pass; no remote peer was used. |
| Reject invalid live WebSocket frame headers | ✅ | Live TLS reads now reject masked server frames, reserved bits, unsupported fragmentation, and oversized control frames before waiting for payload bytes. The generic parser still decodes masked client frames for the local peer. Parser and live-reader regressions, connector tests, and `./tools/check.sh` pass; no remote peer was used. |

---

## Abbey runtime foundation train

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Provider-neutral agent runtime contracts | ✅ | `abi-agent-runtime` owns model requests/events, tool descriptions, policy, audit, usage, budgets, cancellation, bounded capture, and deterministic providers. `RunContext::checkpoint` gives blocking adapters a read-only cancellation/budget polling boundary without synthetic events or access to cancellation authority. Evidence: 52 unit + 11 external contract tests and warning-denied private-item rustdoc. It deliberately performs **no tool execution** and **no model inference**. |
| Hash-verified model registry contracts | ✅ | `abi-models` owns validated immutable manifests, artifact hashing, external storage policy, resumable-download state, license acceptance, and registry resolution. Its original unsigned loaders remain for explicit local/test provenance; authenticated delivery uses the signed-envelope APIs below. |
| Agent host orchestration | ✅ | `abi-agent-host` composes the existing provider, registry, policy, and audit contracts with an object-safe `ToolExecutor`. Startup compiles full JSON Schemas with external resolution disabled; calls are resolved and validated before audited authorization. The continuation loop rejects duplicate IDs, unknown tools, malformed/schema-invalid JSON, provider-fabricated results, post-terminal events, oversized events/results, and recursive calls beyond finite event/output/tool/deadline/round/run budgets. Evidence: 5 unit + 11 external contract tests, warning-denied clippy, and warning-denied private-item rustdoc. |
| Registry delivery security | ✅ | `abi-models` ships a blocking Rustls HTTPS range transport with redirects disabled, exact `206 Content-Range` validation, body/artifact size ceilings, immutable-revision URLs, hash verification, fsynced partials, and atomic no-clobber publication. Ed25519 envelopes verify exact manifest bytes against an explicit publisher trust store before parsing; content hashes remain mandatory. License records bind accepting principal, license digest, model/revision, and the ordered artifact path/hash set. Evidence: 71 unit + 4 external API tests, warning-denied clippy, and warning-denied private-item rustdoc. No weights, datasets, or adapters are stored here. |
| Local model execution | ✅ | `abi-model-runtime` connects exact principal-bound `abi-models` resolution to `abi-agent-runtime::ModelProvider` through Tokenizers, Safetensors, Candle, and `abi-compute` evidence. Callers select an exact model and device; no default, model/device substitution, or fallback exists. Generated scratch fixtures prove deterministic CPU inference and locally exercised Metal initialization plus native tensor operations. Model, tokenizer, raw-prompt, context, generation, runtime-event, and cancellation bounds apply before their respective allocations or emissions. Load and inference reports keep requested, initialized, executed, fallback, mixed-path, and runtime-verification evidence separate. The first architecture is only the tiny `abi-bigram-v1` offline fixture; Gemma, tool-call generation, model data, CUDA runtime execution, placement profiling, and performance remain unclaimed. Evidence: 14 default external contracts, 14 Metal-feature contracts on macOS, warning-denied clippy, and warning-denied private-item rustdoc. CUDA source is feature-gated through Candle but local compilation is unavailable without `nvcc`. |
| Authenticated worker plane | ✅ | `abi-worker` adds typed worker identity/capability, signed audience-bound job and cancellation envelopes, manager-derived principal-scoped idempotency digests, finite leases/deadlines, cooperative cancellation, bounded ordered result chunks, independently verifiable result digests, quotas, replay resistance, health, and append-only job-control reconstruction. Transport configuration reuses the gateway's owner-protected mandatory-mTLS loader while job-control storage remains independent of WDBX data-plane authority. Evidence: 1 unit + 25 external contract tests, warning-denied clippy, warning-denied private-item rustdoc, and the complete workspace gate. This foundation does **not** provide a network listener, scheduler, model executor, production certificate lifecycle, production cluster, or separate-host proof. |

> This train is additive. It must not change ABI's frozen 14-command CLI or
> 12-tool MCP catalogs. Model weights, datasets, and generated adapters stay
> outside this repository.

---

## Abbey System federation

| Item | Status | Acceptance boundary |
| ---- | ------ | ------------------- |
| Abbey v1 change-set and approval replay | ✅ | Local C2 for the PR #814 kernel slice: 17-event frozen replay in two fresh processes; golden decisions/transcript, 27 field mutations, expiry, regeneration, invalidated states, and dual control. Focused tests: 14 passed; ABI gates and Mintlify validation green on 2026-09-06. Evidence: `docs/superpowers/evidence/2026-09-06-abbey-v1-change-approval-c2.md`. Corpus stays data-only/C1 and CLI/MCP surfaces unchanged. Full federation C2, durable lifecycle/execution, and hosted CI remain unproven by this slice. |
| Federation reconciliation and P1 Abbey Contracts design | ✅ | Donald approved the C0 design on 2026-08-22. The constitutional P0-P7 slug map is fixed without Program 8; no production, deployment, Discord, or home-directory mutation is authorized by the document alone. |
| Program 1 Abbey contract source corpus | ✅ | ABI-local C1 source/contract evidence: 113 manifest-bound artifacts (133,200 bytes), aggregate digest `3ffd487bdc497b7ce54b8c29978a3686dcbffdb66a85957a0ee4f99ba576cdfd`, 89 Python repository/corpus behavior tests, independent Rust verification, four fail-closed temporary mutations, and the complete strict gate. Stable-Rust/Swift consumers, installed artifacts, hosted CI, runtime federation, durable WDBX behavior, authorization, production, and live Discord remain unperformed. |

---

## #647 Rust-rescoped (optional hardening)

See `docs/superpowers/archive/plans/2026-07-31-rust-647-followups.md`.

| Item | Status | Notes |
| ---- | ------ | ----- |
| Lock-across-I/O audit (REST/cluster/MCP) | ✅ | No lock held across TCP I/O; rate-limiter mutex is math-only |
| MCP malformed/empty bearer contracts | ✅ | `abi-mcp` HTTP tests cover empty/Basic/wrong-case/no-space |
| DurableStore concurrency regression test | ✅ | `DurableStore` holds a lifetime-scoped advisory writer lock; 50 concurrent opens return `WriterBusy`, drop releases the lock, and 50 real REST query → joined teardown → reopen/search lifecycles stay green. `open` also waits out a transient `WouldBlock` (250 ms budget, 1 ms steps) so the fork/exec window that duplicates the lock fd into a child cannot masquerade as contention; a genuinely held lock still reports `WriterBusy`. |
| Bench regression gate in `check.sh` | ✅ | `tools/bench_regress.sh`: live Rust HNSW insert/search workload, best p50 of 5, 25% local debug threshold; same OS/arch baseline required (other host classes disclose `SKIP`). Deterministic pass/fail hooks prove the comparator. |

---

## OS control (goal: "Improve OS control safety and flexibility")

| Item | Status | Notes |
| ---- | ------ | ----- |
| 30s execute timeout | ✅ | `exec_command_with_timeout`; stdout/stderr drained on threads so a chatty command cannot deadlock the pipe before the timeout |
| Env filtering | ✅ | `env_clear()` then re-add only non-`ABI_*` / `*SECRET*` / `*TOKEN*` / `*KEY*` / `*PASSWORD*` / `*CREDENTIAL*` |
| Dry-run accepts any command | ✅ | Read-only by design; emits `policy=allowed\|denied` + allowlist note. Execute stays gated |
| Configurable timeout | ✅ | `timeout_secs` in the policy file, bounded 1..=3600; default stays 30s |
| WDBX audit block for executed commands | ✅ | `os/audit.rs` appends vector + `os-cmd:<id>` KV + audit block per **executed** command (never dry-run), including killed timeouts (`timed_out=true`, exit 124). Store injected so tests use a scratch path. Intentional no-store and open/write failures are disclosed distinctly on the `[os-cmd]` line. |
| `~/.abi/os-policy.toml` | ✅ | `os/policy.rs`, strict TOML subset. **Narrow-only**: `allow` is intersected with the compiled `CEILING`, so the file can never grant a command the binary does not already permit. Unknown, duplicate, and malformed keys fail closed. Path overridable via `ABI_OS_POLICY`. |
| Scratch-path hardening for the os-control audit tests | ✅ | 2026-08-01: the two tests built their scratch dir from `{pid}-{thread_id:?}`; both now use `abi_foundation::temp_path::temp_file_path()` (PID + per-process counter), which cannot collide with anything this process made earlier. Hygiene, kept on its own merits. |
| CI flake: `os::audit`/`os` scratch-store `WriterBusy` | ✅ | Fixed by the writer-lock retry in [#772](https://github.com/donaldfilimon/abi/pull/772), **not** by the row above. Measured head-to-head, same harness (`os::` at `--test-threads 8`, 40 runs each): scratch-path hardening alone still failed **3/40**; the writer-lock retry alone and both together failed **0/40**. The "leftover dir from a panicked run / `ThreadId` reuse" theory is not reachable — a dead process cannot hold an `flock` (the kernel releases it at exit) and `scratch_store` already called `remove_dir_all` first. The real holder is live and transient: a `fork` duplicates the lock's fd into the child until `exec` closes it via O_CLOEXEC, which is why renaming the path cannot help. |

**Disclosed, not fixed:** the same ad hoc `temp_dir().join(format!("..{pid}-{thread:?}"))` scratch-path pattern (not `temp_file_path`) also appears in `nn.rs`, `complete.rs`, `wdbx_simulate.rs`, `wdbx/mod.rs`, and the sibling `../wdbx/crates/abi-wdbx/src/retrieval.rs`. None have been observed to flake; left untouched per "smallest verified slice" rather than swept into a broad rename.

---

## Docs hygiene

| Item | Status | Notes |
| ---- | ------ | ----- |
| Archive completed Zig-era `docs/superpowers/plans/*` | ✅ | Moved under `docs/superpowers/archive/plans/` with Archived banners |
| Executable dependency-security scan helper | ✅ | `tools/security/run-dep-scan.sh` runs installed `cargo-audit` or `cargo-deny`, otherwise emits an explicit SKIP; `ABI_DEP_SCAN_REQUIRE=1` makes missing tooling fail closed. The local RustSec scan has no vulnerability advisories after replacing the unmaintained PEM parser/server edge; two explicitly disclosed unmaintained transitive crates remain isolated behind optional TFHE-rs. |

---

## Disclosed residuals (do NOT fake-complete)

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| Native accelerator execution | ◑ | Metal dot/cosine/norm/batch-cosine paths are locally scoped and only count after initialization plus CPU-oracle verification. CoreML loads and executes an output-checked tiny model under a `.cpuAndNeuralEngine` request; placement and ANE residency remain unverified. CUDA/Vulkan adapters compile/report capability but have no verified runtime execution. |
| External shader / MLIR toolchains | ◑ | Validation / textual IR only |
| Mobile `native_dispatch` | ◑ | Simulated desktop profile |
| Production FHE / multi-host sharding | ◑ | DGHV educational refresh and optional TFHE-rs demos are reference-scoped and have no independent cryptographic audit. Exact replication/read repair/rebalance and `cluster local-demo` are single-host multi-process proof, not production multi-host deployment. |
| Full ggml/llama.cpp | ◑ | Demo GGUF container only (char-LM payload) |
| WDBX v2 causal multi-writer program | ◑ | The local product path now includes causal per-writer journals, verified migration with retained backup, authenticated transaction/segment objects, retained-generation rekey, confirmation-gated GC, exact committed-transaction export/import, accelerator-backed batch search with deterministic CPU parity, and exact single-host replica/read-repair/rebalance proof. A 50-process crash/compaction stress test recovers every reported commit and surfaces all recovered conflicts. Deterministic versioned PQ and persisted-autoencoder artifacts are integrated into segment codecs with validation and quality metrics. `abi-compute` supplies cycle-free accelerator contracts and five-state evidence. Additive credential-provider evidence covers the existing macOS Keychain and Windows protected-file paths plus a target-gated, in-process Linux Secret Service implementation. Linux source and tests cross-compile, but no Linux daemon round trip is claimed and the default auth backend is unchanged. The integrated gateway supplies bounded authenticated gRPC plus metadata-only WebSocket events with local TLS/mTLS runtime tests. Remaining boundaries: production separate-host deployment, hosted/Windows/Linux runtime proof, DAST, and independent crypto/security review. |

---

## Recently closed product slices

| Item | Status | Notes |
| ---- | ------ | ----- |
| Live Discord/Twilio TLS clients | ✅ | rustls `wss://` via `abi-connectors::tls_ws`; offline process-local TLS peer tests |
| Metal DOT kernels | ✅ | `metal-kernels` feature; `accelerated=true` when init succeeds; CPU oracle test |
| Windows credential ACL CI | ✅ | PR #794 ran on Windows Server 2025 on 2026-08-19: 2 ACL tests and 7 credential-file tests passed. The test found and pinned the missing protected-DACL flag plus Windows' `AI` bookkeeping form; caller docs now point to that runtime proof. |
| True incremental NN sampler | ✅ | `SampleState::step` + demo GGUF load/sample |

---

## Recently landed

- Loopback browser studio: `abi agent browser --studio` plus `abi_ai::multimodal` image/video/audio/voice/fuse processors. Claim-honest local analysis only; Web Speech STT/TTS stay in the page. Browser plan output now matches `embedded_browser=false`.
- Product residuals: Discord/Twilio TLS WS, Metal DOT kernels, Windows ACL CI, incremental NN + demo GGUF
- Aggressive residual teardown: deleted `modernized/`, `modern-refactor/`, `zig-pin`, `zig-newest-skills`, `zig-build-doctor`, Zig-only cross-compile skill; rewrote `abi` agents + lessons + docs hub for nightly Rust; freed local `zig-out`/`.zig-cache`
- Agent skill drivers ported off `zig-out`/`./build.sh cli` → `./tools/cargo.sh` + `target/debug/abi` (dashboard, backends, plugins, SEA scratch store, WDBX roundtrip, etc.)
- nn demo JSON checkpoint (`--out` / `--checkpoint`); Rust smoke scripts + goals/run-abi/mcp-smoke skills
- **Rust rewrite on `main`** via [#756](https://github.com/donaldfilimon/abi/pull/756) (`34c35d5`)
- FoundationModels Swift `@c` shim + `complete --live --model apple-fm --confirm`
- shaders / mlir / hash / metrics / mobile report surfaces
- local_bridge + MCP HTTP/SSE; wdbx_stats open-failure disclosure for CI
- Discord gateway/routing/WS framing (offline); Twilio ConversationRelay local path
- Zig one-shot teardown

---

## Residual reclassification against CSAPS (2026-08-22)

The seven `◑ Partial / disclosed` rows above were the only available definition
of "abi is not complete". Read against `CSAPS_WDBX_Revised_2026.pdf` — a
**proposed architecture**, not an obligation — several stop being gaps and
become the paper's own recommended posture. Section 9.1 states the design
principle plainly: *prove the architecture digitally first*, and treat
Loihi-class processors, physical reservoirs, and memristor arrays as **optional
research backends that must outperform matched digital emulation on an
end-to-end Pareto metric before adoption**. Section 2.5 says the same for
memristive substrates; section 2.3 says a physical reservoir "is an optimization
experiment, not an architectural dependency"; section 2.4 says "the baseline
architecture therefore requires no neuromorphic processor".

| Residual | Reclassified | Why |
| --- | --- | --- |
| Native accelerator execution (Metal verified against CPU oracle; CUDA/Vulkan compile-only; ANE residency unverified) | **Correct by design** | Section 9.1's stance exactly. Deterministic CPU parity is the acceptance test; unverified exotic placement is the honest state, and claiming it would violate R11's complete-system accounting. |
| External shader / MLIR toolchains, validation only | **Correct by design** | Same principle. No CSAPS requirement depends on textual IR being executed. |
| Mobile `native_dispatch` simulated on a desktop profile | **Correct by design, disclosed** | CSAPS-1 is explicitly a nonhazardous digital testbed on conventional programmable hardware. Mobile is not on its critical path. |
| Production FHE / multi-host sharding (DGHV educational, TFHE-rs demo, single-host multi-process cluster proof) | **Correct by design, with one real gap** | Reference-scoped crypto matches the paper's posture. The genuine gap is **independent cryptographic and security review**, which section 6.6's threat model needs and which no amount of local work supplies. |
| Full ggml/llama.cpp (demo GGUF container, char-LM payload) | **Out of scope, not a gap** | CSAPS never requires a production LLM. The deliberative path may call one; hosting it is not the substrate's job. |
| Local OpenAI bridge + MCP HTTP transport | **Closed since this assessment** | Persistent MCP 2024-11-05 HTTP+SSE sessions now deliver JSON-RPC responses as SSE `message` events; the stdio parity contract covers the transport. Streamable HTTP (2025-03-26) is not claimed. |
| WDBX v2 causal multi-writer program | **Genuine gap, large — and now measured** | See `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`. The row's own listed boundaries (separate-host deployment, hosted/Windows/Linux runtime proof, DAST, independent crypto review) remain accurate, and the analysis adds the substrate-contract gaps: canonical CBOR/COSE, sorted parent hashes, episode-level signing, regime and version fields, contradiction and quarantine edges, evidence-weighted retrieval, the selective write gate, and section 6.9 retention semantics. |

**Net effect at the 2026-08-22 assessment.** Of seven disclosed residuals,
four were the paper's recommended engineering posture rather than shortfalls,
one was out of scope, and two were implementation gaps. The MCP transport gap
has since closed; the WDBX conformance gap and external review boundaries
remain disclosed above.

**Not reclassified as done, and deliberately so:** independent cryptographic
review, independent security review, and DAST. Those cannot be self-certified.
They are `🔴 Blocked` on an external party, not `◑`.

### Aviva persona surface

Reviewed as part of the same pass. Aviva is one of the three routed personas in
`crates/abi-ai` (`soul.rs`, `identity.rs`, `router.rs`, `modulator.rs`,
`keywords.rs`), and `abi-ai` is **pure**: no WDBX dependency, no I/O, fully
deterministic, which is what makes `ai_run` byte-reproducible and
golden-testable. The extraction did not touch it, and it must stay that way —
under constitutional invariant A1 abi-ai owns persona routing and modulation
while Abbey owns persona definition, so introducing I/O or a substrate
dependency here would both break determinism and blur an authority boundary.
No change was required and none was made.

## GPU surface fix-and-improve train

- [x] Survey `crates/abi-gpu` (7 files, ~2,000 lines) and establish runtime ground truth from `abi backends` rather than prose.
- [x] Run the abi-gpu feature matrix: default, `--all-features`, `--no-default-features`, `+metal-kernels`, `+cuda-adapter,vulkan-adapter`. All five exit 0 at 20 passed / 0 failed.
- [x] Relabel the `abi backends` compatibility matrix (`usable=` → `compat … service_available=`) so accelerator rows stop reading as accelerator-presence claims. `./tools/check.sh` exit 0, `check: all green`.
- [x] Confirm `abi_compute::capabilities()` accelerator availability is a deliberate pinned contract and leave it unchanged.
- [x] Make the Metal-inactive degradation path testable on a Metal-capable host. `MetalAccelerator` now reaches kernels through a function-pointer seam (`MetalKernels::REAL` in production, a test-only inactive set otherwise). Mutation-checked: injecting `REAL` into the inactive constructor makes the new test fail at `!before.initialized()`. 21 tests pass in all five feature configurations; `./tools/check.sh` exit 0, `check: all green`.
- [x] Decide whether two capability tables (`abi-gpu` evidence ladder vs `abi-compute` compat table) should remain separate, or whether `abi backends` should print the ladder as `abi wdbx compute` already does. Cross-repo; Donald's call. **Decided 2026-09-16 (Donald): keep them separate.** `abi backends` no longer repeats the `compat` rows; it prints one pointer line to `abi wdbx compute info`, which already owned both the compat table and the `evidence` ladder (verified on the built binary before the change).
- [x] Commit the `abi-cli/src/backends.rs` relabel (landed as `ee435e9e`, on `origin/main`; 3 unrelated docs files in the tree belong to another session and are preserved outside the commit).


## 2026-10-03 training-foundation verification checklist

- [x] Preserve canonical HEAD/index and every prior change; establish coordinated sole source ownership and isolate build artifacts from live MCP processes.
- [x] Read all 29 initial changed paths and subsystem interactions; record confirmed F1/H1 findings and scoped episode review.
- [x] Observe F1's 1-pass/6-fail regression run, apply the retained-history guard and observe 7 passes.
- [x] Observe H1's 1-pass/3-fail regression run, apply cancellation-first callback admission checks and observe 4 passes.
- [x] Run the meaningful foundation suites: contracts 23, registry 39, artifacts 6, integrated foundation 2, complete host 29; all pass, none ignored.
- [x] Preserve actual baseline/failure/repair logs and source artifacts outside the shared tree; keep historical receipts intact.

Final source qualification is determined by the external `abi-final-result.json`:
actual gate exit 0, stable complete ABI/sibling input manifests and no blocking
independent review findings are required. The corresponding source receipt records
suite counts, ignored/skipped checks and remaining proof gaps. The unchecked old
foundation plan's initial-draft RED steps are historical proof gaps, not claims of
executed failures; the recorded execution ruling and confirmed repair RED/GREEN
apply to this current review.

Actual Abbey/Aviva/ABI/Quesar weights, real optimization, independent numerical
quality and teacher/student evaluation, licensed base/trainer selection, resource
commitments and serving/deployment remain outside this bounded source foundation.


## 2026-10-08 integration and publication verification

Donald authorized integration of pending source and worktrees into main and
origin/main, with launcher updates after validation. The full `./tools/check.sh`
gate passed twice on the pending ABI source, including the run with
`WDBX_REVISION=7ddeb3d1389ec0e9fab5ac5397acf0da91a01174`. The required sibling
passed `bash tools/check.sh` before that exact source revision was published.
This qualifies the retained training foundation, callback admission guards and
per-edge resolution projection at their source/local-test boundaries. CUDA
compilation remains unavailable without nvcc; independent audits, actual trained
weights, separate-host operation and live provider acceptance remain unproved.
Remote CI and installed-artifact verification are separate subsequent checks.

## 2026-10-20 self-learning CLI hardening

- [x] Regress and reject `complete --learn` requests that would silently dispatch
  live transport or a local-bridge model instead of learning. Keep the existing
  command catalog and normal local-learning path unchanged.
- [x] Require the learning smoke to prove persistence, router adaptation and
  cross-process evidence recall. Use unique scratch stores, override inherited
  persistence, lock builds to the executed target, and reject misleading markers
  inside generated response text. Eight isolated script regressions and the real
  two-process smoke pass (second-pass evidence_count=1).
- [x] Complete the final primary gate: `PATH="$HOME/.cargo/bin:$PATH"
  ./tools/check.sh` exited 0 (1,085 Rust executions, no failures/ignored;
  181 Python tests, 8 existing skips). CUDA compilation remains unavailable
  without nvcc; the sibling WDBX unused-import warning is unchanged. Integrate
  only this slice on canonical main, preserving existing host/runtime/gateway/
  ledger work. No remote action or model weight training is included.

- [x] (subagent follow-up) Small high-value self-learn CLI enhancements on main:
  - Removed silent `Ok(None)` fallback inside `run_learn` for `complete --learn`
    (now emits clear "requires a persistent WDBX store" error, exit 1, exactly like
    `improve`; prevents non-learning dispatch while preserving local complete path).
  - Learn fields now ride the FIRST metadata line with the rest of the metadata
    (`requested_model=... persisted=... learn=true evidence_count=N adapted=B
    self_learn_cycle=... recall_ready=...`). A dedicated learn-only first line was
    tried and reverted: the checked-in SEA smoke inspects only the first output
    line, so the split hid `provider=local`/`persisted=` from it and failed a
    healthy run with "missing metadata: provider=local". The first line is the
    anti-spoofing boundary — the generated response text may repeat arbitrary
    markers, so a later line must never be able to satisfy a persistence check.
    `learning_tests.rs::learn_markers_all_share_the_first_output_line` pins the
    shape the smoke depends on.
  - Additional test coverage: new `learning_without_persistent_store_emits_clear_error`
    in `learning_tests.rs`; extended `rejected_learning...` asserts for the learn=
    /evidence/adapted fields.
  - Verified: recent USAGE notes for --learn/improve (internal error strings + help),
    `cargo check -p abi-cli`, clippy clean on abi-cli, focused `test -p abi-cli`
    complete:: + improve (18/18 complete incl new, 7 improve), golden help, sample
    runs (evidence recall on 2nd pass, clear errors).
  - Only edited abi-cli self-learn paths; unrelated dirty files preserved; stayed on
    canonical main (no branch). Updated this ledger.
