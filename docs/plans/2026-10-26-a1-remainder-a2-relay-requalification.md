# A1 Remainder and A2 Relay Recovery Requalification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind ABI's full gate to recorded sibling revisions, reconcile C01–C44 into one execution ledger, and requalify durable execution / relay recovery (C03–C06) with current evidence plus the regression tests the existing suites lack.

**Architecture:** A1 is evidence work. It captures the consumed ABI/WDBX source set, runs `./tools/check.sh` in an isolated target, writes a tracked receipt and a 44-row ledger. A2 maps every A2 spec bullet to existing tests, reruns them, then adds four regression tests at the fault barriers that have none. One of them, node-certificate expiry after the authority callback, is a real defect with a one-closure fix. The other three are characterization tests: they are expected to pass, and if one fails it is reported as a defect. A final bound gate closes A2.

**Tech Stack:** Rust 1.99.1 stable (edition 2024) via `./tools/cargo.sh`; `abi-worker`, `abi-control-plane`, `abi-agent-host`; SQLite outbox with the test-only VFS fault injector (`crate::delivery_outbox::faults`); Python 3.11+ policy tests.

**Spec:** `docs/plans/2026-10-25-complete-abi-and-lylx-handoff.md` (sections A1 and A2), with `docs/verification/2026-10-23-completion-matrix.md` (C01–C44) and `docs/spec/encrypted-relay-recovery.md` (invariants 1–10).

Plan location: `docs/plans/` overrides the skill default `docs/superpowers/plans/`, per repository convention. Scope: the spec spans A1–A11 and Stage B. This plan is deliberately limited to the A1 remainder and A2, as two task groups. Nothing after A2 is covered here.

## Global Constraints

- Run Cargo only through `./tools/cargo.sh`. Never run bare `cargo` or `rustup run`.
- Toolchain: stable Rust 1.99.1 from `rust-toolchain.toml`. No nightly and no toolchain overrides.
- Every dependency-resolving Cargo stage takes `--locked`, and every `cargo test` ends with `< /dev/null`.
- Work on local `main` in the canonical checkout. Create no branches or worktrees. Never push. Never resume CI or change the CI `WDBX_REVISION` pin (`2969d71a2c3a81a060dcfc6cf53b4617f13c8b94`).
- End every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never edit `../wdbx` or `../abbey`, and never open `~/.abi/`. Gate runs use `ABI_WDBX_PATH=:memory:` with `ABI_WDBX_PERSIST` unset.
- The gate invocation is exactly: `PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/.work/<run>/target" env -u ABI_WDBX_PERSIST ABI_WDBX_PATH=:memory: ./tools/check.sh < /dev/null`.
- A gate log containing `dependency-scan: SKIP` is a failed gate, even if it exits 0.
- A moved WDBX invalidates gate evidence. If WDBX `HEAD`, or the status or tree of its consumed paths (`Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `crates`), differs between the pre-gate and post-gate captures, the receipt is invalid and the gate reruns.
- WDBX's own gate and the Abbey root/desktop gates are owner-held. Record them as external/blocked and never execute them.
- Never relabel external work as done. A C-row status changes only with a current evidence path dated in this run. Historical receipts (`/Users/donald/.codex/verification/...`, `/tmp/...`) never support a status change on their own.
- Every `.rs` file stays at or under 1000 lines (`tools/check_rust_sizes.sh`). New tests go in new files or child modules, not in near-limit files.
- Do not fabricate failing tests. A characterization test that passes on first run is recorded as such, with no source change.

## Review Focus

1. **WDBX moves during the gate.** A parallel source-pin-review writer is active in `../wdbx`; it was at `952bb66` with dirty docs when this plan was written. If the post-gate capture differs from the pre-gate capture in the consumed section, the receipt is invalid. Owner: Task 1, Step 6 (and Task 8, Step 2).
2. **The dependency scanner is missing from PATH.** The scan prints `SKIP` instead of running. Expect `dependency-scan: RUN cargo-audit` and no `SKIP` line. Owner: Task 1, Steps 1 and 7.
3. **A ledger row inherits a status from a historical receipt.** Any row whose status differs from the matrix must cite a 2026-10-26 evidence path. Owner: Task 2, Step 4.
4. **A new test pushes a file past 1000 lines.** Check with `bash tools/check_rust_sizes.sh` after each A2 task. Owner: Tasks 4–7, verification steps.
5. **A test hangs on the auth sign-in stdin read.** Every focused test command in Tasks 3–7 ends with `< /dev/null`. Owner: all A2 run steps.

---

## A1 — Bind the baseline and reconcile C01–C44

### Task 1: Consumed-source capture and bound full gate (C01, C02, C42)

**Files:**
- Create: `docs/verification/2026-10-26-a1-gate-receipt.md`
- Scratch (ignored): `.work/a1-20261026/{abi-pre,abi-post,wdbx-pre,wdbx-post,abbey-observed}.txt`, `check.log`, `exit.txt`

**Interfaces:**
- Consumes: none.
- Produces: the receipt path and the recorded ABI/WDBX revisions that Task 2 cites for C01/C02/C42. The capture procedure (Steps 2, 6–7) is reused verbatim by Task 8 with the run name `a2-20261026`.

- [ ] **Step 1: Preflight**

Run:
```bash
git status --short --branch
git check-ignore -q .work/a1-20261026 && echo ignored
test -x "$HOME/.cargo/bin/cargo-audit" && echo audit-present
mkdir -p .work/a1-20261026
```
Expected: `## main...` with no modified or untracked entries, `ignored` (`.work/` is ignored only through the user's global gitignore), and `audit-present`. If any line is missing, stop.

- [ ] **Step 2: Capture the pre-gate source set**

Run:
```bash
R=.work/a1-20261026
{ git rev-parse HEAD; git status --porcelain=v1 --untracked-files=all; } > $R/abi-pre.txt
{ echo "## consumed"; git -C ../wdbx rev-parse HEAD HEAD:crates HEAD:Cargo.lock HEAD:Cargo.toml HEAD:rust-toolchain.toml;
  git -C ../wdbx status --porcelain=v1 --untracked-files=all -- Cargo.toml Cargo.lock rust-toolchain.toml crates;
  echo "## other"; git -C ../wdbx status --porcelain=v1 --untracked-files=all; } > $R/wdbx-pre.txt
{ git -C ../abbey rev-parse HEAD; git -C ../abbey status --porcelain=v1 | head -50; } > $R/abbey-observed.txt
sed -n '/## consumed/,/## other/p' $R/wdbx-pre.txt
```
Expected: the consumed section lists five object IDs and no status lines. If consumed-path status lines appear, WDBX is mid-edit in consumed inputs: stop and wait for its writer, then re-run this step. Dirty paths under `## other` (docs only) are recorded, not blocking.

- [ ] **Step 3: Run the full gate (expect over 30 minutes; run in the background, with a timeout of at least 90 minutes)**

Run:
```bash
R=.work/a1-20261026
( PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$PWD/$R/target" env -u ABI_WDBX_PERSIST ABI_WDBX_PATH=:memory: ./tools/check.sh < /dev/null > $R/check.log 2>&1; echo "exit=$?" > $R/exit.txt )
cat $R/exit.txt
```
Expected: `exit=0`.

- [ ] **Step 4: If the gate failed, record it and stop**

If `exit.txt` is not `exit=0`, keep `check.log` and find the failing stage with `grep -n '==> ' $R/check.log | tail -3`. Write the receipt (Step 8) with status `FAILED at <stage>`, commit it (Step 10), and continue to Task 2 with C02 left unchanged. Do not repair source in this task.

- [ ] **Step 5: Extract the gate results**

Run:
```bash
R=.work/a1-20261026
grep -c 'check: all green' $R/check.log
grep -E '^dependency-scan: (RUN|SKIP|FAIL)' $R/check.log
grep -E 'benchmark regression gate: PASS' $R/check.log
grep -E '^test result:' $R/check.log | awk '{p+=$4; f+=$6; i+=$8} END {print "rust", p, f, i}'
grep -E '^(Ran [0-9]+ tests|OK|FAILED)' $R/check.log | head -4
shasum -a 256 $R/check.log
```
Expected:
- `1`
- `dependency-scan: RUN cargo-audit (...)`, with no `SKIP` or `FAIL` line
- `benchmark regression gate: PASS`
- `rust <N> 0 0`. Record N; the last recorded value was 1,241 at `aaf2929`.
- `Ran <M> tests`, then `OK (skipped=<K>)`. Record M and K; the last recorded values were 186 and 8.

- [ ] **Step 6: Capture the post-gate source set and compare**

Run Step 2's first two command groups again, writing `abi-post.txt` and `wdbx-post.txt`. Then run:
```bash
R=.work/a1-20261026
diff $R/abi-pre.txt $R/abi-post.txt && echo abi-stable
diff <(sed -n '/## consumed/,/## other/p' $R/wdbx-pre.txt) <(sed -n '/## consumed/,/## other/p' $R/wdbx-post.txt) && echo wdbx-stable
```
Expected: `abi-stable` and `wdbx-stable`. If either is missing, the evidence is invalid. Record the diff, discard the pass/fail verdict, and redo Steps 2–6 once. If it is unstable again, write the receipt as `INVALID: sibling moved` and stop A1's gate claim.

- [ ] **Step 7: Confirm the dependency scan actually ran**

Run: `grep -c 'dependency-scan: SKIP' .work/a1-20261026/check.log`
Expected: `0`.

- [ ] **Step 8: Write `docs/verification/2026-10-26-a1-gate-receipt.md`**

Required fields:
- ABI `HEAD`, and ABI status before and after (both clean).
- WDBX `HEAD`; the `crates`, `Cargo.lock`, `Cargo.toml` and `rust-toolchain.toml` object IDs; the consumed-path status (empty); and the verbatim `## other` dirty list.
- A statement that WDBX moved from the matrix-recorded `96c4b131faed45eadcaef215a31af53776282f81`, with the new revision, or that it is unchanged.
- The CI pin `2969d71`, stated as unchanged.
- Abbey revision and dirty count, labelled "observed only — owner gates not run".
- `rustc` and `cargo` versions from the log's first stage, plus `cargo-audit` version.
- The exact gate command, `exit`, Rust N/0/0, Python M with K skips, dependency-scan line, benchmark line, CUDA note (no `nvcc`), `check.log` SHA-256 and its local `.work` path.
- The verdict: "local source evidence for C02 bound to the recorded ABI/WDBX pair; not WDBX's own gate, not Abbey's gates, not a frozen release cohort".

- [ ] **Step 9: Run the docs checks**

Run:
```bash
python3 -m unittest discover -s tools/tests -p 'test_docs*.py'
git diff --check
```
Expected: `OK` (skips allowed) and no output.

- [ ] **Step 10: Commit**

```bash
git add docs/verification/2026-10-26-a1-gate-receipt.md
git commit -m "docs(verification): record A1 bound gate receipt

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: C01–C44 execution ledger

**Files:**
- Create: `docs/verification/2026-10-26-c01-c44-execution-ledger.md`
- Modify: `docs/verification/2026-10-23-completion-matrix.md`. Append one dated bullet to "Execution record" pointing to the ledger and the receipt. Do not change table rows in this task.
- Modify: `tasks/todo.md`. Add one new dated `[x]` entry under "Approved complete-ABI execution". Do not edit the existing `aaf2929` entry.

**Interfaces:**
- Consumes: Task 1's receipt path and revisions.
- Produces: the ledger table that Task 8 updates in rows C03–C06 only. The column order is fixed as `| ID | Requirement | Status | Evidence path | Owner | Next action | Local/External |`.

- [ ] **Step 1: Write the ledger header**

The header holds: authority (the matrix), the status legend copied verbatim from the matrix (Q/U/M/E), the bound source set (from Task 1), and these rules:
- A status differing from the matrix requires a 2026-10-26 evidence path.
- External rows never close locally.
- Ledger and matrix must agree, and the matrix table remains the requirement text.

- [ ] **Step 2: Write the 44 rows**

For each row:
- **Status:** copy it from the matrix, including its scope qualifier such as "Q local Mac".
- **Owner:** copy it from the matrix.
- **Evidence path:** the current repository path that proves the present status. Use a test file or source module for U/Q rows, the Task 1 receipt for C01, C02 and C42, and `none` for M rows. Historical external receipts may be listed as "historical:" but never as the sole support for a Q.
- **Local/External:** `External` for every E row and for these rows whose remaining evidence is owner- or environment-held:
  - C07 (Abbey `RuntimeStore` owner)
  - C11 and C12 (live accounts)
  - C13 (Linux/Windows backends)
  - C15, C16 and C17 (WDBX owner)
  - C18 (real issuer)
  - C33 (Abbey/WDBX installed journey)
  - C37 (WDBX restore)
  
  Every other row is `Local`.
- **Next action:** one imperative line naming the spec stage (A2…A11) that owns it. For C03–C06 it reads "A2 requalification (this plan, Tasks 3–8)".
- **C42:** status `E`, next action "WDBX own gate + Abbey root/desktop gates by owners against WDBX `<Task 1 rev>`; pin promotion after exact-rev pass".

- [ ] **Step 3: Record the owner-held gates as an external/blocked step (do not execute)**

Add a "Blocked external inputs" section with two checkboxes, left unchecked:
- `[ ] WDBX owner: own full gate at <Task 1 WDBX rev> — receipt path pending`
- `[ ] Abbey owner: root and desktop gates against <Task 1 WDBX rev> and Abbey <observed rev> — receipt path pending`

Note that A1's exit criterion ("ABI's gate, WDBX's own gate and Abbey root/desktop gates") remains unmet until both receipts exist.

- [ ] **Step 4: Verify the row count and status parity**

Run:
```bash
L=docs/verification/2026-10-26-c01-c44-execution-ledger.md
grep -cE '^\| C[0-9]{2} \|' $L
diff <(grep -E '^\| C[0-9]{2} \|' docs/verification/2026-10-23-completion-matrix.md | awk -F'|' '{gsub(/ /,"",$2); gsub(/^ +| +$/,"",$5); print $2, $5}') \
     <(grep -E '^\| C[0-9]{2} \|' $L | awk -F'|' '{gsub(/ /,"",$2); gsub(/^ +| +$/,"",$4); print $2, $4}')
```
Expected: `44`, and no diff output. Task 2 changes no status. Any difference is an inherited relabel; fix it before committing.

- [ ] **Step 5: Run the docs checks**

Run `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'` and `git diff --check`.
Expected: `OK` and no output. User-run step: `.agents/skills/docs-validate/validate.sh` is user-invocable only (network `npx`). List it as pending in the ledger unless the user runs it and supplies the `RESULT:` line.

- [ ] **Step 6: Commit**

```bash
git add docs/verification/2026-10-26-c01-c44-execution-ledger.md docs/verification/2026-10-23-completion-matrix.md tasks/todo.md
git commit -m "docs(verification): add C01-C44 execution ledger

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## A2 — Durable execution and relay recovery requalification (C03–C06)

### Task 3: A2 coverage map and focused requalification run

**Files:**
- Create: `docs/verification/2026-10-26-a2-relay-requalification.md`

**Interfaces:**
- Consumes: none.
- Produces: the coverage table (`| A2 bullet | Existing tests | New test (Task) | Result |`) that Tasks 4–7 append to and Task 8 cites from ledger rows C03–C06.

- [ ] **Step 1: Write the coverage map**

Map each bullet to the existing tests found at plan time:

| A2 bullet | Existing tests (file) |
| --- | --- |
| Short/zero writes, sync failure | `short_writes_complete_the_entire_frame_before_success`, `zero_write_never_acknowledges_or_reuses_the_handle`, `failed_sync_never_acknowledges_even_when_complete_frame_survives`, `partial_frame_failure_poisoning_refuses_reuse_and_reopen` (`abi-worker/src/file_store/fault_tests.rs`); `actual_sqlite_full_partial_write_and_sync_failures_poison` (`delivery_outbox/tests.rs`) |
| Corrupt/truncated records | `corrupted_and_partial_frames_refuse_recovery`, `external_truncation_poisons_handle_and_blocks_further_appends`, `reordered_valid_frames_and_hostile_lengths_refuse` (`durable_control.rs`); `malformed_control_history_fails_reconstruction`, `tampered_result_digest_in_control_history_fails_reconstruction` (`contracts.rs`); `modified_packet_history_refuses_instead_of_resetting` (`delivery_outbox/tests.rs`) |
| Failed terminal persistence | `persisted_terminal_with_lost_ack_reopens_as_cached_without_execution`, `missing_terminal_reopens_as_unresolved_without_execution` (`terminal_durability.rs`); outbox-side gap → Task 7 |
| Process kill | `process_exit_after_admission_preserves_non_dispatching_retry` (`durable_control.rs`); `reconnect_process_terminal_survives_worker_and_authority_kills`, `reconnect_process_kills_do_not_resume_prepared_admitted_or_partial_generation` (`service/tests/reconnect_process.rs`); `parent_kill_import_transaction_preserves_complete_old_or_new_view` (`delivery_consumer/interruption_tests.rs`) |
| Lost responses | `reconnect_process_http_replays_after_execution_expiry_with_current_credentials`, `reconnect_process_terminal_survives_worker_and_authority_kills`; `restart_replays_identical_packets_after_execution_deadline_and_prior_ack` (`relay/tests/recovery.rs`) |
| Duplicate delivery | `signed_conflicting_duplicate_and_foreign_binding_preserve_journal`; `streaming_frames_are_ordered_private_idempotent_and_do_not_resume_after_restart`; `completed_ciphertext_retry_does_not_charge_the_memory_quota_twice` |
| Exhausted quotas | `active_principal_capability_and_output_quotas_are_enforced`, `replay_revalidates_active_concurrency_and_terminal_capacity`, `admission_reserves_control_capacity_for_a_terminal_event` (`contracts.rs`); `quotas_enforce_two_active_and_ten_queued_per_account`; `replay_capacity_refusal_preserves_commitments_and_closed_manifest`; `reconnect_process_corruption_lock_and_reservation_failure_refuse_before_inference`; `twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded` |
| Recheck cancellation/revocation/time/lease before dispatch, incl. after callbacks | `elapsed_lease_before_dispatch_never_invokes_provider`, `authority_callback_consuming_lease_never_dispatches_or_readmits`, `authority_callback_with_remaining_lease_executes_once`, `pre_cancelled_observed_inference_never_calls_provider` (`inference.rs`); `authority_permit_must_still_be_fresh_after_durable_admission`; `governed_tool_tenant_mutation_and_fresh_revocation_refuse_effects`; gaps → Tasks 4, 5, 6 |
| Exact packets and recorded results across reopen | `exclusive_create_reopen_retains_exact_packets_and_expiry_tombstone`; `file_reopen_preserves_completed_and_unresolved_evidence`; `committed_inference_is_cached_after_real_store_reopen`; `cancellation_survives_restart_and_blocks_success_replay` |
| Uncertain effect stays unresolved | `replayed_admission_can_be_abandoned_once_without_readmission`, `unresolved_history_rejects_contradictory_terminal_events` (`recovery.rs`); `direct_execution_after_uncertainty_refuses_before_prepare_or_provider`; `reconnect_process_saved_completion_refuses_missing_or_admitted_control_history` |
| Admission ≠ redelivery authority | `replay_grant_round_trip_cannot_become_execution_permit_and_expires_exclusively` (`delivery_types/tests.rs`); `completed_lookup_after_deadline_is_read_only` (`recovery.rs`) |

Record this known, design-bounded limit: on the inference path, revocation reaches a node only through the five-second `DispatchPermit` (`relay.rs`, `expires_at: (issued_at + 5)`) and, after Task 4, through local certificate expiry. Live revocation inside an already-issued permit window is not observable on the node. The plan does not claim it.

- [ ] **Step 2: Run the focused suites**

Run:
```bash
./tools/cargo.sh test --locked -p abi-worker < /dev/null 2>&1 | tee .work/a1-20261026/a2-worker.log | grep -E '^test result:'
./tools/cargo.sh test --locked -p abi-control-plane --lib < /dev/null 2>&1 | tee .work/a1-20261026/a2-control-plane.log | grep -E '^test result:'
node crates/abi-control-plane/web/tests/recovery.cjs
```
Expected: every `test result:` line reads `ok. N passed; 0 failed; 0 ignored`. Record each N. The node script exits 0. Confirm that every test named in the map appears in the logs as `... ok`:
```bash
for t in short_writes_complete_the_entire_frame_before_success reconnect_process_terminal_survives_worker_and_authority_kills replay_grant_round_trip_cannot_become_execution_permit_and_expires_exclusively; do grep -h "$t ... ok" .work/a1-20261026/a2-*.log >/dev/null && echo "$t ok"; done
```
Extend the loop to every name in the map. Expected: one `ok` line per name.

- [ ] **Step 3: Commit the map with results**

Run `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'` and `git diff --check`; expect `OK` and no output. Then:
```bash
git add docs/verification/2026-10-26-a2-relay-requalification.md
git commit -m "docs(verification): map A2 recovery bullets to current tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Relay inference rechecks node certificate after the authority callback (C04, C06 — defect)

The defect, read from source at `63d2bc0`: `RelayRuntime::infer_dispatch` (`crates/abi-control-plane/src/relay_runtime.rs`) verifies the node certificate once, at the first clock read. The authority callback passed to `WorkerManager::infer_observed` then re-verifies only the `DispatchPermit`. A permit can outlive the certificate by up to five seconds, because `relay.rs` does not clamp permit expiry to certificate expiry. So a node whose credential expires between admission and dispatch still invokes the provider.

**Files:**
- Create: `crates/abi-control-plane/src/relay/tests/dispatch_recheck.rs`
- Modify: `crates/abi-control-plane/src/relay/tests.rs`. Extract `submission_at(f: &Fixture, account: &Account, node: &NodeCertificate, submitted_at: i64, expires_at: i64) -> Submission`, keep `submission(..)` delegating with `(100, 200)`, and add `mod dispatch_recheck;` beside `mod recovery;`.
- Modify: `crates/abi-control-plane/src/relay_runtime.rs`, the authority closure inside `infer_dispatch`.

**Interfaces:**
- Consumes: the existing test helpers `fixture()`, `poll(&mut Fixture, bool, i64) -> Dispatch`, `permit(&mut Fixture, bool, &str, i64) -> DispatchPermit` and `CountProvider` from `relay/tests.rs`.
- Produces: `submission_at` (as above). The runtime contract becomes: "the authority closure refuses unless both the permit and the node certificate verify at the callback's trusted time".

- [ ] **Step 1: Write the failing test**

```rust
//! Dispatch rechecks node credential validity after the authority callback.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

#[test]
fn node_certificate_expiring_during_authority_recheck_never_dispatches() {
    let mut f = fixture();
    assert_eq!(f.node_a.expires_at, 3700); // enrolled at 100, exclusive expiry
    let s = submission_at(&f, &f.alice, &f.node_a, 3695, 3800);
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 3696).unwrap();
    let dispatch = poll(&mut f, true, 3697);
    let permit = permit(&mut f, true, &id, 3698);
    assert!(permit.expires_at > f.node_a.expires_at); // permit outlives the certificate
    let mut runtime = crate::RelayRuntime::new(
        f.node_a.clone(), f.key_a.clone(), RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(), "local-echo-v1", abi_worker::MemoryControlStore::new(), 3698,
    ).unwrap();
    let provider = CountProvider(AtomicUsize::new(0));
    // Clock reads: pre-check, pre-admission, admitted_at, started_at (authority time), dispatch_at, completed_at.
    let mut ticks = [3698, 3698, 3698, 3700, 3700, 3700].into_iter();
    assert!(runtime.execute(&dispatch, &permit, &provider, || ticks.next().unwrap_or(3700)).is_err());
    assert_eq!(provider.0.load(SeqCst), 0);
    // The durable admission stays unresolved; an exact retry never dispatches.
    assert!(matches!(runtime.execute(&dispatch, &permit, &provider, || 3699), Ok(None)));
    assert_eq!(provider.0.load(SeqCst), 0);
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `./tools/cargo.sh test --locked -p abi-control-plane --lib -- relay::tests::dispatch_recheck < /dev/null`
Expected: FAIL at `assert!(runtime.execute(...).is_err())`. The provider ran once and `execute` returned `Ok(Some(_))`. A panic before the first `execute`, for example `poll(..)` returning no dispatch or `permit(..)` returning no permit, means the fixture time window is wrong, not that the defect is absent. Fix the fixture times and rerun. If the test passes instead, the defect analysis is wrong: record it as characterization and skip Step 3.

- [ ] **Step 3: Recheck the certificate inside the authority closure**

In `infer_dispatch`, change the `|time| { ... }` closure so that, with `let seconds = time / 1000;`, it requires both `permit.verify(&self.authority_key, seconds)` and `self.certificate.verify(&self.authority_key, &b.audience, seconds)`. It maps either failure to `abi_worker::WorkerError::Authentication("dispatch permit or node certificate no longer valid".into())`.

Disjoint field captures (`self.authority_key`, `self.certificate`) coexist with the `&mut self.manager` borrow. Do not reorder the existing pre-admission checks.

- [ ] **Step 4: Run the test and the full library suite**

Run:
```bash
./tools/cargo.sh test --locked -p abi-control-plane --lib -- relay::tests::dispatch_recheck < /dev/null
./tools/cargo.sh test --locked -p abi-control-plane --lib < /dev/null 2>&1 | grep -E '^test result:'
```
Expected: `1 passed`, then `ok. N passed; 0 failed; 0 ignored`, with N one higher than Task 3's control-plane count.

- [ ] **Step 5: Run lint and the size check**

Run:
```bash
./tools/cargo.sh clippy --locked -p abi-control-plane --all-targets -- -D warnings
./tools/cargo.sh fmt --all -- --check
bash tools/check_rust_sizes.sh
```
Expected: all three exit 0.

- [ ] **Step 6: Record the result and commit**

Append the row "Recheck after callback: certificate expiry | — | `node_certificate_expiring_during_authority_recheck_never_dispatches` (Task 4) | RED→GREEN" to the Task 3 map.

```bash
git add crates/abi-control-plane/src/relay/tests.rs crates/abi-control-plane/src/relay/tests/dispatch_recheck.rs crates/abi-control-plane/src/relay_runtime.rs docs/verification/2026-10-26-a2-relay-requalification.md
git commit -m "fix(relay): recheck node certificate after dispatch authority callback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Worker authority-callback refusal and cancellation (C04 — characterization)

**Files:**
- Create: `crates/abi-worker/tests/authority_recheck.rs` (with `#[allow(dead_code)] mod support;` as in `inference.rs`)

**Interfaces:**
- Consumes: `support::{NOW, envelope, manager}`; `WorkerManager::{infer, infer_with_authority, infer_observed, recorded}`; `abi_worker::{CancellationToken, InferenceOutcome, JobStatus, RecordedExecution, WorkerError}`.
- Produces: none. These tests are test-only.

- [ ] **Step 1: Write the tests**

Use a local `CountingProvider(AtomicUsize)` wrapping `EchoProvider` (same shape as `inference.rs`).

- `authority_refusal_leaves_admission_unresolved_and_never_dispatches`:
  - `infer_with_authority(&req, &p, || NOW, |_| Err(WorkerError::Authentication("revoked".into())))` matches `Err(WorkerError::Authentication(_))`.
  - The provider count is 0.
  - `worker.recorded(&req)` matches `Ok(RecordedExecution::Admitted(_))`.
  - A retry with `|_| panic!("existing admission must not request fresh authority")` returns `InferenceOutcome::InProgress(_)`.
  - The provider count is still 0.
- `cancellation_during_authority_callback_never_dispatches_and_records_failure`:
  - `let cancel = CancellationToken::new();` then `infer_observed(&req, &p, || NOW, |_| { cancel.cancel(); Ok(()) }, &cancel, |_| Ok(()))` returns `InferenceOutcome::Completed(r)`.
  - `r.status == JobStatus::Failed` and `r.chunks.is_empty()`.
  - The provider count is 0.
  - `worker.infer(&req, &p, || NOW)` returns `InferenceOutcome::Cached(r2)` with `r2 == r`.
  - The provider count is still 0.

- [ ] **Step 2: Run them**

Run: `./tools/cargo.sh test --locked -p abi-worker --test authority_recheck < /dev/null`
Expected: `2 passed; 0 failed`. This is a characterization result: `inference.rs` propagates `authority(started_at)?` before dispatch and gates `run_provider` on `!cancel.is_cancelled()`. If either test FAILS, do not change the assertions. Record the failure output in the A2 map as a C04 defect and stop A2 for a separate repair plan.

- [ ] **Step 3: Run lint and the size check, then commit**

Run `./tools/cargo.sh clippy --locked -p abi-worker --all-targets -- -D warnings` and `bash tools/check_rust_sizes.sh`; expect exit 0. Append the map rows (Result: "characterization PASS").

```bash
git add crates/abi-worker/tests/authority_recheck.rs docs/verification/2026-10-26-a2-relay-requalification.md
git commit -m "test(worker): pin authority-callback refusal and cancellation before dispatch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Governed adapter cancellation during the final authority refresh (C06/C08 boundary)

**Files:**
- Create: `crates/abi-control-plane/src/service/tests/governed_tools/cancel_recheck.rs`
- Modify: `crates/abi-control-plane/src/service/tests/governed_tools.rs`. Add `mod cancel_recheck;` after `mod support;`. A child module may use the parent's private `case`, `governance`, `refresh`, `FileAdapter`, `Facts` and `Local`.
- Modify (only if Step 2 fails): `crates/abi-control-plane/src/relay_runtime/tools.rs` (`FreshFacts`, `execute_governed`).

**Interfaces:**
- Consumes: `DurableRelayRuntime::execute_governed(&mut self, &Dispatch, &SignedDeliveryPolicy, &mut GovernedToolContext<'_>, &CancellationToken) -> Result<GovernedToolOutcome>`; the fixtures as used by `governed_tool_pending_approval_then_creates_scratch_file_once`.
- Produces: if repaired, `FreshFacts` gains the field `cancel: &'a CancellationToken`.

- [ ] **Step 1: Write the test `governed_cancellation_during_final_authority_refresh_never_reaches_adapter`**

Use a local helper `async fn approved(cancel_at: Option<usize>) -> (crate::Result<GovernedToolOutcome>, usize /*effects*/, usize /*refresh calls in approved call*/, crate::Result<GovernedToolOutcome> /*retry*/)`. It:
- builds a fresh `fixture().await`;
- reaches `Pending` and records the human approval exactly as the existing approval test does;
- then wraps `refresh(&f, &c)` in a counting closure that calls `token.cancel()` inside call number `cancel_at` before returning the permit;
- invokes `execute_governed` with that `token`;
- then invokes it once more with a fresh `CancellationToken::new()` as the retry.

Assertions:
- Calibration, `approved(None)`: the outcome is `Recorded(r)` with `r.status == JobStatus::Succeeded`, effects == 1, and `k = calls ≥ 1`.
- Cancelled, `approved(Some(k))`, cancelling inside the last refresh before the effect: effects == 0; the outcome is `Err(_)`, `Ok(Unresolved)` or `Ok(Recorded(r))` with `r.status != Succeeded`; and the retry is not `Recorded` with status `Succeeded`, with effects still 0.

> **Ruling (2026-10-27):** cancel point moved to last pre-adapter refresh; call-8 cancel is post-effect by design — cost if wrong: one reworked test.
> Refresh k (call 8 of 8 in the approved call) is the post-effect observation (`abi-agent-host` `governed.rs`, after `dispatch.execute`). Cancelling there correctly yields effects == 1 and `Unresolved`.
> The corrected test does two things. First, it locates the last pre-adapter refresh at runtime: the `facts.observe` in the second `live_authority_valid` after stage 12, which is the last refresh before the effect file exists. It asserts effects == 0 and a non-success outcome there. Second, it pins the post-effect cancel: effects == 1, `Ok(Unresolved)`, and no successful retry.
> Step 3 is withdrawn. Applying it did not change the post-effect case and regressed an earlier cancel point. The corrected test passes on unchanged source as a characterization test.

- [ ] **Step 2: Run it**

Run: `./tools/cargo.sh test --locked -p abi-control-plane --lib -- governed_tools::cancel_recheck < /dev/null`
Expected: either PASS or FAIL with effects == 1.
- PASS means `abi-agent-host`'s stage-12 guard already observes `cancel` after the facts callback. Record it as characterization and go to Step 4.
- FAIL means a cancellation raised inside the authority refresh reached the adapter. That is a defect; go to Step 3.

- [ ] **Step 3 (only on FAIL): Refuse facts after cancellation**

Add `cancel: &'a CancellationToken` to `FreshFacts`, and initialize it from `execute_governed`'s `cancel`. In `FreshFacts::observe`, after the refreshed permit verifies and before `self.inner.observe(request)`, return `Err(LifecycleError::Unavailable)` when `self.cancel.is_cancelled()`. Re-run Step 2; expect PASS.

- [ ] **Step 4: Run the suite, lint and the size check**

Run:
```bash
./tools/cargo.sh test --locked -p abi-control-plane --lib -- governed_tools < /dev/null
./tools/cargo.sh clippy --locked -p abi-control-plane --all-targets -- -D warnings
bash tools/check_rust_sizes.sh
```
Expected: `0 failed`, and the other two commands exit 0.

- [ ] **Step 5: Record and commit**

Append the map row with Result "characterization PASS" or "RED→GREEN".

```bash
git add crates/abi-control-plane/src/service/tests/governed_tools.rs crates/abi-control-plane/src/service/tests/governed_tools/cancel_recheck.rs docs/verification/2026-10-26-a2-relay-requalification.md
git add -u crates/abi-control-plane/src/relay_runtime/tools.rs
git commit -m "test(governed): cancellation inside final authority refresh never reaches adapter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
If Step 3 ran, use the subject `fix(governed): refuse facts after cancellation in authority refresh`.

### Task 7: Outbox terminal-persistence failure and poisoned-handle guard (C06 — characterization)

**Files:**
- Modify: `crates/abi-control-plane/src/service/tests/direct_recovery_guard.rs`. Add one test, reusing its private `dispatch(..)` helper and `Count` provider; the file stays near 230 lines.

**Interfaces:**
- Consumes:
  - `crate::delivery_outbox::faults::{Guard, Fault::{Full, PartialWrite, Sync}}`; `Guard::new(&Path)` must precede `FileRelayOutbox::create`, and `guard.arm(fault)` fails the next targeted write.
  - `FileRelayOutbox::{create, open}(&Path, OutboxLimits)`, `poisoned()`, `entry(&str)`.
  - `FileControlStore::open(&Path, FileControlStoreLimits)`.
  - `RelayRuntime::new(cert, key, RelayKey, [u8; 32], &str, store, i64)`.
  - `DurableRelayRuntime::{new, execute_durable, reconcile_local}`.
  - `OutboxPacket::{Delta, Terminal}`, `OutboxPacket::id(&JobBinding)`, `RecordedOutcome::Completed`.
  - `enroll(&f, &browser, 116)`, whose encryption key is `RelayKey::from_seed(&[117; 32])`.
- Produces: none.

- [ ] **Step 1: Write the test `terminal_outbox_failure_reopens_with_one_terminal_and_refuses_fresh_dispatch`**

For each `fault` in `[Full, PartialWrite, Sync]`, using a fresh fixture, browser and `enroll(.., 116)`, with `time = now()`, outbox `terminal.sqlite` and worker log `terminal.log`:

1. Create `Guard::new(&outbox_path)`, then the `DurableRelayRuntime` from `runtime(&f, &cert, &key, encryption, "terminal.log")` and `FileRelayOutbox::create`.
2. Call `execute_durable(&d, &policy, &permit, &provider, clock, &cancel, |_| Ok(()))`. The clock closure returns `time` and calls `guard.arm(fault)` exactly once, on its first read after `provider.0 == 1`, so the next outbox write is the terminal append in `finish_local`.
   - Assert the result is `Err`, `provider == 1` and `durable.outbox.poisoned()`.
3. Build a second fresh dispatch with `dispatch(&f, &cert, time, time)`. `execute_durable` on the same runtime returns `Err`, and `provider` stays 1.
4. Drop the runtime and the guard. Reopen with `FileControlStore::open`, `RelayRuntime::new(.., RelayKey::from_seed(&[117; 32]), ..)` and `FileRelayOutbox::open`. Call `DurableRelayRuntime::new(..)` and `reconcile_local(time)`; both are `Ok`. The first job's entry then satisfies:
   - exactly one `OutboxPacket::Terminal`;
   - `manifest.outcome == RecordedOutcome::Completed`;
   - `provider == 1`.
   
   Collect `ids1 = packets.iter().map(|p| p.id(binding))`.
5. Drop and reopen again, then call `reconcile_local(time)`. Assert `ids2 == ids1`: no resealing and no new terminal. `provider` stays 1.

- [ ] **Step 2: Run it**

Run: `./tools/cargo.sh test --locked -p abi-control-plane --lib -- direct_recovery_guard < /dev/null`
Expected: `2 passed`. This is a characterization result: `finish_local` seals a missing terminal from `RecordedExecution::Completed` exactly once, and `PrivateDb::check` refuses a poisoned handle. If it FAILS, keep the test unchanged, record the failing assertion and fault in the A2 map as a C06 defect, and stop A2 for a separate repair plan. Do not improvise a storage change.

- [ ] **Step 3: Run lint and the size check, then commit**

Run `./tools/cargo.sh clippy --locked -p abi-control-plane --all-targets -- -D warnings` and `bash tools/check_rust_sizes.sh`; expect exit 0. Append the map row.

```bash
git add crates/abi-control-plane/src/service/tests/direct_recovery_guard.rs docs/verification/2026-10-26-a2-relay-requalification.md
git commit -m "test(relay): pin outbox terminal-persistence failure recovery without re-execution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7b: Governed tool path rechecks node certificate before each effect (C06/C08 — defect, added by ruling)

> **Ruling (2026-10-27):** added from independent review I1; same defect class as Task 4 on the tools path — cost if wrong: one extra recheck per effect.

`execute_governed` (`crates/abi-control-plane/src/relay_runtime/tools.rs`) verified the node certificate once, in `validate_tool`, before preview and local approval. Each pre-effect `FreshFacts::observe` verified only the refreshed `DispatchPermit`. A permit issued just before certificate expiry stays valid for up to 5 s, so an approved tool effect could run with an expired node credential.

**Files:**
- Create: `crates/abi-control-plane/src/service/tests/governed_tools/cert_recheck.rs`
- Modify: `crates/abi-control-plane/src/service/tests/governed_tools.rs`. Add `case_timed` (choose the submission time from the certificate) and `refresh_at` (permit issue time); `case_for` and `refresh` delegate.
- Modify: `crates/abi-control-plane/src/relay_runtime/tools.rs`. `FreshFacts` carries the node certificate, and `observe` requires both the permit and the certificate to verify at the same trusted second.

**Test:** `node_certificate_expiring_after_approval_never_reaches_tool_adapter`. The job is submitted at `E − 2`, where `E` is the certificate expiry. The local approval is decided at `E − 0.4 s`, and refreshed permits are issued at `E − 1` (valid to `E + 4`).
- Calibration: with no expiry, the effect runs once.
- The clock moves to `E` inside refresh 1: effects 0 and `Err`.
- The clock moves to `E` inside the last pre-adapter refresh: effects 0 and a non-success outcome.

RED was observed first: effects 1 and `Recorded(Succeeded)`.

**Minor M1 (separate commit):** recheck the inference authority at the worker's dispatch-time clock read, not only at the read before the callback. It has its own failing test.

### Task 8: A2 exit — bound gate on the changed source and ledger update

**Files:**
- Create: `docs/verification/2026-10-26-a2-gate-receipt.md`
- Modify: `docs/verification/2026-10-26-c01-c44-execution-ledger.md` (rows C03–C06 only); `docs/verification/2026-10-23-completion-matrix.md` (rows C05/C06 only if their status changes, plus one execution-record bullet); `tasks/todo.md` (one new dated entry).

**Interfaces:**
- Consumes: Task 1's capture and gate procedure; the A2 map from Tasks 3–7.
- Produces: the final A2 evidence.

- [ ] **Step 1: Confirm a clean tree**

Run `git status --short --branch`. Expected: no modified or untracked entries; `main` is ahead by the Task 1–7 commits.

- [ ] **Step 2: Run the bound gate**

Repeat Task 1, Steps 2–7 verbatim with `R=.work/a2-20261026`. Expected: the same outcomes: `exit=0`, `check: all green`, `dependency-scan: RUN cargo-audit`, zero `SKIP`, `benchmark regression gate: PASS`, `rust <N> 0 0`, `abi-stable`, `wdbx-stable`. Record whether WDBX `HEAD` equals Task 1's; a difference must be stated in the receipt.

- [ ] **Step 3: Write the A2 receipt**

Write `docs/verification/2026-10-26-a2-gate-receipt.md` with the same fields as Task 1, Step 8, plus the list of A2 commits and the per-task results (RED→GREEN or characterization).

- [ ] **Step 4: Update ledger rows C03–C06**

Set the evidence path to the A2 map plus the A2 receipt for all four rows.
- C03 and C04 keep "Q local Unix" and "Q local Mac".
- C05 and C06 change from `U` to `Q local Mac` only if all of the following hold:
  - every mapped test passed in the Step 2 log;
  - Tasks 5–7 recorded no defect;
  - Step 2 was valid.
  
  Otherwise they stay `U`, with the blocking item named.

Each row's text states its limits: no power-loss, separate-host, public-deployment or Windows/Linux runtime proof. Mirror any C05/C06 status change in the matrix table so the two documents agree. Re-run Task 2, Step 4's parity diff; expect differences only on the rows changed here, each citing a 2026-10-26 path.

- [ ] **Step 5: Run the docs checks and commit**

Run `python3 -m unittest discover -s tools/tests -p 'test_docs*.py'` and `git diff --check`; expect `OK` and no output.

```bash
git add docs/verification/2026-10-26-a2-gate-receipt.md docs/verification/2026-10-26-c01-c44-execution-ledger.md docs/verification/2026-10-23-completion-matrix.md tasks/todo.md
git commit -m "docs(verification): record A2 relay recovery requalification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Next

A3: join the sealed process adapter to Abbey's real `RuntimeStore` and local approval authority after the Abbey owner's source/lock handoff. This plan does not start A3. The A1 owner-held gate receipts (WDBX's own gate, Abbey root/desktop) remain open inputs.
