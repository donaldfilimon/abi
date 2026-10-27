# A2 relay recovery requalification — 2026-10-26

Authority: `../plans/2026-10-26-a1-remainder-a2-relay-requalification.md` (Tasks 3–7),
spec `../plans/2026-10-25-complete-abi-and-lylx-handoff.md` section A2, matrix rows
C03–C06 in `2026-10-23-completion-matrix.md`, and invariants 1–10 in
`../spec/encrypted-relay-recovery.md`.

Scope: focused local requalification on the current Mac only. This record does not
change any matrix or ledger status. The bound full gate and the C03–C06 ledger update
belong to Task 8 of the plan and were not run here.

## Source set

- ABI: local `main` at `ebd976e` (clean tree at start).
- Sibling WDBX (compiled as a dependency only, never edited): `952bb66`, consumed
  paths (`Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `crates`) clean; dirty
  docs only (`docs/completion/substrate-release/README.md`, an untracked
  `docs/completion/execution/evidence/2026-10-26-source-pin-migration-review/`). A
  parallel WDBX writer was active, so this is not a bound gate receipt.
- Toolchain: `./tools/cargo.sh` (Rust 1.99.1 stable), isolated
  `CARGO_TARGET_DIR=.work/a2/target`, `CARGO_BUILD_JOBS=2`, stdin closed.

## Coverage map

| A2 bullet | Existing tests | New test (Task) | Result |
| --- | --- | --- | --- |
| Short/zero writes, sync failure | `short_writes_complete_the_entire_frame_before_success`, `zero_write_never_acknowledges_or_reuses_the_handle`, `failed_sync_never_acknowledges_even_when_complete_frame_survives`, `partial_frame_failure_poisoning_refuses_reuse_and_reopen` (`crates/abi-worker/src/file_store/fault_tests.rs`); `actual_sqlite_full_partial_write_and_sync_failures_poison` (`crates/abi-control-plane/src/delivery_outbox/tests.rs`) | — | PASS |
| Corrupt/truncated records | `corrupted_and_partial_frames_refuse_recovery`, `external_truncation_poisons_handle_and_blocks_further_appends`, `reordered_valid_frames_and_hostile_lengths_refuse` (`crates/abi-worker/tests/durable_control.rs`); `malformed_control_history_fails_reconstruction`, `tampered_result_digest_in_control_history_fails_reconstruction` (`crates/abi-worker/tests/contracts.rs`); `modified_packet_history_refuses_instead_of_resetting` (`crates/abi-control-plane/src/delivery_outbox/tests.rs`) | — | PASS |
| Failed terminal persistence | `persisted_terminal_with_lost_ack_reopens_as_cached_without_execution`, `missing_terminal_reopens_as_unresolved_without_execution` (`crates/abi-worker/tests/terminal_durability.rs`) | outbox-side gap → Task 7 | PASS |
| Process kill | `process_exit_after_admission_preserves_non_dispatching_retry` (`crates/abi-worker/tests/durable_control.rs`); `reconnect_process_terminal_survives_worker_and_authority_kills`, `reconnect_process_kills_do_not_resume_prepared_admitted_or_partial_generation` (`crates/abi-control-plane/src/service/tests/reconnect_process.rs`); `parent_kill_import_transaction_preserves_complete_old_or_new_view` (`crates/abi-control-plane/src/delivery_consumer/interruption_tests.rs`) | — | PASS |
| Lost responses | `reconnect_process_http_replays_after_execution_expiry_with_current_credentials`, `reconnect_process_terminal_survives_worker_and_authority_kills` (`crates/abi-control-plane/src/service/tests/reconnect_process/`); `restart_replays_identical_packets_after_execution_deadline_and_prior_ack` (`crates/abi-control-plane/src/relay/tests/recovery.rs`) | — | PASS |
| Duplicate delivery | `signed_conflicting_duplicate_and_foreign_binding_preserve_journal` (`crates/abi-control-plane/src/delivery_consumer/interruption_tests.rs`); `streaming_frames_are_ordered_private_idempotent_and_do_not_resume_after_restart` (`crates/abi-control-plane/src/relay/tests/streaming.rs`); `completed_ciphertext_retry_does_not_charge_the_memory_quota_twice` (`crates/abi-control-plane/src/relay/tests.rs`) | — | PASS |
| Exhausted quotas | `active_principal_capability_and_output_quotas_are_enforced`, `replay_revalidates_active_concurrency_and_terminal_capacity`, `admission_reserves_control_capacity_for_a_terminal_event` (`crates/abi-worker/tests/contracts.rs`); `quotas_enforce_two_active_and_ten_queued_per_account` (`crates/abi-control-plane/src/relay/tests.rs`); `replay_capacity_refusal_preserves_commitments_and_closed_manifest` (`crates/abi-control-plane/src/relay/tests/recovery.rs`); `reconnect_process_corruption_lock_and_reservation_failure_refuse_before_inference` (`crates/abi-control-plane/src/service/tests/reconnect_process/authority_cases.rs`); `twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded` (`crates/abi-control-plane/src/service/tests/reconnect_fairness.rs`) | — | **FAILED (open)**: `twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded` hit its 15 s wall-clock `tokio::time::timeout` (`Elapsed(())` at `reconnect_fairness.rs:198`) in 3 of 3 runs; all others PASS |
| Recheck cancellation/revocation/time/lease before dispatch, incl. after callbacks | `elapsed_lease_before_dispatch_never_invokes_provider`, `authority_callback_consuming_lease_never_dispatches_or_readmits`, `authority_callback_with_remaining_lease_executes_once`, `pre_cancelled_observed_inference_never_calls_provider` (`crates/abi-worker/tests/inference.rs`); `authority_permit_must_still_be_fresh_after_durable_admission` (`crates/abi-control-plane/src/relay/tests.rs`); `governed_tool_tenant_mutation_and_fresh_revocation_refuse_effects` (`crates/abi-control-plane/src/service/tests/governed_tools.rs`) | gaps → Tasks 4, 5, 6 | PASS |
| Exact packets and recorded results across reopen | `exclusive_create_reopen_retains_exact_packets_and_expiry_tombstone` (`crates/abi-control-plane/src/delivery_outbox/tests.rs`); `file_reopen_preserves_completed_and_unresolved_evidence` (`crates/abi-worker/tests/recovery.rs`); `committed_inference_is_cached_after_real_store_reopen` (`crates/abi-worker/tests/inference.rs`); `cancellation_survives_restart_and_blocks_success_replay` (`crates/abi-control-plane/src/relay/tests/recovery.rs`) | — | PASS |
| Uncertain effect stays unresolved | `replayed_admission_can_be_abandoned_once_without_readmission`, `unresolved_history_rejects_contradictory_terminal_events` (`crates/abi-worker/tests/recovery.rs`); `direct_execution_after_uncertainty_refuses_before_prepare_or_provider` (`crates/abi-control-plane/src/service/tests/direct_recovery_guard.rs`); `reconnect_process_saved_completion_refuses_missing_or_admitted_control_history` (`crates/abi-control-plane/src/service/tests/reconnect_process/authority_cases.rs`) | — | PASS |
| Admission ≠ redelivery authority | `replay_grant_round_trip_cannot_become_execution_permit_and_expires_exclusively` (`crates/abi-control-plane/src/delivery_types/tests.rs`); `completed_lookup_after_deadline_is_read_only` (`crates/abi-worker/tests/recovery.rs`) | — | PASS |

Known, design-bounded limit: on the inference path, revocation reaches a node only
through the five-second `DispatchPermit` (`crates/abi-control-plane/src/relay.rs`,
`expires_at: (issued_at + 5)`) and, after Task 4, through local certificate expiry.
Live revocation inside an already-issued permit window is not observable on the node.
This record does not claim it.

## Focused run (Task 3, Step 2)

Commands (run from the ABI root with the isolated target above):

```bash
./tools/cargo.sh test --locked -p abi-worker < /dev/null
./tools/cargo.sh test --locked -p abi-control-plane --lib < /dev/null
node crates/abi-control-plane/web/tests/recovery.cjs
```

Results:

- `abi-worker`: exit 0. `test result:` lines — lib 13, `contracts` 25,
  `durable_control` 12, `inference` 16, `recovery` 7, `terminal_durability` 2,
  doc-tests 0; every line `0 failed; 0 ignored`.
- `abi-control-plane --lib`: exit 101. `test result: FAILED. 85 passed; 1 failed;
  0 ignored`. The one failure is
  `service::tests::reconnect_fairness::twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded`,
  `called Result::unwrap() on an Err value: Elapsed(())` at
  `reconnect_fairness.rs:198:6` (its 15 s wall-clock budget). The
  `recovery_child` panics in the log are the intended child-process fault
  scenarios; their parent tests passed.
- Reruns of that test: `-- reconnect_fairness` (1 passed, 1 failed; same
  `Elapsed`) and `-- --exact service::tests::reconnect_fairness::twenty_retained_…`
  (0 passed, 1 failed; same `Elapsed`, finished in 53.71 s).
- Host load average during all three runs was about 145–187 (other writers were
  building concurrently). No source was changed. The cause is not established. The
  test is wall-clock-bounded, so load is the likely cause, but it cannot be
  separated from a fairness regression without a rerun on a quiet host. Until that
  rerun passes, the bullet stays open.
- `node crates/abi-control-plane/web/tests/recovery.cjs`: exit 0, 4 tests, 4 pass.
- Name check: 44 of the 45 mapped test names appear as `... ok` in the logs. The
  remaining name is the timed-out test above.

Logs (ignored, local): `.work/a2/a2-worker.log`, `.work/a2/a2-control-plane.log`,
`.work/a2/a2-control-plane-fairness-rerun.log`,
`.work/a2/a2-control-plane-fairness-rerun2.log`, `.work/a2/a2-node.log`.

## Added regression tests (Tasks 4–7)

| A2 bullet | Existing tests | New test (Task) | Result |
| --- | --- | --- | --- |
