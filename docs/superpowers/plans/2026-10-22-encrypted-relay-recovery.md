# Durable Encrypted Relay Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Tasks 1–8 have local implementations and focused verification. The
final complete gate and source receipt are recorded in the execution ledger;
this does not close the parent ecosystem plan.

**Goal:** Redeliver the exact encrypted output of enrolled tenant workers after
network loss or process restart, without executing inference a second time.

**Architecture:** Keep a synced, bounded encrypted outbox on each user node and
retain only routing/lifecycle/delivery commitments centrally. Reconnect reconciles
recorded worker outcomes and republishes saved packets under short-lived delivery
permits; it never renews execution authority or resumes interrupted generation.
Client imports deduplicate verified packets durably before rendering.

**Tech Stack:** Existing Rust 1.99.1/edition 2024 workspace, rusqlite, Tokio,
reqwest, Ed25519 and RFC 9180 HPKE. No new cryptographic primitive, hosted payload
store, service binary, top-level CLI command or MCP tool.

**Spec:** [Encrypted relay recovery](../../spec/encrypted-relay-recovery.md),
scoped under the [approved ecosystem plan](2026-10-22-abi-production-ecosystem.md).
Read both; this document specifies implementation order, interfaces and tests.

## Global Constraints

- Work on canonical `main`; preserve all pre-existing dirty and untracked source.
- Always `./tools/cargo.sh`; resolving commands use `--locked`, tests `< /dev/null`.
- Stable Rust 1.99.1, edition 2024; each Rust file at most 1,000 lines.
- Preserve CLI 14/MCP 12 catalogs and current v1 crypto/signature encodings.
- Private prompts, output packets, keys and inference stay on user infrastructure.
- Outbox: 64 MiB database, 1 MiB reserved encoded artifacts/job, 256 retained jobs,
  at most 16 delta frames plus one terminal packet. Preserve existing 65,536-byte
  maximum decrypted result and 384-KiB actual encoded HTTP body/response ceilings.
- Delivery retention: default 24 hours from submission; configurable 60 seconds
  through seven days, immutable at admission. Current credentials must remain
  valid independently. Replay grants last at most 30 seconds and grant no execution.
- Preserve 2-active/10-queued defaults and cancellation-request/stop-ack distinction.
- No live store, resource purchases, CI resumption, publication, training, live
  account testing or deployment. Unix local acceptance does not qualify Windows,
  Linux runtime, separate hosts, storage power loss or production operation.
- Implementation status must be supported by the source/test matrix below.
  Final qualification requires the complete gate and immutable source receipt;
  this plan alone is not evidence of a passing gate.
- Follow the current session's source/commit authorization. Checkpoints below
  require scoped reviewable diffs and receipts; do not commit unrelated pending
  account/worker work merely to satisfy a skill's generic commit step.

## Review Focus

1. Lost ack plus central epoch change: an old acknowledged cursor must not hide
   missing transient payloads; Task 4 tests commitment versus availability.
2. Two-store crash gap: outbox/control history disagreement must not start another
   provider or deadlock every future job; Tasks 2–3 test all ordering boundaries.
3. Cancellation or revocation racing replay: retained ciphertext is not renewed
   execution authority or permission to publish success; Tasks 4–5 test both.
4. Credential/model/time changes: historical lookup ignores execution expiry but
   current delivery auth still applies; Tasks 1, 4 and 7 test these independently.
5. UI/storage crash after import: cursor advancement and packet storage must be
   atomic; Task 6 tests duplicate/conflicting pages and restart before rendering.

## Source baseline and file responsibilities

The original planning baseline had live encrypted frames and durable worker
results with transient relay packets. The current continuation adds the private
outbox, replay-only authority, post-deadline delivery and client-import journal.
`WorkerManager::submit` retains its execution-time admission checks; recovery
uses the separate read-only historical API. The relay schema is version 2.

| Original proposed files | Responsibility |
| --- | --- |
| Create `crates/abi-worker/src/manager/recorded.rs`; modify `manager.rs`, `store.rs`, `types.rs`, `lib.rs` | Read-only recorded outcome lookup and explicit prior-process abandonment |
| Create `crates/abi-worker/tests/recovery.rs` | Exact historical lookup, tombstone and quota contracts |
| Create `crates/abi-control-plane/src/delivery_types.rs` | Versioned delivery policy, manifests, packets, grants and replies |
| Create `crates/abi-control-plane/src/delivery_outbox/{mod.rs,store.rs,tests.rs,faults.rs}` | Private synced outbox, reservations, retention and test-only filesystem faults |
| Modify `relay_runtime.rs`; create `relay_recovery_client.rs` | Write-before-send execution integration and reconnect-only delivery driver |
| Create `relay_recovery/{mod.rs,tests.rs}`; modify `relay.rs`, `relay_stream.rs`, `relay_schema.sql`, `relay_http.rs` | Metadata migration, replay authority, transient restoration and HTTP compatibility |
| Modify `relay_types.rs`, `relay_client.rs`, `lib.rs` | Additive signed recovery action and bounded transport status classification |
| Create `delivery_consumer.rs` and `web/tests/recovery.cjs`; modify native examples and `web/app.js` | Verified import/cursors, local decryption, unavailable versus partial status |
| Create `tests/reconnect_process.rs` and `tests/support/recovery_child.rs` in the control-plane crate | Actual process crash/network/restart acceptance |
| Modify `RELAY.md`, README/claims/public-api docs, task and verification ledgers | Accurate setup, recovery limits, compatibility and evidence |

Names below are proposed public Rust APIs. `AuthorityResult<T>` means the existing
control-plane `crate::Result<T>`; worker APIs return `Result<T, WorkerError>`.
Implementers must not silently invent alternative wire names between tasks.

## Implemented local outcome (2026-10-22)

| Task | Implemented outcome | Evidence entry points |
| --- | --- | --- |
| 1 | Exact read-only worker lookup, prior-process unresolved transition and additive signed delivery contracts | `crates/abi-worker/tests/recovery.rs`; control-plane recovery protocol tests |
| 2 | Private bounded FULL-sync ciphertext outbox, stable sidecar lock, reservations, immutable artifacts, tombstones and poison refusal | `src/delivery_outbox/{mod,store,tests,faults}.rs` in `abi-control-plane` |
| 3 | Preparation before admission, sync before emit, recorded-result terminal reconstruction and no-provider local reconciliation | `src/relay_runtime/recovery.rs` and its tests |
| 4 | Version-2 metadata, delivery-only grants, transactional commitment/audit and transient rehydration under current authority | `src/relay_recovery/`; `src/relay/tests/recovery.rs` |
| 5 | Recovery before fresh poll, fresh proof per retry, bounded backoff/publication and persisted stop observations | `src/relay_recovery_client.rs`; native worker example |
| 6 | Atomic verified encrypted packet/cursor import, local durable helpers and strict browser wrapper/ack flows | `src/delivery_consumer.rs`; `examples/relay_job.rs`; `web/tests/recovery.cjs` |
| 7 | Actual local child-process kills/restarts and TCP response loss with durable provider counts and tenant/key isolation | `src/service/tests/reconnect_process.rs` |
| 8 | Source-scoped docs, independent review repairs, focused suites and final gate/source-receipt workflow | `docs/verification/2026-10-22-production-ecosystem.md` |

Paths beginning `src/` above are relative to `crates/abi-control-plane/`.
Focused worker/control-plane and browser suites and warning-denied package
Clippy have passed. The execution ledger is authoritative for final gate and
archive results. Review repairs cover legacy-v1 access to durable jobs,
persistence of cancellation observed in both state and stream replies, and
transactional publication audit failures.

The detailed task recipes below retain the original implementation sequence. They are not a
claim that every proposed test name or initial test-first RED run was executed;
use the implemented outcome matrix, actual test source and receipts for that
history. Process-kill and injected SQLite failures do not qualify storage power
loss, separate hosts or public production operation.

- [x] Final whole-workspace gate, required scanner, docs validation and immutable
  source receipt: completion is recorded in the execution ledger.

## Task 1: Freeze delivery contracts and recorded worker recovery

**Interfaces produced**

- `RecordedExecution::{Missing, Admitted(JobLease), Completed(JobResult), Cancelled,
  Unresolved}` in `abi-worker`.
- `WorkerManager::recorded(&self, envelope: &JobEnvelope) -> Result<RecordedExecution,
  WorkerError>`: exact historical lookup, no time-based admission or mutation.
- `WorkerManager::recover_interrupted(&mut self, envelope: &JobEnvelope,
  observed_at_unix_ms: i64) -> Result<RecordedExecution, WorkerError>`: only
  replayed prior-process active admissions may become unresolved tombstones.
- `DeliveryPolicy`, `DurableSubmission`, `SignedDeliveryPolicy`, `TerminalManifest`,
  `OutboxPacket::{Delta { frame }, Terminal { packet }}`,
  `ReplayGrant`, `RecoveryRequest`, `RecoveryAction::{Inspect, Publish}`,
  `RecoveryReply`, `DeliveryPage`, `DeliveryAck`, `DeliveryStatus` in
  `delivery_types.rs`, with fields/encodings fixed by the linked spec.
- `PacketId { job_id, kind: PacketKind, sequence, digest }` and canonical SHA-256 packet and
  manifest commitments; signatures bind scope, submission digest and version.

- Write worker tests: `completed_lookup_after_deadline_is_read_only`,
  `changed_envelope_principal_nonce_or_scope_refuses`,
  `historical_model_revision_does_not_require_current_model`,
  `replayed_admission_can_be_abandoned_once_without_readmission`,
  `current_process_admission_cannot_be_abandoned_as_replayed`. Assert zero store
  appends during lookup, zero new leases/provider calls, and freed active quota
  with old job/nonce/request identity permanently refused after abandonment.
- Run `./tools/cargo.sh test --locked -p abi-worker --test recovery < /dev/null`.
  Observe the missing API/test failure, then implement lookup using the manager's
  exact recorded envelope/request identity. Do not move execution-time checks in
  `submit` or weaken fresh admissions. Add the tagged unresolved control event;
  replay must preserve it and reject contradictory terminal transitions. Health
  must exclude unresolved jobs from active/succeeded/cancelled counts and expose
  their unresolved count explicitly.
- Write protocol golden/round-trip tests in
  `delivery_types/tests.rs`: v1 bytes unchanged; scope mutation, expired grant,
  grant-as-dispatch-permit, unknown version and grant epoch mismatch refuse.
- Implement canonical typed contracts; add optional recovery reply fields and
  signed `NodeAction::Recovery`. Keep fresh nonce/action-proof validation intact.
- Rerun worker recovery, old worker tests and control-plane protocol tests.
  Checkpoint: typed contracts and exact historical evidence, no recovery transport.

## Task 2: Implement the user-node encrypted outbox

**Interfaces consumed:** Task 1 packet/manifest/policy types and existing private
path validation. **Interfaces produced:**

```rust
FileRelayOutbox::create(path: &Path, limits: OutboxLimits) -> AuthorityResult<Self>;
FileRelayOutbox::open(path: &Path, limits: OutboxLimits) -> AuthorityResult<Self>;
FileRelayOutbox::prepare(&mut self, dispatch: &Dispatch,
    policy: &SignedDeliveryPolicy) -> AuthorityResult<()>;
FileRelayOutbox::append(&mut self, job_id: &str,
    packet: &OutboxPacket) -> AuthorityResult<()>;
FileRelayOutbox::pending(&self, now: i64) -> AuthorityResult<Vec<TerminalManifest>>;
FileRelayOutbox::packet(&self, job_id: &str,
    id: PacketId) -> AuthorityResult<Option<OutboxPacket>>;
FileRelayOutbox::record_ack(&mut self, ack: &RecoveryReply) -> AuthorityResult<()>;
FileRelayOutbox::expire(&mut self, now: i64) -> AuthorityResult<()>;
```

- Write tests for exclusive create/reopen, owner-only DB/journals, symlink and
  writable-ancestor refusal, duplicate identity versus conflict, 16-frame and
  encoded reservation limits, terminal closure, ack-before-restart retention,
  expired-packet deletion with tombstone preservation, and missing-file refusal.
- Run the focused `delivery_outbox` tests and observe failures before adding
  storage. Implement schema/version validation, stable exclusive lock, FULL-sync
  transactions, bounded SQL pages and reservation before `prepare` succeeds.
  Persist exact encoded packets; no resealing inside the storage layer.
- Add test-only SQLite VFS fault wrapping in `faults.rs`, guarded to prevent
  parallel global VFS registration races. Inject `SQLITE_FULL`, partial `xWrite`
  and `SQLITE_IOERR_FSYNC` from `xSync`; verify failed commit poisons the handle,
  uncertain state refuses dispatch, and reopening either validates exact history
  or explicitly refuses. A mocked return before an actual write is not an fsync
  failure test. Do not introduce production VFS hooks or environment toggles.
- Run `./tools/cargo.sh test --locked -p abi-control-plane --lib delivery_outbox
  < /dev/null` and private path tests. Checkpoint: packets survive node restart
  without relying on network or changing the worker control log.

## Task 3: Journal output before it can leave the worker

**Interfaces consumed:** Tasks 1–2. **Interfaces produced:**

- `DurableRelayRuntime<S>` owns the existing `RelayRuntime<S>` and
  `FileRelayOutbox`; validates their same node/service identity at startup.
- `RelayRuntime::recorded_dispatch(&self, dispatch: &Dispatch) ->
  AuthorityResult<RecordedExecution>` rebuilds the exact original envelope on
  the user node from its encrypted request. It uses the historical model revision,
  not the newly configured inference revision; signature/scope/digest checks
  still apply. It never calls `submit` or accepts a new dispatch permit.
- `DurableRelayRuntime::reconcile_local(&mut self, now: i64) ->
  AuthorityResult<Vec<TerminalManifest>>`, with no provider parameter.
- `DurableRelayRuntime::execute_durable(&mut self, dispatch: &Dispatch,
  policy: &SignedDeliveryPolicy, permit: &DispatchPermit,
  provider: &dyn ModelProvider, clock: impl FnMut() -> i64,
  cancel: &CancellationToken, emit: impl FnMut(PacketId) -> AuthorityResult<()> + Send)
  -> AuthorityResult<()>`.

- Write fault-boundary tests for crashes after Prepared, after admission,
  after each saved delta, after control completion, before terminal packet sync,
  and after terminal sync. Count provider calls and compare ciphertext bytes.
- Implement preparation/reservation before admission. The current durable
  manager still owns admission and post-sync authority guards. Delta observer:
  seal once, sync outbox, then notify the one-packet transport channel by ID.
  Observer/write failure cancels locally and never advertises success.
- Persist a terminal packet/manifest only after the worker completion commit.
  Reconcile a missing unpublished terminal from read-only `Completed` evidence;
  compare every saved delta against that exact result's prefix. If a packet
  exists, return it; never re-encrypt it. Contradiction poisons recovery.
- For replayed admission without terminal, call `recover_interrupted` only
  after exclusive startup recovery; persist unresolved prefix and release quota.
  Prepared-with-Missing is abandoned, not dispatched. Current live inference must
  stop/join before its outcome is sealed or a cancellation is acknowledged.
- Verify `cached_result_recovery_after_deadline_never_calls_infer`,
  `existing_terminal_ciphertext_is_byte_identical_after_reopen`, and
  `unresolved_job_does_not_block_every_subsequent_fresh_job` with counted providers.
  Checkpoint: local execution and transport have a durable write-before-send join.

## Task 4: Add delivery-only authority and transient rehydration

**Interfaces consumed:** Task 1 contracts. **Interfaces produced:**

- `Relay::submit_durable(&mut self, authority: &mut Authority, account: &Account,
  submission: DurableSubmission, now: i64) -> AuthorityResult<JobReceipt>`.
- `Relay::recover(&mut self, authority: &mut Authority, cert: &NodeCertificate,
  request: &RecoveryRequest, now: i64) -> AuthorityResult<RecoveryReply>`.
- `Relay::delivery_page(&mut self, authority: &mut Authority, account: &Account,
  job_id: &str, after: u32, now: i64) -> AuthorityResult<DeliveryPage>`.
- `Relay::ack_delivery(&mut self, authority: &mut Authority, account: &Account,
  ack: &DeliveryAck, now: i64) -> AuthorityResult<DeliveryStatus>`.

- Write migration tests: v1 jobs stay ephemeral, unknown schema refuses,
  version-2 migration is atomic, cancellation facts survive restart, and older
  schema readers fail closed. Test that post-upgrade mutations forbid authority
  rollback to an older backup; use forward fixes after activity. Never restore
  old worker history as an operational rollback.
- Implement schema version 2 and a new service epoch. Separate execution and
  delivery states/cursors; never requeue an offered or previously dispatched job.
  Preserve original cancellation markers even when execution becomes unresolved.
- Implement replay-only Inspect/grants with current enrollment, principal,
  tenant/node, capability, policy and terminal consistency checks. Packet-only
  post-execution-deadline access is allowed until delivery expiry; no renewed
  dispatch permit is issued. Reject absent/foreign jobs, expired credentials,
  revoked/deleted identities and changed immutable policies.
- Implement Publish: verify packet digest/signature and grant; insert new
  commitment+audit transactionally or match an existing exact commitment, then
  restore missing transient bytes within current memory quota. Duplicate replay
  does not increment frame, active/queued or audit counts. A full buffer returns
  explicit retryable capacity refusal and leaves the worker packet durable.
- Write tests for old ack/new epoch, repeated terminal after lost ack,
  cancellation before success publication, terminal conflict, prefix extension,
  clock regression, seven-day policy cap, server saturation and tenant crossing.
  Cancellation accepts only its authorized prefix/stop fact, never success.
- Wire the v2 account routes and additive node action. Test 384-KiB encoded
  limits and paginated responses using actual serialization, including JSON byte
  array expansion. Checkpoint: restart repair restores delivery, not execution.

## Task 5: Reconnect and recover before polling for fresh jobs

**Interfaces consumed:** Tasks 2–4. **Interfaces produced:**

```rust
RelayClient::recover_pending<S: JobControlStore + Send + 'static>(
    &self, runtime: &mut DurableRelayRuntime<S>, now: i64
) -> AuthorityResult<RecoverySummary>;
```

`RecoverySummary` counts restored, pending, unavailable and refused jobs only;
no prompt/result content. Transport classification distinguishes retryable
network/timeouts/429/503 from refusal, version, scope or packet conflict.

- Write TCP tests that drop the response after frame/terminal commit, restart
  the central process after prior upload acks, and rotate its epoch while the
  worker stays alive. Assert identical packet bytes and provider count exactly 1.
- Implement reconnect backoff 250 ms, 500 ms, 1 s, 2 s, capped at 5 s with
  OS-random jitter in [0, 250] ms. Bound work to 16 packet publications/round,
  one in flight, original 15-second request timeout and delivery deadline. Sign
  every retry freshly; never retry the same node proof or an inference call.
- Recover all retained manifests before Poll. An old epoch/expired replay
  grant obtains a new replay-only grant. Retry a full server buffer without
  evicting the durable source. Auth/revocation/corruption/conflict stops the job
  explicitly; local persistence uncertainty stops all new dispatch.
- On lost authority contact, stop current generation cooperatively as today;
  retain saved prefix/outcome for later delivery. Reconnect never resumes it.
  Keep cancellation inspection active when connected and persist its observation
  before stop acknowledgement. Test cancellation during upload and backoff.
- Update `examples/relay_worker.rs` with explicit outbox initialization versus
  reopen, safe missing-file refusal for existing installations, and metadata-only
  reconnect status. No automatic `--initialize` on error. Test graceful shutdown
  and dropped runner without orphaned live work.
- Checkpoint: repeated disconnect/reconnect cycles do not increase executions
  and do not silently bypass revoked/expired enrollment.

## Task 6: Durable verified import and recovery-aware clients

**Interfaces consumed:** DeliveryPage/DeliveryAck, existing node/reply keys.
**Interfaces produced:**

```rust
VerifiedDeliveryStore::create(path: &Path, binding: &JobBinding,
    node_key: [u8; 32]) -> AuthorityResult<Self>;
VerifiedDeliveryStore::open(path: &Path) -> AuthorityResult<Self>;
VerifiedDeliveryStore::import(&mut self, page: &DeliveryPage,
    reply_key: &RelayKey) -> AuthorityResult<DeliveryAck>;
```

- Write tests for identical page imported repeatedly, lost cursor commit,
  crash after commit/before render, wrong key/scope, changed digest at same
  sequence, missing prefix and terminal inconsistent with the imported deltas.
- Implement owner-private bounded encrypted packet/cursor transactions. Verify
  signatures/AEAD before advancing the contiguous cursor; persist packet and
  cursor atomically. Keep plaintext out of that journal. Render from the verified
  saved prefix; stdout is a replayable view, not an exactly-once output effect.
- Extend native prepare/import/open helpers to opt into durable delivery,
  retain immutable policy and keys, and return an ack artifact for browser upload.
  Add a strict durable-wrapper upload validator alongside the existing v1 sealed
  validator; unexpected/plaintext fields must still refuse before any POST.
  The browser never possesses the reply private key and cannot assert that it
  decrypted a packet. Ack metadata means explicit owning-client receipt only;
  it does not authorize outbox deletion before its policy deadline.
- Browser watch distinguishes running, recovering, partial/unresolved,
  unavailable worker, expired delivery and acknowledged completion. Polling remains
  snapshot-based. Preserve logout cancellation and stale-response guards; no
  automatic output download or private keys in browser storage.
- Run `node --test crates/abi-control-plane/web/tests/*.cjs`, syntax checks and
  Rust consumer tests. Checkpoint: reconnect delivers one verified packet per
  identity; duplicate transport does not duplicate imported content.

## Task 7: Qualify crash recovery with actual processes

**Interfaces consumed:** All preceding tasks. No production API additions.

- Implement child process roles for authority, worker and durable counted
  fixture provider. Use scratch 0700 directories and an out-of-band durable
  invocation marker; failure of that marker aborts the fixture rather than
  hiding another invocation. Kill children at deterministic boundary signals.
- Exercise every row of the spec's failure matrix through actual local TCP,
  including simultaneous two-tenant jobs, repeated service/worker restarts and
  response loss after committed writes. Distinct reply keys must decrypt only
  their own verified prefixes and terminal results.
- Cover late replay after execution expiry but before delivery expiry,
  expired credential refusal, changed configured model revision with a recorded
  result, cancellation/revocation/deletion during disconnect, poison/corruption,
  lock contention and full client/server/outbox budgets. No arbitrary sleeps as
  synchronization; all child waits and network fixtures have bounded timeouts.
- Scan central DB, rollback journal, copied backup and captured logs for
  plaintext canaries and exact saved ciphertext packets. Verify client/outbox
  journals contain only their intended encrypted artifacts/metadata. Never use
  live stores, real IdP/provider credentials or remote hosts for local evidence.
- Run `./tools/cargo.sh test --locked -p abi-control-plane --test
  reconnect_process < /dev/null`; run full worker and control-plane suites.
  Accept only zero unauthorized dispatches and no provider counter above 1 for
  any recovered job. Record partial/unresolved outcomes instead of hiding them.

## Task 8: Documentation, review and release-source receipt

- Update `crates/abi-control-plane/RELAY.md` with enrollment/outbox upgrade,
  immutable delivery retention, reconnect behavior, v1/v2 compatibility and
  expired-credential limits. Describe unavailable durable bytes separately from
  successful execution and imported output. Document safe recovery of poisoned
  stores and forbidden history rollback/reset.
- Update `docs/contracts/public-api.mdx`, external-claims audit, README and
  `tasks/{todo,goals}.md` from actual evidence. Retention/deletion of WDBX memory,
  OS preemption, useful model weights, public rollout and separate-host acceptance
  remain separate owner deliverables. Synchronize instruction files only if
  durable conventions actually change.
- Request one independent review of the whole recovery change, emphasizing
  the five Review Focus cases, immutable cryptographic contexts, no-provider
  recovery APIs and two-store crash ordering. Reproduce important findings with
  failing tests, repair, then rerun affected suites.
- Run the complete `./tools/check.sh`, required scanner, docs-policy tests,
  `.agents/skills/docs-validate/validate.sh`, Node browser tests and
  `git diff --check`. Rebuild legacy golden embeddings if touched. Preserve
  failures, actual exits, compiler/features and platform exclusions.
- Save ABI source manifest/archive plus logs in a new external verification
  receipt; capture sibling revisions/status as metadata unless their own exact
  gates are actually rerun. Link this continuation in the ecosystem ledger.
  Qualify only this local slice; do not mark the ecosystem plan complete.

## Completion criteria and dependency order

Tasks 1 → 2 → 3 and 1 → 4 converge at 5 → 6 → 7 → 8. Execute sequentially on the
canonical checkout unless the user selects isolated parallel ownership. Each
checkpoint includes its exact diff and test result; no mock-only recovery claim.

Complete when an existing authorized job can survive loss/restart and reconstruct
its committed encrypted output on its original client within retention/current
authority, while every retry leaves provider invocations unchanged. A crashed
in-flight job may remain partial/unresolved; it must not restart inference.

Out of scope: guaranteed execution of a lost queued input, resumed generation,
public service deployment, automatic enrollment/key renewal, independent security
assessment, multi-host acceptance, WDBX backup/erasure implementation, OS sandbox
preemption and useful original trained checkpoints. Those original ecosystem
milestones remain in the approved parent plan rather than being implicitly
completed by output recovery.
