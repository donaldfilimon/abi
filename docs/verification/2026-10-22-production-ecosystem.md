# Production ecosystem execution ledger

Plan: docs/superpowers/plans/2026-10-22-abi-production-ecosystem.md

- [x] 1. Captured ABI/WDBX/Abbey candidate qualification and gap register.
  Later canonical sibling edits are excluded; CI/pin/publication remain pending.
- [ ] 2. Public accounts, private node enrollment and browser client.
  - [x] Local Rust OIDC account/enrollment service, browser and node example.
  - [x] Encrypted outbound relay, job quota enforcement and isolated local test jobs.
  - [ ] Actual public-provider and separately hosted qualification.
- [ ] 3. Durable workers, transport, bounded execution and governance.
  - [x] Private Unix durable worker control history and reopen/retry tests.
  - [x] Local authenticated bounded inference with durable cached results.
  - [x] Outbound encrypted relay and fresh signed dispatch permits.
  - [x] Encrypted incremental results and acknowledged cooperative cancellation.
  - [x] Private encrypted outbox, local reconnect/replay and verified client imports.
  - [ ] Separate-host partitions and Abbey governed-tool approval join.
- [ ] 4. WDBX retention/key custody and governed live adapters.
- [ ] 5. Rust original-model training, evaluation and runtime.
- [ ] 6. Cross-platform packages, public operation and external assessment.

## 2026-10-22

- Baseline ABI HEAD 69910bf: full tools/check.sh passed during planning.
  ABI and WDBX were clean; Abbey was clean and 42 commits ahead of origin/main.
  This is local ABI evidence, not a complete ecosystem receipt.
- Ruling: implement on canonical main per the approved plan and repository
  instructions, rather than the execution skill's default worktree.
- Ruling: implement durable worker storage before enrollment, because the
  existing worker manager's only store is volatile. This changes task order,
  not the release boundary; enrollment must consume durable replay state.
- Durable storage requirements: explicit create versus reopen, exclusive
  writer lock, versioned bounded hash-chained records, sync before success,
  strict corruption/truncation refusal, and no automatic missing-log reset.
- Full training, spending, live accounts, CI, publishing and deployment remain
  pending explicit authorization. Windows/Linux and external audit acceptance
  require their actual environments; do not substitute cross-compilation.

## Implemented worker slice

- `FileControlStore` implements explicit create/reopen, exclusive writer lock,
  current-user private regular files, bounded hash-chained records and sync
  before acknowledgment. Corrupt/partial/missing history refuses; uncertain
  writes poison the handle. Checksums do not defeat owner rewrite or rollback.
- Unix support is locally exercised. Windows refuses before creating a log
  until private-file ACL validation exists. Directory privacy and stable log
  path are caller preconditions; no production certificate lifecycle is implied.
- Control history retains result bytes for exact retries on user infrastructure.
  This slice does not supply encrypted logs, compaction, retention or deletion
  policy; those copies must be included in the future erasure/key-custody work.
  It must not be used to persist private results in the public control plane.
- `WorkerManager::infer` admits through the existing signed manager and only
  dispatches fresh leases. Exact retries return cached results or prior leases.
  Strict inference inputs, signed/configured output bounds, cooperative runtime
  budgets, trusted pre/post-dispatch time and final token usage are enforced.
  It offers no tools. Provider revision binding remains a startup responsibility.
- Six initial store regressions failed against the placeholder, then passed.
  Inference placeholder: four failures plus two refusal controls. A default
  record-budget regression failed for valid 1 MiB output and passed with an
  8 MiB encoded-record ceiling. Independent review found a final token usage
  overshoot: actual failing regression retained, then corrected.
- Added real file/reopen, process-exit, write-failure poison, hostile-length,
  reordered-frame, chunk-limit, tool-refusal, diagnostic-canary, late-completion
  and durable inference-cache coverage. Short-write/sync-failure injection
  remains a coverage gap. Process exit is not proof of power-loss recovery.
- Independent read-only review found no actionable store defect and one P2
  inference final-token defect, repaired with a regression. No security audit,
  original trained model, remote execution or whole-product completion claimed.
- Added only existing libc/nix dependencies for private Unix file validation;
  ABI and Abbey lockfile package entries changed without version resolution.

## Qualification evidence and source drift

External evidence directory:
`/Users/donald/.codex/verification/abi-production-20261022/`.

- `source-manifest.json` captures the starting source cohort after saving the
  plan: ABI 69910bf, WDBX 36bc99b, Abbey 19d934d (full revisions in manifest).
- Canonical qualification encountered concurrent WDBX work referencing a
  temporarily missing crypto/profile.rs. Preserve `worker-clippy.log`; do not
  attribute that failure to the ABI slice or modify its owner's source.
- Isolated source snapshots preserve the captured WDBX/Abbey revisions.
  The first ABI store-only cohort gate and WDBX gate exited 0; logs are
  `abi-cohort-gate.log` and `wdbx-cohort-gate.log`. These do not qualify the
  subsequently added inference adapter or later canonical owner changes.
- Abbey's first locked check refused the new worker dependency graph;
  `abbey-cohort-gate.log` retains that failure. Candidate qualification includes
  the minimal Abbey lockfile adjustment, not an unchanged historical revision.
- Final worker, ABI and Abbey acceptance results are recorded below once run.

### Qualified ABI worker candidate

- Final `abi-final-gate.log`: complete ABI gate exit 0, 1,113 Rust executions,
  zero failures/ignored. Python policy suite: 186 cases, eight skipped because
  the personal central-sync driver is absent. Required cargo-audit actually
  ran, sizes/format/strict Clippy, sibling contracts, Metal tests/docs, live
  local benchmark guard, rustdoc and locked release builds passed. CUDA
  compilation was unavailable without nvcc. No policy exceptions changed.
- Final focused worker suite: 51 executions (2 unit, 25 existing contracts,
  12 durable-control and 12 inference), zero failures/ignored. Strict package
  Clippy passed. The final usage regression and maximum-result regression are
  included. Details: `worker-suite-final.log`, `worker-clippy-final.log`.
- Captured WDBX gate: exit 0, 1,076 Rust executions, zero failures/ignored.
  Existing warning-level diagnostics remain disclosed in the full log.
- `qualification-receipt.json` records source comparisons and release artifact
  hashes. All captured ABI/WDBX inputs remained unchanged through their gates.
  `abi-store-only-tree.txt` retains the earlier store-only Git tree identity.
- Artifact smoke exit 0: release CLI help, launcher stdio tools/list (12 tools),
  signatures of CLI/MCP and both native dylibs. Memory-only store with
  persistence disabled. `artifact-smoke.json` retains the exact boundary.
- Mintlify validation and three docs policy tests passed. `git diff --check`
  and source size guard passed. This report's result additions are a reporting
  delta after qualification, not a new executable source cohort.
- Completed isolated build caches were removed to recover disk space; source
  snapshots, receipts, logs and ABI release artifacts are retained.

### Consumer completion and retained setup failures

- `abbey-qualified-gate.log`: full root gate exit 0, 5,656 Rust executions,
  zero failures/ignored, 187 Python tests. Default, WDBX, personal and accelerator
  modes, warning-denied docs, codegen, claims, read-only boundaries and isolated
  installer/rollback acceptance passed. No release-strip warnings. Existing
  soft size warnings and unavailable Windows/Linux cross targets remain in log.
- The first full consumer run passed Rust stages but failed the generated
  desktop README check (`abbey-final-locked-gate.log`, exit 1). Generator output
  removed two extra blank lines only; no capability count/status changed. The
  canonical consumer changes from this slice are the two lock dependencies
  (nix explicitly version-disambiguated) and those two generated blank lines.
- First desktop run (`abbey-desktop-gate.log`, exit 1): five packaged-runner
  fixture errors because the copied qualification directory had an unborn Git
  HEAD. Restored the captured Abbey base ref through read-only object alternates;
  no source files or canonical Git refs changed. This setup error is retained,
  not attributed to product source. Snapshot recreation needs the captured Git
  metadata before those fixtures; root and desktop artifacts are locally scoped.
- `abbey-desktop-qualified-gate.log`: desktop gate exit 0, 145 Rust executions,
  100 Python and 178 Bun tests, typecheck/frontend/bundle checks, real linked
  desktop binary and scratch-daemon proof. Opt-in graphical/packaged-app
  acceptance and notarization were not performed by this default gate.
- Final source comparison: ABI 1,283 entries, WDBX 1,218, Abbey 910; zero byte,
  mode or file-set mismatches. The ABI directory symlink is included by link
  target and lstat mode, verified against the captured Git index, rather than
  incorrectly treated as an omitted directory. Five distinct adjacent WDBX
  package identities passed locked offline Cargo metadata verification.
- `qualified-source-cohort.tar.gz` and `source-archive-sha256.txt` retain the
  exact qualified files and receipts independently of future canonical edits.
  The canonical report and two ABI worker summaries received a documentation
  delta afterward; Mintlify, docs policy and whitespace checks passed on that
  delta. No executable source changed after the final ABI gate.
- Source scope is macOS ARM64 Rust 1.99.1. Full ecosystem completion remains
  open: the local provider executor has no public transport or governed tool
  integration, and no original useful model or public deployment was produced.

## Account and enrollment continuation (2026-10-22)

- Baseline `accounts-baseline.log`: full canonical ABI gate passed before this
  continuation. Prior captured three-repository source qualification remains a
  separate immutable receipt, not proof for these new sources.
- Ruling: add a separate `abi-control-plane` workspace crate, not new frozen
  CLI/MCP commands. Use private SQLite authority metadata, independent of user
  memory and durable worker result history. No private application payload is
  accepted or persisted by this service.
- Ruling: issue versioned Ed25519 application credentials now; X.509/mTLS node
  connection integration remains an explicit transport deliverable. Local
  approval signs the complete tenant/account/node-key/capability/expiry preview.
- Implemented verified OIDC code flow/PKCE, browser-bound state/nonce, strict
  asymmetric JWT verification, private hashed sessions and CSRF/Origin checks;
  isolated accounts even when two identities have the same verified email.
- Initial OIDC dependency introduced unfixed RSA advisory RUSTSEC-2023-0071.
  Removed it in favor of oauth2 plus jsonwebtoken/AWS-LC. Required scanner runs
  again without a new exception; existing package versions are preserved.
- Implemented explicit database initialization/reopen, bounded transactions and
  append-only administrative audit, single-use pairing with separate poll
  secret, mandatory exact local key signature, one-hour audience-bound node
  credentials, tenant-scoped browser listing/revocation/deletion and fresh
  possession verification with used-nonce persistence. Active SQLite records
  use secure deletion; backups and physical media erasure are separate work.
- Browser serves account signup/login through the configured provider, pairing
  preview, enrollment status, revoke, logout and delete controls. Local terminal
  example generates/stores a private node key, requires the account UUID as
  confirmation, verifies the issued credential and proves possession once.
- Initial account RED receipt contains two actual assertion failures; subsequent
  GREEN exercises restart and tenant separation. Independent reviewer found
  anonymous audit exhaustion and anonymous challenge-slot exhaustion. Two new
  failing regressions are retained in `accounts-review-red.log`; anonymous
  requests now append no administrative audit and reserve no per-node slots.
  Removed the irreversible logical audit row cap; finite database capacity
  still requires operational monitoring/archive work before public access.
- A second read-only review of the revised authority, new JWT verifier and
  enrollment example found no further actionable defect. This is not an
  independent security assessment or release approval.
- JWT critical-header regression and expanded issuer/authorized-party/time/
  access-token-hash refusal cases retain RED/GREEN evidence separately.
- Remaining: account abuse controls, proxy limits, configurable job queues,
  issuer JWKS refresh/key rotation, audit archive/backup recovery acceptance,
  credential renewal, platform credential facilities and Windows/Linux runtime
  acceptance. No actual provider account, public deployment or job execution
  is included. The two-tenant end-to-end jobs criterion remains unchecked.

### Final account/enrollment verification

- `accounts-final-full-gate.log`: full canonical `./tools/check.sh` exited 0
  after the final source changes: 1,133 Rust test executions, zero failures or
  ignored Rust tests; 186 Python tests with the eight existing personal-sync
  skips. The 20 account/enrollment tests pass within that gate. Rustfmt,
  warnings-denied Clippy, required dependency scanner (507 dependencies),
  legacy CLI/MCP goldens, WDBX contract tests, Metal checks, benchmark guard,
  rustdoc and release CLI/MCP builds passed. CUDA remains untested without nvcc.
- `accounts-final-tests.log`: all 20 focused account/enrollment tests pass,
  including actual local signed-provider token exchange with PKCE validation,
  browser-bound callback replay refusal, HTTP node confirmation/authentication,
  private pathname ancestors, used-nonce persistence and active-record erasure.
- `accounts-path-red.log`: writable-ancestor regression failed before repair.
  Private file validation and SQLite now use one trusted canonical pathname;
  root/current-owned sticky directories are accepted, other replaceable
  ancestor paths refused. Follow-up read-only review found no actionable defect.
- `accounts-cli-smoke.json`: actual built binary initializes private state,
  refuses overwrite without altering the database, refuses a non-loopback
  listener before provider contact, rejects public configuration permissions,
  and does not echo its configuration-secret canary in failure diagnostics.
- `accounts-docs.log`: Mintlify validation passed. Docs-policy tests (3) and
  `git diff --check` passed. Instruction review confirmed the three instruction
  files consistently describe 18 local crates and the separate service.
- `accounts-source-manifest.json` and `accounts-source.tar.gz`: final ABI source
  fingerprints and archive plus current sibling revision/dirty-state metadata.
  These new sources do not replace the earlier immutable worker qualification.
  No dependency package identity was removed/upgraded; 65 identities were added
  relative to HEAD, including the pre-existing uncommitted worker additions.
- Validation is local macOS ARM64. No CI, publication, live provider/account,
  model training, platform installer or public deployment was performed.

## Encrypted tenant relay continuation (2026-10-22)

- Baseline `relay-baseline.log`: full ABI gate passed before this continuation.
  Existing source changes and sibling owner work were preserved on main.
- Invoked Abbey system-prompt skill was read. Its prompt-copy workflow does
  not target this service; retain its explicit consent and measured-claim
  boundaries. No Discord/hosted prompt update or live-bot operation is included.
- Ruling: keep payload ciphertext in bounded transient memory and persist only
  routing/status/cryptographic commitments. After service restart, outstanding
  jobs become unresolved. Availability is sacrificed rather than persisting
  private payloads centrally or silently dispatching a second execution.
- Ruling: implement bounded outbound HTTP polling as worker transport. User
  nodes require no inbound listener. TLS uses the configured origin; explicit
  loopback development is the only HTTP exception. Polls/action proofs and
  each job phase recheck current tenant, enrollment, capability and revocation.
- New enrollment claims optionally bind an X25519 key, preserving canonical
  serialization/signatures of legacy key-less claims. Legacy nodes refuse
  encrypted jobs. HPKE uses RFC 9180 X25519/HKDF-SHA256/ChaCha20-Poly1305,
  binding all account/tenant/node/job/request/result context. Private response
  keys remain on submitting clients. No operator training-pool admission exists.
- Durable states distinguish queued, offered and dispatched. A fresh signed
  five-second permit is committed at dispatch; lost offers and uncertain
  post-dispatch outcomes are never automatically repeated. Exact completion
  retries return recorded metadata without consuming a second memory quota.
- Reused worker envelopes and FileControlStore; node identity, tenant namespace,
  model revision, output limits and deterministic job identity are startup-bound.
  File history remains on the user node and may contain private result bytes.
  Only model.infer is supported; tool/script/provider tool requests refuse.
- Independent review found that a slow admission sync could outlive the permit.
  `relay-permit-red.log` preserves the actual failing timing regression. New
  `infer_with_authority` runs its guard after durable admission and before the
  provider call; expired authority leaves unresolved admission and zero calls.
- Initial HPKE backend introduced an unmaintained proc-macro-error2 dependency;
  replaced with hpke 0.13 and its standard RFC suite, without scanner exceptions.
  `relay-dependencies.log` preserves the failure; final scanner runs separately.
- Browser uploads/downloads sealed job artifacts; local relay_job helper owns
  plaintext preparation/decryption. relay_worker is an explicit local-echo-v1
  probe with private FileControlStore and pinned authority key. It is not a
  useful trained model, generic shell executor or OS sandbox.
- `relay-http-acceptance.log`: two verified mock-issuer tenants enroll separate
  keys, submit jobs, execute through actual outbound TCP with separate histories,
  decrypt their own results and refuse cross-tenant access. Provider counters
  and reopened history prove cached retries do not execute again. Plaintext
  canaries and ciphertext prefixes are absent from the central database.
- Capacity failures, malformed/tampered crypto, full queues, revoked nodes,
  cancellation, foreign results, action mutation and restart ambiguity have
  local acceptance at that checkpoint. The following streaming continuation
  qualifies incremental output and cooperative cancellation; real cross-host partitions, durable Abbey approval consumption, public access,
  key renewal and platform acceptance remain open. No spending, CI resumption,
  live accounts, publication, model training or deployment occurred.

### Relay verification and source receipt

- `relay-final-qualified-gate.log`: complete ABI gate passed on the final relay
  source, with 1,142 Rust test executions, zero Rust failures or ignored tests,
  and 186 Python policy tests (eight existing environment-dependent skips).
  This includes 29 control-plane tests and the required sibling WDBX contracts,
  legacy CLI/MCP checks, dependency scan, Metal checks, documentation and release
  builds. A sibling WDBX unused-import warning remains outside this slice.
- `relay-browser-tests.log`: three browser boundary tests passed, including
  refusing plaintext or unexpected fields before an HTTP submission. Mintlify
  validation (`relay-docs.log`), documentation policy and diff checks passed.
- `relay-result-retry-red.log` and `relay-result-status-red.log` preserve failing
  completion-retry regressions before repair: repeated reports now avoid double
  charging memory and repeated submissions report buffered result availability.
- `relay-dependencies-final.log`: required scanner passed without new exceptions.
  Compared with HEAD, no existing package identity was removed or upgraded;
  84 identities were added, including the earlier account/worker additions.
- `relay-source-manifest.json` and `relay-source.tar.gz` retain the final ABI
  source fingerprints and archive, plus sibling revision/dirty-state metadata.
  They supplement the earlier immutable worker and account receipts; sibling
  metadata does not represent new full Abbey or WDBX qualification.

## Encrypted streaming and cancellation continuation (2026-10-22)

- Extended the enrolled-node protocol with independently HPKE-encrypted,
  node-signed `abi.relay-stream.v1` deltas. Complete binding and sequence are
  authenticated; strict ordering, 16-frame and per-job byte ceilings, global
  ciphertext accounting and exact retry commitments refuse gaps/mutation or
  capacity exhaustion before accepting a frame. Payloads remain transient;
  SQLite stores only commitments, byte counts and lifecycle metadata.
- Native streaming uses a blocking provider task, one-frame async channel,
  acknowledgement backpressure and concurrent 250-ms cancellation inspection.
  Transport failure or runner drop signals cooperative cancellation. Browser
  watch polls encrypted counts/status; snapshots decrypt only with the native
  `open-stream` helper and locally retained node/reply keys.
- Dispatched cancellation records `cancel_requested` and retains active quota.
  Only an authenticated worker `Stopped` acknowledgement after provider return
  records `cancelled`. Local worker history retains a failure-terminal result,
  rather than inventing a signed worker cancellation or successful inference.
  Worker loss, expiry/revocation or restart leaves uncertainty unresolved.
- `stream-cancel-race-red.log` preserves a real TCP race failure: cancellation
  between inspection and frame upload prevented stop acknowledgement. The
  worker now reconciles authority after provider return and on terminal-report
  refusal. Focused acceptance passes for that race and for an idle cooperative
  provider, while an independent tenant completes its encrypted stream.
- `stream-qualified-gate.log` preserves the failing pre-cancelled dispatch
  regression: the provider was invoked once before observing cancellation.
  The worker now checks the token immediately before invocation;
  `stream-precancel-green.log` proves zero calls and a durable failed result.
- Added frame ordering/tampering/duplicate/budget/restart tests, observer refusal
  and pre-cancelled provider tests, and browser watch/logout tests. Independent
  read-only source review found no actionable regression; it did not run tests.
- Interface decision: bounded snapshot polling carries incremental frames; no
  SSE/WebSocket promise is added. Delivery across restart, hard process
  termination, separate-host partitions, governed effects and public operation
  remain open. No live accounts, CI, publication, training or deployment ran.

### Streaming verification and source receipt

- `stream-final-qualified-gate.log`: complete ABI gate passed, 1,149 Rust test
  executions, zero failures/ignored, including 34 control-plane tests and 14
  inference tests. Required dependency scan, legacy CLI/MCP contracts, sibling
  WDBX contracts, local Metal checks, benchmark gate, rustdoc and release builds
  passed. CUDA remains unavailable without nvcc. The pre-existing sibling WDBX
  unused-import warning remains disclosed.
- `stream-browser-tests.log`: five tests passed for sealed-only upload, encrypted
  frame watching through cancellation acknowledgement and stale-watch shutdown.
  `stream-final-policy.log`: all 186 Python policy tests passed with eight existing
  environment-dependent skips. `stream-final-docs.log`: Mintlify validation passed.
  Diff checks passed. The policy suite was rerun after explicitly allowlisting
  the control-plane setup and relay Markdown guides, which the broad Markdown
  ignore rule had previously excluded from source archives.
- Independent review preceded the final gate's pre-cancelled invocation failure;
  that additional bug was repaired and its regression passes in the final gate.
  No new dependencies or lockfile changes were needed for this continuation.
- `stream-source-manifest.json` and `stream-source.tar.gz`: final ABI source,
  modes and hashes, plus current sibling revision/status metadata. These are
  additive receipts, not replacement qualification for earlier source cohorts
  or evidence of new full Abbey/WDBX/platform gates.

## Durable encrypted replay planning continuation (2026-10-22; historical baseline)

- User requested the full continuation plan for durable stream redelivery and
  worker reconnect without duplicate execution. Saved the scoped specification
  at `docs/spec/encrypted-relay-recovery.md` and eight-task implementation plan at
  `docs/superpowers/plans/2026-10-22-encrypted-relay-recovery.md`.
- Proposed boundary: exact synced ciphertext outbox on user infrastructure,
  read-only historical worker lookup, unresolved tombstones for interrupted
  admissions, replay-only current-authority grants, central transient rehydration
  and durable verified client imports. The control plane still persists no
  application payloads. This was the planning baseline; the implemented local
  continuation below supersedes that status.
- Refreshed planning baseline: `reconnect-plan-baseline.log`, complete ABI local
  gate exit 0. This qualifies the existing streaming source, not the proposed
  reconnect feature. Pre-existing unrelated changes remain preserved on main.

## Durable encrypted recovery implementation (2026-10-22)

- Implemented the eight-task recovery continuation in `abi-worker` and
  `abi-control-plane`. Worker historical lookup validates exact recorded scope
  without execution-time admission or a provider. Only replayed prior-process
  admission can become an unresolved tombstone; quota is released without
  readmitting the old job or nonce.
- The private node outbox reserves encoded capacity before admission, syncs
  exact sealed deltas/terminal packets before transport, retains upload metadata
  without premature deletion, and expires ciphertext while keeping identity,
  digest and outcome tombstones. Defaults: 64 MiB database, 1 MiB reservation
  per job, 256 retained identities, 16 deltas plus one terminal. Stable private
  sidecar locking, checksums, bounded pages and fail-closed poison handling are
  exercised. Test-only SQLite VFS wrappers perform real partial writes and
  syncs before injected write/sync errors; full-storage refusal is also tested.
- Durable runtime reconciliation joins outbox and private control history.
  Recorded completion can fill a missing unpublished terminal after checking
  the retained prefix; exact existing ciphertext is reused. Prepared-with-missing
  admission and prior-process in-flight work never invoke the provider during
  recovery. Cancellation observations from inspection and stream replies are
  persisted before local stop signaling.
- Central relay schema v2 stores immutable delivery policy, signed manifests,
  cancellation/lifecycle and packet commitments, while application payloads
  remain transient. Replay-only grants require current ownership, enrollment,
  capability, credential, cancellation and retention checks. Current epoch and
  packet availability are separate from old upload commitments. Publication
  commits new metadata and administrative audit transactionally before exposing
  transient bytes; exact duplicate replay does not double-charge quotas/audits.
- `recover_pending` precedes fresh polling, with fresh signed node proofs,
  bounded single-flight publication, transport classification and capped jittered
  backoff. The native worker explicitly distinguishes fresh journal creation,
  ordinary reopen and stopped legacy-installation outbox-only initialization.
  Missing existing stores never trigger automatic empty-state creation.
- Verified client imports atomically save encrypted packets and contiguous
  cursors after signature/AEAD/scope/prefix/terminal checks. Native durable
  prepare/import/open helpers keep reply keys local and export receipt metadata;
  strict browser durable wrappers/acknowledgements, manual encrypted downloads,
  snapshot recovery status and logout guards are covered by browser tests.
- Focused worker/control-plane suites and warning-denied all-target package
  Clippy passed (`recovery-focused2.log`, `recovery-clippy5.log`): 70
  control-plane tests and 60 worker tests, zero failures/ignored. Actual local
  child-process/TCP acceptance is retained in `recovery-process-tests.log`
  (eight tests), including killed prepared/admitted/partial/completed workers,
  service epochs, lost committed responses and durable invocation counters.
  Browser suite passes eight tests; outbox/client-import focused tests include
  real cursor-sync failure and reopen before rendering. Initial recipe RED
  runs are not claimed where no actual failing receipt was retained.
- Independent review identified three repaired boundaries: legacy v1 views
  bypassing durable delivery authority, cancellation observation/state response
  persistence, and publication audit atomicity. Regressions cover each repair.
- Documentation validation passed: three docs-policy tests; all 186 Python
  policy cases with eight existing environment skips; Mintlify config/page
  validation; and `git diff --check`. Logs: `recovery-doc-policy.log`,
  `recovery-policy.log`, `recovery-docs.log`.
- Final complete `./tools/check.sh`, required scanner and source
  manifest/archive receipts are tracked separately below when completed. The
  implementation and focused successes alone do not assert a new whole-gate
  or immutable three-repository qualification.
- Retention is fixed at admission (default 24 hours; library 60 seconds–seven
  days), independent of execution and credential expiry. One-hour credentials
  have no automatic renewal. User-node control logs may contain plaintext
  results. Outbox expiry is not WDBX memory erasure, backup deletion or physical
  secure erase. No CI resumption, live account/provider test, publication,
  resource purchase, training or deployment occurred in this continuation.
- Local Unix/macOS fixture scope only: storage power loss, separate hosts,
  Windows/Linux runtime, public IdP/deployment, useful trained models, governed
  live effects and independent security assessment remain separate acceptance.

## Remaining release gaps

| Work | Owner | Required acceptance | State |
| --- | --- | --- | --- |
| Current canonical three-repository cohort | ABI/WDBX/Abbey owners | Stable handoff, source fingerprints and all repository gates | In progress; concurrent owner work |
| OIDC accounts, node enrollment, encrypted relay, browser client | ABI | Two verified tenants pair/revoke and execute isolated jobs | Partial: account/enrollment/browser and local encrypted two-tenant jobs; public/separate-host qualification open |
| Outbound transport, scheduler, durable dispatch/governance join | ABI + Abbey | Real reconnect/crash/approval/revocation tests without repeated effect | Partial: durable local inference plus outbound encrypted relay; encrypted streaming, cooperative cancellation and durable local replay/client imports tested; local Abbey approval/tool composition now tested below; separate-host and general platform adapters open |
| Private durable worker history on Windows | ABI | Real Windows ACL and replay tests | Not implemented; explicit refusal |
| Retention, erasure, backups and key custody | WDBX | Owner-defined contracts and replica/retained-generation deletion tests | Owner implementation required |
| Governed desktop/file/process/messaging adapters | ABI + Abbey | Approved/refused real effects on all platforms, designated accounts | Not implemented by this slice |
| Rust transformer, tokenizer, trainer, original weights and evaluation | ABI model pipeline | Reproducible pilots, rights manifest, measured evaluation and runtime | Not implemented; full training needs resource authorization |
| Packages, public operation and independent assessment | Release/operator/external reviewer | Platform install/rollback, separate hosts, recovery and external findings closure | Not performed |

### Final recovery review additions

- Consumer journal format 2 pins the authority key and rejects missing, changed
  or incorrectly signed delivery policies before cursor advancement. Seven
  consumer tests cover the import boundary and reopen. Previously imported output
  stays locally readable after cancellation; new terminal imports are refused.
- Cross-store consistency is checked during construction and reconciliation.
  Missing/admitted history cannot authorize a saved completed result; direct
  execution also refuses a runtime with an unresolved live outcome. Cancellation
  persistence failures remain latched rather than cleared by later responses.
- Eleven reconnect tests pass (`recovery-process-and-fairness-tests.log`), including
  twenty retained jobs, sixteen total exchanges per round, fresh-poll progress,
  relay epoch invalidation, five-second native proof expiry against the existing
  sixty-four-live-proof bound, and contradictory-history refusal. Both observed
  cancellation receipt states persist locally
  (`recovery-cancellation-monitor-tests.log`).

The direct runtime guard regression also passes: an uncertain execution refuses
an independently valid fresh dispatch before outbox preparation or provider
invocation (`service::tests::direct_recovery_guard`).

### Final recovery qualification receipt

- Final `./tools/check.sh` exited **0**, all green:
  `recovery-final-gate3.log`. Stable Rust/cargo 1.99.1 on aarch64-apple-darwin;
  1,198 passing Rust cases across the gate's 74 test summaries, zero Rust
  failures or ignored cases. The control-plane library has 76 passing cases.
  The complete gate includes 186 Python policy cases (eight existing skips),
  required cargo-audit execution, contract/golden validation, size/fmt/Clippy,
  build/test/doc/release stages, sibling WDBX contract tests, Darwin Metal
  tests/docs, and benchmark regression checks. No CUDA compiler/hardware
  acceptance is inferred. Existing sibling WDBX unused-import warning is retained
  in the log; the gate exits successfully.
- Browser final receipt: `recovery-browser-qualified.log`, eight passed.
  Docs-policy final receipt: `recovery-doc-policy-qualified.log`, three passed.
  Mintlify final receipt: `recovery-docs-final.log`, PASS. Final whitespace
  validation and source receipt are retained with these logs.
- `recovery-source-manifest.json` and `recovery-source.tar.gz` capture the final
  tracked and nonignored untracked ABI sources, per-file identities, git status,
  toolchain and sibling revision/status metadata. `recovery-validation.json`
  records command exits and log digests. All files are under
  `/Users/donald/.codex/verification/abi-production-20261022/`.
  Earlier baseline/failure/checkpoint receipts remain intact. Sibling metadata
  does not imply new complete WDBX or Abbey root/desktop qualification.
- All eight recovery tasks are complete for this local source slice. The parent
  ecosystem plan remains open with the release gaps listed above.


## Durable governed-tool continuation (2026-10-22, in progress)

The full ecosystem objective remains active. Contract:
[durable governed tools](../spec/durable-governed-tools.md); implementation and
acceptance plan:
[governed-tool continuation](../superpowers/plans/2026-10-22-durable-governed-tools.md).

- Implemented local source boundaries: shared exact local approval contract;
  sibling Abbey immutable previews/explicit decisions and lifecycle composition;
  startup-owned live host dispatch with fresh full authority after prepare and
  stage 12; tool-only relay runtime, durable worker admission, bounded actual
  tool results and private outbox composition.
- Focused live-host verification passed during this continuation: seven unit,
  27 contract and 39 governed tests, plus package all-target Clippy with warnings
  denied. This receipt predates subsequent integration edits and does not assert
  their final qualification. Control-plane and Abbey focused acceptance and the
  final complete gates/source receipts remain in progress.
- Review rulings: canonical main and unrelated edits preserved; `A1Actor` only
  in the relay bridge, higher levels fail closed; no network decision authority;
  immutable source-bounded preview expiry; current tool permit renewal for the
  same job without re-admission; reconnect preserves pristine local pending
  proposals and refuses sticky cancellation/abandonment. Models are unchanged.
- Initial checkpoint remaining acceptance (later receipts below supersede the native row):
  actual native outbound tool driver, secure browser/native
  local authority UI, concrete scoped platform adapters beyond scratch files,
  full-chain process kills/network loss proving no duplicate effects, complete
  ABI and sibling Abbey gates and immutable source/log receipts. No public,
  separate-host, Windows/Linux, power-loss or external assessment claim is made.

Earlier recovery qualification above covers that prior local inference/replay
slice; it does not qualify this new governed-tool continuation or close the
parent objective.

### Governed execution validation checkpoints

- Baseline `governance-baseline.log` exposed two existing CLI tests racing global
  WDBX overrides. The no-store case now uses the foundation lock and explicitly
  disables persistence; the app fixture explicitly enables its private scratch
  store. Assertions were preserved. Four complete 199-case CLI library runs
  passed after repair. No live user store was opened.
- Focused ABI host validation: seven unit, 27 driver-contract and 39 governed
  cases pass, including actual returned-output validation, revocation/precondition
  changes after prepare/stage 12, and error/panic/lost-terminal uncertainty.
- Control-plane library: 83 cases pass (`governance-cp-tests.log`), including
  durable-only tool admission, tenant/capability isolation, refreshed same-job
  permits after human delay, pending preview reuse, real scratch-file effects,
  cancellation/abandonment and repeated recorded execution without new effects.
- Actual Abbey composition: `governance-abbey-integration.log`, five passed;
  `governance-abbey-unit.log`, 22 passed. The integration uses real Abbey SQLite,
  encrypted tool dispatch and durable worker/outbox stores. Child processes exit
  immediately after prepare and after syncing the real file effect, before
  terminal persistence; reopen preserves interrupted/unresolved accounting and
  the effect counter never increases on retry.
- Independent source review covered exact actor/grant/preview matching, owner
  isolation, fresh pre-effect authority, the two-store crash gap and the native
  scratch assembly. Review repairs added actual result delivery, coordinated byte
  budgets, source-bounded immutable preview expiry, tool-only fresh permits,
  sticky outbox refusal and opaque local preview lookup IDs.
- The complete ABI gate passed (`governance-abi-gate2.log`, exit 0), including the
  required scanner. Its earlier attempt stopped at a new native-helper Clippy
  style error; that receipt remains preserved. WDBX's complete stable-1.99.1
  gate also passed (`governance-wdbx-gate.log`, exit 0). Abbey root/desktop gates
  and the final immutable cohort receipt remain pending at this checkpoint.
- The native Abbey example compiles, passes its offline real-file fixture and
  all-target Clippy. This does not qualify interactive network acceptance,
  Windows/Linux execution, general file/process/desktop/Discord/Twilio adapters,
  designated live accounts or public deployment. Those requirements remain open
  in the governance continuation plan and parent ecosystem plan.

### Native governed relay acceptance

`governance-native-pty-final.log` records a passing actual native TCP/PTY workflow:
real node enrollment and local confirmation, nonterminal approval refusal,
encrypted durable tool submission, exact local preview/approval, one synced
scratch-file effect, explicit encrypted terminal delivery, verified client import,
and identical ciphertext/imported output after both service and worker restart.
Foreign-account and anonymous reads refuse, and the private parameter canary is
absent from central persistence. Accounts/sessions are seeded local fixtures;
this receipt does not qualify public OIDC signup. Native artifact identities are
in `governance-native-artifacts.json`.

The first native attempt timed out waiting for redelivery. The example's full
`JobResult` debug dump was replaced with bounded outcome metadata, and the PTY
harness now drains output while polling. The passing rerun does not independently
establish the sole cause of the earlier timeout. That failure receipt is retained
as `governance-native-pty.log`. The added explicit denial workflow is tracked as
a separate pending acceptance until its final run completes.

Abbey's first whole-gate attempt reached a legacy migration assertion expecting
schema 10; migration 11 requires the coordinated exact-version assertion change.
The focused real-daemon migration test passes. A later mixed-source cohort added
golden commitment regressions that exposed missing canonical-byte and aggregate
hash verification. Actual validation was implemented, preserving those tests;
1,265 Abbey library cases and warning-denied Clippy pass. The strict golden
profile and source identities are recorded in `abbey-golden-validation-receipt.md`.
No source corpus or expected commitment was changed to make the tests pass.

`governance-abbey-gate3.log` completed with exit 0 and `check.sh: OK`,
including default, WDBX, personal-edition and accelerator cohorts, actual installer
rollback and edition coexistence. Windows/Linux cross targets were unavailable.
A subsequent local-authority ownership hardening is being validated separately;
this earlier receipt is retained rather than represented as its final proof.

Final native acceptance `governance-native-denial-final.log` passed with explicit
local denial, no denied admission/intent/effect, persistent denial after both
restarts, exactly one approved file effect, identical ciphertext and client import,
foreign/anonymous refusal and absent central plaintext. The local approval wrapper
also now scopes execution reads/updates and compensation origins to the complete
local owner; 27 governed-store tests, warning-denied Clippy and scoped review pass.

The next full Abbey attempt (`governance-abbey-gate4.log`) exposed a golden-corpus
path alias regression. Raw empty/dot/dotdot path segments are now rejected before
normalization; all seven contract tests and Clippy pass. The failed receipt is
preserved, and final root/desktop gates run sequentially on this final source.

### Final local governed-tool qualification (2026-10-23)

- ABI `./tools/check.sh`: exit 0, `governance-abi-gate3.log`, 1,209 Rust
  executions; required dependency scanner ran, Metal and frozen goldens passed.
- WDBX pinned-toolchain `bash tools/check.sh`: exit 0,
  `governance-wdbx-gate2.log`, 1,136 Rust executions.
- Abbey `./check.sh`: exit 0, `governance-abbey-gate6.log`, 5,884 Rust
  executions across four feature cohorts, 212 Python cases, actual installed
  accelerator layout, rollback and edition coexistence.
- Abbey `./desktop/check.sh`: exit 0, `governance-abbey-desktop-final.log`,
  frontend/build/security/daemon checks and 145 Rust executions.
- Final native TCP/PTY receipt `governance-native-qualified.log`: exit 0,
  exact local approve/deny, one approved effect, no denied admission/intent/effect,
  sticky denial and identical ciphertext/client import after both restarts.
  `governance-native-qualified-artifacts.json` binds the executable hashes.

A concurrent Abbey Metal lifecycle test changed during qualification; its exact
current `--features accel --test daemon_model_lifecycle` target was rebuilt and
both cases pass (`governance-concurrent-metal-test.log`). Concurrent WDBX
specimen-studio edits are preserved and outside this ABI dependency acceptance.
The complete local source archive identifies those dirty files; it is not a
published immutable three-owner handoff and no dependency/CI pins were promoted.
Windows/Linux cross targets and CUDA nvcc were unavailable. Public OIDC login is
not exercised by the seeded-account native receipt. General platform adapters,
separate-host/full-chain fault acceptance, training and public operation remain
open under the owner/acceptance register. Earlier failed receipts remain intact.


### Governed full-chain interruption continuation (2026-10-23)

Task 8 acceptance extends the existing local scratch-file boundary. It uses the
real Abbey local SQLite authority, private worker log, encrypted outbox, native
loopback TCP/PTY assembly and verified consumer journal. No production wire
format, model path or frozen CLI/MCP catalog is changed. Parent-driven hard
kills supplement the older explicit-exit fixtures.

| Interruption point | Acceptance test / mechanism | Required observed recovery |
| --- | --- | --- |
| Preview committed, before decision | Abbey `interruptions::kill_after_preview_persistence` | Exact preview/expiry survive; no admission/effect until explicit local approval |
| Exact decision committed, before admission | Abbey `interruptions::kill_after_approval_persistence` | Exact approved decision survives; one freshly authorized execution |
| Worker admission committed, before prepare | Abbey `interruptions::kill_after_worker_admission` | Unresolved; zero adapter calls; no readmission |
| Approval consumed and intent committed | Abbey `interruptions::kill_after_lifecycle_prepare` | Consumed approval and interrupted intent; zero effects; no retry |
| File effect synced, before lifecycle terminal | Abbey `interruptions::kill_after_synced_effect` | One durable invocation/effect; interrupted/unresolved; no retry |
| Lifecycle terminal committed, before worker completion | Abbey `interruptions::kill_after_lifecycle_terminal` | Lifecycle success retained; worker unresolved; no invented worker result or repeat effect |
| Worker completion committed, before outbox | Abbey `interruptions::kill_after_worker_completion` | Cached completion reconstructs delivery without approval reads or fresh fact refresh |
| Complete outbox committed, before transport | Abbey `interruptions::kill_after_complete_outbox_before_transport` | Exact packet/manifest bytes survive restart |
| Relay publication accepted, response lost | Native smoke plus `governed_response_loss.py` TCP proxy | Upstream successful recorded response precedes dropped socket; worker restart uses durable outbox without fresh approval |
| Consumer import before SQLite commit | ABI `parent_kill_import_transaction_preserves_complete_old_or_new_view` | Previous complete journal/ACK/output survive; retry imports atomically |
| Consumer import after SQLite commit, before ACK | Same ABI parent-kill test, second checkpoint | Complete terminal journal survives; repeated imports preserve exact ACK/journal/output |
| Consumer ACK accepted, response lost | Native smoke TCP proxy | Successful upstream ACK precedes dropped socket; repeated ACKs, including after authority restart, succeed without changing effect/output |

Each Abbey checkpoint publishes a synced marker and is killed/reaped by its
parent; three independent reopen/replay cycles assert durable invocation count
before adapter entry, exact file bytes and modification time, and stable
persisted ciphertext/manifest. Native smoke proves persistent file count and
bytes/inode/mtime, not a native adapter invocation counter. It also verifies the
accepted publication packet hash matches the packet delivered to the consumer,
identical manifest/packet bytes after restart, explicit denial, account isolation,
and absence of private parameter canaries from central SQLite persistence.
Consumer controls reject validly signed conflicting duplicates, foreign account
bindings and changed job bindings without advancing the saved journal.

The first kill-fixture run exposed a fixture lookup using the original `call`
instead of its opaque persisted call ID; the fixture now reads the persisted ID.
The initial native loss run observed that lost publication acknowledgement exits
the demo worker. The final harness explicitly restarts it once and verifies no
new approval; this is restart recovery, not uninterrupted worker availability.
Initial failures are retained separately from final passing receipts.

Focused acceptance passes and independent task/final matrix reviews approve all
twelve rows. Further consumer oracle hardening compares complete imported state
against a separate reference journal and literal terminal output; its scoped
re-review and final ABI gate pass. Task 8 is checked for this composed local
scratch acceptance. Evidence directory:
`/Users/donald/.codex/verification/abi-governed-task8-20261023/`.
General platform adapters, separate-host partitions, public OIDC login,
Windows/Linux execution and power-loss qualification remain open. This local
acceptance does not complete the production ecosystem objective.


#### Final Task 8 qualification receipts

Evidence is retained in
`/Users/donald/.codex/verification/abi-governed-task8-20261023/`:

- ABI `./tools/check.sh`: exit 0, 1,212 Rust executions, zero failed/ignored;
  required dependency scanner ran. `abi-consumer-final-receipt.json` binds the
  exact source hashes and raw final-wave logs. All 1,351 captured ABI source
  identities/modes were unchanged during that gate; subsequent Task 7 planning
  and implementation are a separate source window.
- Abbey `./check.sh < /dev/null`: exit 0, `check.sh: OK`, all default, WDBX,
  personal-edition and accelerator cohorts, Rustdoc, policy/codegen and isolated
  installer/rollback acceptance. `abbey-root-receipt.json` and
  `qualified-source-manifest.json` retain the gate log identity and dirty source
  cohort. The gate started while test-only assertions were being hardened;
  final Task 8 focused tests and warning-denied Clippy were rerun on settled
  source, and their exact hashes remain unchanged at gate completion.
- Abbey `./tools/cargo.sh test --locked --test governed_relay < /dev/null`:
  15 passed, zero failed/ignored, including all eight parent-driven SIGKILL rows.
  Focused warning-denied Clippy passes. Final logs:
  `abbey-governed-kills-20261023-final.log`, `abbey-governed-kill-clippy.log`.
- ABI consumer focused suite: 10 passed; warning-denied package Clippy passes.
  Logs: `abi-consumer-finalwave-tests.log`, `abi-consumer-finalwave-clippy.log`.
  Host/worker focused verification also passes (133 executions, zero failed).
- Native rebuilt TCP/PTY acceptance passes:
  `abbey-governed-network-loss.log` and `abbey-governed-network-receipt.json`
  bind successful response-loss assertions to all four executable hashes.
  Wrapper rebuild commands and complete invocation are retained in
  `abi-governed-network-report.md`.
- Independent sibling scoped review and final twelve-point matrix review pass;
  ABI consumer/reopen reviews and the final independent-oracle re-review pass.
  Review reports, earlier failed fixtures, fix-wave logs and initial native
  response-loss failure are retained alongside passing receipts.
- Documentation policy suites, `git diff --check`, and
  `.agents/skills/docs-validate/validate.sh` pass; final logs are retained in the
  same directory. No CLI/MCP golden, protocol or production execution behavior
  was changed by this acceptance continuation.

`acceptance-source-sha256.json` and `owned-source/` preserve final test source;
`starting-source-manifest.json` and `qualified-source-manifest.json` retain
tracked/untracked identities, modes, dirty status and exact heads for ABI, Abbey
and WDBX. Concurrent unrelated metadata-PTY and specimen-studio edits were
preserved, not attributed to this slice. This is local dirty-source qualification,
not a published immutable three-owner cohort or CI pin promotion. Windows/Linux
cross targets and CUDA nvcc were unavailable; the existing sibling
`AdmissionSimulator` unused-import warning remains disclosed. General adapters,
separate-host partitions, power-loss acceptance and the parent ecosystem remain
open. No commit, CI resumption, live account action or deployment occurred.
